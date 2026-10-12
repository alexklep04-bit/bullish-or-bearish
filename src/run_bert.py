"""Fine-tune BERT-base and FinBERT on Bullish/Bearish/Neutral tweets.

Usage:
  python src/run_bert.py --model bert     [--seeds 1 2 3] [--frac 1.0]
  python src/run_bert.py --model finbert  [--seeds 1 2 3] [--frac 1.0]
  python src/run_bert.py --model prosus-zeroshot   # off-the-shelf FinBERT, no training
"""
import argparse
import copy
import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification,
                          BertTokenizer, get_linear_schedule_with_warmup, logging)

from common import eval_sets, evaluate_all, load_twitter, metrics, normalize

logging.set_verbosity_error()
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

CHECKPOINTS = {
    "bert": "google-bert/bert-base-uncased",          # general-domain BERT (Devlin et al., 2019)
    "finbert": "yiyanghkust/finbert-pretrain",        # BERT pre-trained on 4.9B tokens of financial text (Yang et al., 2020)
    "prosus-zeroshot": "ProsusAI/finbert",            # FinBERT already fine-tuned on PhraseBank (Araci, 2019)
}


def load(name: str, num_labels: int = 3):
    ckpt = CHECKPOINTS[name]
    if name == "finbert":
        # this checkpoint ships only vocab.txt and an old-style config -> load with explicit BERT classes
        tok = BertTokenizer.from_pretrained(ckpt, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(ckpt, num_labels=num_labels)
    else:
        tok = AutoTokenizer.from_pretrained(ckpt)
        kw = {} if name == "prosus-zeroshot" else {"num_labels": num_labels}
        model = AutoModelForSequenceClassification.from_pretrained(ckpt, **kw)
    return tok, model.to(DEVICE)


def make_predict_fn(tok, model, label_map=None, batch_size=128):
    @torch.no_grad()
    def predict(texts):
        model.eval()
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = [normalize(t) for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            preds.append(model(**enc).logits.argmax(-1).cpu().numpy())
        preds = np.concatenate(preds)
        return np.array([label_map[p] for p in preds]) if label_map else preds
    return predict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finetune(name, seed, frac, epochs, lr, batch_size, sets):
    set_seed(seed)
    train, _, _ = load_twitter()
    if frac < 1.0:
        train, _ = train_test_split(train, train_size=frac, stratify=train["label"], random_state=seed)
    tok, model = load(name)

    texts = [normalize(t) for t in train["text"]]
    labels = train["label"].tolist()

    def collate(idx):
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in idx])
        return enc

    loader = DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    total = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    predict = make_predict_fn(tok, model)

    best_f1, best_state, best_epoch, history = -1, None, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
        dev_f1 = metrics(sets["dev"]["label"], predict(sets["dev"]["text"].tolist()))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running / len(loader), "dev_macro_f1": dev_f1})
        print(f"  [{name} seed={seed} frac={frac}] epoch {epoch}: loss={running / len(loader):.4f} dev F1={dev_f1:.4f}",
              flush=True)
        if dev_f1 > best_f1:  # model selection on dev only
            best_f1, best_epoch = dev_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    tag = name if frac == 1.0 else f"{name}_frac{frac}"
    evaluate_all(tag, seed, predict, sets, extra={
        "best_epoch": best_epoch, "history": history, "train_time_s": train_time,
        "n_train": len(train), "hparams": {"lr": lr, "epochs": epochs, "batch_size": batch_size, "max_len": MAX_LEN},
    })
    del model, best_state
    if DEVICE == "mps":
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CHECKPOINTS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training data (learning curve)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    sets = eval_sets()
    if args.model == "prosus-zeroshot":
        tok, model = load(args.model)
        # ProsusAI head: 0=positive, 1=negative, 2=neutral  ->  ours: 1=Bullish, 0=Bearish, 2=Neutral
        evaluate_all("finbert-prosus-zeroshot", 0, make_predict_fn(tok, model, label_map={0: 1, 1: 0, 2: 2}), sets,
                     extra={"note": "no training on tweets; PhraseBank scores are NOT out-of-domain (model was trained on it)"})
        return

    # smaller training subsets get more epochs so the number of updates stays reasonable
    # (100%/50%/25% -> 4 epochs, 10% -> 10 epochs)
    epochs = args.epochs or min(10, max(4, round(1 / args.frac)))
    for seed in args.seeds:
        finetune(args.model, seed, args.frac, epochs, args.lr, args.batch_size, sets)


if __name__ == "__main__":
    main()
