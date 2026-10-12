"""CNN (Kim, 2014) and BiLSTM text classifiers on GloVe word embeddings.

Variants
  cnn-rand       : embeddings randomly initialised and learned
  cnn-static     : GloVe 100d, frozen
  cnn-nonstatic  : GloVe 100d, fine-tuned
  bilstm         : GloVe 100d fine-tuned -> 1-layer BiLSTM(128) -> max-pool over time

Usage: python src/run_neural.py [--models cnn-rand cnn-static cnn-nonstatic bilstm] [--seeds 1 2 3] [--frac 1.0]
"""
import argparse
import copy
import gzip
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common import eval_sets, evaluate_all, load_twitter, metrics, normalize, tokenize

torch.set_num_threads(4)
DEVICE = "cpu"  # small models; keeps the GPU free for the BERT runs
GLOVE = Path.home() / ".cache/glove/glove-wiki-gigaword-100.gz"
EMB_DIM, MAX_LEN = 100, 50


# ---------------------------------------------------------------- vocabulary & embeddings
def build_vocab(texts):
    counts = Counter(tok for t in texts for tok in tokenize(normalize(t)))
    itos = ["<pad>", "<unk>"] + sorted(counts)
    return {w: i for i, w in enumerate(itos)}


def load_glove(vocab):
    """Return (matrix, coverage). Words missing from GloVe get U(-0.25, 0.25) like in Kim (2014)."""
    rng = np.random.default_rng(0)
    mat = rng.uniform(-0.25, 0.25, (len(vocab), EMB_DIM)).astype(np.float32)
    mat[0] = 0.0
    found = 0
    with gzip.open(GLOVE, "rt", encoding="utf8") as f:
        next(f)  # header: "<n_words> <dim>"
        for line in f:
            word, rest = line.split(" ", 1)
            if word in vocab:
                mat[vocab[word]] = np.asarray(rest.split(), dtype=np.float32)
                found += 1
    return torch.tensor(mat), found / (len(vocab) - 2)


def encode(texts, vocab):
    ids = [[vocab.get(tok, 1) for tok in tokenize(normalize(t))][:MAX_LEN] or [1] for t in texts]
    lengths = torch.tensor([len(x) for x in ids])
    out = torch.zeros(len(ids), max(5, int(lengths.max())), dtype=torch.long)  # >=5 so the width-5 filters fit
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x)
    return out, lengths


# ---------------------------------------------------------------- models
class KimCNN(nn.Module):
    def __init__(self, emb, freeze, n_filters=100, sizes=(3, 4, 5), n_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb.clone(), freeze=freeze, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb.shape[1], n_filters, k) for k in sizes)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_filters * len(sizes), n_classes)

    def forward(self, x, lengths):
        e = self.emb(x).transpose(1, 2)                                      # (B, D, T)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]   # max-over-time pooling
        return self.out(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, emb, hidden=128, n_classes=3, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb.clone(), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(emb.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.drop(self.emb(x)), lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, padding_value=-1e4)
        return self.out(self.drop(h.max(dim=1).values))                     # max-pool over time steps


def build(name, glove, rand_emb):
    if name == "cnn-rand":
        return KimCNN(rand_emb, freeze=False)
    if name == "cnn-static":
        return KimCNN(glove, freeze=True)
    if name == "cnn-nonstatic":
        return KimCNN(glove, freeze=False)
    if name == "bilstm":
        return BiLSTM(glove)
    raise ValueError(name)


# ---------------------------------------------------------------- training
def run(name, seed, frac, vocab, glove, sets, epochs=15, patience=3, batch_size=50, lr=1e-3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train, _, _ = load_twitter()
    if frac < 1.0:
        train, _ = train_test_split(train, train_size=frac, stratify=train["label"], random_state=seed)
    rand_emb = torch.empty_like(glove).uniform_(-0.25, 0.25)
    model = build(name, glove, rand_emb).to(DEVICE)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X, L = encode(train["text"].tolist(), vocab)
    y = torch.tensor(train["label"].values)

    @torch.no_grad()
    def predict(texts):
        model.eval()
        xb, lb = encode(texts, vocab)
        return model(xb.to(DEVICE), lb).argmax(-1).cpu().numpy()

    best_f1, best_state, best_epoch, bad, history = -1, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            lb = L[idx]
            xb = X[idx][:, :max(5, int(lb.max()))]
            loss = loss_fn(model(xb.to(DEVICE), lb), y[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        dev_f1 = metrics(sets["dev"]["label"], predict(sets["dev"]["text"].tolist()))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(y), "dev_macro_f1": dev_f1})
        if dev_f1 > best_f1:
            best_f1, best_epoch, bad = dev_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:  # early stopping on dev macro-F1
                break
    model.load_state_dict(best_state)
    tag = name if frac == 1.0 else f"{name}_frac{frac}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    evaluate_all(tag, seed, predict, sets, extra={
        "best_epoch": best_epoch, "history": history, "train_time_s": time.time() - t0,
        "n_train": len(train), "trainable_params": n_params})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cnn-rand", "cnn-static", "cnn-nonstatic", "bilstm"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    sets = eval_sets()
    train, _, _ = load_twitter()
    vocab = build_vocab(train["text"])  # vocabulary from the full training split
    glove, coverage = load_glove(vocab)
    print(f"vocab={len(vocab)}  GloVe coverage={coverage:.1%}", flush=True)
    for name in args.models:
        for seed in args.seeds:
            run(name, seed, args.frac, vocab, glove, sets)


if __name__ == "__main__":
    main()
