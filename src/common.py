"""Shared utilities: data loading, regex preprocessing, splits and metrics.

Label scheme used everywhere: 0 = Bearish, 1 = Bullish, 2 = Neutral
(the native scheme of the Twitter dataset; PhraseBank is mapped onto it).
"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LABELS = ["Bearish", "Bullish", "Neutral"]
PB_MAP = {"negative": 0, "positive": 1, "neutral": 2}
SEEDS = [1, 2, 3]

# ---------------------------------------------------------------- regex preprocessing
URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,6}(?:\.[A-Za-z])?\b")
PCT_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?\s?%")
NUM_RE = re.compile(r"[-+]?\$?\d+(?:[.,]\d+)*")
SPACE_RE = re.compile(r"\s+")


def normalize(text: str, numbers: bool = False) -> str:
    """Regex normalisation of tweet noise.

    URLs, @user mentions and $CASHTAGS carry no sentiment, so they are replaced
    by placeholder tokens. With numbers=True percentages and numbers are also
    collapsed into <pct>/<num> (used only in the ablation study).
    """
    text = html.unescape(text)
    text = URL_RE.sub(" <url> ", text)
    text = USER_RE.sub(" <user> ", text)
    text = CASHTAG_RE.sub(" <ticker> ", text)
    if numbers:
        text = PCT_RE.sub(" <pct> ", text)
        text = NUM_RE.sub(" <num> ", text)
    return SPACE_RE.sub(" ", text).strip()


TOKEN_RE = re.compile(r"<\w+>|[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*%?|[^\sa-z\d]")


def tokenize(text: str) -> list[str]:
    """Regex word tokenizer for the non-transformer models."""
    return TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------- data
def load_twitter(dev_size: float = 0.15, seed: int = 42):
    """Official train split -> train/dev (stratified); official 'valid' split -> test."""
    full = pd.read_csv(DATA / "twitter_fin/sent_train.csv")
    test = pd.read_csv(DATA / "twitter_fin/sent_valid.csv")
    train, dev = train_test_split(full, test_size=dev_size, stratify=full["label"], random_state=seed)
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


def load_phrasebank(agree: str = "50Agree") -> pd.DataFrame:
    rows = []
    path = DATA / f"phrasebank/FinancialPhraseBank-v1.0/Sentences_{agree}.txt"
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                sent, lab = line.rsplit("@", 1)
                rows.append((sent.strip(), PB_MAP[lab.strip()]))
    return pd.DataFrame(rows, columns=["text", "label"])


def eval_sets():
    """All evaluation sets: in-domain dev/test + out-of-domain PhraseBank."""
    _, dev, test = load_twitter()
    return {
        "dev": dev,
        "test": test,
        "pb_50": load_phrasebank("50Agree"),
        "pb_all": load_phrasebank("AllAgree"),
    }


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    directional = np.isin(y_true, [0, 1])
    # "sign error": a truly bullish item called bearish or vice versa --
    # the most expensive mistake if the output were used as a trading signal
    opposite = ((y_true == 0) & (y_pred == 1)) | ((y_true == 1) & (y_pred == 0))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "f1_bearish": f[0], "f1_bullish": f[1], "f1_neutral": f[2],
        "prec_bearish": p[0], "prec_bullish": p[1],
        "rec_bearish": r[0], "rec_bullish": r[1],
        "sign_error_rate": opposite.sum() / max(directional.sum(), 1),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def save_result(model: str, seed, scores: dict, extra: dict | None = None):
    """One JSON per (model, seed); collected later by make_report_assets.py."""
    out = {"model": model, "seed": seed, "scores": scores, **(extra or {})}
    name = f"{model}__seed{seed}.json".replace("/", "_")
    with open(RESULTS / name, "w") as f:
        json.dump(out, f, indent=1, default=float)


def save_predictions(model: str, seed, split: str, texts, y_true, y_pred):
    pred_dir = RESULTS / "predictions"
    pred_dir.mkdir(exist_ok=True)
    pd.DataFrame({"text": texts, "label": y_true, "pred": y_pred}).to_csv(
        pred_dir / f"{model}__seed{seed}__{split}.csv".replace("/", "_"), index=False)


def evaluate_all(model: str, seed, predict_fn, sets: dict, extra: dict | None = None, verbose=True):
    """Run predict_fn(list_of_texts) on every eval set, save scores + test predictions."""
    scores = {}
    for split, df in sets.items():
        y_pred = predict_fn(df["text"].tolist())
        scores[split] = metrics(df["label"], y_pred)
        if split == "test":
            save_predictions(model, seed, split, df["text"], df["label"], y_pred)
    save_result(model, seed, scores, extra)
    if verbose:
        s = scores
        print(f"{model:<28} seed={seed}  dev F1={s['dev']['macro_f1']:.3f}  "
              f"test acc={s['test']['accuracy']:.3f} F1={s['test']['macro_f1']:.3f}  "
              f"PB50 F1={s['pb_50']['macro_f1']:.3f}  PBall F1={s['pb_all']['macro_f1']:.3f}", flush=True)
    return scores
