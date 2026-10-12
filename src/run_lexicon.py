"""Rule-based baselines (no machine learning).

1. Loughran-McDonald (LM) finance lexicon: score = #positive - #negative words
   (positive words preceded by a negation within 3 tokens count as negative,
   following Loughran & McDonald, 2011).
2. LM + hand-written finance regex rules: analyst actions (upgrade/downgrade),
   guidance/target changes, price moves, and a numeric rule that compares
   "to $X from $Y" to decide the direction of a revision.

Rules were written by inspecting the *training* split only.
"""
import re

import numpy as np
import pandas as pd

from common import DATA, eval_sets, evaluate_all, load_twitter, metrics, normalize, tokenize

lm = pd.read_csv(DATA / "lexicon/LM.csv")
LM_POS = set(lm.loc[lm["Positive"] > 0, "Word"].str.lower())
LM_NEG = set(lm.loc[lm["Negative"] > 0, "Word"].str.lower())
NEGATIONS = {"no", "not", "never", "none", "neither", "nor", "nobody", "isn't", "wasn't", "aren't", "don't",
             "doesn't", "didn't", "won't", "can't", "cannot", "without"}

OBJ = r"(?:price target|target|pt|guidance|outlook|forecasts?|estimates?|dividend|rating|stake|payout)"
BULLISH_RULES = [
    r"\bupgrad(?:e|es|ed|ing)\b",
    rf"\b(?:rais(?:e|es|ed|ing)|lift(?:s|ed)?|boost(?:s|ed)?|hik(?:e|es|ed)|increas(?:e|es|ed))\b[^.]{{0,30}}\b{OBJ}",
    rf"\b{OBJ}\s+(?:raised|lifted|boosted|hiked|increased)\b",
    r"\b(?:beat|beats|tops|topped|exceed(?:s|ed)?|surpass(?:es|ed)?)\b",
    r"\b(?:jump(?:s|ed)?|soar(?:s|ed|ing)?|surg(?:e|es|ed|ing)|rall(?:y|ies|ied)|climb(?:s|ed)?|ris(?:e|es|ing)|rose"
    r"|gain(?:s|ed)?|advanc(?:e|es|ed)|rebound(?:s|ed)?|bounc(?:e|es|ed)|spik(?:e|es|ed)|higher|rocket(?:s|ed)?)\b",
    r"\bup \d+(?:\.\d+)?\s?%",
    r"\b(?:outperform|overweight|buy rating|bullish|upbeat|optimism|record high|all-time high|new highs?)\b",
]
BEARISH_RULES = [
    r"\bdowngrad(?:e|es|ed|ing)\b",
    rf"\b(?:cut(?:s|ting)?|lower(?:s|ed)?|slash(?:es|ed)?|reduc(?:e|es|ed)|trim(?:s|med)?)\b[^.]{{0,30}}\b{OBJ}",
    rf"\b{OBJ}\s+(?:cut|lowered|slashed|reduced|trimmed)\b",
    r"\b(?:miss|misses|missed)\b",
    r"\b(?:fall(?:s|ing)?|fell|drop(?:s|ped)?|plung(?:e|es|ed)|tumbl(?:e|es|ed)|sink(?:s)?|sank|slid(?:e|es)?"
    r"|slump(?:s|ed)?|slip(?:s|ped)?|declin(?:e|es|ed)|retreat(?:s|ed)?|crash(?:es|ed)?|lower|dump(?:s|ed|ing)?)\b",
    r"\bdown \d+(?:\.\d+)?\s?%",
    r"\b(?:underperform|underweight|sell rating|bearish|warns?|warning|recession|lawsuit|sues|probe|layoffs?|fears?)\b",
]
BULL_RE = [re.compile(p, re.I) for p in BULLISH_RULES]
BEAR_RE = [re.compile(p, re.I) for p in BEARISH_RULES]
# "price target raised to $40 from $32" -> compare the two numbers
REVISION_RE = re.compile(r"\bto \$?(\d+(?:\.\d+)?) from \$?(\d+(?:\.\d+)?)", re.I)


def lm_score(text: str) -> int:
    toks = tokenize(text)
    score = 0
    for i, t in enumerate(toks):
        if t in LM_POS:
            negated = any(w in NEGATIONS for w in toks[max(0, i - 3):i])
            score += -1 if negated else 1
        elif t in LM_NEG:
            score -= 1
    return score


def rule_score(text: str) -> int:
    score = sum(bool(r.search(text)) for r in BULL_RE) - sum(bool(r.search(text)) for r in BEAR_RE)
    m = REVISION_RE.search(text)
    if m:
        new, old = float(m.group(1)), float(m.group(2))
        score += 2 * int(np.sign(new - old))
    return score


def to_label(score: float) -> int:
    return 1 if score > 0 else 0 if score < 0 else 2


def make_predict(rule_weight: float | None):
    def predict(texts):
        out = []
        for t in texts:
            t = normalize(t)
            s = lm_score(t)
            if rule_weight is not None:
                s += rule_weight * rule_score(t)
            out.append(to_label(s))
        return np.array(out)
    return predict


def main():
    sets = eval_sets()
    train, _, _ = load_twitter()
    evaluate_all("lexicon_LM", 0, make_predict(None), sets)

    # weight of regex rules vs. LM words, chosen on the training split
    best = max([1, 2, 3, 5], key=lambda w: metrics(train["label"], make_predict(w)(train["text"].tolist()))["macro_f1"])
    print(f"chosen rule weight (on train): {best}")
    evaluate_all("lexicon_LM+regex", 0, make_predict(best), sets, extra={"rule_weight": best})


if __name__ == "__main__":
    main()
