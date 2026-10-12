"""TF-IDF + logistic regression, with a preprocessing ablation.

Variants
  raw        : lower-cased text, no regex normalisation
  norm       : URLs / @users / $TICKERS replaced by placeholder tokens
  norm+num   : additionally numbers and percentages -> <num> / <pct>
  norm+char  : 'norm' word 1-2 grams + character 2-5 grams
C and class weighting are tuned on the dev split.
"""
import argparse
import json
from functools import partial

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import LABELS, RESULTS, eval_sets, evaluate_all, load_twitter, metrics, normalize, tokenize

VARIANTS = {
    "raw": dict(prep=lambda t: t, char=False),
    "norm": dict(prep=normalize, char=False),
    "norm+num": dict(prep=partial(normalize, numbers=True), char=False),
    "norm+char": dict(prep=normalize, char=True),
}


class TfidfModel:
    def __init__(self, prep, char, C, class_weight):
        self.prep = prep
        self.vecs = [TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                     ngram_range=(1, 2), min_df=2, sublinear_tf=True)]
        if char:
            self.vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True))
        self.clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000)

    def _x(self, texts, fit=False):
        texts = [self.prep(t) for t in texts]
        mats = [v.fit_transform(texts) if fit else v.transform(texts) for v in self.vecs]
        return hstack(mats).tocsr()

    def fit(self, texts, y):
        self.clf.fit(self._x(texts, fit=True), y)
        return self

    def predict(self, texts):
        return self.clf.predict(self._x(texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training data (learning curve)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets = eval_sets()
    train, dev, _ = load_twitter()
    if args.frac < 1.0:
        train, _ = train_test_split(train, train_size=args.frac, stratify=train["label"], random_state=args.seed)
    for name in args.variants:
        cfg = VARIANTS[name]
        grid = [(C, cw) for C in [0.5, 1, 2, 5, 10, 20] for cw in [None, "balanced"]]
        scored = []
        for C, cw in grid:
            m = TfidfModel(cfg["prep"], cfg["char"], C, cw).fit(train["text"].tolist(), train["label"])
            scored.append((metrics(dev["label"], m.predict(dev["text"].tolist()))["macro_f1"], C, cw, m))
        f1, C, cw, best = max(scored, key=lambda s: s[0])
        tag = f"tfidf_lr[{name}]" if args.frac == 1.0 else f"tfidf_lr[{name}]_frac{args.frac}"
        evaluate_all(tag, args.seed, best.predict, sets, extra={"C": C, "class_weight": cw, "n_train": len(train)})

        if name == "norm" and args.frac == 1.0:  # most informative n-grams per class, for the report
            feats = np.array(best.vecs[0].get_feature_names_out())
            top = {LABELS[k]: feats[np.argsort(-best.clf.coef_[k])[:25]].tolist() for k in range(3)}
            with open(RESULTS / "tfidf_top_features.json", "w") as f:
                json.dump(top, f, indent=1)
            for k, v in top.items():
                print(f"  top {k}: {', '.join(v[:15])}")


if __name__ == "__main__":
    main()
