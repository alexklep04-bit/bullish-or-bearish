"""Case study: test tweets that mention big banks (JPMorgan, Goldman Sachs, BlackRock, ...).

The role of the bank in each of the 64 such test tweets was labelled by hand
(data/bank_mentions_test_roles.csv, made by reading every tweet):
  messenger  - the bank rates, forecasts or trades ANOTHER company or the market
               ("Netflix downgraded to underperform at Wells Fargo")
  subject    - the news is about the bank itself ("JPMorgan expands private credit group")
  incidental - the bank is only mentioned in passing ("Alliance Data taps new CEO from Citi")

Reports (1) how much of the directional signal in these tweets is actually about
the bank and (2) how each model does on them compared with the rest of the test set.

Usage: python src/case_study_banks.py
"""
import numpy as np
import pandas as pd

from common import DATA, LABELS, RESULTS, SEEDS, load_twitter, metrics

MODELS = [  # (result key, display name)
    ("lexicon_LM+regex", "LM lexicon + regex"),
    ("tfidf_lr[norm+char]", "TF-IDF + LogReg"),
    ("cnn-nonstatic", "CNN-non-static"),
    ("bilstm", "BiLSTM"),
    ("bert", "BERT-base"),
    ("finbert", "FinBERT"),
]


def pred_file(key, seed):
    for s in (seed, 0):  # deterministic models only have seed 0
        p = RESULTS / f"predictions/{key}__seed{s}__test.csv"
        if p.exists():
            return p
    return None


def main():
    roles = pd.read_csv(DATA / "bank_mentions_test_roles.csv")
    _, _, test = load_twitter()
    assert (test.loc[roles.test_index, "text"].values == roles.text.values).all(), "role file out of sync with test set"
    is_bank = np.zeros(len(test), dtype=bool)
    is_bank[roles.test_index] = True

    lines = [f"Test tweets mentioning one of 11 big banks: {len(roles)} of {len(test)} ({len(roles) / len(test):.1%})",
             pd.crosstab(roles.role, roles.label_name, margins=True).to_string()]
    directional = roles[roles.label_name != "Neutral"]
    about_bank = (directional.role == "subject").sum()
    lines.append(f"\nDirectional (Bullish/Bearish) bank tweets: {len(directional)}; about the bank itself: {about_bank} "
                 f"({about_bank / len(directional):.0%}); bank only the messenger: "
                 f"{(directional.role == 'messenger').sum()} ({(directional.role == 'messenger').mean():.0%})")
    lines.append("Mentions per bank: " + ", ".join(f"{b} {n}" for b, n in
                                                     roles.bank.str.split(", ").explode().value_counts().items()))

    rows, examples = [], roles[["test_index", "bank", "role", "label_name", "text"]].copy()
    for key, name in MODELS:
        acc_b, f1_b, acc_r, f1_r = [], [], [], []
        for seed in SEEDS:
            p = pred_file(key, seed)
            if p is None:
                continue
            pred = pd.read_csv(p).pred.values
            acc_b.append(metrics(test.label[is_bank], pred[is_bank])["accuracy"])
            f1_b.append(metrics(test.label[is_bank], pred[is_bank])["macro_f1"])
            acc_r.append(metrics(test.label[~is_bank], pred[~is_bank])["accuracy"])
            f1_r.append(metrics(test.label[~is_bank], pred[~is_bank])["macro_f1"])
            if seed == 1 or "seed0" in p.name:
                examples[name] = [LABELS[x] for x in pred[roles.test_index]]
            if "seed0" in p.name:
                break
        if acc_b:
            rows.append({"Model": name, "Runs": len(acc_b),
                         "Bank tweets acc": f"{100 * np.mean(acc_b):.1f}", "Bank tweets F1": f"{100 * np.mean(f1_b):.1f}",
                         "Other tweets acc": f"{100 * np.mean(acc_r):.1f}", "Other tweets F1": f"{100 * np.mean(f1_r):.1f}"})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "table_case_study_banks.csv", index=False)
    examples.to_csv(RESULTS / "case_study_banks_predictions.csv", index=False)
    lines += ["\nModel accuracy / macro-F1 on bank tweets vs. the rest of the test set (mean over seeds)",
              table.to_string(index=False)]
    (RESULTS / "case_study_banks.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
