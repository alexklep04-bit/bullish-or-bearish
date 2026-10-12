"""Exploratory look at the two financial sentiment datasets.

Usage: python3 explore_data.py <data_dir>
"""
import re
import sys
from pathlib import Path

import pandas as pd

data_dir = Path(sys.argv[1])

# ---------- Twitter Financial News Sentiment (zeroshot, MIT) ----------
TW_LABELS = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
tw_train = pd.read_csv(data_dir / "twitter_fin/sent_train.csv")
tw_valid = pd.read_csv(data_dir / "twitter_fin/sent_valid.csv")
for df in (tw_train, tw_valid):
    df["sentiment"] = df["label"].map(TW_LABELS)
tw = pd.concat([tw_train.assign(split="train"), tw_valid.assign(split="valid")])

print("=" * 70)
print("TWITTER FINANCIAL NEWS SENTIMENT")
print("=" * 70)
print(f"train = {len(tw_train)}, valid = {len(tw_valid)}, total = {len(tw)}")
print(pd.crosstab(tw["sentiment"], tw["split"], margins=True))
print("\nClass share (%):")
print((tw["sentiment"].value_counts(normalize=True) * 100).round(1))

tw["n_words"] = tw["text"].str.split().str.len()
print("\nWords per tweet:", tw["n_words"].describe()[["mean", "50%", "max"]].round(1).to_dict())

# regex-based surface statistics (ties in with the regex lecture)
cashtag = re.compile(r"\$[A-Z]{1,5}\b")
url = re.compile(r"https?://\S+")
pct = re.compile(r"[-+]?\d+(?:\.\d+)?%")
print("\nRegex surface features (share of tweets containing):")
print(f"  cashtag ($TICKER): {tw['text'].str.contains(cashtag).mean():.1%}")
print(f"  URL:               {tw['text'].str.contains(url).mean():.1%}")
print(f"  percentage:        {tw['text'].str.contains(pct).mean():.1%}")

tickers = tw["text"].str.findall(cashtag).explode().dropna()
print("\nTop 15 cashtags:", tickers.value_counts().head(15).to_dict())

# company names are matched case-insensitively, tickers case-sensitively
# (otherwise "GS" would also match "earninGS", "holdinGS", ...)
companies = {
    "JPMorgan": r"(?i:jp ?morgan|j\.p\. morgan)|\bJPM\b",
    "BlackRock": r"(?i:blackrock)|\bBLK\b",
    "Goldman": r"(?i:goldman)|\$GS\b",
    "Tesla": r"(?i:tesla)|\bTSLA\b",
}
for name, pattern in companies.items():
    m = tw["text"].str.contains(pattern, regex=True)
    print(f"Tweets mentioning {name:<10}: {m.sum():>4}  ->  {tw.loc[m, 'sentiment'].value_counts().to_dict()}")

print("\nRandom examples per class:")
for lab in ["Bullish", "Bearish", "Neutral"]:
    print(f"\n  [{lab}]")
    for t in tw[tw["sentiment"] == lab].sample(4, random_state=7)["text"]:
        print("   -", t[:140])

# ---------- Financial PhraseBank (Malo et al., 2014) ----------
print("\n" + "=" * 70)
print("FINANCIAL PHRASEBANK")
print("=" * 70)
pb_dir = data_dir / "phrasebank/FinancialPhraseBank-v1.0"
for agree in ["50Agree", "66Agree", "75Agree", "AllAgree"]:
    rows = []
    with open(pb_dir / f"Sentences_{agree}.txt", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if "@" in line:
                sent, lab = line.rsplit("@", 1)
                rows.append((sent.strip(), lab.strip()))
    pb = pd.DataFrame(rows, columns=["text", "sentiment"])
    dist = pb["sentiment"].value_counts().to_dict()
    print(f"{agree:<9} n={len(pb):>5}  {dist}")

print("\nExamples (AllAgree):")
for lab in ["positive", "negative", "neutral"]:
    for t in pb[pb["sentiment"] == lab].sample(2, random_state=3)["text"]:
        print(f"  [{lab}] {t[:140]}")
