"""How BERT's general WordPiece vocabulary and FinBERT's finance vocabulary split the tweets.

Usage: python src/tokenizer_stats.py
"""
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, logging

from common import RESULTS, load_twitter, normalize, tokenize

logging.set_verbosity_error()
EXAMPLE = "Autodesk downgraded to underweight from neutral at JPMorgan"


def main():
    bert = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    finbert = BertTokenizer.from_pretrained("yiyanghkust/finbert-pretrain", do_lower_case=True)
    train, _, _ = load_twitter()
    texts = [normalize(t) for t in train.text]

    lines = [f"Vocabulary size: BERT {len(bert):,}, FinBERT {len(finbert):,}"]
    for name, tok in [("BERT", bert), ("FinBERT", finbert)]:
        n_sub = np.array([len(tok.tokenize(t)) for t in texts])
        words = [w for t in texts for w in tokenize(t) if w.isalpha()]
        split = np.mean([len(tok.tokenize(w)) > 1 for w in words])
        lines.append(f"{name:<8} sub-words per tweet: mean {n_sub.mean():.1f}, median {np.median(n_sub):.0f};  "
                     f"alphabetic words split into >1 piece: {split:.1%} of {len(words):,}")
    lines.append(f"\nExample: {EXAMPLE}")
    lines.append(f"  BERT    ({len(bert.tokenize(EXAMPLE))}): {' '.join(bert.tokenize(EXAMPLE))}")
    lines.append(f"  FinBERT ({len(finbert.tokenize(EXAMPLE))}): {' '.join(finbert.tokenize(EXAMPLE))}")
    for w in ["downgraded", "underweight", "overweight", "bearish", "bullish", "dividend", "premarket", "buyback",
              "ebitda", "jpmorgan", "blackrock", "goldman"]:
        lines.append(f"  {w:<12} BERT: {' '.join(bert.tokenize(w)):<24} FinBERT: {' '.join(finbert.tokenize(w))}")
    (RESULTS / "tokenizer_stats.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
