#!/usr/bin/env bash
# Download the third-party data used in this project. It is not stored in the repository:
# see data/README.md for sources and licences.
#
# Usage: ./get_data.sh [data_dir]      (default: ./data)
set -euo pipefail

DATA="${1:-$(cd "$(dirname "$0")" && pwd)/data}"
HF="https://huggingface.co/datasets"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DATA/twitter_fin" "$DATA/phrasebank" "$DATA/lexicon"

echo "1/4 Twitter Financial News Sentiment (zeroshot, MIT)"
for f in sent_train.csv sent_valid.csv sent_dataset_meta.txt; do
  curl -fsSL -o "$DATA/twitter_fin/$f" "$HF/zeroshot/twitter-financial-news-sentiment/resolve/main/$f"
done

echo "2/4 Financial PhraseBank v1.0 (Malo et al., 2014; CC BY-NC-SA 3.0)"
curl -fsSL -o "$TMP/fpb.zip" "$HF/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
unzip -o -q "$TMP/fpb.zip" -d "$TMP/fpb"
rm -rf "$DATA/phrasebank/FinancialPhraseBank-v1.0"
mv "$TMP/fpb/FinancialPhraseBank-v1.0" "$DATA/phrasebank/"

echo "3/4 Loughran-McDonald master dictionary (as shipped in the pysentiment2 package)"
python3 -m pip download -q --no-deps "pysentiment2==0.1.1" -d "$TMP/pkg"
unzip -o -q -j "$TMP/pkg/pysentiment2-0.1.1-py3-none-any.whl" "pysentiment2/static/LM.csv" -d "$DATA/lexicon"

echo "4/4 GloVe 100-d word vectors (128 MB, cached in ~/.cache/glove)"
mkdir -p "$HOME/.cache/glove"
if [ ! -f "$HOME/.cache/glove/glove-wiki-gigaword-100.gz" ]; then
  curl -fsSL -o "$HOME/.cache/glove/glove-wiki-gigaword-100.gz" \
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
fi
echo "done: data in $DATA"
