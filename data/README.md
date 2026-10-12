# Data

Third-party data is **not** stored in this repository. Run `./get_data.sh` from the repository root to download it:

| Folder | Content | Source | Licence |
| --- | --- | --- | --- |
| `twitter_fin/` | Twitter Financial News Sentiment: 9,543 training and 2,388 validation tweets labelled Bearish (0), Bullish (1), Neutral (2) | [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT |
| `phrasebank/` | Financial PhraseBank v1.0: 4,846 financial news sentences, four annotator-agreement levels | Malo et al. (2014), [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | CC BY-NC-SA 3.0 |
| `lexicon/LM.csv` | Loughran–McDonald master dictionary (2,355 negative, 354 positive words) | Loughran & McDonald (2011), via the [pysentiment2](https://pypi.org/project/pysentiment2/) package | free for academic use |
| `~/.cache/glove/` | GloVe 6B, 100-d word vectors | Pennington et al. (2014), via [gensim-data](https://github.com/RaRe-Technologies/gensim-data) | PDDL |

## Files made in this project

`bank_mentions_test_roles.csv`: the 64 test tweets that name one of 11 large banks (JPMorgan, Goldman Sachs,
BlackRock, Morgan Stanley, Citi, BofA/Merrill, Wells Fargo, UBS, Deutsche Bank, Barclays, Credit Suisse).
Bank names were found with regular expressions; the role of the bank was labelled by hand after reading each tweet:

- `messenger`: the bank rates, forecasts or trades another company or the market (*Netflix downgraded to underperform at Wells Fargo*)
- `subject`: the news is about the bank itself (*JPMorgan expands private credit group*)
- `incidental`: the bank is only mentioned in passing (*Alliance Data taps new CEO from Citi*)

`test_index` is the row of the tweet in `twitter_fin/sent_valid.csv`.
