# Bullish or Bearish? Comparing NLP Models from Regular Expressions to FinBERT for Financial News Sentiment

Language Technologies — Homework Assignment (experimental option)

Oct 12, 2026 · @Oleksandr

## Abstract

Quantitative investors turn financial news into trading signals, and the building block is a classifier that labels a headline as bullish, bearish or neutral. This project compares one model from every family of the course on that task: a finance lexicon with regular-expression rules, TF-IDF with logistic regression, a CNN and a BiLSTM on GloVe embeddings, and fine-tuned BERT-base and FinBERT. The models are trained on 8,111 financial news tweets and tested on 2,388 held-out tweets and, without retraining, on 4,846 sentences of the Financial PhraseBank. Fine-tuned Transformers reach 83–84 macro-F1, 5–6 points above the classical models, which are statistically tied with each other. Finance pre-training does not help in-domain, but it makes FinBERT the most robust model on formal news (86.0 macro-F1) and the most data-efficient one. Regular expressions add 17 points to a lexicon at no training cost. Error analysis and a case study of tweets that name JPMorgan, Goldman Sachs and BlackRock show that numbers, negation and the target company of the news remain open problems.

## 1 Introduction

Text is one of the oldest sources of "alternative data" in quantitative investing. BlackRock's systematic team writes that its early text signals simply counted positive and negative words in broker reports, earnings calls, filings and news, and that it now uses Transformer language models (Savi, Shen & Tsaig, 2023). J.P. Morgan's *Big Data and AI Strategies* report likewise lists news sentiment among the core alternative datasets for trading strategies (Kolanovic & Krishnamachari, 2017).

The building block of such a signal is a classifier that reads one headline and decides whether it is good news (bullish), bad news (bearish) or neither (neutral) for the stock. This project builds that block with one model from every family covered in the course, retracing the same path from word counts to Transformers. The models are compared on short financial news tweets, with a formal news dataset as a robustness test. Four research questions guide the experiments:

1. **RQ1 – Model families.** How do rule-based (regular expressions + lexicon), linear (TF-IDF), convolutional, recurrent and Transformer models compare on Bullish / Bearish / Neutral classification?
2. **RQ2 – Domain pre-training.** Does a BERT model pre-trained on financial text (FinBERT) beat general-domain BERT when both are fine-tuned on the same data?
3. **RQ3 – Domain shift.** How well do models trained on tweets transfer to sentences from formal financial news?
4. **RQ4 – Data efficiency.** How much labelled data does each approach need?

Fine-tuned Transformers beat every earlier approach by 5–6 macro-F1 points, and the gap widens under domain shift and with little training data, the two settings in which finance pre-training helps. Hand-written regular expressions add 17 points to a finance lexicon at no training cost.

## 2 Related work

**Lexicons.** Loughran and McDonald (2011) showed that general sentiment dictionaries misread financial text (*liability* or *tax* are not negative in a 10-K) and built finance-specific word lists. Their dictionary (2,355 negative and 354 positive words) is still the standard rule-based baseline.

**Annotated data.** Malo et al. (2014) released the Financial PhraseBank: 4,846 news sentences labelled from an investor's point of view by 16 annotators with a finance background (3 researchers and 13 master's students). It is the standard benchmark for financial sentiment.

**Neural classifiers.** Kim (2014) showed that a one-layer CNN over pre-trained word vectors such as GloVe (Pennington et al., 2014) is a strong sentence classifier, comparing random, frozen ("static") and fine-tuned ("non-static") embeddings. LSTMs (Hochreiter & Schmidhuber, 1997) read the sentence as a sequence instead.

**Transformers.** BERT (Devlin et al., 2019) pre-trains a Transformer encoder (Vaswani et al., 2017) by masked language modelling and is then fine-tuned. Two finance variants share the name FinBERT: Araci (2019) adapted BERT to financial news and fine-tuned it on PhraseBank, while Yang et al. (2020) pre-trained BERT on 4.9 billion tokens of filings, earnings-call transcripts and analyst reports with a finance vocabulary. This project uses the first off-the-shelf and fine-tunes the second.

## 3 Data

All models are trained on the **Twitter Financial News Sentiment** dataset ([Hugging Face](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), MIT licence): 11,931 English tweets from financial news accounts, labelled Bearish, Bullish or Neutral. They read like a trading-desk news feed of rating changes, price targets, earnings surprises and market moves, for example *Synaptics stock price target raised to $40 from $32 at J.P. Morgan* (Bullish) or *Autodesk downgraded to underweight from neutral at JPMorgan* (Bearish). The official validation split is the **test set**; a stratified 15% of the training split is the **development set** used for every model choice. The **Financial PhraseBank** (Malo et al., 2014) serves only as an out-of-domain test set, with positive/negative/neutral mapped to Bullish/Bearish/Neutral (Table 1).

*Table 1. Data sets, their role and class shares.*

| Set | Role | Texts | Bearish | Bullish | Neutral | Words per text |
| --- | --- | --- | --- | --- | --- | --- |
| Twitter train | training | 8,111 | 15.1% | 20.1% | 64.7% | 12.2 |
| Twitter dev | model selection | 1,432 | 15.1% | 20.2% | 64.7% | 12.1 |
| Twitter test | in-domain test | 2,388 | 14.5% | 19.9% | 65.6% | 12.3 |
| PhraseBank ≥50% agree | out-of-domain test | 4,846 | 12.5% | 28.1% | 59.4% | 23.1 |
| PhraseBank 100% agree | out-of-domain test | 2,264 | 13.4% | 25.2% | 61.4% | 22.4 |

Two properties shape the evaluation. The classes are imbalanced: always answering Neutral gives 65.6% test accuracy but only 26.4% macro-F1, so **macro-F1 is the main metric**. And the sets differ in register: tweets are short and noisy (47% contain a URL, 15% a $CASHTAG), while PhraseBank sentences are twice as long, formal, and about companies listed in Helsinki. Only 38 test tweets (1.6%) match a training tweet once URLs are removed.

## 4 Methods

One model is chosen from each family in the course, ordered from transparent rules to large pre-trained networks (Table 2). All of them see the same regex-normalised text (Section 5).

*Table 2. Models compared.*

| Model | Input representation | Classifier | Trained weights |
| --- | --- | --- | --- |
| LM lexicon | Loughran–McDonald word counts | sign of (positive − negative) | 0 |
| LM lexicon + regex | + 14 finance regex patterns | sign of weighted score | 1 |
| TF-IDF + logistic regression | word 1–2-grams, char 2–5-grams | multinomial logistic regression | 0.17 M |
| CNN-rand / -static / -non-static | random / frozen / tuned GloVe (100-d) | Kim CNN: widths 3, 4, 5 × 100 filters | 1.55 / 0.12 / 1.55 M |
| BiLSTM | tuned GloVe (100-d) | 2 × 128 LSTM, max-pooling | 1.67 M |
| BERT-base | WordPiece, general vocabulary | 12-layer encoder + linear head | 110 M |
| FinBERT | WordPiece, finance vocabulary | same, pre-trained on financial text | 110 M |
| FinBERT-Prosus | WordPiece | already fine-tuned on PhraseBank | 0 (zero-shot) |

**Rule-based models.** The lexicon model counts Loughran–McDonald words; a positive word within three tokens after a negation (*not*, *no*, *never*…) counts as negative. The regex variant adds 14 patterns written by reading the training split only: analyst actions (`\bupgrad(e|es|ed|ing)\b`), target and guidance revisions, earnings surprises (*beats*, *misses*), price moves (*jumps*, *up 5%*) and rating words (*overweight*). One rule parses numbers: in "*target raised to $40 from $32*" the pattern `to \$?(\d+) from \$?(\d+)` extracts both prices, and their difference gives the direction; the rule weight (3) was chosen on the training split.

**Neural models.** The CNN follows Kim (2014) but uses 100-d GloVe instead of 300-d word2vec vectors, and Adam instead of Adadelta; its three variants isolate the value of pre-trained embeddings. GloVe covers 82.3% of the 14,331 training word types; missing words start from U(−0.25, 0.25).

**Transformers.** BERT-base-uncased and FinBERT (Yang et al., 2020) share one architecture; only the pre-training corpus and vocabulary differ. FinBERT keeps finance words whole: *downgraded* is one FinBERT token but three BERT tokens (*down ##grade ##d*), and *JPMorgan* one versus three (*jp ##mo ##rgan*). FinBERT-Prosus (Araci, 2019) is applied without any training on tweets (positive → Bullish, negative → Bearish).

## 5 Experimental setup

**Preprocessing.** Regular expressions decode HTML entities and replace URLs, @user mentions and $CASHTAGS by `<url>`, `<user>` and `<ticker>`: they carry no sentiment but would flood the vocabulary. Non-Transformer models use a regex word tokenizer; the Transformers use their own WordPiece tokenizers.

**Training and model selection** use the development set only; the test sets are touched once.

- *TF-IDF:* C ∈ {0.5, 1, 2, 5, 10, 20} × class weighting (none / balanced), grid search.
- *CNN, BiLSTM:* Adam (learning rate 10⁻³), batch 50, at most 50 tokens, up to 15 epochs with early stopping after 3 epochs without improvement.
- *BERT, FinBERT:* AdamW (2·10⁻⁵, weight decay 0.01), batch 32, at most 64 sub-words, 4 epochs, 10% warm-up, gradient clipping at 1.0; the best dev epoch is kept.

Neural models are trained with three seeds and reported as mean ± standard deviation; deterministic models run once. For the learning curve, models are retrained on stratified 10%, 25% and 50% subsets of the training split, three seeds each (BERT models get 10 epochs at 10%, about 250 updates).

**Metrics.** Macro-F1 is the main metric, with accuracy alongside. A finance-specific **sign-error rate** counts truly bullish or bearish tweets labelled with the *opposite* direction: for a trading signal, calling bad news good is far costlier than calling it neutral. Macro-F1 differences are tested with a paired bootstrap over test tweets (2,000 resamples per seed; Berg-Kirkpatrick et al., 2012). **Environment:** an Apple M3 laptop, Transformers on its GPU (PyTorch MPS); PyTorch 2.14, Transformers 5.19, scikit-learn 1.7. One BERT fine-tuning run takes 13–23 minutes.

## 6 Results

*Table 3. Results on the Twitter test set (2,388 tweets) and on the Financial PhraseBank, which no model was trained on. Macro-F1, accuracy and sign-error rate in %; mean ± standard deviation over three seeds for neural models. FinBERT-Prosus has no PhraseBank score because it was fine-tuned on PhraseBank.*

| Model | Macro-F1 | Accuracy | Sign errors | PhraseBank 100% | PhraseBank ≥50% |
| --- | --- | --- | --- | --- | --- |
| Always Neutral | 26.4 | 65.6 | 0.0 | 25.4 | 24.8 |
| LM lexicon | 46.0 | 60.1 | 6.8 | 48.3 | 50.4 |
| LM lexicon + regex | 63.4 | 69.2 | 6.4 | 62.9 | 58.1 |
| TF-IDF + logistic regression | 77.5 | 84.3 | 6.7 | 60.8 | 50.6 |
| CNN-rand | 74.9 ± 1.5 | 82.6 ± 0.8 | 8.6 ± 1.3 | 54.9 ± 3.1 | 46.4 ± 2.3 |
| CNN-static | 74.7 ± 0.8 | 82.1 ± 0.6 | 9.7 ± 1.4 | 65.0 ± 2.8 | 53.4 ± 2.4 |
| CNN-non-static | 77.5 ± 0.1 | 83.7 ± 0.0 | 6.9 ± 0.4 | 68.0 ± 1.8 | 56.1 ± 1.4 |
| BiLSTM | 77.3 ± 0.6 | 83.1 ± 0.9 | 7.9 ± 0.8 | 65.6 ± 3.8 | 54.9 ± 2.7 |
| FinBERT-Prosus, zero-shot | 65.4 | 70.1 | 9.4 | – | – |
| BERT-base | **83.9 ± 0.3** | **87.5 ± 0.2** | **3.7 ± 0.3** | 82.5 ± 2.3 | 67.1 ± 1.8 |
| FinBERT | 82.7 ± 0.9 | 86.8 ± 0.9 | 5.3 ± 1.1 | **86.0 ± 0.6** | **69.2 ± 0.8** |

**Model families (RQ1).** Fine-tuned Transformers are best: BERT-base reaches 83.9 and FinBERT 82.7 macro-F1, 5–6 points above the best classical models, and the paired bootstrap puts their difference to TF-IDF above zero for every seed (+3.9 to +6.7 points). TF-IDF (77.5) and the fine-tuned-GloVe CNN (77.5) are tied in every seed, and so are the CNN and the BiLSTM (77.3): on 8,000 short tweets, word order and embeddings add nothing over well-tuned n-grams. GloVe helps the CNN only when fine-tuned (frozen 74.7, random 74.9). Rules alone are weak (46.0), but the 14 regex rules add 17.4 points (95% CI 15.4–19.3) without any training. The ready-made FinBERT-Prosus reaches only 65.4, so a model fine-tuned on formal news does not carry over to tweets. Fine-tuned Transformers also make the fewest sign errors (BERT 3.7%). In the TF-IDF ablation, regex normalisation changes test macro-F1 little (76.5 → 76.9), replacing numbers by placeholders hurts (75.7) because numbers carry direction, and character n-grams help most (77.5).

**Domain pre-training (RQ2).** In-domain, FinBERT does not beat BERT: BERT is ahead by 1.2 points on average, significantly so for one of the three seeds. FinBERT's advantage appears elsewhere. On PhraseBank it scores 86.0 ± 0.6 against 82.5 ± 2.3 for BERT, whose score ranges from 80.0 to 85.6 across seeds, and with 10% of the training data it leads by about 4 points (76.3 vs. 72.5; Figure 1). The two vocabularies cut tweets into a similar number of pieces (19.4 vs. 19.9 sub-words per tweet), so the gain comes from what FinBERT read during pre-training rather than from shorter inputs.

**Domain shift (RQ3).** On PhraseBank's formal sentences the classical models lose 10–20 points. TF-IDF falls from 77.5 to 60.8 because its n-grams memorise tweet phrasing; the CNN with random embeddings loses 20 points but the GloVe CNNs only about 10, so pre-trained word vectors carry knowledge across text styles. The rule-based model, which never saw training data, transfers as well as TF-IDF (62.9). BERT loses 1.4 points on average, while FinBERT is the only model that improves (+3.3). Every trained model scores 10–17 points lower on the ≥50%-agreement subset, whose sentences the annotators themselves disagreed on.

**Data efficiency (RQ4).** With 811 labelled tweets (10% of the training set), FinBERT reaches 76.3 macro-F1, within about one point of what TF-IDF and the CNN achieve with all 8,111, while on the same subsets those two reach only 62.6 and 60.9 (Figure 1). The gap between FinBERT and TF-IDF shrinks from 14 points at 10% to 5 at full size, and FinBERT's lead over BERT (3.8 points at 10%) vanishes at 50% and reverses with the full training set.

&#91;image: Figure 1. Test macro-F1 against the number of labelled training tweets (log scale); mean ± standard deviation over three seeds, each seed drawing its own stratified subset.\]

*Figure 1. Test macro-F1 against the number of labelled training tweets (log scale); mean ± standard deviation over three seeds, each seed drawing its own stratified subset.*

## 7 Discussion

**Error analysis.** Most remaining errors are harmless for a trading signal: 267 of FinBERT's 306 test errors (seed 1, 87%) only confuse a directional tweet with Neutral or vice versa, and many of these tweets are borderline (Figure 2). The 39 sign errors (4.7% of directional tweets) were read one by one and grouped by cause (Table 4).

&#91;image: Figure 2. Confusion matrices on the Twitter test set, normalised by true class and summed over seeds: TF-IDF + logistic regression (left) and fine-tuned FinBERT (right).\]

*Figure 2. Confusion matrices on the Twitter test set, normalised by true class and summed over seeds: TF-IDF + logistic regression (left) and fine-tuned FinBERT (right).*

*Table 4. Causes of FinBERT's 39 sign errors (seed 1), assigned by hand; the true label is in brackets.*

| Cause | Errors | Example |
| --- | --- | --- |
| Mixed or misleading cue words | 8 | *Buy oil service firms, Bernstein says after seven-year bearish view* (Bullish) |
| Market jargon or world knowledge | 8 | *CACI gains a bull on growth acceleration* (Bullish) |
| Numbers that must be compared | 7 | *Kohl's Q3 EPS 78 cents vs. 98 cents* (Bearish) |
| Negation or reversal | 7 | *Global Economy Won't See Recession in 2020* (Bullish) |
| No clear cause or arguable label | 6 | *US Proved Oil & Gas Reserves Hit New Record High* (Bearish) |
| Polarity depends on what moves | 3 | *Saudi Arabia's Unemployment Falls to Lowest in Over Three Years* (Bullish) |

Two causes suggest concrete fixes. Earnings and macro releases ("*X vs. Y*") need arithmetic, which the regex rule of Section 4 already does for price targets and could do for reported-versus-expected figures. Words such as *falls* or *high* flip polarity with the variable they describe (unemployment, costs, jobless claims), which sentence-level labels on 8,000 tweets teach only weakly.

**Case study: JPMorgan, Goldman Sachs and BlackRock.** A trading signal needs sentiment *about a company*, not about a sentence. Regular expressions find 64 test tweets (2.7%) that name one of eleven large banks, among them JPMorgan (10), Goldman Sachs (10) and BlackRock (3). Reading them shows that in 46 (72%) the bank is only the messenger, as in *Netflix downgraded to underperform at Wells Fargo*, and only 15 are about the bank itself. Of the 47 bullish or bearish bank tweets, just 4 (9%) carry sentiment about the bank. A naive signal that assigns a tweet's label to every company it names would therefore mostly trade JPMorgan or Goldman Sachs on news about other firms, even though FinBERT labels these tweets as accurately as the rest of the test set (87.0% vs. 86.8% accuracy). The missing piece is target-dependent sentiment: deciding which company the news is about.

**Limitations.** All results come from one dataset of short, US-centric tweets with a single official split, and some labels are arguable (Table 4). No model was linked to market data, so the economic value of the signal is untested. Large generative language models were outside the scope of this comparison.

## 8 Conclusion

Each step of the course improves financial sentiment classification, but not uniformly. Fine-tuned Transformers are 5–6 macro-F1 points better than the best classical models, while TF-IDF, the GloVe CNN and the BiLSTM are statistically tied on 8,000 short tweets. Under domain shift and with little labelled data the differences grow: classical models lose 10–17 points on formal news, the Transformers hold their level, and FinBERT's finance pre-training gives the most robust and most data-efficient model. Hand-written regular expressions remain useful: they add 17 points to a lexicon without any training and transfer as well as TF-IDF.

For a quant trading application, the error analysis and the bank case study point to the next steps: reasoning about reported-versus-expected numbers, handling negation, and target-dependent sentiment that attaches news to the right company. An event study linking predicted sentiment to next-day returns would then measure the economic value of the signal.

## References

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*. Model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).
- Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012). An empirical investigation of statistical significance in NLP. *EMNLP-CoNLL 2012*, 995–1005.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional Transformers for language understanding. *NAACL-HLT 2019*, 4171–4186. Model: [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased).
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP 2014*, 1746–1751.
- Kolanovic, M., & Krishnamachari, R. T. (2017). *Big Data and AI Strategies: Machine Learning and Alternative Data Approach to Investing*. J.P. Morgan Global Quantitative & Derivatives Strategy.
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796. Data: [takala/financial\_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank).
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP 2014*, 1532–1543.
- Savi, R., Shen, J., & Tsaig, Y. (2023, September 8). How artificial intelligence is transforming investing: BlackRock Systematic. [Article](https://insurancebiz.co.za/news/wealth-and-investments/how-artificial-intelligence-is-transforming-investing-blackrock-systematic.html).
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems 30*.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). FinBERT: A pretrained language model for financial communications. *arXiv:2006.08097*. Model: [yiyanghkust/finbert-pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain).
- Zeroshot (2022). Twitter Financial News Sentiment \[dataset\]. [Hugging Face](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment).
