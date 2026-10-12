"""Collect results/*.json into tables, figures and an error analysis for the report."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import LABELS, RESULTS, ROOT, SEEDS, eval_sets, load_twitter, metrics

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# chart palette (validated categorical slots + ink/grid tokens)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})

MAIN = [  # (result key, display name, family)
    ("lexicon_LM", "LM lexicon", "Rule-based"),
    ("lexicon_LM+regex", "LM lexicon + regex rules", "Rule-based"),
    ("tfidf_lr[norm+char]", "TF-IDF + logistic regression", "Linear"),
    ("cnn-rand", "CNN-rand", "CNN"),
    ("cnn-static", "CNN-static (GloVe)", "CNN"),
    ("cnn-nonstatic", "CNN-non-static (GloVe)", "CNN"),
    ("bilstm", "BiLSTM (GloVe)", "RNN"),
    ("finbert-prosus-zeroshot", "FinBERT-Prosus, zero-shot", "Transformer"),
    ("bert", "BERT-base, fine-tuned", "Transformer"),
    ("finbert", "FinBERT, fine-tuned", "Transformer"),
]
NAME = {k: n for k, n, _ in MAIN}


def load_results():
    rows = []
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.load(open(f))
        if "scores" not in d:  # e.g. tfidf_top_features.json
            continue
        for split, s in d["scores"].items():
            rows.append({"model": d["model"], "seed": d["seed"], "split": split,
                         **{k: v for k, v in s.items() if k != "confusion"},
                         "confusion": np.array(s["confusion"]), "train_time_s": d.get("train_time_s"),
                         "best_epoch": d.get("best_epoch")})
    return pd.DataFrame(rows)


def pm(series, pct=True, digits=1):
    m, s = series.mean(), series.std(ddof=0)
    f = 100 if pct else 1
    return f"{m * f:.{digits}f}" if len(series) == 1 or np.isnan(s) else f"{m * f:.{digits}f} ± {s * f:.{digits}f}"


# ---------------------------------------------------------------- tables
def majority_row():
    """'Always Neutral' reference row (the majority class everywhere)."""
    sets = eval_sets()
    s = {k: metrics(d.label, np.full(len(d), 2)) for k, d in sets.items()}
    t = s["test"]
    f = lambda v: f"{100 * v:.1f}"
    return {"Family": "Reference", "Model": "Always Neutral", "Seeds": "-", "Accuracy": f(t["accuracy"]),
            "Macro-F1": f(t["macro_f1"]), "F1 Bearish": f(t["f1_bearish"]), "F1 Bullish": f(t["f1_bullish"]),
            "F1 Neutral": f(t["f1_neutral"]), "Sign errors %": f(t["sign_error_rate"]),
            "PhraseBank-All F1": f(s["pb_all"]["macro_f1"]), "PhraseBank-50 F1": f(s["pb_50"]["macro_f1"]),
            "Train time (s)": "-"}


def main_table(df):
    out = [majority_row()]
    for key, name, fam in MAIN:
        t = df[(df.model == key) & (df.split == "test")]
        if t.empty:
            continue
        pb = df[(df.model == key) & (df.split == "pb_all")]
        pb50 = df[(df.model == key) & (df.split == "pb_50")]
        leak = key == "finbert-prosus-zeroshot"
        out.append({
            "Family": fam, "Model": name, "Seeds": len(t),
            "Accuracy": pm(t.accuracy), "Macro-F1": pm(t.macro_f1),
            "F1 Bearish": pm(t.f1_bearish), "F1 Bullish": pm(t.f1_bullish), "F1 Neutral": pm(t.f1_neutral),
            "Sign errors %": pm(t.sign_error_rate),
            "PhraseBank-All F1": "(trained on it)" if leak else pm(pb.macro_f1),
            "PhraseBank-50 F1": "(trained on it)" if leak else pm(pb50.macro_f1),
            "Train time (s)": "-" if t.train_time_s.isna().all() else f"{t.train_time_s.mean():.0f}",
        })
    return pd.DataFrame(out)


def ablation_table(df):
    out = []
    for v, desc in [("raw", "No preprocessing"), ("norm", "Regex normalisation (URL/@user/$TICKER)"),
                    ("norm+num", "+ numbers and % -> placeholders"), ("norm+char", "Normalisation + char 2-5-grams")]:
        t = df[(df.model == f"tfidf_lr[{v}]")].set_index("split")
        if t.empty:
            continue
        out.append({"Variant": desc, "Dev macro-F1": f"{100 * t.loc['dev', 'macro_f1']:.1f}",
                    "Test macro-F1": f"{100 * t.loc['test', 'macro_f1']:.1f}",
                    "Test accuracy": f"{100 * t.loc['test', 'accuracy']:.1f}",
                    "PhraseBank-All F1": f"{100 * t.loc['pb_all', 'macro_f1']:.1f}"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures
def fig_main(df):
    keys = [k for k, _, _ in MAIN if not df[(df.model == k) & (df.split == "test")].empty]
    test = [df[(df.model == k) & (df.split == "test")].macro_f1 for k in keys]
    pb = [df[(df.model == k) & (df.split == "pb_all")].macro_f1 for k in keys]
    y = np.arange(len(keys))[::-1]
    h = 0.36
    test_m, test_s = [100 * t.mean() for t in test], [100 * t.std(ddof=0) for t in test]
    leak = [k == "finbert-prosus-zeroshot" for k in keys]
    pb_m = [np.nan if lk else 100 * p.mean() for lk, p in zip(leak, pb)]
    pb_s = [0 if lk else 100 * p.std(ddof=0) for lk, p in zip(leak, pb)]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.barh(y + h / 2 + 0.01, test_m, h, xerr=test_s, color=BLUE, label="Twitter test (in-domain)",
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.barh(y - h / 2 - 0.01, pb_m, h, xerr=pb_s, color=ORANGE, label="PhraseBank, 100% agreement (out-of-domain)",
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    for yi, m, s, p, ps in zip(y, test_m, test_s, pb_m, pb_s):
        # value labels sit past the error bar so the two never collide
        ax.text(m + s + 1.2, yi + h / 2, f"{m:.1f}", va="center", fontsize=7, color=INK2)
        if np.isnan(p):
            ax.text(1, yi - h / 2, "n/a (trained on PhraseBank)", va="center", fontsize=7, color=MUTED)
        else:
            ax.text(p + ps + 1.2, yi - h / 2, f"{p:.1f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [NAME[k] for k in keys])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Macro-F1, % (mean ± std over seeds)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.savefig(FIG / "fig_main_results.png")
    plt.close(fig)


def fig_confusion(df):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    panels = [k for k in ["tfidf_lr[norm+char]", "finbert"] if not df[(df.model == k)].empty]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.9))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, panels):
        cms = df[(df.model == key) & (df.split == "test")].confusion
        cm = np.sum(np.stack(cms.values), axis=0)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(NAME[key], fontsize=9, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion.png")
    plt.close(fig)


LC_SERIES = [("tfidf_lr[norm+char]", "TF-IDF + LogReg", BLUE), ("cnn-nonstatic", "CNN-non-static", ORANGE),
             ("bert", "BERT-base", AQUA), ("finbert", "FinBERT", YELLOW)]
FRACS = [0.1, 0.25, 0.5, 1.0]


def lc_points(df, key):
    """(n_train, mean, std, n_runs) of test macro-F1 per training fraction.

    Each seed draws its own stratified subset; the 100% point is the main run
    (all seeds; TF-IDF on the full data is deterministic, so one run).
    """
    n_full = len(load_twitter()[0])
    pts = []
    for frac in FRACS:
        tag = key if frac == 1.0 else f"{key}_frac{frac}"
        t = df[(df.model == tag) & (df.split == "test")].macro_f1
        if not t.empty:
            pts.append((round(frac * n_full), t.mean(), t.std(ddof=0), len(t)))
    return pts


def fig_learning_curve(df):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ends = []
    for key, label, color in LC_SERIES:
        pts = lc_points(df, key)
        if len(pts) < 2:
            continue
        xs, ys, sd, _ = map(np.array, zip(*pts))
        ax.errorbar(xs, 100 * ys, yerr=100 * sd, color=color, lw=2, marker="o", ms=5, label=label, capsize=2.5,
                    elinewidth=1, markeredgecolor="white", markeredgewidth=1)
        ends.append([100 * ys[-1], 100 * ys[-2], label, xs[-1]])
    if not ends:
        plt.close(fig)
        return
    # direct labels at the line ends, nudged apart when two lines finish at the same value
    # (ties within half a point: the line arriving from lower down keeps the lower label)
    ends.sort(key=lambda e: (round(e[0]), e[1]))
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 1.7)
    for y_lab, _, label, x_end in ends:
        ax.text(x_end * 1.08, y_lab, label, va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([811, 2028, 4056, 8111], ["811\n(10%)", "2,028\n(25%)", "4,056\n(50%)", "8,111\n(100%)"])
    ax.minorticks_off()
    ax.set_xlim(650, 17000)
    ax.set_xlabel("Labelled training tweets (log scale)")
    ax.set_ylabel("Test macro-F1, %")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(FIG / "fig_learning_curve.png")
    plt.close(fig)


def learning_curve_table(df):
    rows = []
    for key, label, _ in LC_SERIES:
        row = {"Model": label}
        for n, m, s, k in lc_points(df, key):
            row[f"{n} tweets"] = f"{100 * m:.1f} ± {100 * s:.1f} (n={k})"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- significance & errors
MCNEMAR_PAIRS = [("finbert", "bert"), ("finbert", "tfidf_lr[norm+char]"), ("bert", "tfidf_lr[norm+char]"),
                 ("cnn-nonstatic", "tfidf_lr[norm+char]"), ("bilstm", "cnn-nonstatic"),
                 ("finbert", "finbert-prosus-zeroshot"), ("lexicon_LM+regex", "lexicon_LM")]


def pred_file(key, seed):
    """Test predictions of one run; deterministic models only have seed 0."""
    for s in (seed, 0):
        p = RESULTS / "predictions" / f"{key}__seed{s}__test.csv".replace("/", "_")
        if p.exists():
            return p
    return None


def mcnemar(a_key, b_key, seed):
    """Exact McNemar test on paired test predictions (n01 = only A right, n10 = only B right)."""
    A, B = pd.read_csv(pred_file(a_key, seed)), pd.read_csv(pred_file(b_key, seed))
    a_ok, b_ok = A.pred == A.label, B.pred == B.label
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return n01, n10, binomtest(n01, n01 + n10, 0.5).pvalue


def macro_f1(y, p):
    cm = np.bincount(3 * y + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))


def bootstrap_f1(a_key, b_key, seed, n_boot=2000):
    """Paired bootstrap over test tweets for the macro-F1 difference A - B: (diff, 95% CI low, high, p)."""
    A, B = pd.read_csv(pred_file(a_key, seed)), pd.read_csv(pred_file(b_key, seed))
    y, pa, pb = A.label.values, A.pred.values, B.pred.values
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = macro_f1(y[idx], pa[idx]) - macro_f1(y[idx], pb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return macro_f1(y, pa) - macro_f1(y, pb), lo, hi, p


def error_analysis(lines):
    best = pd.read_csv(RESULTS / "predictions/finbert__seed1__test.csv")
    best["true"] = best.label.map(dict(enumerate(LABELS)))
    best["predicted"] = best.pred.map(dict(enumerate(LABELS)))
    wrong = best[best.label != best.pred]
    lines.append(f"\nFinBERT (seed 1) test errors: {len(wrong)} / {len(best)}")
    lines.append(wrong.groupby(["true", "predicted"]).size().rename("count").to_string())

    sign = wrong[wrong.label.isin([0, 1]) & wrong.pred.isin([0, 1])]
    lines.append(f"\nSign errors (Bullish <-> Bearish), all {len(sign)}:")
    lines += [f"  [{r.true} -> {r.predicted}] {r.text}" for r in sign.itertuples()]

    for (t, p), g in wrong.groupby(["true", "predicted"]):
        if t in ("Bullish", "Bearish") and p in ("Bullish", "Bearish"):
            continue
        lines.append(f"\nSample {t} -> {p}:")
        lines += [f"  {x}" for x in g.text.sample(min(8, len(g)), random_state=0)]

    # tweets that every main model gets wrong (seed 1 where available)
    preds = {key: pd.read_csv(pred_file(key, 1)).pred.values for key, _, _ in MAIN if pred_file(key, 1)}
    allwrong = np.all([preds[k] != best.label.values for k in preds], axis=0)
    lines.append(f"\nTweets misclassified by ALL {len(preds)} models: {allwrong.sum()}")
    lines += [f"  [{best.true[i]}] {best.text[i]}" for i in np.where(allwrong)[0][:25]]


def main():
    df = load_results()
    lines = []
    mt = main_table(df)
    mt.to_csv(RESULTS / "table_main.csv", index=False)
    lines += ["MAIN RESULTS (test = official Twitter valid split, 2,388 tweets)", mt.to_string(index=False)]
    ab = ablation_table(df)
    ab.to_csv(RESULTS / "table_tfidf_ablation.csv", index=False)
    lines += ["\nTF-IDF PREPROCESSING ABLATION", ab.to_string(index=False)]

    lct = learning_curve_table(df)
    lct.to_csv(RESULTS / "table_learning_curve.csv", index=False)
    lines += ["\nLEARNING CURVE (test macro-F1, mean ± std over seeds)", lct.to_string(index=False)]

    lines.append("\nMcNEMAR TESTS (exact, paired test predictions per seed; n01 = only A right, n10 = only B right)")
    for a, b in MCNEMAR_PAIRS:
        for seed in SEEDS:
            if pred_file(a, seed) and pred_file(b, seed):
                n01, n10, p = mcnemar(a, b, seed)
                lines.append(f"  {a} vs {b} [seed {seed}]: n01={n01} n10={n10} p={p:.2e}")

    lines.append("\nPAIRED BOOTSTRAP, macro-F1 difference A - B in points (95% CI, two-sided p; 2,000 resamples)")
    for a, b in MCNEMAR_PAIRS:
        for seed in SEEDS:
            if pred_file(a, seed) and pred_file(b, seed):
                d, lo, hi, p = bootstrap_f1(a, b, seed)
                lines.append(f"  {a} vs {b} [seed {seed}]: {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] p={p:.3f}")

    if (RESULTS / "predictions/finbert__seed1__test.csv").exists():
        error_analysis(lines)

    for fn in (fig_main, fig_confusion, fig_learning_curve):
        fn(df)
    (RESULTS / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
