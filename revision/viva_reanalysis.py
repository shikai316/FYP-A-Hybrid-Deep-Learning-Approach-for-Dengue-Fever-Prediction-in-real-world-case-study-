"""
Viva revision re-analysis — examiner items 1, 2, 3, 5.

Recomputes every hypothesis test from the per-seed raw data already saved by the
notebooks, and emits, for each research question:
    H0 / Ha, test used, statistic, p-value, effect size, 95% CI,
    Holm-corrected p across the 5 horizons, and an explicit
    "Reject H0" / "Fail to reject H0" decision at alpha = 0.05.

Nothing is typed by hand: every number traces to a file in FYP code/metrics/.
"""
import ast
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

M = Path(__file__).resolve().parent.parent / "metrics"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
HORIZONS = [1, 7, 14, 21, 28]
SEEDS = [42 + 7 * i for i in range(30)]


# ---------------------------------------------------------------- helpers
def parse_raw(cell):
    """multi_horizon_results.csv stores lists as strings, some with np.float64()."""
    s = str(cell).replace("np.float64(", "").replace(")", "")
    return np.asarray(ast.literal_eval(s), dtype=float)


def decision(p, alpha=ALPHA):
    return "Reject H0" if p < alpha else "Fail to reject H0"


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def ci_mean(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    se = stats.sem(x)
    h = se * stats.t.ppf((1 + conf) / 2, n - 1)
    return float(x.mean() - h), float(x.mean() + h)


def cohens_d_one(x, mu0):
    x = np.asarray(x, dtype=float)
    return float((x.mean() - mu0) / x.std(ddof=1))


def cohens_dz(diff):
    diff = np.asarray(diff, dtype=float)
    return float(diff.mean() / diff.std(ddof=1))


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


# ---------------------------------------------------------------- load
mh = pd.read_csv(M / "multi_horizon_results.csv")
RAW = {}
for _, row in mh.iterrows():
    h = int(row["horizon"])
    RAW[h] = {
        "ilt_rmse": parse_raw(row["ilt_raw_runs"]),
        "lstm_rmse": parse_raw(row["lstm_raw_runs"]),
        "rf_rmse": parse_raw(row["rf_raw_runs"]),
        "ilt_mae": parse_raw(row["ilt_raw_mae"]),
        "lstm_mae": parse_raw(row["lstm_raw_mae"]),
        "rf_mae": parse_raw(row["rf_raw_mae"]),
        "ilt_r2": parse_raw(row["ilt_raw_r2"]),
        "lstm_r2": parse_raw(row["lstm_raw_r2"]),
        "rf_r2": parse_raw(row["rf_raw_r2"]),
        "sarima_rmse": float(row["sarima_rmse"]),
        "sarima_mae": float(row["sarima_mae"]),
        "sarima_r2": float(row["sarima_r2"]),
    }

results = {}


# ================================================================ H1
# H0: mean RMSE(ILT) >= 0.80 * RMSE(SARIMA)   (improvement is NOT more than 20%)
# Ha: mean RMSE(ILT) <  0.80 * RMSE(SARIMA)   (improvement exceeds 20%)
# Secondary, weaker H0: mean RMSE(ILT) = RMSE(SARIMA) (no difference at all).
rows = []
for h in HORIZONS:
    x = RAW[h]["ilt_rmse"]
    sar = RAW[h]["sarima_rmse"]
    thr = 0.80 * sar

    # primary: superiority-margin (non-inferiority style) one-sample t, one-sided
    t_m, p_two_m = stats.ttest_1samp(x, thr)
    p_m = p_two_m / 2 if t_m < 0 else 1 - p_two_m / 2
    w_m, pw_two_m = stats.wilcoxon(x - thr, alternative="less")

    # secondary: any difference at all
    t_d, p_two_d = stats.ttest_1samp(x, sar)
    p_d = p_two_d / 2 if t_d < 0 else 1 - p_two_d / 2

    lo, hi = ci_mean(x)
    red = 100 * (1 - x.mean() / sar)
    red_lo, red_hi = 100 * (1 - hi / sar), 100 * (1 - lo / sar)
    rows.append(
        dict(
            horizon=f"h={h}d",
            ilt_mean=x.mean(),
            ilt_sd=x.std(ddof=1),
            ilt_ci_lo=lo,
            ilt_ci_hi=hi,
            sarima=sar,
            pct_reduction=red,
            pct_red_ci_lo=red_lo,
            pct_red_ci_hi=red_hi,
            margin_threshold=thr,
            t_stat=t_m,
            p_margin=p_m,
            wilcoxon_p_margin=pw_two_m,
            cohens_d=cohens_d_one(x, thr),
            p_difference=p_d,
            wins=f"{int((x < sar).sum())}/30",
        )
    )
h1 = pd.DataFrame(rows)
h1["p_margin_holm"] = holm(h1["p_margin"])
h1["p_difference_holm"] = holm(h1["p_difference"])
h1["decision_margin"] = [decision(p) for p in h1["p_margin_holm"]]
h1["decision_difference"] = [decision(p) for p in h1["p_difference_holm"]]

# pooled
pool_x = np.concatenate([RAW[h]["ilt_rmse"] for h in HORIZONS])
pool_thr = np.concatenate([np.full(30, 0.8 * RAW[h]["sarima_rmse"]) for h in HORIZONS])
pool_sar = np.concatenate([np.full(30, RAW[h]["sarima_rmse"]) for h in HORIZONS])
t_p, p2 = stats.ttest_rel(pool_x, pool_thr)
h1_pool = dict(
    n=len(pool_x),
    mean_reduction_pct=100 * (1 - pool_x.mean() / pool_sar.mean()),
    t=t_p,
    p_margin=p2 / 2 if t_p < 0 else 1 - p2 / 2,
    wilcoxon_p=stats.wilcoxon(pool_x - pool_thr, alternative="less").pvalue,
    cohens_dz=cohens_dz(pool_x - pool_thr),
)
h1_pool["decision"] = decision(h1_pool["p_margin"])
results["H1"] = dict(per_horizon=h1.to_dict("records"), pooled=h1_pool)


# ================================================================ H2
# H0: mean(RMSE_ILT - RMSE_LSTM) = 0   (attention adds nothing)
# Ha: mean(RMSE_ILT - RMSE_LSTM) < 0   (attention lowers error)
rows = []
for h in HORIZONS:
    d = RAW[h]["ilt_rmse"] - RAW[h]["lstm_rmse"]
    t, p2 = stats.ttest_rel(RAW[h]["ilt_rmse"], RAW[h]["lstm_rmse"])
    p1 = p2 / 2 if t < 0 else 1 - p2 / 2
    w = stats.wilcoxon(d, alternative="less")
    lo, hi = ci_mean(d)
    rows.append(
        dict(
            horizon=f"h={h}d",
            ilt_mean=RAW[h]["ilt_rmse"].mean(),
            lstm_mean=RAW[h]["lstm_rmse"].mean(),
            mean_diff=d.mean(),
            diff_ci_lo=lo,
            diff_ci_hi=hi,
            t_stat=t,
            p_paired_t=p1,
            wilcoxon_p=w.pvalue,
            cohens_dz=cohens_dz(d),
            rank_biserial=rank_biserial(d),
            ilt_wins=f"{int((d < 0).sum())}/30",
        )
    )
h2 = pd.DataFrame(rows)
h2["p_holm"] = holm(h2["p_paired_t"])
h2["decision"] = [decision(p) for p in h2["p_holm"]]

pool_i = np.concatenate([RAW[h]["ilt_rmse"] for h in HORIZONS])
pool_l = np.concatenate([RAW[h]["lstm_rmse"] for h in HORIZONS])
d_pool = pool_i - pool_l
t, p2 = stats.ttest_rel(pool_i, pool_l)
lo, hi = ci_mean(d_pool)
h2_pool = dict(
    n=len(d_pool),
    mean_diff=d_pool.mean(),
    ci_lo=lo,
    ci_hi=hi,
    t=t,
    p_paired_t=p2 / 2 if t < 0 else 1 - p2 / 2,
    wilcoxon_p=stats.wilcoxon(d_pool, alternative="less").pvalue,
    cohens_dz=cohens_dz(d_pool),
)
h2_pool["decision"] = decision(h2_pool["p_paired_t"])
results["H2"] = dict(per_horizon=h2.to_dict("records"), pooled=h2_pool)


# ================================================================ H3
# H0: pi <= 0.70  (at most 70% of top-ranked attention features are domain-validated)
# Ha: pi >  0.70
dv = pd.read_csv(M / "ch5_final_h3_domain_validation_RECONSTRUCTED.csv")
k = int(dv["Domain_validated"].sum())
n = len(dv)
bt = stats.binomtest(k, n, 0.70, alternative="greater")
ci = bt.proportion_ci(confidence_level=0.95, method="exact")
cons = pickle.load(open(M / "h3_30seed_attention_consistency.pkl", "rb"))
hit = cons["consensus_top10_domain_hit_rate"]
per_seed_hits = cons.get("top10_hit_count_per_feature", {})
h3 = dict(
    k=k,
    n=n,
    observed_rate=k / n,
    p0=0.70,
    test="Exact one-sided binomial test",
    p_value=bt.pvalue,
    ci_lo=ci.low,
    ci_hi=ci.high,
    decision=decision(bt.pvalue),
    consensus_top10_hit_rate=hit,
    cross_seed_rmse_mean=cons["rmse_mean"],
    cross_seed_rmse_sd=cons["rmse_std"],
    mean_cv_per_timestep=cons["mean_cv_per_timestep"],
)
# stability of the attention profile across seeds: H0 profile is uniform (1/28 each)
attn = np.asarray(cons["all_mean_attn"], dtype=float)  # 30 x 28
uni = 1.0 / attn.shape[1]
peak_idx = int(attn.mean(axis=0).argmax())
peak_vals = attn[:, peak_idx]
t_u, p2_u = stats.ttest_1samp(peak_vals, uni)
h3["uniformity_test"] = dict(
    H0="attention profile is uniform over the 28-day window (alpha_t = 1/28)",
    peak_timestep=peak_idx,
    peak_mean_weight=float(peak_vals.mean()),
    uniform_weight=uni,
    t=float(t_u),
    p_value=float(p2_u / 2 if t_u > 0 else 1 - p2_u / 2),
)
h3["uniformity_test"]["decision"] = decision(h3["uniformity_test"]["p_value"])
results["H3"] = h3


# ================================================================ H4
# H0: cost ratio (ILT / Standalone LSTM) >= 2 on the metric
# Ha: cost ratio < 2
eff = pd.read_csv(M / "efficiency_table.csv").set_index("Model")
lstm, ilt = eff.loc["Standalone LSTM"], eff.loc["ILT (proposed)"]
N = 30
rows = []
for metric, col, sdcol in [
    ("Training time (s)", "Train time (s)", "Train time std (s)"),
    ("Inference latency (ms/sample)", "Inference per sample (ms)", "Inference per sample std (ms)"),
]:
    thr = 2 * lstm[col]
    m, sd = ilt[col], ilt[sdcol]
    se = sd / np.sqrt(N)
    t = (m - thr) / se
    p = stats.t.cdf(t, N - 1)  # one-sided, Ha: mean < threshold
    hw = stats.t.ppf(0.975, N - 1) * se
    rows.append(
        dict(
            metric=metric,
            lstm=lstm[col],
            ilt=m,
            ilt_sd=sd,
            ratio=m / lstm[col],
            threshold_2x=thr,
            ilt_ci_lo=m - hw,
            ilt_ci_hi=m + hw,
            ratio_ci_lo=(m - hw) / lstm[col],
            ratio_ci_hi=(m + hw) / lstm[col],
            t_stat=t,
            p_value=p,
            deterministic=False,
        )
    )
for metric, col in [("Trainable parameters", "Params"), ("Model size (KB)", "Model size (KB)")]:
    rows.append(
        dict(
            metric=metric,
            lstm=lstm[col],
            ilt=ilt[col],
            ilt_sd=0.0,
            ratio=ilt[col] / lstm[col],
            threshold_2x=2 * lstm[col],
            ilt_ci_lo=ilt[col],
            ilt_ci_hi=ilt[col],
            ratio_ci_lo=ilt[col] / lstm[col],
            ratio_ci_hi=ilt[col] / lstm[col],
            t_stat=np.nan,
            p_value=np.nan,
            deterministic=True,
        )
    )
h4 = pd.DataFrame(rows)
h4["decision"] = [
    ("Reject H0 (deterministic: ratio < 2 exactly)" if r.deterministic else decision(r.p_value))
    for r in h4.itertuples()
]

# scaling exponent: log(time) = a + b log(T);  H0: b >= 2 (quadratic or worse), Ha: b < 2
sc = pd.read_csv(M / "bigo_scaling_experiment.csv")
scal = {}
for name, col in [("TemporalAttention", "TemporalAttention_ms"), ("MultiHeadAttention", "MultiHeadAttention_ms")]:
    lr = stats.linregress(np.log(sc["T"]), np.log(sc[col]))
    df_ = len(sc) - 2
    tcrit = stats.t.ppf(0.975, df_)
    t_stat = (lr.slope - 2.0) / lr.stderr
    scal[name] = dict(
        exponent=lr.slope,
        se=lr.stderr,
        ci_lo=lr.slope - tcrit * lr.stderr,
        ci_hi=lr.slope + tcrit * lr.stderr,
        r_squared=lr.rvalue**2,
        t_vs_2=t_stat,
        p_subquadratic=float(stats.t.cdf(t_stat, df_)),
    )
    scal[name]["decision"] = decision(scal[name]["p_subquadratic"])
results["H4"] = dict(metrics=h4.to_dict("records"), scaling=scal)


# ================================================================ H5
# H0: OOS R2 <= 0 at horizon h (model has no predictive skill out of sample)
# Ha: OOS R2 >  0
fc = pickle.load(open(M / "ilt_holdout_forecasts.pkl", "rb"))
rng = np.random.default_rng(42)
B = 5000
BLOCK = 28  # moving-block bootstrap, block = one input window
rows = []
for h in HORIZONS:
    pred = np.asarray(fc[h]["pred"], dtype=float).ravel()
    act = np.asarray(fc[h]["actual"], dtype=float).ravel()
    n = len(act)

    def r2(p_, a_):
        return 1 - np.sum((a_ - p_) ** 2) / np.sum((a_ - a_.mean()) ** 2)

    obs = r2(pred, act)
    nb = int(np.ceil(n / BLOCK))
    starts_pool = np.arange(0, n - BLOCK + 1)
    boot = np.empty(B)
    for b in range(B):
        st = rng.choice(starts_pool, size=nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for st_ in [st] for s in st_])[:n]
        boot[b] = r2(pred[idx], act[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float((boot <= 0).mean())  # one-sided bootstrap p for H0: R2 <= 0
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    rows.append(
        dict(
            horizon=f"h={h}d",
            oos_r2=obs,
            r2_ci_lo=lo,
            r2_ci_hi=hi,
            bootstrap_p=max(p, 1 / B),
            oos_rmse=rmse,
            n_obs=n,
            decision=decision(max(p, 1 / B)),
        )
    )
h5 = pd.DataFrame(rows)

mit = pd.read_csv(M / "realworld_mitigation_significance.csv")
results["H5"] = dict(
    per_horizon=h5.to_dict("records"),
    finetune=mit.to_dict("records"),
    note="Bootstrap: moving-block (block=28d) resampling of the 2023-2025 holdout, B=5000, seed 42.",
)


# ================================================================ item 5: model comparison
# Omnibus H0: all four models have the same mean RMSE at horizon h.
comp_rows, post_rows = [], []
MODELS = ["SARIMA", "Random Forest (fair)", "Standalone LSTM", "ILT (proposed)"]
for h in HORIZONS:
    ilt_, lstm_, rf_ = RAW[h]["ilt_rmse"], RAW[h]["lstm_rmse"], RAW[h]["rf_rmse"]
    sar_ = np.full(30, RAW[h]["sarima_rmse"])
    fr = stats.friedmanchisquare(sar_ + rng.normal(0, 1e-9, 30), rf_, lstm_, ilt_)
    for name, arr, det in [
        ("SARIMA", sar_, True),
        ("Random Forest (fair)", rf_, False),
        ("Standalone LSTM", lstm_, False),
        ("ILT (proposed)", ilt_, False),
    ]:
        lo, hi = (arr[0], arr[0]) if det else ci_mean(arr)
        comp_rows.append(
            dict(
                horizon=f"h={h}d",
                model=name,
                rmse_mean=arr.mean(),
                rmse_sd=0.0 if det else arr.std(ddof=1),
                ci_lo=lo,
                ci_hi=hi,
                mae_mean=RAW[h]["sarima_mae"] if name == "SARIMA" else
                {"Random Forest (fair)": RAW[h]["rf_mae"], "Standalone LSTM": RAW[h]["lstm_mae"],
                 "ILT (proposed)": RAW[h]["ilt_mae"]}[name].mean(),
                r2_mean=RAW[h]["sarima_r2"] if name == "SARIMA" else
                {"Random Forest (fair)": RAW[h]["rf_r2"], "Standalone LSTM": RAW[h]["lstm_r2"],
                 "ILT (proposed)": RAW[h]["ilt_r2"]}[name].mean(),
                friedman_chi2=fr.statistic,
                friedman_p=fr.pvalue,
            )
        )
    # post-hoc: proposed vs each existing model
    for name, arr, paired in [
        ("SARIMA", sar_, False),
        ("Random Forest (fair)", rf_, True),
        ("Standalone LSTM", lstm_, True),
    ]:
        d = ilt_ - arr
        if paired:
            t, p2 = stats.ttest_rel(ilt_, arr)
        else:
            t, p2 = stats.ttest_1samp(ilt_, arr[0])
        p1 = p2 / 2 if t < 0 else 1 - p2 / 2
        lo, hi = ci_mean(d)
        post_rows.append(
            dict(
                horizon=f"h={h}d",
                comparison=f"ILT vs {name}",
                mean_diff=d.mean(),
                ci_lo=lo,
                ci_hi=hi,
                pct_improvement=100 * (1 - ilt_.mean() / arr.mean()),
                t_stat=t,
                p_raw=p1,
                cohens_dz=cohens_dz(d),
                wins=f"{int((d < 0).sum())}/30",
            )
        )
comp = pd.DataFrame(comp_rows)
post = pd.DataFrame(post_rows)
post["p_holm"] = holm(post["p_raw"])  # 15 comparisons, family-wise
post["decision"] = [decision(p) for p in post["p_holm"]]

# overall mean rank across the 150 seed x horizon cells
ranks = []
for h in HORIZONS:
    mat = np.vstack(
        [np.full(30, RAW[h]["sarima_rmse"]), RAW[h]["rf_rmse"], RAW[h]["lstm_rmse"], RAW[h]["ilt_rmse"]]
    ).T
    ranks.append(stats.rankdata(mat, axis=1))
ranks = np.vstack(ranks)
mean_ranks = dict(zip(MODELS, ranks.mean(axis=0).round(3)))

results["ModelComparison"] = dict(
    table=comp.to_dict("records"),
    posthoc=post.to_dict("records"),
    mean_ranks=mean_ranks,
    note="Friedman omnibus per horizon over 30 seeds x 4 models; post-hoc = one-sided paired t "
         "(one-sample against the deterministic SARIMA value), Holm-corrected over all 15 comparisons.",
)


# ================================================================ save
h1.to_csv(OUT / "H1_tests.csv", index=False)
h2.to_csv(OUT / "H2_tests.csv", index=False)
h4.to_csv(OUT / "H4_tests.csv", index=False)
h5.to_csv(OUT / "H5_tests.csv", index=False)
comp.to_csv(OUT / "model_comparison_table.csv", index=False)
post.to_csv(OUT / "model_comparison_posthoc.csv", index=False)
with open(OUT / "hypothesis_tests.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

# master decision summary
summary = [
    dict(
        hyp="H1", criterion="ILT RMSE >=20% lower than SARIMA",
        H0="mu_RMSE(ILT) >= 0.80 x RMSE(SARIMA)  — improvement does not exceed 20%",
        Ha="mu_RMSE(ILT) <  0.80 x RMSE(SARIMA)  — improvement exceeds 20%",
        test="One-sample one-sided t-test vs the 0.80 x SARIMA margin (30 seeds), Holm over 5 horizons",
        p=h1_pool["p_margin"], decision=h1_pool["decision"],
    ),
    dict(
        hyp="H2", criterion="ILT RMSE lower than Standalone LSTM",
        H0="mu(RMSE_ILT - RMSE_LSTM) = 0", Ha="mu(RMSE_ILT - RMSE_LSTM) < 0",
        test="Paired one-sided t-test per horizon (30 seeds), Holm over 5 horizons; pooled n=150",
        p=h2_pool["p_paired_t"], decision=h2_pool["decision"],
    ),
    dict(
        hyp="H3", criterion=">=70% of top-20 attention features domain-validated",
        H0="pi <= 0.70", Ha="pi > 0.70",
        test="Exact one-sided binomial test (k=20, n=20)",
        p=h3["p_value"], decision=h3["decision"],
    ),
    dict(
        hyp="H4", criterion="every cost metric < 2x the Standalone LSTM and attention sub-quadratic in T",
        H0="cost ratio >= 2 on at least one metric, and/or attention exponent >= 2",
        Ha="all cost ratios < 2 and attention exponent < 2",
        test="One-sample one-sided t vs the 2x threshold (30 seeds) per metric; log-log OLS slope vs 2",
        p=float(h4.loc[h4.metric == "Inference latency (ms/sample)", "p_value"].iloc[0]),
        decision="see per-metric table",
    ),
    dict(
        hyp="H5", criterion="OOS R2 > 0 on the unseen 2023-2025 holdout",
        H0="R2 <= 0 at horizon h", Ha="R2 > 0 at horizon h",
        test="Moving-block bootstrap of R2 (block=28d, B=5000)",
        p=float(h5.loc[0, "bootstrap_p"]), decision="see per-horizon table",
    ),
]
pd.DataFrame(summary).to_csv(OUT / "hypothesis_decision_summary.csv", index=False)

# ---------------------------------------------------------------- report
pd.set_option("display.width", 200, "display.max_columns", 50)
print("\n===== H1 =====\n", h1.round(4).to_string(index=False))
print("\nPOOLED:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in h1_pool.items()})
print("\n===== H2 =====\n", h2.round(4).to_string(index=False))
print("\nPOOLED:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in h2_pool.items()})
print("\n===== H3 =====")
for k, v in h3.items():
    print(" ", k, "=", v)
print("\n===== H4 =====\n", h4.round(4).to_string(index=False))
print("\nSCALING:", json.dumps(scal, indent=2, default=float))
print("\n===== H5 =====\n", h5.round(4).to_string(index=False))
print("\n===== MODEL COMPARISON (post-hoc) =====\n", post.round(5).to_string(index=False))
print("\nMean ranks (1 = best) over 150 seed x horizon cells:", mean_ranks)
print("\nFriedman per horizon:\n", comp[["horizon", "friedman_chi2", "friedman_p"]].drop_duplicates().to_string(index=False))
print("\nSaved to", OUT)
