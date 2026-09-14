"""
viva_revision_common.py
=======================
Shared plumbing for the three post-viva revision notebooks (17, 18, 19).

Written 2026-08-04 in response to the viva-voce feedback of 31 Jul 2026:
    item 6 -> Notebook 17 (fine-tuning patch)
    item 8 -> Notebook 18 (all five windows, both readings)
    item 9 -> Notebook 19 (dataset splitting analysis)

Design rules followed here:
  * Feature engineering is byte-for-byte the pipeline used by Notebook 10
    (including its 99th-percentile weather cap), so any number produced by
    Notebook 17 is directly comparable with the existing H5 Arm-A figures.
    Notebooks 18 and 19 use the same function for internal consistency.
  * Model architecture is imported from canonical_pipeline.py — the single
    source of truth — so nothing is redefined and allowed to drift.
  * SEEDS is the same fixed list used everywhere else: [42 + 7i for i in range(30)].
  * Every helper returns per-seed arrays, never just a mean, so the workbook
    can show distributions with both mean and median (viva item 7).
"""

import gc
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from canonical_pipeline import (  # noqa: F401  (re-exported for the notebooks)
    SEQ_LEN,
    TemporalAttention,
    build_lstm_transformer,
    build_standalone_lstm,
    make_sequences,
)

N_SEEDS = 30
SEEDS = [42 + 7 * i for i in range(N_SEEDS)]
HORIZONS = [1, 7, 14, 21, 28]
ALPHA = 0.05

BENCH_CSV = "singapore_dengue_weather_2013_2022.csv"
HOLDOUT_CSV = "singapore_dengue_weather_2023_2025.csv"


# ──────────────────────────────────────────────────────────────────────────
# 1.  Feature engineering — identical to Notebook 10's build_features()
# ──────────────────────────────────────────────────────────────────────────
def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["temperature", "temperature_max", "temperature_min",
                "rainfall", "humidity", "windspeed", "windspeed_max"]:
        df[col] = df[col].clip(upper=df[col].quantile(0.99))

    if "week_of_year" not in df.columns:
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    df["temp_humidity"] = df["temperature"] * df["humidity"] / 100
    df["rain_humidity"] = df["rainfall"] * df["humidity"] / 100
    df["temp_rain"] = df["temperature"] * df["rainfall"]
    df["temp_range"] = df["temperature_max"] - df["temperature_min"]

    for lag in [14, 21, 28]:
        df[f"rain_lag_{lag}"] = df["rainfall"].shift(lag)
    for lag in [14, 21]:
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)
    df["humidity_lag_14"] = df["humidity"].shift(14)

    for lag in [14, 21, 28]:
        df[f"cases_lag_{lag}"] = df["dengue_cases"].shift(lag)
    for w in [7, 14, 28]:
        df[f"cases_mean_{w}"] = df["dengue_cases"].shift(14).rolling(w).mean()
        df[f"cases_std_{w}"] = df["dengue_cases"].shift(14).rolling(w).std()
        df[f"cases_max_{w}"] = df["dengue_cases"].shift(14).rolling(w).max()
    df["cases_ema_14"] = df["dengue_cases"].shift(14).ewm(span=14).mean()
    df["cases_ema_28"] = df["dengue_cases"].shift(14).ewm(span=28).mean()
    df["cases_diff_14"] = df["dengue_cases"].diff(14)

    for w in [7, 14, 28]:
        df[f"temp_mean_{w}"] = df["temperature"].rolling(w).mean()
        df[f"rain_sum_{w}"] = df["rainfall"].rolling(w).sum()
        df[f"humidity_mean_{w}"] = df["humidity"].rolling(w).mean()

    df["temp_diff"] = df["temperature"].diff()
    df["rain_diff"] = df["rainfall"].diff()
    return df.dropna().reset_index(drop=True)


def feature_sets(df: pd.DataFrame):
    """(rf_features, dl_seq_features, dl_feat_features) — same rule as canonical_pipeline."""
    excl = ["date", "dengue_cases", "year"]
    all_cols = [c for c in df.columns if c not in excl]
    case_pfx = ["cases_lag_", "cases_mean_", "cases_std_", "cases_max_", "cases_ema_", "cases_diff_"]
    rf = [c for c in all_cols if not any(c.startswith(p) for p in case_pfx) and c != "cases_7d_avg"]
    return rf, all_cols, rf


# ──────────────────────────────────────────────────────────────────────────
# 2.  Metrics + error decomposition
# ──────────────────────────────────────────────────────────────────────────
def score(y_true, y_pred) -> dict:
    """RMSE / MAE / R2 plus the decomposition used to justify a negative R2.

    MSE = bias^2 + variance-of-error, and R2 = 1 - MSE / Var(y_true).
    Reporting `bias` and `scale_ratio` separately shows *why* R2 went negative:
    a systematic level offset (bias) behaves very differently from noise.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    var_y = float(np.var(y_true))
    bias = float(err.mean())
    return dict(
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        bias=bias,
        bias_sq_share=float(bias ** 2 / mse) if mse > 0 else np.nan,
        err_var=float(err.var()),
        pred_mean=float(y_pred.mean()),
        actual_mean=float(y_true.mean()),
        pred_sd=float(y_pred.std()),
        actual_sd=float(y_true.std()),
        scale_ratio=float(y_pred.std() / y_true.std()) if y_true.std() > 0 else np.nan,
        var_y=var_y,
        n=int(len(y_true)),
    )


def fit_target_scaler(y, clip_pct=95):
    """MinMax(0.02, 0.98) target scaler, optionally clipped at a percentile.

    clip_pct=None disables clipping (the plain Notebook 08 behaviour).
    """
    y = np.asarray(y, dtype=float)
    if clip_pct is not None:
        y = np.clip(y, 0, np.percentile(y, clip_pct))
    sc = MinMaxScaler(feature_range=(0.02, 0.98))
    sc.fit(y.reshape(-1, 1))
    return sc


def inverse(scaler, p_scaled):
    return np.maximum(scaler.inverse_transform(np.asarray(p_scaled).reshape(-1, 1)).ravel(), 0)


# ──────────────────────────────────────────────────────────────────────────
# 3.  Training helpers (per-seed, memory-safe)
# ──────────────────────────────────────────────────────────────────────────
def train_model(seed, builder, Xs_tr, Xf_tr, y_tr_scaled, val_frac=0.15,
                epochs=100, patience=20, lr=1e-3, verbose=0, **kw):
    """Train one seeded model; returns the fitted keras model."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    n_seq, n_feat = Xs_tr.shape[-1], Xf_tr.shape[-1]
    m = builder(Xs_tr.shape[1], n_seq, n_feat, pfx=f"s{seed}_", learning_rate=lr, **kw)
    vs = max(int(len(Xs_tr) * val_frac), 30)
    m.fit(
        [Xs_tr[:-vs], Xf_tr[:-vs]], y_tr_scaled[:-vs],
        validation_data=([Xs_tr[-vs:], Xf_tr[-vs:]], y_tr_scaled[-vs:]),
        epochs=epochs, batch_size=32, verbose=verbose,
        callbacks=[
            EarlyStopping("val_loss", patience=patience, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=0),
        ],
    )
    return m


def predict(m, Xs, Xf, scaler):
    return inverse(scaler, m.predict([Xs, Xf], verbose=0).ravel())


def release(m):
    """Free the model and the TF graph — without this the kernel dies part-way
    through a 30-seed loop (the memory leak fixed in Notebook 10 on 2026-06-26)."""
    del m
    tf.keras.backend.clear_session()
    gc.collect()


# ──────────────────────────────────────────────────────────────────────────
# 4.  Statistics — every test reports H0, p, and an explicit decision
# ──────────────────────────────────────────────────────────────────────────
from scipy import stats  # noqa: E402


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    adj, running = np.empty(n), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def decide(p, alpha=ALPHA):
    return "Reject H0" if p < alpha else "Fail to reject H0"


def ci_mean(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    h = stats.sem(x) * stats.t.ppf((1 + conf) / 2, len(x) - 1)
    return float(x.mean() - h), float(x.mean() + h)


def paired_test(a, b, alternative="less"):
    """Paired comparison of two per-seed arrays. Ha: mean(a - b) < 0 by default."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t, p2 = stats.ttest_rel(a, b)
    p1 = p2 / 2 if (t < 0) == (alternative == "less") else 1 - p2 / 2
    lo, hi = ci_mean(d)
    try:
        w = stats.wilcoxon(d, alternative=alternative).pvalue
    except ValueError:
        w = np.nan
    return dict(
        mean_a=float(a.mean()), mean_b=float(b.mean()), mean_diff=float(d.mean()),
        ci_lo=lo, ci_hi=hi, t_stat=float(t), p_paired_t=float(p1), wilcoxon_p=float(w),
        cohens_dz=float(d.mean() / d.std(ddof=1)), a_wins=f"{int((d < 0).sum())}/{len(d)}",
    )


def one_sample_vs(x, mu0, alternative="greater"):
    """H0: mean(x) <= mu0 (alternative='greater')  or  mean(x) >= mu0 ('less')."""
    x = np.asarray(x, float)
    t, p2 = stats.ttest_1samp(x, mu0)
    p1 = p2 / 2 if (t > 0) == (alternative == "greater") else 1 - p2 / 2
    lo, hi = ci_mean(x)
    return dict(
        mean=float(x.mean()), sd=float(x.std(ddof=1)), ci_lo=lo, ci_hi=hi,
        mu0=float(mu0), t_stat=float(t), p_value=float(p1),
        cohens_d=float((x.mean() - mu0) / x.std(ddof=1)), decision=decide(p1),
    )


def summarise(x):
    x = np.asarray(x, float)
    lo, hi = ci_mean(x)
    return dict(mean=float(x.mean()), median=float(np.median(x)), sd=float(x.std(ddof=1)),
                ci_lo=lo, ci_hi=hi, min=float(x.min()), max=float(x.max()), n=int(len(x)))


# ──────────────────────────────────────────────────────────────────────────
# 5.  Box plot with BOTH mean and median (viva item 7)
# ──────────────────────────────────────────────────────────────────────────
def boxplot_mean_median(ax, data, labels, ylabel="", title="", fmt="{:.2f}", colors=None):
    """House style for every box plot in the revision: orange median line,
    red diamond mean with the value printed, grey per-seed dots.

    Empty series are dropped with a warning rather than raising: a splitting
    strategy or arm that produced no usable folds should not take the whole
    figure down with it.
    """
    import matplotlib.pyplot as plt  # noqa: F401

    keep = [i for i, d in enumerate(data) if len(np.asarray(d).ravel()) > 0]
    if len(keep) < len(data):
        dropped = [labels[i] for i in range(len(data)) if i not in keep]
        print(f"  [boxplot] no data for {dropped} — omitted from '{title or ylabel}'")
    if not keep:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="0.5")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return None
    data = [np.asarray(data[i], float).ravel() for i in keep]
    labels = [labels[i] for i in keep]
    if colors:
        colors = [colors[i] for i in keep]

    bp = ax.boxplot(
        data, labels=labels, showmeans=True, patch_artist=True, widths=0.6,
        meanprops=dict(marker="D", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=5),
        medianprops=dict(color="#ff7f0e", linewidth=2),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor((colors[i] if colors else "#2ca02c"))
        patch.set_alpha(0.35)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, d in enumerate(data, start=1):
        d = np.asarray(d, float)
        ax.scatter(i + np.random.default_rng(i).normal(0, 0.045, len(d)), d,
                   s=6, color="0.45", alpha=0.45, zorder=1, linewidths=0)
        ax.text(i, d.max() + 0.02 * span, fmt.format(d.mean()), ha="center", va="bottom",
                fontsize=7.5, color="#d62728", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return bp


LEGEND_NOTE = ("orange line = median · red diamond = mean (the analysis is stated on the mean) "
               "· grey dots = individual seeds")

print(f"viva_revision_common loaded — {N_SEEDS} seeds {SEEDS[0]}..{SEEDS[-1]}, horizons {HORIZONS}")
