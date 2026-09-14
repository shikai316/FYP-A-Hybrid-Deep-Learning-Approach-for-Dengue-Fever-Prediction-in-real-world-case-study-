"""
16_rf_feature_parity_multihorizon.py
======================================
Viva follow-up test: "Is it fair to restrict Random Forest to weather-only
features? If RF got the SAME features ILT uses (including case-history),
would it close the gap?"

This script trains TWO Random Forest variants, with IDENTICAL hyperparameters
and the IDENTICAL chronological train/test split, at the SAME 5 forecast
horizons used everywhere else in this project (1, 7, 14, 21, 28 days):

  1. RF (weather-only, 44 features)   - the existing "fair" baseline
                                         (Notebooks 03/08/11/12/13).
  2. RF (full-feature, 60 features)   - same RF, given the SAME feature
                                         columns as ILT's sequence branch
                                         (i.e. including case_lag_*,
                                         cases_mean_*, cases_std_*, etc.)

Only the feature set changes between (1) and (2). Same RandomForestRegressor
hyperparameters, same random_state, same horizon-shift convention. This
isolates the FEATURE-SET effect from the ARCHITECTURE effect, which is the
correct way to answer "is RF bad because it's a worse algorithm, or because
it was starved of inputs?"

SARIMA is also refit fresh here (not loaded from the old models/sarima_model.pkl,
which was trained on a different, non-canonical feature pipeline with a
different dropna() cutoff and therefore a different train/test row count) so
that every number in the comparison table comes from the exact same
date-aligned split as the two RF variants.

For context, the existing 30-seed-mean ILT / Standalone LSTM numbers
(Notebook 15, metrics/full_reproduction_*_30run.csv) are loaded alongside -
NOT retrained here, since that requires TensorFlow/GPU time this script does
not need and the authoritative numbers already exist.

Design notes / what this script deliberately does NOT do:
  - It does NOT flatten the 28-day window into RF (that would be a 1,680-
    column model and a different, much larger experiment). It gives RF the
    SAME per-day engineered columns ILT's sequence branch sees (e.g.
    cases_lag_14 IS "case count 14 days ago", cases_mean_28 IS "rolling mean
    of the last 28 days") as a flat snapshot - the natural reading of
    "use the same features as ILT" for a non-sequence model.
  - It does NOT retune RF's hyperparameters for the full-feature variant.
    Keeping n_estimators/max_depth/etc. fixed at the project's existing
    values is what makes this a clean ablation (only one thing changes).
  - It DOES support repeating each RF fit over multiple seeds (RF_SEEDS
    below) to report mean +/- std rather than a single lucky/unlucky run,
    in the same spirit as the project's 30-seed DL protocol. Originally
    shipped with a single seed (42) to keep runtime short; promoted to a
    true-random 30-seed Monte Carlo set (matching Notebooks 08/12/15's
    convention) after viva prep flagged the single-seed result as an
    unverified robustness gap -- see RF_SEEDS below.

`engineer_features()` and `get_feature_sets()` below are copied verbatim from
canonical_pipeline.py (the project's single source of truth for feature
engineering) rather than imported, ONLY so this script has zero TensorFlow
dependency and can run anywhere sklearn/pandas/pmdarima are installed. If
canonical_pipeline.py's feature engineering ever changes, re-sync these two
functions from there.

Outputs:
  metrics/rf_feature_parity_multihorizon.csv   - full per-horizon table
  plots/rf_feature_parity_multihorizon.png     - RMSE & R^2 vs horizon chart
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

CSV_PATH = "singapore_dengue_weather_2013_2022.csv"
TRAIN_FRAC = 0.8
HORIZONS = [1, 7, 14, 21, 28]          # identical convention to Notebooks 08/15
# 30-seed Monte Carlo robustness check (true random draws, not reproducible by
# design -- matches Notebooks 08/12/15's canonical seeding convention). Replaces
# the original single-seed [42] default once this finding became a viva-facing
# claim ("RF given the same features as ILT wins at every horizon") that needed
# the same statistical rigor as every other headline result in this project.
N_SEEDS = 30
RF_SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)
print(f"RF_SEEDS ({N_SEEDS}, freshly drawn this run): {RF_SEEDS}")
RF_KWARGS = dict(
    n_estimators=300, max_depth=12,
    min_samples_split=10, min_samples_leaf=5,
    max_features="sqrt", n_jobs=-1,
)

os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ----------------------------------------------------------------------------
# Feature engineering - verbatim copy from canonical_pipeline.py
# (kept inline to avoid pulling in the TensorFlow import chain for a
#  Random-Forest-only experiment; see module docstring)
# ----------------------------------------------------------------------------
def engineer_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

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
        df[f"cases_lag_{lag}"] = df["dengue_cases"].shift(lag)

    for window in [7, 14, 28]:
        df[f"cases_mean_{window}"] = df["dengue_cases"].shift(14).rolling(window).mean()
        df[f"cases_std_{window}"] = df["dengue_cases"].shift(14).rolling(window).std()
        df[f"cases_max_{window}"] = df["dengue_cases"].shift(14).rolling(window).max()

    for window in [7, 14, 28]:
        df[f"temp_mean_{window}"] = df["temperature"].rolling(window).mean()
        df[f"rain_sum_{window}"] = df["rainfall"].rolling(window).sum()
        df[f"humidity_mean_{window}"] = df["humidity"].rolling(window).mean()

    for lag in [14, 21, 28]:
        df[f"rain_lag_{lag}"] = df["rainfall"].shift(lag)
    for lag in [14, 21]:
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)
    df["humidity_lag_14"] = df["humidity"].shift(14)

    df["cases_ema_14"] = df["dengue_cases"].shift(14).ewm(span=14).mean()
    df["cases_ema_28"] = df["dengue_cases"].shift(14).ewm(span=28).mean()
    df["cases_diff_14"] = df["dengue_cases"].diff(14)
    df["temp_diff"] = df["temperature"].diff()
    df["rain_diff"] = df["rainfall"].diff()

    return df.dropna().reset_index(drop=True)


def get_feature_sets(df_enhanced: pd.DataFrame):
    """Returns (rf_features [44], dl_seq_features [60], dl_feat_features [=rf_features])."""
    exclude_cols = ["date", "dengue_cases", "year"]
    feature_columns = [c for c in df_enhanced.columns if c not in exclude_cols]

    case_pfx = ["cases_lag_", "cases_mean_", "cases_std_", "cases_max_",
                "cases_ema_", "cases_diff_"]
    case_exact = ["cases_7d_avg"]
    rf_features = [c for c in feature_columns
                   if not any(c.startswith(p) for p in case_pfx) and c not in case_exact]
    dl_seq_features = feature_columns
    return rf_features, dl_seq_features, rf_features


def horizon_shift(X, y, h):
    """Same convention as Notebook 08's run_horizon(): X[:-h] predicts y[h:]."""
    return X[: len(X) - h], y[h:]


def fit_rf_multi_seed(X_tr, y_tr, X_te, y_te, seeds):
    """Fit RF over several seeds, return per-seed metrics + train metrics for the first seed."""
    test_rmse, test_mae, test_r2 = [], [], []
    train_rmse_first, train_r2_first = None, None
    for i, seed in enumerate(seeds):
        rf = RandomForestRegressor(random_state=seed, **RF_KWARGS)
        rf.fit(X_tr, y_tr)
        p_te = rf.predict(X_te)
        test_rmse.append(np.sqrt(mean_squared_error(y_te, p_te)))
        test_mae.append(mean_absolute_error(y_te, p_te))
        test_r2.append(r2_score(y_te, p_te))
        if i == 0:
            p_tr = rf.predict(X_tr)
            train_rmse_first = np.sqrt(mean_squared_error(y_tr, p_tr))
            train_r2_first = r2_score(y_tr, p_tr)
    return dict(
        test_rmse_mean=np.mean(test_rmse), test_rmse_std=np.std(test_rmse),
        test_mae_mean=np.mean(test_mae), test_mae_std=np.std(test_mae),
        test_r2_mean=np.mean(test_r2), test_r2_std=np.std(test_r2),
        train_rmse=train_rmse_first, train_r2=train_r2_first,
    )


def main():
    print("=" * 78)
    print("  RF Feature-Parity Test - does giving RF the same features as ILT help?")
    print("=" * 78)

    df = engineer_features(CSV_PATH)
    rf_features, dl_seq_features, _ = get_feature_sets(df)
    print(f"Engineered dataset: {df.shape[0]} rows")
    print(f"  rf_features (weather-only):     {len(rf_features)} cols")
    print(f"  dl_seq_features (full/ILT-par): {len(dl_seq_features)} cols")
    assert len(rf_features) == 44, f"Expected 44 weather-only features, got {len(rf_features)}"
    assert len(dl_seq_features) == 60, f"Expected 60 full features, got {len(dl_seq_features)}"

    train_size = int(len(df) * TRAIN_FRAC)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    print(f"Chronological split: train={len(train_df)} days, test={len(test_df)} days "
          f"(train ends {train_df['date'].iloc[-1].date()}, "
          f"test ends {test_df['date'].iloc[-1].date()})")

    y_train_full = train_df["dengue_cases"].values
    y_test_full = test_df["dengue_cases"].values

    weather_scaler = RobustScaler().fit(train_df[rf_features].values)
    full_scaler = RobustScaler().fit(train_df[dl_seq_features].values)
    X_train_weather = weather_scaler.transform(train_df[rf_features].values)
    X_test_weather = weather_scaler.transform(test_df[rf_features].values)
    X_train_full = full_scaler.transform(train_df[dl_seq_features].values)
    X_test_full = full_scaler.transform(test_df[dl_seq_features].values)

    # -- SARIMA: refit fresh on THIS split for row-perfect alignment --
    print("\nFitting SARIMA fresh on the canonical split (for date alignment "
          "with the RF variants above)...")
    from pmdarima import auto_arima
    sarima_model = auto_arima(
        train_df["dengue_cases"], seasonal=True, m=7,
        max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=1, D=0,
        trace=False, error_action="ignore", suppress_warnings=True, stepwise=True)
    sarima_pred_full = sarima_model.predict(n_periods=len(test_df))
    print(f"  SARIMA order={sarima_model.order}, seasonal={sarima_model.seasonal_order}")

    # -- 30-seed DL reference numbers (loaded, not retrained - see docstring) --
    dl_ref = {}
    rmse_csv = "metrics/full_reproduction_rmse_30run.csv"
    r2_csv = "metrics/full_reproduction_r2_30run.csv"
    if os.path.exists(rmse_csv) and os.path.exists(r2_csv):
        rmse_df = pd.read_csv(rmse_csv)
        r2_df = pd.read_csv(r2_csv)
        for h in HORIZONS:
            dl_ref[h] = dict(
                ilt_rmse=rmse_df[f"ILT_h{h}d"].mean(),
                ilt_r2=r2_df[f"ILT_h{h}d"].mean(),
                lstm_rmse=rmse_df[f"Standalone LSTM_h{h}d"].mean(),
                lstm_r2=r2_df[f"Standalone LSTM_h{h}d"].mean(),
            )
        print(f"  Loaded 30-seed ILT / Standalone LSTM reference numbers from {rmse_csv}")
    else:
        print("  (30-seed reference CSVs not found - table will omit ILT/LSTM reference columns)")

    rows = []
    for h in HORIZONS:
        print(f"\n--- Horizon h={h}d ---")

        # SARIMA
        s_pred = sarima_pred_full[h:]
        s_actual = y_test_full[h:]
        n = min(len(s_pred), len(s_actual))
        s_rmse = np.sqrt(mean_squared_error(s_actual[:n], s_pred[:n]))
        s_r2 = r2_score(s_actual[:n], s_pred[:n])
        print(f"  SARIMA              RMSE={s_rmse:.2f}  R2={s_r2:.4f}")

        # RF weather-only (44 features) - existing baseline, reproduced here
        Xw_tr, yw_tr = horizon_shift(X_train_weather, y_train_full, h)
        Xw_te, yw_te = horizon_shift(X_test_weather, y_test_full, h)
        rf_w = fit_rf_multi_seed(Xw_tr, yw_tr, Xw_te, yw_te, RF_SEEDS)
        print(f"  RF (weather-only,44) RMSE={rf_w['test_rmse_mean']:.2f}"
              f"+/-{rf_w['test_rmse_std']:.2f}  R2={rf_w['test_r2_mean']:.4f}"
              f"+/-{rf_w['test_r2_std']:.4f}  (train R2={rf_w['train_r2']:.4f})")

        # RF full-feature (60 features, ILT-parity, incl. case-history)
        Xf_tr, yf_tr = horizon_shift(X_train_full, y_train_full, h)
        Xf_te, yf_te = horizon_shift(X_test_full, y_test_full, h)
        rf_f = fit_rf_multi_seed(Xf_tr, yf_tr, Xf_te, yf_te, RF_SEEDS)
        print(f"  RF (full,60,ILT-par) RMSE={rf_f['test_rmse_mean']:.2f}"
              f"+/-{rf_f['test_rmse_std']:.2f}  R2={rf_f['test_r2_mean']:.4f}"
              f"+/-{rf_f['test_r2_std']:.4f}  (train R2={rf_f['train_r2']:.4f})")

        row = dict(
            horizon=h,
            sarima_rmse=s_rmse, sarima_r2=s_r2,
            rf_weather_rmse_mean=rf_w["test_rmse_mean"], rf_weather_rmse_std=rf_w["test_rmse_std"],
            rf_weather_r2_mean=rf_w["test_r2_mean"], rf_weather_r2_std=rf_w["test_r2_std"],
            rf_weather_train_rmse=rf_w["train_rmse"], rf_weather_train_r2=rf_w["train_r2"],
            rf_full_rmse_mean=rf_f["test_rmse_mean"], rf_full_rmse_std=rf_f["test_rmse_std"],
            rf_full_r2_mean=rf_f["test_r2_mean"], rf_full_r2_std=rf_f["test_r2_std"],
            rf_full_train_rmse=rf_f["train_rmse"], rf_full_train_r2=rf_f["train_r2"],
        )
        if h in dl_ref:
            row.update(
                ilt_rmse_30seed=dl_ref[h]["ilt_rmse"], ilt_r2_30seed=dl_ref[h]["ilt_r2"],
                lstm_rmse_30seed=dl_ref[h]["lstm_rmse"], lstm_r2_30seed=dl_ref[h]["lstm_r2"],
            )
        rows.append(row)

    results = pd.DataFrame(rows)
    out_csv = "metrics/rf_feature_parity_multihorizon.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # -- Summary table --
    print("\n" + "=" * 100)
    print("  SUMMARY - RMSE (test set), all horizons")
    print("=" * 100)
    header = f"{'h':>4} | {'SARIMA':>10} | {'RF weather(44)':>16} | {'RF full(60)':>16}"
    if "ilt_rmse_30seed" in results.columns:
        header += f" | {'Std LSTM(30s)':>14} | {'ILT(30s)':>10}"
    print(header)
    print("-" * 100)
    for _, r in results.iterrows():
        line = (f"{int(r['horizon']):>4} | {r['sarima_rmse']:>10.2f} | "
                f"{r['rf_weather_rmse_mean']:>9.2f}+/-{r['rf_weather_rmse_std']:<4.2f} | "
                f"{r['rf_full_rmse_mean']:>9.2f}+/-{r['rf_full_rmse_std']:<4.2f}")
        if "ilt_rmse_30seed" in results.columns:
            line += f" | {r['lstm_rmse_30seed']:>14.2f} | {r['ilt_rmse_30seed']:>10.2f}"
        print(line)

    print("\n" + "=" * 100)
    print("  SUMMARY - R^2 (test set), all horizons")
    print("=" * 100)
    header = f"{'h':>4} | {'SARIMA':>10} | {'RF weather(44)':>16} | {'RF full(60)':>16}"
    if "ilt_r2_30seed" in results.columns:
        header += f" | {'Std LSTM(30s)':>14} | {'ILT(30s)':>10}"
    print(header)
    print("-" * 100)
    for _, r in results.iterrows():
        line = (f"{int(r['horizon']):>4} | {r['sarima_r2']:>10.4f} | "
                f"{r['rf_weather_r2_mean']:>9.4f}+/-{r['rf_weather_r2_std']:<4.4f} | "
                f"{r['rf_full_r2_mean']:>9.4f}+/-{r['rf_full_r2_std']:<4.4f}")
        if "ilt_r2_30seed" in results.columns:
            line += f" | {r['lstm_r2_30seed']:>14.4f} | {r['ilt_r2_30seed']:>10.4f}"
        print(line)

    # -- Interpretation --
    print("\n" + "=" * 100)
    print("  INTERPRETATION")
    print("=" * 100)
    gap_closed_pct = (
        (results["rf_weather_rmse_mean"] - results["rf_full_rmse_mean"])
        / results["rf_weather_rmse_mean"] * 100
    )
    for h, pct, r2f, r2w in zip(results["horizon"], gap_closed_pct,
                                results["rf_full_r2_mean"], results["rf_weather_r2_mean"]):
        print(f"  h={h:>2}d: giving RF the full feature set changed test RMSE by {pct:+.1f}% "
              f"(R^2 {r2w:.3f} -> {r2f:.3f})")
    if "ilt_rmse_30seed" in results.columns:
        still_behind = (results["rf_full_rmse_mean"] > results["ilt_rmse_30seed"]).all()
        print(f"\n  RF (full-feature) still behind ILT at every horizon: {still_behind}")

    # -- Chart --
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        hs = results["horizon"].values

        ax = axes[0]
        ax.plot(hs, results["sarima_rmse"], "^-", color="#95a5a6", label="SARIMA")
        ax.errorbar(hs, results["rf_weather_rmse_mean"], yerr=results["rf_weather_rmse_std"],
                    fmt="o-", color="#2ecc71", label="RF (weather-only, 44)")
        ax.errorbar(hs, results["rf_full_rmse_mean"], yerr=results["rf_full_rmse_std"],
                    fmt="D-", color="#e67e22", label="RF (full feature, 60, ILT-parity)")
        if "ilt_rmse_30seed" in results.columns:
            ax.plot(hs, results["lstm_rmse_30seed"], "v-", color="#9b59b6", label="Standalone LSTM (30-seed mean)")
            ax.plot(hs, results["ilt_rmse_30seed"], "s-", color="#3498db", lw=2.5, label="ILT (30-seed mean)")
        ax.set_xlabel("Forecast Horizon (days)"); ax.set_ylabel("RMSE")
        ax.set_title("RMSE vs Horizon - does feature parity close the gap?", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(hs)

        ax = axes[1]
        ax.plot(hs, results["sarima_r2"], "^-", color="#95a5a6", label="SARIMA")
        ax.errorbar(hs, results["rf_weather_r2_mean"], yerr=results["rf_weather_r2_std"],
                    fmt="o-", color="#2ecc71", label="RF (weather-only, 44)")
        ax.errorbar(hs, results["rf_full_r2_mean"], yerr=results["rf_full_r2_std"],
                    fmt="D-", color="#e67e22", label="RF (full feature, 60, ILT-parity)")
        if "ilt_r2_30seed" in results.columns:
            ax.plot(hs, results["lstm_r2_30seed"], "v-", color="#9b59b6", label="Standalone LSTM (30-seed mean)")
            ax.plot(hs, results["ilt_r2_30seed"], "s-", color="#3498db", lw=2.5, label="ILT (30-seed mean)")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Forecast Horizon (days)"); ax.set_ylabel("R^2")
        ax.set_title("R^2 vs Horizon - does feature parity close the gap?", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(hs)

        plt.tight_layout()
        plt.savefig("plots/rf_feature_parity_multihorizon.png", dpi=150, bbox_inches="tight")
        print("\nSaved plots/rf_feature_parity_multihorizon.png")
    except Exception as e:
        print(f"\n(chart skipped: {e})")

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
