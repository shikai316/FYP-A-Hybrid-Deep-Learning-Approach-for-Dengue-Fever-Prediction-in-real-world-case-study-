"""
Examiner item 7 — every box plot redrawn showing BOTH the mean and the median,
with the mean labelled, because the performance analysis is stated on the mean.

Convention used in every figure produced here:
    orange line  = median
    red diamond  = mean   (value printed above the box)
    grey dots    = the 30 individual seed results
"""
import ast
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

M = Path(__file__).resolve().parent.parent / "metrics"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 7, 14, 21, 28]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

MEANPROPS = dict(marker="D", markerfacecolor="#d62728", markeredgecolor="#d62728", markersize=5)
MEDIANPROPS = dict(color="#ff7f0e", linewidth=2)


def parse_raw(cell):
    return np.asarray(ast.literal_eval(str(cell).replace("np.float64(", "").replace(")", "")), dtype=float)


mh = pd.read_csv(M / "multi_horizon_results.csv")
RAW = {}
for _, r in mh.iterrows():
    h = int(r["horizon"])
    RAW[h] = {
        "ILT (proposed)": parse_raw(r["ilt_raw_runs"]),
        "Standalone LSTM": parse_raw(r["lstm_raw_runs"]),
        "Random Forest (fair)": parse_raw(r["rf_raw_runs"]),
        "SARIMA": np.full(30, float(r["sarima_rmse"])),
        "ilt_mae": parse_raw(r["ilt_raw_mae"]),
        "lstm_mae": parse_raw(r["lstm_raw_mae"]),
        "rf_mae": parse_raw(r["rf_raw_mae"]),
        "ilt_r2": parse_raw(r["ilt_raw_r2"]),
        "lstm_r2": parse_raw(r["lstm_raw_r2"]),
        "rf_r2": parse_raw(r["rf_raw_r2"]),
        "sarima_mae": float(r["sarima_mae"]),
        "sarima_r2": float(r["sarima_r2"]),
    }


def annotate(ax, data, positions, fmt="{:.2f}", dy=0.02, jitter=True, colors=None):
    """Draw seed dots and print the mean above each box."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, (d, pos) in enumerate(zip(data, positions)):
        if jitter and np.std(d) > 0:
            ax.scatter(
                pos + np.random.default_rng(i).normal(0, 0.045, len(d)),
                d, s=6, color="0.45", alpha=0.45, zorder=1, linewidths=0,
            )
        ax.text(
            pos, np.max(d) + dy * span, fmt.format(np.mean(d)),
            ha="center", va="bottom", fontsize=7.5, color="#d62728", fontweight="bold",
        )


def legend(ax):
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color="#ff7f0e", lw=2, label="median"),
            Line2D([], [], marker="D", color="#d62728", ls="", ms=5, label="mean (analysis is on the mean)"),
            Line2D([], [], marker="o", color="0.45", ls="", ms=4, label="individual seed (n=30)"),
        ],
        loc="best", fontsize=7.5, framealpha=0.9,
    )


# ------------------------------------------------------------ B1: all models x all horizons (RMSE)
fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), sharey=True)
models = ["SARIMA", "Random Forest (fair)", "Standalone LSTM", "ILT (proposed)"]
short = ["SARIMA", "RF", "LSTM", "ILT"]
cols = ["#8c8c8c", "#9467bd", "#1f77b4", "#2ca02c"]
for ax, h in zip(axes, HORIZONS):
    data = [RAW[h][m] for m in models]
    bp = ax.boxplot(data, showmeans=True, meanprops=MEANPROPS, medianprops=MEDIANPROPS,
                    patch_artist=True, widths=0.6, labels=short)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    ax.set_title(f"h = {h} d")
    annotate(ax, data, range(1, 5))
    if ax is axes[0]:
        ax.set_ylabel("Test RMSE (30 seeds)")
axes[0].set_ylim(20, 78)
legend(axes[0])
fig.suptitle("Figure B1 — RMSE distribution by model and forecast horizon (mean ◆ and median shown)", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "B1_rmse_all_models_all_horizons.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ B2: ILT vs LSTM only (H2 focus)
fig, ax = plt.subplots(figsize=(11, 4.4))
data, labels, positions, colors = [], [], [], []
for i, h in enumerate(HORIZONS):
    data += [RAW[h]["Standalone LSTM"], RAW[h]["ILT (proposed)"]]
    labels += [f"LSTM\nh={h}d", f"ILT\nh={h}d"]
    positions += [i * 2.6 + 1, i * 2.6 + 1.9]
    colors += ["#1f77b4", "#2ca02c"]
bp = ax.boxplot(data, positions=positions, showmeans=True, meanprops=MEANPROPS,
                medianprops=MEDIANPROPS, patch_artist=True, widths=0.7, labels=labels)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
ax.set_ylabel("Test RMSE (30 seeds)")
ax.set_ylim(24, 46)
annotate(ax, data, positions)
legend(ax)
ax.set_title("Figure B2 — H2: ILT vs Standalone LSTM, per horizon (mean ◆ and median shown)")
fig.tight_layout()
fig.savefig(OUT / "B2_h2_ilt_vs_lstm.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ B3: paired differences (H2)
fig, ax = plt.subplots(figsize=(8.5, 4.4))
diffs = [RAW[h]["ILT (proposed)"] - RAW[h]["Standalone LSTM"] for h in HORIZONS]
bp = ax.boxplot(diffs, showmeans=True, meanprops=MEANPROPS, medianprops=MEDIANPROPS,
                patch_artist=True, widths=0.6, labels=[f"h={h}d" for h in HORIZONS])
for patch in bp["boxes"]:
    patch.set_facecolor("#2ca02c")
    patch.set_alpha(0.3)
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_ylabel("RMSE(ILT) − RMSE(LSTM), paired by seed")
annotate(ax, diffs, range(1, 6), fmt="{:+.2f}")
legend(ax)
ax.set_title("Figure B3 — H2: paired per-seed difference; below 0 means the attention module helps")
fig.tight_layout()
fig.savefig(OUT / "B3_h2_paired_differences.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ B4: MAE and R2 by model
for metric, key, ylab, ylim in [
    ("MAE", "mae", "Test MAE (30 seeds)", (10, 52)),
    ("R2", "r2", "Test R² (30 seeds)", (-0.45, 0.85)),
]:
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.4), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        data = [
            np.full(30, RAW[h][f"sarima_{key}"]),
            RAW[h][f"rf_{key}"],
            RAW[h][f"lstm_{key}"],
            RAW[h][f"ilt_{key}"],
        ]
        bp = ax.boxplot(data, showmeans=True, meanprops=MEANPROPS, medianprops=MEDIANPROPS,
                        patch_artist=True, widths=0.6, labels=short)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        ax.set_title(f"h = {h} d")
        ax.set_ylim(*ylim)
        annotate(ax, data, range(1, 5), fmt="{:.3f}" if key == "r2" else "{:.2f}")
        if ax is axes[0]:
            ax.set_ylabel(ylab)
    legend(axes[0])
    fig.suptitle(f"Figure B4{'a' if key == 'mae' else 'b'} — {metric} by model and horizon (mean ◆ and median shown)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / f"B4{'a' if key == 'mae' else 'b'}_{key}_all_models.png", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------ B5: H3 per-seed quality
cons = pickle.load(open(M / "h3_30seed_attention_consistency.pkl", "rb"))
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (d, lab, fmt) in zip(
    axes[:2], [(np.array(cons["all_rmse"]), "Test RMSE", "{:.2f}"), (np.array(cons["all_r2"]), "Test R²", "{:.3f}")]
):
    bp = ax.boxplot([d], showmeans=True, meanprops=MEANPROPS, medianprops=MEDIANPROPS,
                    patch_artist=True, widths=0.5, labels=[lab])
    bp["boxes"][0].set_facecolor("#2ca02c")
    bp["boxes"][0].set_alpha(0.35)
    ax.set_ylabel(lab)
    annotate(ax, [d], [1], fmt=fmt)
    ax.set_title(f"H3 model quality — {lab}")
attn = np.asarray(cons["all_mean_attn"], dtype=float)
axes[2].boxplot([attn[:, t] for t in range(attn.shape[1])], showmeans=True, meanprops=dict(MEANPROPS, markersize=3),
                medianprops=MEDIANPROPS, widths=0.6,
                labels=[str(t + 1) if (t + 1) % 4 == 0 else "" for t in range(attn.shape[1])])
axes[2].axhline(1 / attn.shape[1], color="k", ls="--", lw=1, label="uniform 1/28")
axes[2].set_xlabel("time step in the 28-day input window")
axes[2].set_ylabel("attention weight α")
axes[2].set_title("H3 attention profile across 30 seeds")
axes[2].legend(fontsize=7.5)
legend(axes[0])
fig.suptitle("Figure B5 — H3: per-seed model quality and attention profile (mean ◆ and median shown)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "B5_h3_consistency.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ B6: H4 optimiser comparison
eff = pickle.load(open(M / "efficiency_results_FIXED.pkl", "rb"))
opt = eff["optimiser_results"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, key, ylab, fmt in [
    (axes[0], "all_rmse", "Test RMSE (30 seeds)", "{:.2f}"),
    (axes[1], "all_train_time", "Training time (s, 30 seeds)", "{:.1f}"),
]:
    data = [np.asarray(o[key], dtype=float) for o in opt]
    names = [o["Optimiser"] for o in opt]
    bp = ax.boxplot(data, showmeans=True, meanprops=MEANPROPS, medianprops=MEDIANPROPS,
                    patch_artist=True, widths=0.6, labels=names)
    for patch in bp["boxes"]:
        patch.set_facecolor("#ff7f0e")
        patch.set_alpha(0.3)
    ax.set_ylabel(ylab)
    annotate(ax, data, range(1, len(data) + 1), fmt=fmt)
legend(axes[0])
fig.suptitle("Figure B6 — H4: optimiser comparison over 30 seeds (mean ◆ and median shown)", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "B6_h4_optimiser.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ B7: H4 scaling with fitted exponents
sc = pd.read_csv(M / "bigo_scaling_experiment.csv")
fig, ax = plt.subplots(figsize=(6.5, 4.4))
from scipy import stats as st

for col, lab, c in [
    ("TemporalAttention_ms", "TemporalAttention (proposed)", "#2ca02c"),
    ("MultiHeadAttention_ms", "Full self-attention", "#d62728"),
]:
    lr = st.linregress(np.log(sc["T"]), np.log(sc[col]))
    ax.loglog(sc["T"], sc[col], "o-", color=c, label=f"{lab}: T^{lr.slope:.2f}")
    ax.loglog(sc["T"], np.exp(lr.intercept) * sc["T"] ** lr.slope, "--", color=c, alpha=0.5)
ax.loglog(sc["T"], sc["TemporalAttention_ms"].iloc[0] * (sc["T"] / sc["T"].iloc[0]) ** 2,
          ":", color="0.4", label="quadratic reference T²")
ax.set_xlabel("sequence length T")
ax.set_ylabel("forward-pass time (ms)")
ax.legend(fontsize=8)
ax.set_title("Figure B7 — H4: attention cost scaling vs the quadratic reference")
fig.tight_layout()
fig.savefig(OUT / "B7_h4_scaling.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ B8: H5 out-of-sample R2 with bootstrap CI
h5 = pd.read_csv(Path(__file__).resolve().parent / "H5_tests.csv")
rw = pd.read_csv(M / "realworld_generalisation.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.4))
x = np.arange(len(HORIZONS))
ax.errorbar(x, h5["oos_r2"], yerr=[h5["oos_r2"] - h5["r2_ci_lo"], h5["r2_ci_hi"] - h5["oos_r2"]],
            fmt="D", color="#d62728", capsize=5, label="ensemble forecast R² ± 95% block-bootstrap CI")
ax.plot(x, rw["OOS_R2"], "o--", color="#1f77b4", label="30-seed mean R² (nb10)")
ax.plot(x, rw["Within_R2"], "s--", color="0.5", label="within-sample R² (2013–2022)")
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xticks(x, [f"h={h}d" for h in HORIZONS])
ax.set_ylabel("R²")
ax.legend(fontsize=8)
ax.set_title("Figure B8 — H5: out-of-sample skill on the 2023–2025 holdout, with uncertainty")
fig.tight_layout()
fig.savefig(OUT / "B8_h5_oos_r2.png", bbox_inches="tight")
plt.close(fig)

print("figures written to", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name, f"{p.stat().st_size/1024:.0f} KB")
