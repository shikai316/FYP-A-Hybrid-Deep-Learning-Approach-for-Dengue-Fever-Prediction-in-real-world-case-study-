import json
import uuid

PATH = '11_hyperparameter_tuning_analysis_FIXED.ipynb'
nb = json.load(open(PATH, encoding='utf-8'))
cells = nb['cells']


def get_src(i):
    return ''.join(cells[i]['source'])


def set_src(i, text):
    cells[i]['source'] = text.splitlines(keepends=True)


def replace_unique(i, old, new):
    src = get_src(i)
    cnt = src.count(old)
    assert cnt == 1, f"cell {i}: expected exactly 1 match, found {cnt}\n---OLD---\n{old}"
    set_src(i, src.replace(old, new))


def new_code_cell(src):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'id': uuid.uuid4().hex[:8],
        'metadata': {},
        'outputs': [],
        'source': src.splitlines(keepends=True),
    }


def new_md_cell(src):
    return {
        'cell_type': 'markdown',
        'id': uuid.uuid4().hex[:8],
        'metadata': {},
        'source': src.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------
# Cell 10: add an optional `seed` parameter to evaluate_dl_config so
# the new 30-seed validation cells can reuse it without touching the
# existing grid-search behaviour (all existing call sites omit `seed`,
# so they keep using the fixed seed=42 exactly as before).
# ---------------------------------------------------------------------
replace_unique(10,
    "def evaluate_dl_config(lstm_units, lr, batch_size, epochs=50, patience=12):\n"
    "    tf.random.set_seed(42); np.random.seed(42)",

    "def evaluate_dl_config(lstm_units, lr, batch_size, epochs=50, patience=12, seed=42):\n"
    "    tf.random.set_seed(seed); np.random.seed(seed)"
)

# ---------------------------------------------------------------------
# Append new "Part 4" section at the end: re-validate ONLY the two
# final chosen configs (RF best_p, DL best_units/best_lr/best_batch)
# across N_SEEDS=30 true-random seeds. Grid search itself (Parts 1-2)
# stays single-seed, per agreed scope (cost: 30x would be wasteful for
# config *selection*; only the final reported numbers need 30-seed
# validation).
# ---------------------------------------------------------------------

md_part4 = """---
## Part 4 — Final Validation with True-Random Seeds (headline numbers)

The grid search above (Parts 1–2) intentionally uses a **single fixed seed** per
configuration — standard, cost-efficient practice for hyperparameter *selection*
(re-running every grid point 30x would multiply the search cost ~30x for no benefit,
since the grids already compare configs against each other on the same held-out
CV/test data).

This section takes only the two **final chosen configurations** — `best_p` for
Random Forest, and `(best_units, best_lr, best_batch)` for the ILT — and re-runs
*each of those* across `N_SEEDS = 30` **true-random** seeds (drawn fresh every
kernel run via `random.sample`, not a fixed formula). The resulting mean ± std is
the validated headline number to quote in Chapter 5, consistent with the 30-seed
protocol used in Notebooks 08, 12, and 15.
"""

code_seeds = """import random

# ---------------------------------------------------------------------
# 30-seed convention: TRUE random draw, not a fixed formula -- every
# kernel run produces a DIFFERENT list of 30 seeds (Python's `random`
# module is seeded from OS entropy at interpreter start; nothing here
# is hardcoded or reproducible by design). Same convention as
# Notebooks 08, 12, 15.
# ---------------------------------------------------------------------
N_SEEDS = 30
SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)
print(f"SEEDS (this run) = {SEEDS}")
"""

code_rf_30 = """rf_30_rmse, rf_30_mae, rf_30_r2 = [], [], []
for s in SEEDS:
    m = RandomForestRegressor(**best_p, max_features='sqrt', random_state=s, n_jobs=1)
    m.fit(X_r_tr, y_r_tr)
    pred = m.predict(X_r_te)
    rf_30_rmse.append(np.sqrt(mean_squared_error(y_r_te, pred)))
    rf_30_mae.append(mean_absolute_error(y_r_te, pred))
    rf_30_r2.append(r2_score(y_r_te, pred))

rf_30_rmse_mean, rf_30_rmse_std = float(np.mean(rf_30_rmse)), float(np.std(rf_30_rmse))
rf_30_mae_mean = float(np.mean(rf_30_mae))
rf_30_r2_mean = float(np.mean(rf_30_r2))
print(f"RF tuned {best_p}, validated over {N_SEEDS} true-random seeds:")
print(f"  RMSE = {rf_30_rmse_mean:.4f} +/- {rf_30_rmse_std:.4f}")
print(f"  MAE  = {rf_30_mae_mean:.4f}")
print(f"  R2   = {rf_30_r2_mean:.4f}")
"""

code_dl_30 = """dl_30_rmse, dl_30_mae, dl_30_r2, dl_30_epoch = [], [], [], []
for s in SEEDS:
    r = evaluate_dl_config(best_units, best_lr, best_batch, seed=s)
    dl_30_rmse.append(r['RMSE']); dl_30_mae.append(r['MAE']); dl_30_r2.append(r['R2'])
    dl_30_epoch.append(r['best_epoch'])
    print(f"  seed={s}: RMSE={r['RMSE']:.2f}, R2={r['R2']:.4f}, epoch={r['best_epoch']}")

dl_30_rmse_mean, dl_30_rmse_std = float(np.mean(dl_30_rmse)), float(np.std(dl_30_rmse))
dl_30_mae_mean = float(np.mean(dl_30_mae))
dl_30_r2_mean = float(np.mean(dl_30_r2))
print(f"\\nILT tuned (units={best_units}, lr={best_lr:.0e}, bs={best_batch}), "
      f"validated over {N_SEEDS} true-random seeds:")
print(f"  RMSE = {dl_30_rmse_mean:.4f} +/- {dl_30_rmse_std:.4f}")
print(f"  MAE  = {dl_30_mae_mean:.4f}")
print(f"  R2   = {dl_30_r2_mean:.4f}")
"""

code_summary = """print("="*72)
print(f"  HYPERPARAMETER TUNING -- VALIDATED HEADLINE NUMBERS ({N_SEEDS} true-random seeds)")
print("="*72)
print(f"\\n{'Model':<25} {'RMSE (mean+/-std)':>22} {'MAE':>8} {'R2':>8}")
print('-'*68)
print(f"{'RF (default, 1 run)':<25} {rmse_default:>22.4f} {mae_default:>8.4f} {r2_default:>8.4f}")
print(f"{'RF (tuned, '+str(N_SEEDS)+' seeds)':<25} {f'{rf_30_rmse_mean:.4f}+/-{rf_30_rmse_std:.4f}':>22} {rf_30_mae_mean:>8.4f} {rf_30_r2_mean:>8.4f}")
print(f"{'ILT (default, 1 run)':<25} {default_row['RMSE']:>22.4f} {default_row['MAE']:>8.4f} {default_row['R2']:>8.4f}")
print(f"{'ILT (tuned, '+str(N_SEEDS)+' seeds)':<25} {f'{dl_30_rmse_mean:.4f}+/-{dl_30_rmse_std:.4f}':>22} {dl_30_mae_mean:>8.4f} {dl_30_r2_mean:>8.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Default vs Tuned (validated over {N_SEEDS} true-random seeds)', fontsize=13, fontweight='bold')
for ax, name, dflt, tuned_mean, tuned_std in [
        (axes[0], 'Random Forest', rmse_default, rf_30_rmse_mean, rf_30_rmse_std),
        (axes[1], 'ILT (canonical)', default_row['RMSE'], dl_30_rmse_mean, dl_30_rmse_std)]:
    bars = ax.bar(['Default\\n(1 run)', f'Tuned\\n({N_SEEDS} seeds)'], [dflt, tuned_mean],
                   yerr=[0, tuned_std], capsize=6, color=['#6B9EC8', '#4CAF50'], edgecolor='white', width=0.5)
    for bar, val in zip(bars, [dflt, tuned_mean]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title(name); ax.set_ylabel('Test RMSE (cases/day)'); ax.set_ylim(0, max(dflt, tuned_mean+tuned_std)*1.25)
    ax.grid(axis='y', alpha=0.3)
    delta = dflt - tuned_mean
    ax.set_xlabel(f'Delta RMSE = {delta:+.2f} ({delta/dflt*100:+.1f}%)')
plt.tight_layout()
plt.savefig('plots/hp_comparison_summary_30seed_FIXED.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\n✅ Saved plots/hp_comparison_summary_30seed_FIXED.png")

tuning_results['rf_tuned_metrics_30seed'] = {
    'RMSE_mean': rf_30_rmse_mean, 'RMSE_std': rf_30_rmse_std,
    'MAE_mean': rf_30_mae_mean, 'R2_mean': rf_30_r2_mean,
    'n_seeds': N_SEEDS, 'seeds': SEEDS, 'all_rmse': rf_30_rmse, 'all_mae': rf_30_mae, 'all_r2': rf_30_r2,
}
tuning_results['dl_tuned_metrics_30seed'] = {
    'RMSE_mean': dl_30_rmse_mean, 'RMSE_std': dl_30_rmse_std,
    'MAE_mean': dl_30_mae_mean, 'R2_mean': dl_30_r2_mean,
    'n_seeds': N_SEEDS, 'seeds': SEEDS, 'all_rmse': dl_30_rmse, 'all_mae': dl_30_mae, 'all_r2': dl_30_r2,
    'all_best_epoch': dl_30_epoch,
}
with open('metrics/hp_tuning_results_FIXED.pkl', 'wb') as f:
    pickle.dump(tuning_results, f)
print("✅ Updated metrics/hp_tuning_results_FIXED.pkl with 30-seed validated numbers")
"""

cells.append(new_md_cell(md_part4))
cells.append(new_code_cell(code_seeds))
cells.append(new_code_cell(code_rf_30))
cells.append(new_code_cell(code_dl_30))
cells.append(new_code_cell(code_summary))

# ---------------------------------------------------------------------
# Reset all code-cell outputs/execution counts -- new cells added,
# whole notebook needs a fresh re-run.
# ---------------------------------------------------------------------
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Patched", PATH)
print("Modified cell 10 (added seed param). Appended 5 new cells (Part 4: 30-seed validation).")
print("Total cells now:", len(cells))
