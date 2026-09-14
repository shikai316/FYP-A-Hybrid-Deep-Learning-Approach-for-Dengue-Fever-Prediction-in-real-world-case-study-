import json

PATH = '12_overfitting_analysis_30SEED.ipynb'
nb = json.load(open(PATH, encoding='utf-8'))
cells = nb['cells']


def get_src(i):
    return ''.join(cells[i]['source'])


def set_src(i, text):
    cells[i]['source'] = text.splitlines(keepends=True)
    # clear stale outputs/execution_count since semantics changed
    if cells[i]['cell_type'] == 'code':
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None


def replace_unique(i, old, new):
    src = get_src(i)
    assert src.count(old) == 1, f"cell {i}: expected exactly 1 match, found {src.count(old)}\n---OLD---\n{old}"
    set_src(i, src.replace(old, new))


# ---------------------------------------------------------------------
# Cell 1: SEEDS generation — fixed formula -> true random draw, no fixed
# master seed, so the list differs every kernel run.
# ---------------------------------------------------------------------
replace_unique(1,
    "# ---------------------------------------------------------------------------\n"
    "# 30-seed convention (matches notebooks 08 and 15 -- the canonical project standard)\n"
    "# ---------------------------------------------------------------------------\n"
    "N_SEEDS = 30\n"
    "SEEDS = [42 + i * 7 for i in range(N_SEEDS)]\n"
    "print(f\"N_SEEDS = {N_SEEDS}\")\n"
    "print(f\"SEEDS   = {SEEDS}\")\n",

    "# ---------------------------------------------------------------------------\n"
    "# 30-seed convention (matches notebooks 08 and 15 -- the canonical project standard).\n"
    "# Seeds are TRUE random draws, not a fixed formula: Python's `random` module is\n"
    "# seeded from OS entropy at interpreter start, so re-running this cell (or the\n"
    "# whole notebook) produces a DIFFERENT list of 30 seeds every time -- nothing\n"
    "# here is hardcoded or reproducible by design.\n"
    "# ---------------------------------------------------------------------------\n"
    "N_SEEDS = 30\n"
    "SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)\n"
    "print(f\"N_SEEDS = {N_SEEDS}\")\n"
    "print(f\"SEEDS   = {SEEDS}  (freshly drawn this run -- will differ next run)\")\n"
)

# add `random` import alongside the other stdlib imports
replace_unique(1,
    "import os, sys, time, pickle, warnings\n",
    "import os, sys, time, pickle, warnings, random\n"
)

# ---------------------------------------------------------------------
# Cell 5: RF gap -- track the actual seed used per fit
# ---------------------------------------------------------------------
replace_unique(5,
    "rf_30 = {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': []}",
    "rf_30 = {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'seed': []}"
)
replace_unique(5,
    "    rf_30['train_r2'].append(r2_score(y_r_tr, tr_pred))\n"
    "    rf_30['test_r2'].append(r2_score(y_r_te, te_pred))\n",
    "    rf_30['train_r2'].append(r2_score(y_r_tr, tr_pred))\n"
    "    rf_30['test_r2'].append(r2_score(y_r_te, te_pred))\n"
    "    rf_30['seed'].append(seed)\n"
)

# ---------------------------------------------------------------------
# Cell 6: DL gap checkpoint -- track actual seed used (resume-safe even
# though SEEDS itself is regenerated fresh on every kernel restart)
# ---------------------------------------------------------------------
replace_unique(6,
    "    return {name: {'best_train_loss': [], 'best_val_loss': [], 'test_rmse': [], 'test_r2': []} for name, _ in model_builders}",
    "    return {name: {'best_train_loss': [], 'best_val_loss': [], 'test_rmse': [], 'test_r2': [], 'seed': []} for name, _ in model_builders}"
)
replace_unique(6,
    "        dl_gap_results[name]['test_rmse'].append(rmse)\n"
    "        dl_gap_results[name]['test_r2'].append(r2)\n"
    "    save_dl_gap_ckpt(dl_gap_results)\n",
    "        dl_gap_results[name]['test_rmse'].append(rmse)\n"
    "        dl_gap_results[name]['test_r2'].append(r2)\n"
    "        dl_gap_results[name]['seed'].append(seed)\n"
    "    save_dl_gap_ckpt(dl_gap_results)\n"
)

# ---------------------------------------------------------------------
# Cell 9: RF learning curve -- no checkpoint, but record seeds used per
# fraction for transparency in the final saved table.
# ---------------------------------------------------------------------
replace_unique(9,
    "train_fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]\n"
    "lc_rf_train_mean, lc_rf_train_std, lc_rf_test_mean, lc_rf_test_std, lc_n = [], [], [], [], []",
    "train_fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]\n"
    "lc_rf_train_mean, lc_rf_train_std, lc_rf_test_mean, lc_rf_test_std, lc_n = [], [], [], [], []\n"
    "lc_rf_seeds_used = []"
)
replace_unique(9,
    "    lc_rf_train_mean.append(np.mean(tr_rmses)); lc_rf_train_std.append(np.std(tr_rmses))\n"
    "    lc_rf_test_mean.append(np.mean(te_rmses)); lc_rf_test_std.append(np.std(te_rmses))\n",
    "    lc_rf_train_mean.append(np.mean(tr_rmses)); lc_rf_train_std.append(np.std(tr_rmses))\n"
    "    lc_rf_test_mean.append(np.mean(te_rmses)); lc_rf_test_std.append(np.std(te_rmses))\n"
    "    lc_rf_seeds_used.append(list(SEEDS))\n"
)

# ---------------------------------------------------------------------
# Cell 10: DL learning curve checkpoint -- track actual seed used
# ---------------------------------------------------------------------
replace_unique(10,
    "    return {frac: {'train_loss': [], 'val_loss': [], 'test_rmse': []} for frac in fracs_dl}",
    "    return {frac: {'train_loss': [], 'val_loss': [], 'test_rmse': [], 'seed': []} for frac in fracs_dl}"
)
replace_unique(10,
    "        dl_lc_results[frac]['test_rmse'].append(rmse)\n"
    "        save_dl_lc_ckpt(dl_lc_results)\n",
    "        dl_lc_results[frac]['test_rmse'].append(rmse)\n"
    "        dl_lc_results[frac]['seed'].append(seed)\n"
    "        save_dl_lc_ckpt(dl_lc_results)\n"
)

# ---------------------------------------------------------------------
# Cell 16: boxplot checkpoint structure -- add seed tracking
# ---------------------------------------------------------------------
replace_unique(16,
    "    return {name: {'loss': [], 'val_loss': []} for name, _ in model_builders}",
    "    return {name: {'loss': [], 'val_loss': [], 'seed': []} for name, _ in model_builders}"
)

# ---------------------------------------------------------------------
# Cell 17: boxplot sweep loop -- record seed actually used
# ---------------------------------------------------------------------
replace_unique(17,
    "        boxplot_results[name]['loss'].append(loss)\n"
    "        boxplot_results[name]['val_loss'].append(val_loss)\n"
    "    save_boxplot_ckpt(boxplot_results)\n",
    "        boxplot_results[name]['loss'].append(loss)\n"
    "        boxplot_results[name]['val_loss'].append(val_loss)\n"
    "        boxplot_results[name]['seed'].append(seed)\n"
    "    save_boxplot_ckpt(boxplot_results)\n"
)

# ---------------------------------------------------------------------
# Cell 18: save matrices pkl -- use actually-recorded seeds, not the
# (possibly since-regenerated) top-level SEEDS variable
# ---------------------------------------------------------------------
replace_unique(18,
    "with open('metrics/overfit_boxplot_matrices_30SEED.pkl', 'wb') as f:\n"
    "    pickle.dump({'matrices': boxplot_matrices, 'epoch_steps': CKPT_EPOCHS, 'n_seeds': N_SEEDS, 'seeds': SEEDS}, f)",
    "with open('metrics/overfit_boxplot_matrices_30SEED.pkl', 'wb') as f:\n"
    "    pickle.dump({'matrices': boxplot_matrices, 'epoch_steps': CKPT_EPOCHS, 'n_seeds': N_SEEDS,\n"
    "                 'seeds_used': {name: boxplot_results[name]['seed'] for name, _ in model_builders}}, f)"
)

# ---------------------------------------------------------------------
# Cell 24: final summary -- replace blind 'seeds': SEEDS with the actual
# per-stage recorded seed lists (correct even across a resumed/restarted
# run, where the top-level SEEDS would have been freshly regenerated)
# ---------------------------------------------------------------------
replace_unique(24,
    "overfit_summary = {\n"
    "    'n_seeds': N_SEEDS, 'seeds': SEEDS,\n",
    "overfit_summary = {\n"
    "    'n_seeds': N_SEEDS,\n"
    "    'seeds_used': {\n"
    "        'random_forest_gap': rf_30['seed'],\n"
    "        'dl_gap': {name: dl_gap_results[name]['seed'] for name, _ in model_builders},\n"
    "        'boxplot_sweep': {name: boxplot_results[name]['seed'] for name, _ in model_builders},\n"
    "    },\n"
)

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Patched", PATH)
print("Modified cells: 1, 5, 6, 9, 10, 16, 17, 18, 24")
