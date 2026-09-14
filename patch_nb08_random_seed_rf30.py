import json

PATH = '08_multi_horizon_comparison_30run_RESULT_REVIEWED.ipynb'
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


# ---------------------------------------------------------------------
# Cell 2: add `random` import
# ---------------------------------------------------------------------
replace_unique(2,
    "import numpy as np, pandas as pd, warnings, os, pickle\n",
    "import numpy as np, pandas as pd, warnings, os, pickle, random\n"
)

# ---------------------------------------------------------------------
# Cell 5a: insert true-random SEEDS generation before train_dl
# ---------------------------------------------------------------------
replace_unique(5,
    'def train_dl(builder, name, Xts_seq, Xts_feat, yt, Xvs_seq, Xvs_feat, yv,\n'
    '             Xte_seq, Xte_feat, y_te_raw, sl, n_seq, n_feat, pfx,\n'
    '             epochs=80, patience=15, n_runs=30):\n'
    '    """Train n_runs times with different seeds, return mean metrics."""',

    '# ---------------------------------------------------------------------\n'
    '# 30-seed convention: TRUE random draw, not a fixed formula -- every\n'
    "# kernel run produces a DIFFERENT list of 30 seeds (Python's `random`\n"
    '# module is seeded from OS entropy at interpreter start; nothing here\n'
    '# is hardcoded or reproducible by design). Shared by train_dl (LSTM,\n'
    '# ILT) and train_rf_multiseed (RF) below so all three models draw\n'
    '# from the same freshly-generated 30-seed pool.\n'
    '# ---------------------------------------------------------------------\n'
    'N_SEEDS = 30\n'
    'SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)\n'
    'print(f"SEEDS (this run) = {SEEDS}")\n'
    '\n'
    '\n'
    'def train_dl(builder, name, Xts_seq, Xts_feat, yt, Xvs_seq, Xvs_feat, yv,\n'
    '             Xte_seq, Xte_feat, y_te_raw, sl, n_seq, n_feat, pfx,\n'
    '             epochs=80, patience=15, n_runs=30):\n'
    '    """Train n_runs times with different seeds, return mean metrics."""'
)

# ---------------------------------------------------------------------
# Cell 5b: train_dl uses SEEDS instead of the 42 + run*7 formula
# ---------------------------------------------------------------------
replace_unique(5,
    "        seed = 42 + run * 7\n",
    "        seed = SEEDS[run]\n"
)

# ---------------------------------------------------------------------
# Cell 5c: add train_rf_multiseed helper, mirroring train_dl, right
# after train_dl's own definition ends
# ---------------------------------------------------------------------
replace_unique(5,
    'return rmse, mae, r2, all_rmse, all_mae, all_r2\n\n'
    '# ─────────────────────────────────────────────────────────────────────\n'
    '# MAX_RETRIES: max times to retrain LST',

    'return rmse, mae, r2, all_rmse, all_mae, all_r2\n\n\n'
    "def train_rf_multiseed(X_tr, y_tr, X_te, y_te, name='RF (weather only)', n_runs=30):\n"
    '    """Fit n_runs Random Forests with different seeds, return mean metrics (mirrors train_dl)."""\n'
    '    all_rmse, all_mae, all_r2 = [], [], []\n'
    '    for run in range(n_runs):\n'
    '        seed = SEEDS[run]\n'
    '        rf = RandomForestRegressor(n_estimators=300, max_depth=12,\n'
    '            min_samples_split=10, min_samples_leaf=5,\n'
    "            max_features='sqrt', random_state=seed, n_jobs=-1)\n"
    '        rf.fit(X_tr, y_tr)\n'
    '        p = rf.predict(X_te)\n'
    '        all_rmse.append(np.sqrt(mean_squared_error(y_te, p)))\n'
    '        all_mae.append(mean_absolute_error(y_te, p))\n'
    '        all_r2.append(r2_score(y_te, p))\n'
    '    rmse, mae, r2 = np.mean(all_rmse), np.mean(all_mae), np.mean(all_r2)\n'
    '    std = np.std(all_rmse)\n'
    '    print(f"  {name:<22} RMSE={rmse:.2f}±{std:.2f}, MAE={mae:.2f}, R²={r2:.4f}  ({n_runs} runs)")\n'
    '    return rmse, mae, r2, all_rmse, all_mae, all_r2\n\n'
    '# ─────────────────────────────────────────────────────────────────────\n'
    '# MAX_RETRIES: max times to retrain LST'
)

# ---------------------------------------------------------------------
# Cell 5d: replace the inline single-seed RF fit inside run_horizon
# with a call to train_rf_multiseed (30-seed treatment)
# ---------------------------------------------------------------------
replace_unique(5,
    "    rf = RandomForestRegressor(n_estimators=300, max_depth=12,\n"
    "        min_samples_split=10, min_samples_leaf=5,\n"
    "        max_features='sqrt', random_state=42, n_jobs=-1)\n"
    "    rf.fit(X_r_tr, y_r_tr)\n"
    "    rf_p = rf.predict(X_r_te)\n"
    "    rf_rmse = np.sqrt(mean_squared_error(y_r_te, rf_p))\n"
    "    rf_mae = mean_absolute_error(y_r_te, rf_p)\n"
    "    rf_r2 = r2_score(y_r_te, rf_p)\n"
    "    print(f\"  {'RF (weather only)':<22} RMSE={rf_rmse:.2f}, MAE={rf_mae:.2f}, R²={rf_r2:.4f}\")\n",

    "    rf_rmse, rf_mae, rf_r2, rf_raw_runs, rf_raw_mae, rf_raw_r2 = train_rf_multiseed(\n"
    "        X_r_tr, y_r_tr, X_r_te, y_r_te, name='RF (weather only)', n_runs=N_SEEDS)\n"
)

# ---------------------------------------------------------------------
# Cell 5e: surface rf_raw_* in run_horizon's return dict
# ---------------------------------------------------------------------
replace_unique(5,
    "            'lstm_raw_runs': lstm_raw_runs,\n"
    "            'lstm_raw_mae':  lstm_raw_mae,\n"
    "            'lstm_raw_r2':   lstm_raw_r2,\n"
    "            'ilt_raw_runs':  ilt_raw_runs,\n",

    "            'rf_raw_runs':   rf_raw_runs,\n"
    "            'rf_raw_mae':    rf_raw_mae,\n"
    "            'rf_raw_r2':     rf_raw_r2,\n"
    "            'lstm_raw_runs': lstm_raw_runs,\n"
    "            'lstm_raw_mae':  lstm_raw_mae,\n"
    "            'lstm_raw_r2':   lstm_raw_r2,\n"
    "            'ilt_raw_runs':  ilt_raw_runs,\n"
)

# ---------------------------------------------------------------------
# Cell 18: generalize the raw 30-run export table from 2 columns
# (ILT, LSTM) to 3 (ILT, LSTM, RF) per horizon
# ---------------------------------------------------------------------
old_block = (
    "metrics_keys = {\n"
    "    'RMSE': ('ilt_raw_runs',  'lstm_raw_runs'),\n"
    "    'MAE':  ('ilt_raw_mae',   'lstm_raw_mae'),\n"
    "    'R2':   ('ilt_raw_r2',    'lstm_raw_r2'),\n"
    "}\n"
    "\n"
    "for metric_name, (ilt_key, lstm_key) in metrics_keys.items():\n"
    "    print(f\"\\n=== RAW 30-RUN {metric_name} (for Excel) ===\")\n"
    "    \n"
    "    # Header row\n"
    "    print(f\"{'Run':<5}\", end=\"\")\n"
    "    for r in all_results:\n"
    "        h = int(r['horizon'])\n"
    "        print(f\"  ILT_h{h}d   LSTM_h{h}d\", end=\"\")\n"
    "    print()\n"
    "    \n"
    "    # Data rows\n"
    "    for run_idx in range(30):\n"
    "        print(f\"{run_idx+1:<5}\", end=\"\")\n"
    "        for r in all_results:\n"
    "            ilt_v  = r[ilt_key][run_idx]\n"
    "            lstm_v = r[lstm_key][run_idx]\n"
    "            print(f\"  {ilt_v:8.4f}  {lstm_v:8.4f}\", end=\"\")\n"
    "        print()\n"
    "    \n"
    "    # Mean row\n"
    "    print(f\"{'Mean':<5}\", end=\"\")\n"
    "    for r in all_results:\n"
    "        ilt_mean  = np.mean(r[ilt_key])\n"
    "        lstm_mean = np.mean(r[lstm_key])\n"
    "        print(f\"  {ilt_mean:8.4f}  {lstm_mean:8.4f}\", end=\"\")\n"
    "    print()\n"
    "    \n"
    "    # Std row\n"
    "    print(f\"{'Std':<5}\", end=\"\")\n"
    "    for r in all_results:\n"
    "        ilt_std  = np.std(r[ilt_key])\n"
    "        lstm_std = np.std(r[lstm_key])\n"
    "        print(f\"  {ilt_std:8.4f}  {lstm_std:8.4f}\", end=\"\")\n"
    "    print()\n"
    "\n"
)

new_block = (
    "metrics_keys = {\n"
    "    'RMSE': [('ilt_raw_runs', 'ILT'), ('lstm_raw_runs', 'LSTM'), ('rf_raw_runs', 'RF')],\n"
    "    'MAE':  [('ilt_raw_mae',  'ILT'), ('lstm_raw_mae',  'LSTM'), ('rf_raw_mae',  'RF')],\n"
    "    'R2':   [('ilt_raw_r2',   'ILT'), ('lstm_raw_r2',   'LSTM'), ('rf_raw_r2',   'RF')],\n"
    "}\n"
    "\n"
    "for metric_name, key_label_pairs in metrics_keys.items():\n"
    "    print(f\"\\n=== RAW 30-RUN {metric_name} (for Excel) ===\")\n"
    "    \n"
    "    # Header row\n"
    "    print(f\"{'Run':<5}\", end=\"\")\n"
    "    for r in all_results:\n"
    "        h = int(r['horizon'])\n"
    "        for _, label in key_label_pairs:\n"
    "            print(f\"  {label}_h{h}d\".ljust(12), end=\"\")\n"
    "    print()\n"
    "    \n"
    "    # Data rows\n"
    "    for run_idx in range(30):\n"
    "        print(f\"{run_idx+1:<5}\", end=\"\")\n"
    "        for r in all_results:\n"
    "            for key, _ in key_label_pairs:\n"
    "                print(f\"  {r[key][run_idx]:10.4f}\", end=\"\")\n"
    "        print()\n"
    "    \n"
    "    # Mean row\n"
    "    print(f\"{'Mean':<5}\", end=\"\")\n"
    "    for r in all_results:\n"
    "        for key, _ in key_label_pairs:\n"
    "            print(f\"  {np.mean(r[key]):10.4f}\", end=\"\")\n"
    "    print()\n"
    "    \n"
    "    # Std row\n"
    "    print(f\"{'Std':<5}\", end=\"\")\n"
    "    for r in all_results:\n"
    "        for key, _ in key_label_pairs:\n"
    "            print(f\"  {np.std(r[key]):10.4f}\", end=\"\")\n"
    "    print()\n"
    "\n"
)

replace_unique(18, old_block, new_block)

# ---------------------------------------------------------------------
# Reset ALL code-cell outputs/execution counts -- the seed-generation
# scheme and RF treatment both changed, so previously RESULT_REVIEWED
# outputs are now stale and the whole notebook needs a fresh re-run.
# ---------------------------------------------------------------------
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Patched", PATH)
print("Modified cells: 2, 5, 18. All outputs cleared (needs full re-run).")
