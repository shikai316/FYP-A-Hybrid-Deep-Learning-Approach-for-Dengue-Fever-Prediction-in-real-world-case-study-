import json
import uuid

PATH = '13_computational_efficiency_analysis_FIXED.ipynb'
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


# ---------------------------------------------------------------------
# Markdown cell 0: document the 30-seed protocol + why SARIMA is exempt
# ---------------------------------------------------------------------
replace_unique(0,
    "5. Optimiser comparison (Adam vs SGD vs RMSProp) on the **canonical ILT architecture**\n",

    "5. Optimiser comparison (Adam vs SGD vs RMSProp) on the **canonical ILT architecture**\n"
    "\n"
    "### Update (2026-06-23): 30-seed protocol applied\n"
    "Random Forest (§1), the two DL models (§2), and the optimiser comparison (§5) are now each\n"
    "measured across `N_SEEDS = 30` independent **true-random** seeds and reported as mean ± std,\n"
    "consistent with the project-wide 30-seed protocol (Notebooks 08, 09, 10, 11, 12, 15). SARIMA\n"
    "(§1) is unchanged — `auto_arima`'s stepwise search is a deterministic procedure with no\n"
    "random-seed concept, so there is nothing to re-run across seeds.\n"
)

# ---------------------------------------------------------------------
# Insert N_SEEDS/SEEDS setup as a new cell right after cell 1 (imports +
# data load) and before cell 2 (markdown header for section 1).
# ---------------------------------------------------------------------
seeds_setup_src = """import random

# ---------------------------------------------------------------------
# 30-seed convention: TRUE random draw, not a fixed formula -- every
# kernel run produces a DIFFERENT list of 30 seeds (Python's `random`
# module is seeded from OS entropy at interpreter start; nothing here
# is hardcoded or reproducible by design). Same convention as
# Notebooks 08, 09, 10, 11, 12, 15. Applied below to: Random Forest
# (Section 1), the two DL models (Section 2), and the optimiser
# comparison (Section 5). SARIMA is exempt (deterministic, no seed).
# ---------------------------------------------------------------------
N_SEEDS = 30
SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)
print(f"SEEDS (this run) = {SEEDS}")
"""
cells.insert(2, new_code_cell(seeds_setup_src))

# ---------------------------------------------------------------------
# Cell 3 (now still index 3, since we inserted AFTER cell 1 / BEFORE old
# cell 2 which is markdown -- old cell 2 (md) -> new index 3, old cell 3
# (code, SARIMA+RF) -> new index 4). Recompute indices via search instead
# of hardcoding, to avoid off-by-one mistakes.
# ---------------------------------------------------------------------
def find_cell_index(snippet):
    matches = [i for i, c in enumerate(cells) if snippet in ''.join(c['source'])]
    assert len(matches) == 1, f"expected 1 match for snippet, found {len(matches)}: {snippet[:60]!r}"
    return matches[0]


idx_sarima_rf = find_cell_index("# Random Forest — FAIR feature set")
replace_unique(idx_sarima_rf,
    "# Random Forest — FAIR feature set (weather/calendar/population only, Section 3.3.1)\n"
    "ram_before = get_ram_mb()\n"
    "X_r_tr, y_r_tr = d['X_train_rf'][:-1], d['y_train'][1:]\n"
    "X_r_te, y_r_te = d['X_test_rf'][:-1], d['y_test'][1:]\n"
    "t0 = time.time()\n"
    "rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=10,\n"
    "                            min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=1)\n"
    "rf.fit(X_r_tr, y_r_tr)\n"
    "rf_train_time = time.time() - t0\n"
    "rf_ram = get_ram_mb() - ram_before\n"
    "\n"
    "t0 = time.time()\n"
    "for _ in range(REPS):\n"
    "    rf_pred = rf.predict(X_r_te)\n"
    "rf_infer_time = (time.time() - t0) / REPS\n"
    "rf_infer_per_sample = rf_infer_time / len(X_r_te) * 1000\n"
    "rf_bytes = len(pkl.dumps(rf))\n"
    "rf_rmse = np.sqrt(mean_squared_error(y_r_te, rf_pred))\n"
    "rf_r2 = r2_score(y_r_te, rf_pred)\n"
    "\n"
    "efficiency_records.append({'Model': 'Random Forest (fair)', 'Type': 'ML',\n"
    "    'Train time (s)': rf_train_time, 'Inference total (ms)': rf_infer_time*1000,\n"
    "    'Inference per sample (ms)': rf_infer_per_sample,\n"
    "    'Params': rf.n_estimators * rf.estimators_[0].tree_.node_count,\n"
    "    'Model size (KB)': rf_bytes/1024, 'RAM delta (MB)': rf_ram,\n"
    "    'RMSE': rf_rmse, 'R2': rf_r2})\n"
    "print(f\"Random Forest  — train={rf_train_time:.1f}s, infer/sample={rf_infer_per_sample:.4f}ms, \"\n"
    "      f\"size={rf_bytes/1024:.1f}KB, RMSE={rf_rmse:.2f}, R2={rf_r2:.4f}  (fair: weather/calendar/pop only)\")",

    "# Random Forest — FAIR feature set (weather/calendar/population only, Section 3.3.1)\n"
    "# 30-seed protocol: timing AND accuracy measured across N_SEEDS true-random seeds,\n"
    "# reported as mean +/- std.\n"
    "X_r_tr, y_r_tr = d['X_train_rf'][:-1], d['y_train'][1:]\n"
    "X_r_te, y_r_te = d['X_test_rf'][:-1], d['y_test'][1:]\n"
    "\n"
    "rf_train_times, rf_infer_times, rf_infer_per_samples = [], [], []\n"
    "rf_rmses, rf_r2s, rf_byte_sizes, rf_param_counts = [], [], [], []\n"
    "rf_ram = None\n"
    "for s in SEEDS:\n"
    "    ram_before = get_ram_mb()\n"
    "    t0 = time.time()\n"
    "    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_split=10,\n"
    "                                min_samples_leaf=5, max_features='sqrt', random_state=s, n_jobs=1)\n"
    "    rf.fit(X_r_tr, y_r_tr)\n"
    "    rf_train_times.append(time.time() - t0)\n"
    "    rf_ram = get_ram_mb() - ram_before  # RAM footprint isn't seed-sensitive; last measurement kept\n"
    "\n"
    "    t0 = time.time()\n"
    "    for _ in range(REPS):\n"
    "        rf_pred = rf.predict(X_r_te)\n"
    "    infer_time = (time.time() - t0) / REPS\n"
    "    rf_infer_times.append(infer_time)\n"
    "    rf_infer_per_samples.append(infer_time / len(X_r_te) * 1000)\n"
    "    rf_byte_sizes.append(len(pkl.dumps(rf)))\n"
    "    rf_param_counts.append(rf.n_estimators * rf.estimators_[0].tree_.node_count)\n"
    "    rf_rmses.append(np.sqrt(mean_squared_error(y_r_te, rf_pred)))\n"
    "    rf_r2s.append(r2_score(y_r_te, rf_pred))\n"
    "\n"
    "rf_train_time, rf_train_time_std = float(np.mean(rf_train_times)), float(np.std(rf_train_times))\n"
    "rf_infer_time = float(np.mean(rf_infer_times))\n"
    "rf_infer_per_sample, rf_infer_per_sample_std = float(np.mean(rf_infer_per_samples)), float(np.std(rf_infer_per_samples))\n"
    "rf_bytes = float(np.mean(rf_byte_sizes))\n"
    "rf_param_count = float(np.mean(rf_param_counts))\n"
    "rf_rmse, rf_rmse_std = float(np.mean(rf_rmses)), float(np.std(rf_rmses))\n"
    "rf_r2, rf_r2_std = float(np.mean(rf_r2s)), float(np.std(rf_r2s))\n"
    "\n"
    "efficiency_records.append({'Model': 'Random Forest (fair)', 'Type': 'ML',\n"
    "    'Train time (s)': rf_train_time, 'Train time std (s)': rf_train_time_std,\n"
    "    'Inference total (ms)': rf_infer_time*1000,\n"
    "    'Inference per sample (ms)': rf_infer_per_sample, 'Inference per sample std (ms)': rf_infer_per_sample_std,\n"
    "    'Params': rf_param_count,\n"
    "    'Model size (KB)': rf_bytes/1024, 'RAM delta (MB)': rf_ram,\n"
    "    'RMSE': rf_rmse, 'RMSE std': rf_rmse_std, 'R2': rf_r2, 'R2 std': rf_r2_std})\n"
    "print(f\"Random Forest  — train={rf_train_time:.2f}+/-{rf_train_time_std:.2f}s, \"\n"
    "      f\"infer/sample={rf_infer_per_sample:.4f}+/-{rf_infer_per_sample_std:.4f}ms, \"\n"
    "      f\"size={rf_bytes/1024:.1f}KB, RMSE={rf_rmse:.2f}+/-{rf_rmse_std:.2f}, R2={rf_r2:.4f}  \"\n"
    "      f\"(fair: weather/calendar/pop only, {N_SEEDS} seeds)\")"
)

# ---------------------------------------------------------------------
# Cell 5 (DL timing, 2 models) -- full 30-seed rewrite
# ---------------------------------------------------------------------
idx_dl_timing = find_cell_index("dl_configs = [")
old_dl = (
    "FIXED_EPOCHS = 30  # fixed budget (no early stopping) so DL training time is directly comparable\n"
    "\n"
    "dl_configs = [\n"
    "    ('Standalone LSTM', build_standalone_lstm),\n"
    "    ('ILT (proposed)',  build_lstm_transformer),\n"
    "]\n"
    "\n"
    "for name, builder in dl_configs:\n"
    "    tf.random.set_seed(42); np.random.seed(42); gc.collect()\n"
    "    ram_before = get_ram_mb()\n"
    "\n"
    "    model = builder(SEQ_LEN, d['n_seq_features'], d['n_feat_features'])\n"
    "    n_params = model.count_params()\n"
    "    _ = model.predict([Xts_seq[:1], Xts_feat[:1]], verbose=0)  # warmup\n"
    "\n"
    "    t0 = time.time()\n"
    "    model.fit([Xts_seq, Xts_feat], yt, validation_data=([Xvs_seq, Xvs_feat], yv),\n"
    "              epochs=FIXED_EPOCHS, batch_size=32, verbose=0)\n"
    "    train_time = time.time() - t0\n"
    "    ram_after = get_ram_mb()\n"
    "\n"
    "    t0 = time.time()\n"
    "    for _ in range(REPS):\n"
    "        pred_s = model.predict([X_te_s, X_te_f], verbose=0, batch_size=256).flatten()\n"
    "    infer_time = (time.time() - t0) / REPS\n"
    "    infer_per_sample = infer_time / len(pred_s) * 1000\n"
    "\n"
    "    single_input = [X_te_s[:1], X_te_f[:1]]\n"
    "    t0 = time.time()\n"
    "    for _ in range(100):\n"
    "        model.predict(single_input, verbose=0)\n"
    "    single_infer = (time.time() - t0) / 100 * 1000\n"
    "\n"
    "    pred = np.maximum(d['target_scaler'].inverse_transform(pred_s.reshape(-1,1)).flatten(), 0)\n"
    "    rmse = np.sqrt(mean_squared_error(y_te_raw, pred))\n"
    "    r2 = r2_score(y_te_raw, pred)\n"
    "\n"
    "    tmp_path = f'models/tmp_{name.replace(\" \",\"_\").replace(\"(\",\"\").replace(\")\",\"\")}.keras'\n"
    "    model.save(tmp_path)\n"
    "    model_kb = os.path.getsize(tmp_path) / 1024\n"
    "\n"
    "    efficiency_records.append({'Model': name, 'Type': 'DL',\n"
    "        'Train time (s)': train_time, 'Inference total (ms)': infer_time*1000,\n"
    "        'Inference per sample (ms)': infer_per_sample, 'Single-sample infer (ms)': single_infer,\n"
    "        'Params': n_params, 'Model size (KB)': model_kb, 'RAM delta (MB)': ram_after - ram_before,\n"
    "        'RMSE': rmse, 'R2': r2})\n"
    "    print(f\"{name:<20} — train={train_time:.1f}s ({FIXED_EPOCHS}ep), batch infer={infer_per_sample:.4f}ms/sample, \"\n"
    "          f\"single={single_infer:.1f}ms, params={n_params:,}, size={model_kb:.1f}KB, RMSE={rmse:.2f}, R2={r2:.4f}\")\n"
    "\n"
    "eff_df = pd.DataFrame(efficiency_records)\n"
    "print(\"\\n✅ All timing measurements complete (fixed 30-epoch budget — NOT the 30-run averaged headline numbers)\")"
)

new_dl = (
    "FIXED_EPOCHS = 30  # fixed budget (no early stopping) so DL training time is directly comparable\n"
    "\n"
    "dl_configs = [\n"
    "    ('Standalone LSTM', build_standalone_lstm),\n"
    "    ('ILT (proposed)',  build_lstm_transformer),\n"
    "]\n"
    "\n"
    "# 30-seed protocol: each model is built + trained + timed N_SEEDS times with a fresh\n"
    "# true-random seed, and all metrics below are reported as mean +/- std.\n"
    "for name, builder in dl_configs:\n"
    "    train_times, infer_totals_ms, infer_per_samples, single_infers = [], [], [], []\n"
    "    rmses, r2s = [], []\n"
    "    n_params = None; model_kb = None; ram_delta_last = None\n"
    "\n"
    "    for s in SEEDS:\n"
    "        tf.random.set_seed(s); np.random.seed(s); gc.collect()\n"
    "        ram_before = get_ram_mb()\n"
    "\n"
    "        model = builder(SEQ_LEN, d['n_seq_features'], d['n_feat_features'])\n"
    "        n_params = model.count_params()\n"
    "        _ = model.predict([Xts_seq[:1], Xts_feat[:1]], verbose=0)  # warmup\n"
    "\n"
    "        t0 = time.time()\n"
    "        model.fit([Xts_seq, Xts_feat], yt, validation_data=([Xvs_seq, Xvs_feat], yv),\n"
    "                  epochs=FIXED_EPOCHS, batch_size=32, verbose=0)\n"
    "        train_times.append(time.time() - t0)\n"
    "        ram_delta_last = get_ram_mb() - ram_before\n"
    "\n"
    "        t0 = time.time()\n"
    "        for _ in range(REPS):\n"
    "            pred_s = model.predict([X_te_s, X_te_f], verbose=0, batch_size=256).flatten()\n"
    "        infer_time = (time.time() - t0) / REPS\n"
    "        infer_totals_ms.append(infer_time * 1000)\n"
    "        infer_per_samples.append(infer_time / len(pred_s) * 1000)\n"
    "\n"
    "        single_input = [X_te_s[:1], X_te_f[:1]]\n"
    "        t0 = time.time()\n"
    "        for _ in range(100):\n"
    "            model.predict(single_input, verbose=0)\n"
    "        single_infers.append((time.time() - t0) / 100 * 1000)\n"
    "\n"
    "        pred = np.maximum(d['target_scaler'].inverse_transform(pred_s.reshape(-1,1)).flatten(), 0)\n"
    "        rmses.append(np.sqrt(mean_squared_error(y_te_raw, pred)))\n"
    "        r2s.append(r2_score(y_te_raw, pred))\n"
    "\n"
    "        tmp_path = f'models/tmp_{name.replace(\" \",\"_\").replace(\"(\",\"\").replace(\")\",\"\")}.keras'\n"
    "        model.save(tmp_path)\n"
    "        model_kb = os.path.getsize(tmp_path) / 1024\n"
    "\n"
    "    train_time, train_time_std = float(np.mean(train_times)), float(np.std(train_times))\n"
    "    infer_total_ms = float(np.mean(infer_totals_ms))\n"
    "    infer_per_sample, infer_per_sample_std = float(np.mean(infer_per_samples)), float(np.std(infer_per_samples))\n"
    "    single_infer, single_infer_std = float(np.mean(single_infers)), float(np.std(single_infers))\n"
    "    rmse, rmse_std = float(np.mean(rmses)), float(np.std(rmses))\n"
    "    r2, r2_std = float(np.mean(r2s)), float(np.std(r2s))\n"
    "\n"
    "    efficiency_records.append({'Model': name, 'Type': 'DL',\n"
    "        'Train time (s)': train_time, 'Train time std (s)': train_time_std,\n"
    "        'Inference total (ms)': infer_total_ms,\n"
    "        'Inference per sample (ms)': infer_per_sample, 'Inference per sample std (ms)': infer_per_sample_std,\n"
    "        'Single-sample infer (ms)': single_infer, 'Single-sample infer std (ms)': single_infer_std,\n"
    "        'Params': n_params, 'Model size (KB)': model_kb, 'RAM delta (MB)': ram_delta_last,\n"
    "        'RMSE': rmse, 'RMSE std': rmse_std, 'R2': r2, 'R2 std': r2_std})\n"
    "    print(f\"{name:<20} — train={train_time:.2f}+/-{train_time_std:.2f}s ({FIXED_EPOCHS}ep), \"\n"
    "          f\"batch infer={infer_per_sample:.4f}+/-{infer_per_sample_std:.4f}ms/sample, \"\n"
    "          f\"single={single_infer:.2f}+/-{single_infer_std:.2f}ms, params={n_params:,}, size={model_kb:.1f}KB, \"\n"
    "          f\"RMSE={rmse:.2f}+/-{rmse_std:.2f}, R2={r2:.4f}  ({N_SEEDS} seeds)\")\n"
    "\n"
    "eff_df = pd.DataFrame(efficiency_records)\n"
    "print(f\"\\n✅ All timing measurements complete ({N_SEEDS}-seed mean +/- std, fixed {FIXED_EPOCHS}-epoch budget per run)\")"
)
replace_unique(idx_dl_timing, old_dl, new_dl)

# ---------------------------------------------------------------------
# Cell 7 (summary table) -- print mean+/-std where available
# ---------------------------------------------------------------------
idx_summary_table = find_cell_index("COMPUTATIONAL EFFICIENCY COMPARISON")
old_table = (
    'print("=" * 105)\n'
    'print("  COMPUTATIONAL EFFICIENCY COMPARISON  (canonical architecture, Table 3.2 models)")\n'
    'print("=" * 105)\n'
    'print(f"{\'Model\':<22} {\'Train(s)\':>9} {\'Infer/sample(ms)\':>18} {\'Params\':>10} {\'Size(KB)\':>10} {\'RAM(MB)\':>9} {\'RMSE\':>7} {\'R²\':>7}")\n'
    "print('─' * 105)\n"
    "for _, row in eff_df.iterrows():\n"
    "    params_str = f\"{int(row['Params']):,}\" if pd.notna(row['Params']) else 'N/A'\n"
    "    print(f\"{row['Model']:<22} {row['Train time (s)']:>9.1f} {row['Inference per sample (ms)']:>18.4f} \"\n"
    "          f\"{params_str:>10} {row['Model size (KB)']:>10.1f} {row['RAM delta (MB)']:>9.1f} \"\n"
    "          f\"{row['RMSE']:>7.2f} {row['R2']:>7.4f}\")\n"
    "print()\n"
    'print("NOTE: ILT params should be ~41,000 (matches Thesis Section 3.5.2 and Notebook 14\'s Big-O analysis).")\n'
    'print("NOTE: RMSE here is a single 30-fixed-epoch run for timing purposes only — the reported headline")\n'
    'print("      performance numbers must come from Notebook 08\'s 30-run average, not from this notebook.")\n'
    "eff_df.to_csv('metrics/efficiency_table.csv', index=False)"
)
new_table = (
    'print("=" * 115)\n'
    'print(f"  COMPUTATIONAL EFFICIENCY COMPARISON  (canonical architecture, Table 3.2 models, '
    "{N_SEEDS}-seed mean+/-std for RF/DL)\")\n"
    'print("=" * 115)\n'
    'print(f"{\'Model\':<22} {\'Train(s)\':>16} {\'Infer/sample(ms)\':>22} {\'Params\':>10} {\'Size(KB)\':>10} {\'RAM(MB)\':>9} {\'RMSE\':>16} {\'R²\':>7}")\n'
    "print('─' * 115)\n"
    "for _, row in eff_df.iterrows():\n"
    "    params_str = f\"{int(row['Params']):,}\" if pd.notna(row['Params']) else 'N/A'\n"
    "    train_std = row.get('Train time std (s)')\n"
    "    train_str = f\"{row['Train time (s)']:.1f}+/-{train_std:.1f}\" if pd.notna(train_std) else f\"{row['Train time (s)']:.1f}\"\n"
    "    infer_std = row.get('Inference per sample std (ms)')\n"
    "    infer_str = f\"{row['Inference per sample (ms)']:.4f}+/-{infer_std:.4f}\" if pd.notna(infer_std) else f\"{row['Inference per sample (ms)']:.4f}\"\n"
    "    rmse_std = row.get('RMSE std')\n"
    "    rmse_str = f\"{row['RMSE']:.2f}+/-{rmse_std:.2f}\" if pd.notna(rmse_std) else f\"{row['RMSE']:.2f}\"\n"
    "    print(f\"{row['Model']:<22} {train_str:>16} {infer_str:>22} \"\n"
    "          f\"{params_str:>10} {row['Model size (KB)']:>10.1f} {row['RAM delta (MB)']:>9.1f} \"\n"
    "          f\"{rmse_str:>16} {row['R2']:>7.4f}\")\n"
    "print()\n"
    'print("NOTE: ILT params should be ~41,000 (matches Thesis Section 3.5.2 and Notebook 14\'s Big-O analysis).")\n'
    'print(f"NOTE: SARIMA is a single deterministic fit (auto_arima\'s stepwise search has no random-seed concept).")\n'
    'print(f"      RF and DL rows above are mean +/- std across {N_SEEDS} true-random seeds.")\n'
    "eff_df.to_csv('metrics/efficiency_table.csv', index=False)"
)
replace_unique(idx_summary_table, old_table, new_table)

# ---------------------------------------------------------------------
# Cell 9 (bar plots) -- add error bars for train time / infer per sample,
# and switch the efficiency-frontier scatter to an errorbar plot.
# ---------------------------------------------------------------------
idx_plots = find_cell_index("def bar_plot(ax, values, title, ylabel")
replace_unique(idx_plots,
    "def bar_plot(ax, values, title, ylabel, log=False, fmt='.1f'):\n"
    "    bars = ax.bar(x, values, color=colors[:len(models)], edgecolor='white', width=0.6)\n",

    "def bar_plot(ax, values, title, ylabel, log=False, fmt='.1f', yerr=None):\n"
    "    bars = ax.bar(x, values, color=colors[:len(models)], edgecolor='white', width=0.6,\n"
    "                   yerr=yerr, capsize=4 if yerr is not None else 0)\n"
)

replace_unique(idx_plots,
    "bar_plot(fig.add_subplot(gs[0,0]), eff_df['Train time (s)'], 'Training time', 'Seconds', log=True)\n"
    "bar_plot(fig.add_subplot(gs[0,1]), eff_df['Inference per sample (ms)'], 'Inference latency (batch)', 'ms per sample', log=True, fmt='.4f')\n",

    "bar_plot(fig.add_subplot(gs[0,0]), eff_df['Train time (s)'], 'Training time', 'Seconds', log=True,\n"
    "         yerr=eff_df['Train time std (s)'].fillna(0) if 'Train time std (s)' in eff_df.columns else None)\n"
    "bar_plot(fig.add_subplot(gs[0,1]), eff_df['Inference per sample (ms)'], 'Inference latency (batch)', 'ms per sample', log=True, fmt='.4f',\n"
    "         yerr=eff_df['Inference per sample std (ms)'].fillna(0) if 'Inference per sample std (ms)' in eff_df.columns else None)\n"
)

replace_unique(idx_plots,
    "ax_front = fig.add_subplot(gs[1,2])\n"
    "for i, (_, row) in enumerate(eff_df.iterrows()):\n"
    "    ax_front.scatter(row['Train time (s)'], row['RMSE'], s=120, color=colors[i], zorder=3,\n"
    "                      label=row['Model'], edgecolors='white', lw=1.5)\n",

    "ax_front = fig.add_subplot(gs[1,2])\n"
    "for i, (_, row) in enumerate(eff_df.iterrows()):\n"
    "    xerr = row.get('Train time std (s)', 0) or 0\n"
    "    yerr_ = row.get('RMSE std', 0) or 0\n"
    "    ax_front.errorbar(row['Train time (s)'], row['RMSE'], xerr=xerr, yerr=yerr_,\n"
    "                       fmt='o', ms=11, color=colors[i], zorder=3,\n"
    "                       label=row['Model'], ecolor=colors[i], elinewidth=1.5, capsize=4,\n"
    "                       markeredgecolor='white', markeredgewidth=1.5)\n"
)

# ---------------------------------------------------------------------
# Cell 11 (optimiser comparison) -- full 30-seed rewrite
# ---------------------------------------------------------------------
idx_opt = find_cell_index("optimiser_configs = [")
old_opt = (
    "from canonical_pipeline import build_lstm_transformer as _build_ilt\n"
    "\n"
    "optimiser_configs = [\n"
    "    ('Adam',    lambda: Adam(0.001, clipnorm=1.0)),\n"
    "    ('SGD',     lambda: SGD(0.01, momentum=0.9, clipnorm=1.0)),\n"
    "    ('RMSProp', lambda: RMSprop(0.001, clipnorm=1.0)),\n"
    "]\n"
    "\n"
    "opt_records = []\n"
    'print("Comparing optimisers on the CANONICAL ILT architecture (max 50 epochs, early stop patience=15)...\\n")\n'
    "\n"
    "for opt_name, opt_factory in optimiser_configs:\n"
    "    tf.random.set_seed(42); np.random.seed(42)\n"
    "    model = _build_ilt(SEQ_LEN, d['n_seq_features'], d['n_feat_features'])\n"
    "    model.compile(optimizer=opt_factory(), loss='mse')  # override default Adam compile\n"
    "\n"
    "    _ = model.predict([Xts_seq[:1], Xts_feat[:1]], verbose=0)\n"
    "    t0 = time.time()\n"
    "    hist = model.fit([Xts_seq, Xts_feat], yt, validation_data=([Xvs_seq, Xvs_feat], yv),\n"
    "                      epochs=50, batch_size=32, verbose=0,\n"
    "                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),\n"
    "                                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)])\n"
    "    train_time = time.time() - t0\n"
    "    best_ep = int(np.argmin(hist.history['val_loss'])) + 1\n"
    "\n"
    "    pred_s = model.predict([X_te_s, X_te_f], verbose=0).flatten()\n"
    "    pred = np.maximum(d['target_scaler'].inverse_transform(pred_s.reshape(-1,1)).flatten(), 0)\n"
    "    rmse = np.sqrt(mean_squared_error(y_te_raw, pred))\n"
    "    r2 = r2_score(y_te_raw, pred)\n"
    "\n"
    "    opt_records.append({'Optimiser': opt_name, 'Train time (s)': train_time, 'Best epoch': best_ep,\n"
    "        'Best val loss': float(min(hist.history['val_loss'])), 'RMSE': rmse, 'R2': r2,\n"
    "        'loss_history': hist.history['loss'], 'val_loss_history': hist.history['val_loss']})\n"
    "    print(f\"  {opt_name:<10} — train={train_time:.1f}s, best_ep={best_ep}/{len(hist.history['loss'])}, RMSE={rmse:.2f}, R2={r2:.4f}\")\n"
    "\n"
    'print("\\n✅ Optimiser comparison done (single run each — for trend illustration, not a 30-run claim)")'
)
new_opt = (
    "from canonical_pipeline import build_lstm_transformer as _build_ilt\n"
    "\n"
    "optimiser_configs = [\n"
    "    ('Adam',    lambda: Adam(0.001, clipnorm=1.0)),\n"
    "    ('SGD',     lambda: SGD(0.01, momentum=0.9, clipnorm=1.0)),\n"
    "    ('RMSProp', lambda: RMSprop(0.001, clipnorm=1.0)),\n"
    "]\n"
    "\n"
    "opt_records = []\n"
    'print(f"Comparing optimisers on the CANONICAL ILT architecture across {N_SEEDS} true-random seeds "\n'
    '      f"(max 50 epochs, early stop patience=15)...\\n")\n'
    "\n"
    "for opt_name, opt_factory in optimiser_configs:\n"
    "    train_times, best_eps, best_val_losses, rmses, r2s = [], [], [], [], []\n"
    "    loss_hist_last, val_loss_hist_last = None, None\n"
    "\n"
    "    for s in SEEDS:\n"
    "        tf.random.set_seed(s); np.random.seed(s)\n"
    "        model = _build_ilt(SEQ_LEN, d['n_seq_features'], d['n_feat_features'])\n"
    "        model.compile(optimizer=opt_factory(), loss='mse')  # override default Adam compile\n"
    "\n"
    "        _ = model.predict([Xts_seq[:1], Xts_feat[:1]], verbose=0)\n"
    "        t0 = time.time()\n"
    "        hist = model.fit([Xts_seq, Xts_feat], yt, validation_data=([Xvs_seq, Xvs_feat], yv),\n"
    "                          epochs=50, batch_size=32, verbose=0,\n"
    "                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),\n"
    "                                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)])\n"
    "        train_times.append(time.time() - t0)\n"
    "        best_eps.append(int(np.argmin(hist.history['val_loss'])) + 1)\n"
    "        best_val_losses.append(float(min(hist.history['val_loss'])))\n"
    "\n"
    "        pred_s = model.predict([X_te_s, X_te_f], verbose=0).flatten()\n"
    "        pred = np.maximum(d['target_scaler'].inverse_transform(pred_s.reshape(-1,1)).flatten(), 0)\n"
    "        rmses.append(np.sqrt(mean_squared_error(y_te_raw, pred)))\n"
    "        r2s.append(r2_score(y_te_raw, pred))\n"
    "        loss_hist_last, val_loss_hist_last = hist.history['loss'], hist.history['val_loss']\n"
    "\n"
    "    train_time, train_time_std = float(np.mean(train_times)), float(np.std(train_times))\n"
    "    best_ep_mean = float(np.mean(best_eps))\n"
    "    rmse, rmse_std = float(np.mean(rmses)), float(np.std(rmses))\n"
    "    r2, r2_std = float(np.mean(r2s)), float(np.std(r2s))\n"
    "\n"
    "    opt_records.append({'Optimiser': opt_name,\n"
    "        'Train time (s)': train_time, 'Train time std (s)': train_time_std,\n"
    "        'Best epoch': best_ep_mean, 'Best val loss': float(np.mean(best_val_losses)),\n"
    "        'RMSE': rmse, 'RMSE std': rmse_std, 'R2': r2, 'R2 std': r2_std,\n"
    "        'all_rmse': rmses, 'all_train_time': train_times,\n"
    "        'loss_history': loss_hist_last, 'val_loss_history': val_loss_hist_last})\n"
    "    print(f\"  {opt_name:<10} — train={train_time:.2f}+/-{train_time_std:.2f}s, \"\n"
    "          f\"best_ep(avg)={best_ep_mean:.1f}, RMSE={rmse:.2f}+/-{rmse_std:.2f}, R2={r2:.4f}  ({N_SEEDS} seeds)\")\n"
    "\n"
    'print(f"\\n✅ Optimiser comparison done ({N_SEEDS}-seed mean +/- std per optimiser)")'
)
replace_unique(idx_opt, old_opt, new_opt)

# ---------------------------------------------------------------------
# Cell 12 (optimiser plot) -- label with mean+/-std, add error bars
# ---------------------------------------------------------------------
idx_opt_plot = find_cell_index("Optimiser Comparison — Canonical ILT")
old_plot = (
    "for rec in opt_records:\n"
    "    name = rec['Optimiser']\n"
    "    ep = range(1, len(rec['val_loss_history'])+1)\n"
    "    axes[0].plot(ep, rec['val_loss_history'], '-', color=opt_colors[name], lw=2, label=f\"{name} (RMSE={rec['RMSE']:.2f})\")\n"
    "    axes[0].axvline(rec['Best epoch'], color=opt_colors[name], linestyle=':', lw=1, alpha=0.6)\n"
    "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Loss (MSE)')\n"
    "axes[0].set_title('Validation Loss Convergence'); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)\n"
    "\n"
    "opt_names = [r['Optimiser'] for r in opt_records]\n"
    "opt_rmses = [r['RMSE'] for r in opt_records]\n"
    "opt_times = [r['Train time (s)'] for r in opt_records]\n"
    "xo = np.arange(len(opt_names)); w = 0.35\n"
    "ax2b = axes[1].twinx()\n"
    "axes[1].bar(xo - w/2, opt_rmses, w, label='Test RMSE', color=[opt_colors[n] for n in opt_names], edgecolor='white', alpha=0.85)\n"
    "ax2b.bar(xo + w/2, opt_times, w, label='Train time (s)', color=[opt_colors[n] for n in opt_names], edgecolor='white', alpha=0.45)\n"
    "axes[1].set_xticks(xo); axes[1].set_xticklabels(opt_names)\n"
    "axes[1].set_ylabel('Test RMSE (cases/day)'); ax2b.set_ylabel('Training time (s)', color='gray')\n"
    "axes[1].set_title('RMSE vs Training Time per Optimiser')\n"
    "l1,lb1 = axes[1].get_legend_handles_labels(); l2,lb2 = ax2b.get_legend_handles_labels()\n"
    "axes[1].legend(l1+l2, lb1+lb2, fontsize=9); axes[1].grid(axis='y', alpha=0.3)"
)
new_plot = (
    "for rec in opt_records:\n"
    "    name = rec['Optimiser']\n"
    "    ep = range(1, len(rec['val_loss_history'])+1)\n"
    "    axes[0].plot(ep, rec['val_loss_history'], '-', color=opt_colors[name], lw=2,\n"
    "                 label=f\"{name} (RMSE={rec['RMSE']:.2f}+/-{rec['RMSE std']:.2f}, {N_SEEDS} seeds)\")\n"
    "    axes[0].axvline(rec['Best epoch'], color=opt_colors[name], linestyle=':', lw=1, alpha=0.6)\n"
    "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Val Loss (MSE)')\n"
    "axes[0].set_title(f'Validation Loss Convergence (last of {N_SEEDS} seeds shown per optimiser)')\n"
    "axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)\n"
    "\n"
    "opt_names = [r['Optimiser'] for r in opt_records]\n"
    "opt_rmses = [r['RMSE'] for r in opt_records]\n"
    "opt_rmse_stds = [r['RMSE std'] for r in opt_records]\n"
    "opt_times = [r['Train time (s)'] for r in opt_records]\n"
    "opt_time_stds = [r['Train time std (s)'] for r in opt_records]\n"
    "xo = np.arange(len(opt_names)); w = 0.35\n"
    "ax2b = axes[1].twinx()\n"
    "axes[1].bar(xo - w/2, opt_rmses, w, yerr=opt_rmse_stds, capsize=4, label='Test RMSE', color=[opt_colors[n] for n in opt_names], edgecolor='white', alpha=0.85)\n"
    "ax2b.bar(xo + w/2, opt_times, w, yerr=opt_time_stds, capsize=4, label='Train time (s)', color=[opt_colors[n] for n in opt_names], edgecolor='white', alpha=0.45)\n"
    "axes[1].set_xticks(xo); axes[1].set_xticklabels(opt_names)\n"
    "axes[1].set_ylabel('Test RMSE (cases/day)'); ax2b.set_ylabel('Training time (s)', color='gray')\n"
    "axes[1].set_title(f'RMSE vs Training Time per Optimiser (mean +/- std, {N_SEEDS} seeds)')\n"
    "l1,lb1 = axes[1].get_legend_handles_labels(); l2,lb2 = ax2b.get_legend_handles_labels()\n"
    "axes[1].legend(l1+l2, lb1+lb2, fontsize=9); axes[1].grid(axis='y', alpha=0.3)"
)
replace_unique(idx_opt_plot, old_plot, new_plot)

# ---------------------------------------------------------------------
# Reset all code-cell outputs/execution counts -- structure changed,
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
print("Total cells now:", len(cells))
