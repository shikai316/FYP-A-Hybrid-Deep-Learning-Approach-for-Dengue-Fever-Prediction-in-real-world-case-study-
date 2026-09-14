import json

PATH = '10_realworld_improved_v2.ipynb'
nb = json.load(open(PATH, encoding='utf-8'))
cells = nb['cells']

def get_src(i):
    return ''.join(cells[i]['source']) if isinstance(cells[i]['source'], list) else cells[i]['source']

def set_src(i, new_src):
    cells[i]['source'] = new_src.splitlines(keepends=True)
    cells[i]['execution_count'] = None
    cells[i]['outputs'] = []

# ============================================================================
# ROOT CAUSE of "kernel crashed" on the last two cells:
#
# train_and_eval() and train_and_eval_improved() (cell 9) each loop
# `for run in range(n_runs)` (n_runs=30) and call `m = builder(...)` to build
# a BRAND NEW Keras model every iteration, but NEVER release the previous
# model/graph (no tf.keras.backend.clear_session(), no del, no gc.collect()).
# TensorFlow accumulates graph/session state across repeated model creation
# in a loop -- this is a well-known TF/Keras memory leak pattern.
#
# By the time the notebook reaches cell 22 (the new Mitigation Experiment),
# the SAME kernel has already built ~300 models in cell 10 alone
# (5 horizons x 2 calls x 30 seeds), none ever released. Cell 22 then builds
# up to 120 MORE models (2 horizons x 2 arms x 30 seeds) on top of that.
# The saved notebook's own output confirms this: cell 22 dies partway through
# Arm B at h=21 -- right where accumulated RAM finally exceeds what's
# available, and the OS kills the process. Jupyter reports this as an opaque
# "kernel crashed" because the process is killed externally, not via a
# catchable Python exception.
#
# FIX: explicitly release each model and clear the TF graph/session at the
# end of every loop iteration in BOTH functions, plus run gc.collect().
# ============================================================================

# ---- Step 1: add `import gc` to the imports cell (cell 2) ----
src2 = get_src(2)
old_imports = "import random\nwarnings.filterwarnings('ignore')\n"
assert old_imports in src2, "could not find imports anchor in cell 2"
new_imports = "import random\nimport gc\nwarnings.filterwarnings('ignore')\n"
src2 = src2.replace(old_imports, new_imports)
set_src(2, src2)

# ---- Step 2: add cleanup inside train_and_eval_improved's loop (cell 9) ----
src9 = get_src(9)

old_a = (
    "        all_preds.append(p)\n"
    "        all_rmse.append(np.sqrt(mean_squared_error(y_te_raw, p)))\n"
    "        all_mae.append(mean_absolute_error(y_te_raw, p))\n"
    "        all_r2.append(r2_score(y_te_raw, p))\n"
    "\n"
    "    mean_pred = np.mean(all_preds, axis=0)\n"
    "    return np.mean(all_rmse), np.mean(all_mae), np.mean(all_r2), np.std(all_rmse), mean_pred, np.array(all_rmse)\n"
    "\n"
    "\n"
    "# Keep original for within-sample comparison\n"
)
assert src9.count(old_a) == 1, "could not find train_and_eval_improved's loop tail"
new_a = (
    "        all_preds.append(p)\n"
    "        all_rmse.append(np.sqrt(mean_squared_error(y_te_raw, p)))\n"
    "        all_mae.append(mean_absolute_error(y_te_raw, p))\n"
    "        all_r2.append(r2_score(y_te_raw, p))\n"
    "\n"
    "        # MEMORY FIX: release this run's model + TF graph/session before the\n"
    "        # next iteration builds a new one. Without this, TF accumulates graph\n"
    "        # state across all `n_runs` iterations (and across every call to this\n"
    "        # function within the same kernel), eventually exhausting RAM and\n"
    "        # crashing the kernel -- exactly what happened in the Mitigation\n"
    "        # Experiment cells below on the original (unpatched) run.\n"
    "        del m\n"
    "        tf.keras.backend.clear_session()\n"
    "        gc.collect()\n"
    "\n"
    "    mean_pred = np.mean(all_preds, axis=0)\n"
    "    return np.mean(all_rmse), np.mean(all_mae), np.mean(all_r2), np.std(all_rmse), mean_pred, np.array(all_rmse)\n"
    "\n"
    "\n"
    "# Keep original for within-sample comparison\n"
)
src9 = src9.replace(old_a, new_a)

# ---- Step 3: same fix inside train_and_eval's loop (cell 9, second function) ----
old_b = (
    "        all_preds.append(p)\n"
    "        all_rmse.append(np.sqrt(mean_squared_error(y_te_raw, p)))\n"
    "        all_mae.append(mean_absolute_error(y_te_raw, p))\n"
    "        all_r2.append(r2_score(y_te_raw, p))\n"
    "    mean_pred = np.mean(all_preds, axis=0)\n"
    "    return np.mean(all_rmse), np.mean(all_mae), np.mean(all_r2), np.std(all_rmse), mean_pred, np.array(all_rmse)\n"
    "\n"
    "\n"
    "print('Helper functions ready (improved + original)')\n"
)
assert src9.count(old_b) == 1, "could not find train_and_eval's loop tail"
new_b = (
    "        all_preds.append(p)\n"
    "        all_rmse.append(np.sqrt(mean_squared_error(y_te_raw, p)))\n"
    "        all_mae.append(mean_absolute_error(y_te_raw, p))\n"
    "        all_r2.append(r2_score(y_te_raw, p))\n"
    "\n"
    "        # MEMORY FIX: see explanation in train_and_eval_improved above.\n"
    "        del m\n"
    "        tf.keras.backend.clear_session()\n"
    "        gc.collect()\n"
    "    mean_pred = np.mean(all_preds, axis=0)\n"
    "    return np.mean(all_rmse), np.mean(all_mae), np.mean(all_r2), np.std(all_rmse), mean_pred, np.array(all_rmse)\n"
    "\n"
    "\n"
    "print('Helper functions ready (improved + original)')\n"
)
src9 = src9.replace(old_b, new_b)
set_src(9, src9)

# ---- Step 4: add a markdown note right before the Mitigation Experiment section ----
def mdcell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

note = (
    "### Fix (2026-06-26) — kernel-crash root cause patched\n"
    "\n"
    "Running this notebook end-to-end previously crashed the kernel partway through the cells "
    "below (confirmed by this notebook's own saved output: it died mid-way through Arm B at "
    "h=21d). **Root cause:** `train_and_eval` / `train_and_eval_improved` (Cell 4's helper "
    "functions) build a brand-new Keras model every loop iteration (30 seeds x however many "
    "calls) without ever releasing the previous model or clearing the TensorFlow graph/session. "
    "By the time execution reaches this section, the SAME kernel has already silently "
    "accumulated state from ~300 previously-built models in the main loop above, and this "
    "section's own ~120 additional models (2 horizons x 2 arms x 30 seeds) push it over the "
    "available RAM, so the OS kills the process -- Jupyter just reports this opaquely as "
    "\"kernel crashed\" with no Python traceback.\n"
    "\n"
    "**Fix applied:** both training functions now call `tf.keras.backend.clear_session()` + "
    "`gc.collect()` (and `del m`) at the end of every seed iteration, so memory no longer grows "
    "monotonically across the run.\n"
    "\n"
    "**Action needed:** restart the kernel before re-running -- a kernel that already accumulated "
    "memory in a prior run won't recover the leaked memory just because the code changed.\n"
)

# insert right before the markdown cell that introduces the Mitigation Experiment
target_idx = None
for i, c in enumerate(cells):
    s = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    if c['cell_type'] == 'markdown' and 'Mitigation Experiment' in s:
        target_idx = i
        break
assert target_idx is not None, "could not find Mitigation Experiment markdown cell"
cells.insert(target_idx, mdcell(note))

nb['cells'] = cells
json.dump(nb, open(PATH, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"Patched. Total cells now: {len(cells)}")
