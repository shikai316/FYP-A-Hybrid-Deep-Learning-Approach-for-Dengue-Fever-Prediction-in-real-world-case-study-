import json

PATH = '10_realworld_improved_v2.ipynb'
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


def replace_count(i, old, new, expected):
    src = get_src(i)
    cnt = src.count(old)
    assert cnt == expected, f"cell {i}: expected {expected} matches, found {cnt}\n---OLD---\n{old}"
    set_src(i, src.replace(old, new))


# ---------------------------------------------------------------------
# Markdown cell 0: fix "frozen weights, no retraining" claim -- the
# actual code (Cell 4 / index 9-10) retrains the ILT from scratch on
# 2013-2022 + fine-tunes on the 2022 tail, then evaluates OOS on
# 2023-2025. It never trains on 2023-2025 data, but it is NOT frozen.
# ---------------------------------------------------------------------
replace_unique(0,
    "Applies the pre-trained ILT model (weights frozen from 2013–2022 training) to the\n"
    "out-of-sample 2023–2025 dataset with **no retraining**.",

    "Re-trains the ILT model from scratch on the full 2013–2022 dataset (plus a brief\n"
    "fine-tuning pass on the most recent 2022 data), then evaluates **out-of-sample** on\n"
    "the 2023–2025 dataset. No 2023–2025 data is ever used for training or fine-tuning --\n"
    "only for evaluation -- and scalers are carried forward unchanged from the 2013–2022\n"
    "training partition."
)

# ---------------------------------------------------------------------
# Markdown cell 7: same "frozen weights" mislabel
# ---------------------------------------------------------------------
replace_unique(7,
    "Strategy: re-train ILT from scratch on the full 2013-2022 dataset (training + validation),\n"
    "then evaluate on the 2023-2025 dataset using the same scalers.\n"
    "This implements Section 4.3.4: frozen weights from 2013-2022 training phase.",

    "Strategy: re-train ILT from scratch on the full 2013-2022 dataset (training + validation),\n"
    "then evaluate on the 2023-2025 dataset using the same scalers.\n"
    "This implements Section 4.3.4: the model is trained exclusively on 2013-2022 data\n"
    "(never exposed to 2023-2025 during training) before OOS evaluation."
)

# ---------------------------------------------------------------------
# Cell 2 (index 2): import random + true-random SEEDS, replacing the
# single fixed np.random.seed(42)/tf.random.set_seed(42) -- per-run
# seeding now comes from SEEDS[run] inside the training functions.
# ---------------------------------------------------------------------
replace_unique(2,
    "import pickle\n"
    "import warnings\n"
    "import os\n"
    "warnings.filterwarnings('ignore')",

    "import pickle\n"
    "import warnings\n"
    "import os\n"
    "import random\n"
    "warnings.filterwarnings('ignore')"
)

replace_unique(2,
    "np.random.seed(42)\n"
    "tf.random.set_seed(42)\n"
    "\n"
    "# AUDIT FIX (2026-06-21):",

    "# ---------------------------------------------------------------------\n"
    "# 30-seed convention: TRUE random draw, not a fixed formula -- every\n"
    "# kernel run produces a DIFFERENT list of 30 seeds (Python's `random`\n"
    "# module is seeded from OS entropy at interpreter start; nothing here\n"
    "# is hardcoded or reproducible by design). Same convention as\n"
    "# Notebooks 08, 12, 15.\n"
    "# ---------------------------------------------------------------------\n"
    "N_SEEDS = 30\n"
    "SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)\n"
    "print(f\"SEEDS (this run) = {SEEDS}\")\n"
    "\n"
    "# AUDIT FIX (2026-06-21):"
)

# ---------------------------------------------------------------------
# Cell 9 (index 9): the two training functions -- bump default n_runs
# and replace the fixed-formula seed with SEEDS[run]
# ---------------------------------------------------------------------
replace_unique(9,
    "sl, n_seq, n_feat, horizon, n_runs=3,\n"
    "        fine_tune_seq=None, fine_tune_feat=None, fine_tune_y=None):",

    "sl, n_seq, n_feat, horizon, n_runs=N_SEEDS,\n"
    "        fine_tune_seq=None, fine_tune_feat=None, fine_tune_y=None):"
)

replace_unique(9,
    "sl, n_seq, n_feat, horizon, n_runs=3):",
    "sl, n_seq, n_feat, horizon, n_runs=N_SEEDS):"
)

replace_count(9,
    "    for run in range(n_runs):\n"
    "        seed = 42 + run * 7\n",
    "    for run in range(n_runs):\n"
    "        seed = SEEDS[run]\n",
    2
)

# ---------------------------------------------------------------------
# Cell 10 (index 10): bump both call sites from n_runs=3 to N_SEEDS,
# and label the printed results with the run count
# ---------------------------------------------------------------------
replace_unique(10,
    "SEQ_LEN, n_seq_features, n_feat_features, h, n_runs=3)",
    "SEQ_LEN, n_seq_features, n_feat_features, h, n_runs=N_SEEDS)"
)

replace_unique(10,
    "SEQ_LEN, n_seq_features, n_feat_features, h, n_runs=3,\n"
    "        fine_tune_seq=ft_seq, fine_tune_feat=ft_feat, fine_tune_y=ft_y)",

    "SEQ_LEN, n_seq_features, n_feat_features, h, n_runs=N_SEEDS,\n"
    "        fine_tune_seq=ft_seq, fine_tune_feat=ft_feat, fine_tune_y=ft_y)"
)

replace_unique(10,
    "    print(f\"  Within-sample  ILT: RMSE={rmse_b:.2f}+/-{std_b:.2f}, \"\n"
    "          f\"MAE={mae_b:.2f}, R2={r2_b:.4f}\")",

    "    print(f\"  Within-sample  ILT: RMSE={rmse_b:.2f}+/-{std_b:.2f}, \"\n"
    "          f\"MAE={mae_b:.2f}, R2={r2_b:.4f}  ({N_SEEDS} runs)\")"
)

replace_unique(10,
    "    print(f\"  Out-of-sample  ILT: RMSE={rmse_rw:.2f}+/-{std_rw:.2f}, \"\n"
    "          f\"MAE={mae_rw:.2f}, R2={r2_rw:.4f}\")",

    "    print(f\"  Out-of-sample  ILT: RMSE={rmse_rw:.2f}+/-{std_rw:.2f}, \"\n"
    "          f\"MAE={mae_rw:.2f}, R2={r2_rw:.4f}  ({N_SEEDS} runs)\")"
)

# ---------------------------------------------------------------------
# Reset all code-cell outputs/execution counts -- seed scheme + run
# count changed, whole notebook needs a fresh re-run.
# ---------------------------------------------------------------------
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Patched", PATH)
print("Modified cells: md0, md7, 2, 9, 10. All outputs cleared (needs full re-run).")
