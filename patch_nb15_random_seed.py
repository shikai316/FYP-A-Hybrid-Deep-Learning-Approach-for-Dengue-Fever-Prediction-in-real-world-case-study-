import json

PATH = '15_full_reproduction_30seed_5horizon.ipynb'
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
# Cell 1: import random + true-random SEEDS (not a fixed formula)
# ---------------------------------------------------------------------
replace_unique(1,
    "import pickle, time, warnings, os\n",
    "import pickle, time, warnings, os, random\n"
)

replace_unique(1,
    "N_SEEDS = 30\n"
    "SEEDS = [42 + i*7 for i in range(N_SEEDS)]   # identical convention to Notebook 08\n",

    "N_SEEDS = 30\n"
    "# TRUE random draw, not a fixed formula -- every kernel run produces a DIFFERENT\n"
    "# list of 30 seeds (Python's `random` module is seeded from OS entropy at\n"
    "# interpreter start; nothing here is hardcoded or reproducible by design).\n"
    "# Same approach now used in Notebook 08 and Notebook 12.\n"
    "SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)\n"
)

# ---------------------------------------------------------------------
# Cell 4: track the actual seed used per (horizon, model) entry --
# resume-safe even though SEEDS is regenerated fresh on kernel restart
# ---------------------------------------------------------------------
replace_unique(4,
    "    return {h: {'Standalone LSTM': {'rmse': [], 'mae': [], 'r2': [], 'preds': []},\n"
    "                'ILT': {'rmse': [], 'mae': [], 'r2': [], 'preds': []}} for h in HORIZONS}",

    "    return {h: {'Standalone LSTM': {'rmse': [], 'mae': [], 'r2': [], 'preds': [], 'seed': []},\n"
    "                'ILT': {'rmse': [], 'mae': [], 'r2': [], 'preds': [], 'seed': []}} for h in HORIZONS}"
)

# ---------------------------------------------------------------------
# Cell 6: record the seed actually used alongside each result
# ---------------------------------------------------------------------
replace_unique(6,
    "            results[h][name]['preds'].append(pred)\n",
    "            results[h][name]['preds'].append(pred)\n"
    "            results[h][name]['seed'].append(seed)\n"
)

# ---------------------------------------------------------------------
# Reset all code-cell outputs/execution counts -- seed scheme changed,
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
print("Modified cells: 1, 4, 6. All outputs cleared (needs full re-run).")
