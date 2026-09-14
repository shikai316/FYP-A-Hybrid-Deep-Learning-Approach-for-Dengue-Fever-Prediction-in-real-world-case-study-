import json
import uuid

PATH = '09_Interpretability_domain_validation_PATCHED.ipynb'
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
# Cell 4: the single "representative" model now uses a true-random
# seed (SEEDS[0]) instead of hardcoded 42, with a pointer to the new
# Analysis 5 cross-seed consistency section.
# ---------------------------------------------------------------------
replace_unique(4,
    'print(f"Training LSTM-Transformer for h={HORIZON}d...")\n'
    'tf.random.set_seed(42); np.random.seed(42)\n'
    'model = build_model()',

    'print(f"Training LSTM-Transformer for h={HORIZON}d "\n'
    '      f"(representative seed={SEEDS[0]} -- see Analysis 5 below for the full "\n'
    '      f"{N_SEEDS}-seed cross-model consistency check)...")\n'
    'tf.random.set_seed(SEEDS[0]); np.random.seed(SEEDS[0])\n'
    'model = build_model()'
)

# ---------------------------------------------------------------------
# Insert a new code cell right after cell 2 (data prep) and before the
# markdown header "## Train LSTM-Transformer" (current index 3) -- true
# random N_SEEDS/SEEDS setup needed by cell 4 above and by the new
# Analysis 5 section appended at the end.
# ---------------------------------------------------------------------
seeds_setup_src = """import random, pickle

os.makedirs('metrics', exist_ok=True)

# ---------------------------------------------------------------------
# 30-seed convention: TRUE random draw, not a fixed formula -- every
# kernel run produces a DIFFERENT list of 30 seeds (Python's `random`
# module is seeded from OS entropy at interpreter start; nothing here
# is hardcoded or reproducible by design). Same convention as
# Notebooks 08, 12, 15. Used here for: (1) the single "representative"
# model trained below (Cell 4), and (2) the cross-seed feature/attention
# consistency analysis in Analysis 5 at the end of this notebook.
# ---------------------------------------------------------------------
N_SEEDS = 30
SEEDS = random.sample(range(1, 1_000_000), N_SEEDS)
print(f"SEEDS (this run) = {SEEDS}")
"""
cells.insert(3, new_code_cell(seeds_setup_src))

# ---------------------------------------------------------------------
# Append "Analysis 5" at the very end: retrain N_SEEDS=30 independent
# models, extract each one's attention profile + attention-weighted
# feature importance, then report CROSS-SEED (not just cross-sample)
# mean/std -- this is the actual "feature importance consistency"
# metric implied by Section 4.1.4 item #9, complementing the existing
# Analyses 1-4 (which only measured within-model variance across test
# samples of a single trained model).
# ---------------------------------------------------------------------

md_analysis5 = """## Analysis 5: Cross-Seed Attention & Feature-Importance Consistency (30 true-random seeds)

Analyses 1–4 above all describe **one** trained model (the "representative" run from
Cell 4). They measure *within-model* variation (e.g. attention weight std **across test
samples** in Analysis 1). They do not yet answer: *if you retrain this architecture from
scratch with a different random initialisation, do you get the same attention pattern and
the same important features?*

That is what Section 4.1.4 item #9 ("feature importance consistency") actually asks for.
This section trains `N_SEEDS = 30` independent models (same architecture, different
true-random seeds), and reports the **cross-model** mean ± std of the attention profile and
attention-weighted feature importance — i.e. consistency across independent retrainings,
not just across test samples of one model.
"""

code_train_loop = """# ---------------------------------------------------------------------
# Train N_SEEDS=30 independently-seeded models (architecture identical
# to Cell 4 above) and extract each one's full per-sample attention
# matrix + attention-weighted feature importance. This is fully
# self-contained -- it does not reuse the Cell 4 "representative" model
# -- so all 30 runs (including seed index 0) are independent retrainings.
# ---------------------------------------------------------------------
all_mean_attn = []        # (N_SEEDS, 28)          -- per-model average attention profile
all_feat_importance = []  # (N_SEEDS, n_features)  -- per-model attention-weighted feature importance
all_rmse, all_r2 = [], []
all_top10_sets = []

for run, s in enumerate(SEEDS):
    tf.random.set_seed(s); np.random.seed(s)
    m = build_model()
    _ = m.predict([Xts[:1], Xtf[:1]], verbose=0)
    m.fit([Xts, Xtf], yt, validation_data=([Xvs, Xvf], yv),
          epochs=100, batch_size=32, verbose=0,
          callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=0)])

    pred_s_scaled = m.predict([X_te_s, X_te_f], verbose=0).flatten()
    pred_s = np.maximum(global_tgt_scaler.inverse_transform(pred_s_scaled.reshape(-1, 1)).flatten(), 0)
    rmse_s = np.sqrt(mean_squared_error(y_te_raw, pred_s))
    r2_s = r2_score(y_te_raw, pred_s)
    all_rmse.append(rmse_s); all_r2.append(r2_s)

    # Extract attention for all test samples from this model
    _ = m([X_te_s[:1], X_te_f[:1]])
    attn_batches = []
    for i in range(0, len(X_te_s), 64):
        _ = m([X_te_s[i:i+64], X_te_f[i:i+64]], training=False)
        attn_batches.append(m.attn_layer.attention_weights.numpy())
    attn_s = np.concatenate(attn_batches, axis=0).squeeze(-1)  # (n_test, 28)
    all_mean_attn.append(attn_s.mean(axis=0))                   # (28,)

    # Attention-weighted feature importance for this model (same formula as Analysis 2)
    weighted_s = np.abs(X_te_s) * attn_s[:, :, np.newaxis]
    feat_imp_s = weighted_s.sum(axis=1).mean(axis=0)            # (n_features,)
    all_feat_importance.append(feat_imp_s)
    all_top10_sets.append(set(np.argsort(feat_imp_s)[-10:]))

    print(f"  seed {run+1}/{N_SEEDS} (s={s}): RMSE={rmse_s:.2f}, R2={r2_s:.4f}")

all_mean_attn = np.array(all_mean_attn)               # (N_SEEDS, 28)
all_feat_importance = np.array(all_feat_importance)   # (N_SEEDS, n_features)
print(f"\\n✅ Trained {N_SEEDS} independent models. "
      f"Test RMSE = {np.mean(all_rmse):.2f} +/- {np.std(all_rmse):.2f} ({N_SEEDS} runs)")
"""

code_attn_consistency = """cross_seed_mean = all_mean_attn.mean(axis=0)   # (28,) -- consensus attention profile
cross_seed_std = all_mean_attn.std(axis=0)     # (28,) -- how much THIS TIMESTEP varies ACROSS independently-trained models

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
colors = ['#e74c3c' if v > np.percentile(cross_seed_mean, 75) else '#3498db' for v in cross_seed_mean]
ax.bar(range(28), cross_seed_mean, color=colors, alpha=0.8, yerr=cross_seed_std, capsize=2)
ax.axhline(1/28, color='gray', ls='--', alpha=0.5, label='Uniform (1/28)')
ax.set_xlabel('Timestep (0=oldest, 27=most recent)')
ax.set_ylabel('Attention Weight (mean ± std ACROSS 30 SEEDS)')
ax.set_title(f'Cross-Seed Attention Profile Consistency ({N_SEEDS} independently-trained models)', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
im = ax.imshow(all_mean_attn, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Timestep (0=oldest, 27=most recent)')
ax.set_ylabel('Seed / Model Index (1..N_SEEDS)')
ax.set_title('Per-Model Average Attention Profile (one row per independently-trained model)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Attention Weight')

plt.tight_layout()
plt.savefig('plots/h3_cross_seed_attention_consistency.png', dpi=150, bbox_inches='tight')
plt.show()

cv_per_timestep = cross_seed_std / (cross_seed_mean + 1e-12)
print(f"\\nCross-seed consistency (coefficient of variation per timestep, lower = more consistent):")
print(f"  Mean CV across all 28 timesteps: {cv_per_timestep.mean():.3f}")

top5_consensus = np.argsort(cross_seed_mean)[-5:][::-1]
print(f"\\nTop-5 attended days (consensus across {N_SEEDS} independently-trained models):")
for rank, idx in enumerate(top5_consensus):
    print(f"  #{rank+1}: Timestep {idx} ({28-idx} days ago) -- "
          f"weight={cross_seed_mean[idx]:.4f} +/- {cross_seed_std[idx]:.4f}")
"""

code_feature_consistency = """# How often does each feature land in a model's Top-10 attention-weighted
# features? This is the direct "feature importance consistency" metric --
# not just "is the SAME feature always #1", but "across 30 independent
# retrainings, how stable is the set of features the model relies on".
n_features_total = all_feat_importance.shape[1]
top10_count = np.zeros(n_features_total, dtype=int)
for s in all_top10_sets:
    for idx in s:
        top10_count[idx] += 1

order = np.argsort(top10_count)[::-1]
print("="*88)
print(f"  FEATURE IMPORTANCE CONSISTENCY ACROSS {N_SEEDS} INDEPENDENTLY-TRAINED MODELS")
print("="*88)
print(f"  {'Feature':<28} {'Top-10 hit rate':>16}  {'Category':<18} {'Domain?'}")
print(f"  {'-'*82}")
for idx in order[:20]:
    name = dl_seq_features[idx]
    rate = top10_count[idx] / N_SEEDS
    cat = categorize(name)
    is_domain = '✅' if cat in domain_validated else '❌'
    print(f"  {name:<28} {rate*100:>14.0f}%  {cat:<18} {is_domain}")

stable_top10 = [dl_seq_features[idx] for idx in order if top10_count[idx] == N_SEEDS]
print(f"\\n  Features in EVERY single model's Top-10 ({N_SEEDS}/{N_SEEDS} seeds, 100% consistent): {len(stable_top10)}")
for n in stable_top10:
    print(f"    - {n}")

mean_feat_importance = all_feat_importance.mean(axis=0)
consensus_top10 = set(np.argsort(mean_feat_importance)[-10:])
consensus_domain_hits = sum(1 for idx in consensus_top10 if categorize(dl_seq_features[idx]) in domain_validated)
print(f"\\n  Consensus (mean-importance) Top-10 domain-validated hit rate: "
      f"{consensus_domain_hits}/10 = {consensus_domain_hits*10}% "
      f"{'✅ PASS (>70%)' if consensus_domain_hits/10 > 0.7 else '❌ FAIL'}")
"""

code_export_summary = """h3_30seed_results = {
    'n_seeds': N_SEEDS, 'seeds': SEEDS,
    'all_rmse': all_rmse, 'all_r2': all_r2,
    'rmse_mean': float(np.mean(all_rmse)), 'rmse_std': float(np.std(all_rmse)),
    'all_mean_attn': all_mean_attn.tolist(),
    'cross_seed_mean_attn': cross_seed_mean.tolist(),
    'cross_seed_std_attn': cross_seed_std.tolist(),
    'mean_cv_per_timestep': float(cv_per_timestep.mean()),
    'top10_hit_count_per_feature': {dl_seq_features[i]: int(top10_count[i]) for i in range(n_features_total)},
    'consensus_top10_domain_hit_rate': consensus_domain_hits / 10,
}
with open('metrics/h3_30seed_attention_consistency.pkl', 'wb') as f:
    pickle.dump(h3_30seed_results, f)
print("✅ Saved metrics/h3_30seed_attention_consistency.pkl")

print("\\n" + "="*80)
print("  H3 CROSS-SEED ROBUSTNESS -- ADDENDUM TO FINAL SUMMARY")
print("="*80)
print(f"  Model quality is stable across seeds: RMSE = {np.mean(all_rmse):.2f} +/- {np.std(all_rmse):.2f} ({N_SEEDS} runs)")
print(f"  Temporal attention pattern is "
      f"{'STABLE' if cv_per_timestep.mean() < 0.5 else 'SOMEWHAT VARIABLE'} "
      f"across independent re-trainings (mean CV={cv_per_timestep.mean():.3f})")
print(f"  Consensus feature-importance domain hit rate: {consensus_domain_hits}/10 = {consensus_domain_hits*10}%")
print("="*80)
"""

cells.append(new_md_cell(md_analysis5))
cells.append(new_code_cell(code_train_loop))
cells.append(new_code_cell(code_attn_consistency))
cells.append(new_code_cell(code_feature_consistency))
cells.append(new_code_cell(code_export_summary))

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
print("Inserted SEEDS-setup cell after old cell 2. Modified old cell 4 (representative seed).")
print("Appended 5 new cells (Analysis 5: cross-seed consistency).")
print("Total cells now:", len(cells))
