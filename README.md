# Source Code — BMCS3403 Project II

**A Hybrid Deep Learning Approach for Dengue Fever Prediction in a Real-World Case Study**

Ong Shi Kai · Bachelor of Data Science (Honours) · Supervisor: Ts. Dr. Tan Choo Jun

Every notebook in this folder is committed **with its outputs saved**, so the results can be
read without running anything.

---

## 1. Environment

```
Python 3.13
pip install -r requirements.txt
```

Run Jupyter from this folder, not from a subdirectory — all paths are relative to the root.

A GPU is not required. The models are small (≈42k parameters) and the full 30-seed suite
runs on CPU.

---

## 2. Where each thesis result comes from

| Notebook | Produces |
|---|---|
| `00_setup_canonical_baselines_AUDITNOTE.ipynb` | Canonical dataset, splits, scalers, SARIMA fit. **Run this first.** |
| `00b_benchmark_dataset_2013_2022_verification.ipynb` | Provenance and verification of the benchmark dataset — see §7 |
| `01_data_preprocessing_feature_engineering_AUDITNOTE.ipynb` | Feature engineering; documents all six feature families |
| `08_multi_horizon_comparison_30run_RESULT_REVIEWED.ipynb` | **H1, H2** — Table 5.2, the headline RMSE/MAE/R² table |
| `09_Interpretability_domain_validation_PATCHED.ipynb` | **H3** — attention rankings vs documented dengue risk factors |
| `13_computational_efficiency_analysis_FIXED.ipynb` | **H4** — training time, inference latency, model size |
| `14_bigo_complexity_analysis_FIXED.ipynb` | **H4** — empirical scaling exponent β |
| `10_realworld_improved_v2.ipynb` | **H5** — frozen generalisation to 2023–2025 |
| `Q6_17_finetune_patch_experiment.ipynb` | **H5** — the six adaptation arms (A0–A5) |
| `11_hyperparameter_tuning_analysis_FIXED.ipynb` | Hyperparameter sensitivity |
| `12_overfitting_analysis_30SEED.ipynb` | Train/validation gap across 30 seeds |
| `15_full_reproduction_30seed_5horizon.ipynb` | End-to-end reproduction of every reported number |
| `Q8_18_all_windows_coverage.ipynb` | Input-window length sweep |
| `Q9_19_dataset_splitting_analysis.ipynb` | Cross-validation design, rolling-origin evaluation |
| `Singapore_Dengue_Dataset_*.ipynb` | Construction of the 2013–2022 and 2023–2025 datasets |

`canonical_pipeline.py` is the single source of truth: the `TemporalAttention` layer, the
model builders, the data splits and the seeding. Every notebook imports it.

### Where the model training actually happens

The filenames `02_train_…` through `06b_train_…` are misleading: those are early single-run
drafts and **none of the reported results come from them** (see §5). Training happens in two
places.

**Model definitions — `canonical_pipeline.py`:**

| Line | Definition |
|---|---|
| 209 | `class TemporalAttention(layers.Layer)` — W (64×64) + b (64) + u (64) = 4,224 parameters |
| 247 | `build_standalone_lstm(...)` — the architecturally identical control |
| 265 | `build_lstm_transformer(...)` — the ILT |
| 302 | `train_once(builder, seed, ...)` — one seeded training run |
| 48 / 117 / 169 | feature engineering, the 60-feature DL and 44-feature RF sets, splits and scalers |

**Training loops — inside the result notebooks**, each over thirty fixed seeds
(`SEEDS = [42 + 7*i for i in range(30)]`):

| Notebook | Trains | Supports |
|---|---|---|
| `08_multi_horizon_comparison_30run_RESULT_REVIEWED` | ILT, Standalone LSTM, RF, SARIMA | **H1, H2** — Table 5.2 |
| `09_Interpretability_domain_validation_PATCHED` | ILT (attention weights extracted) | **H3** |
| `10_realworld_improved_v2` | ILT | **H5** — frozen generalisation |
| `Q6_17_finetune_patch_experiment` | ILT, six adaptation arms | **H5** — A0–A5 |
| `13_computational_efficiency_analysis_FIXED` | ILT, LSTM, RF, SARIMA | **H4** |
| `11_hyperparameter_tuning_analysis_FIXED` | ILT, RF | Hyperparameter sensitivity |
| `12_overfitting_analysis_30SEED` | ILT, LSTM, RF | Train/validation gap |
| `15_full_reproduction_30seed_5horizon` | ILT, LSTM | End-to-end reproduction |
| `Q8_18`, `Q9_19` | ILT | Window sweep, splitting design |
| `16_rf_feature_parity_multihorizon.py` | RF, SARIMA | Fair-feature Random Forest |

To read the architecture, open `canonical_pipeline.py` lines 209–300. To see it trained and
evaluated, open `08_multi_horizon_comparison_30run_RESULT_REVIEWED.ipynb`.

`16_rf_feature_parity_multihorizon.py` gives the Random Forest the same engineered features
as the deep models, so the comparison isolates architecture rather than feature set.

### Figure numbering — notebook against thesis

Figures were numbered by the notebook that produced them (`Q6_17` → 17.x, `Q8_18` → 18.x,
`Q9_19` → 19.x, and the `revision/` analyses → B*n*). Chapter 5 of the thesis renumbers them
in reading order. The titles printed inside the image files are the **notebook** numbers,
because that is what the code that generated them actually writes; the table below maps each
one to its thesis number.

| Thesis | Title inside the image | Produced by | File |
|---|---|---|---|
| Figure 5.1 | Figure B1 | `revision/viva_boxplots.py` | `revision/plots/B1_rmse_all_models_all_horizons.png` |
| Figure 5.2 | Figure B3 | `revision/viva_boxplots.py` | `revision/plots/B3_h2_paired_differences.png` |
| Figure 5.3 | *(no number in image)* | `09_Interpretability_domain_validation_PATCHED` | `plots/h3_feature_importance.png` |
| Figure 5.4 | *(no number in image)* | `09_Interpretability_domain_validation_PATCHED` | `plots/h3_temporal_attention_profile.png` |
| Figure 5.5 | Figure 18.1 | `Q8_18_all_windows_coverage` | `plots/nb18_h3_all_horizons.png` |
| Figure 5.6 | *(no number in image)* | `14_bigo_complexity_analysis_FIXED` | `plots/bigo_scaling_comparison.png` |
| Figure 5.7 | Figure 18.2 | `Q8_18_all_windows_coverage` | `plots/nb18_h4_all_horizons.png` |
| Figure 5.8 | Figure 19.1 | `Q9_19_dataset_splitting_analysis` | `plots/nb19_splitting_analysis.png` |
| Figure 5.9 | Figure 18.3 | `Q8_18_all_windows_coverage` | `plots/nb18_window_sweep.png` |
| Figure 5.10 | Figure B8 | `revision/viva_boxplots.py` | `revision/plots/B8_h5_oos_r2.png` |
| Figure 5.11 | Figure 17.1 | `Q6_17_finetune_patch_experiment` | `plots/nb17_arms_r2_boxplot.png` |
| Figure 5.12 | Figure 17.3 | `Q6_17_finetune_patch_experiment` | `plots/nb17_arms_forecast_overlay.png` |

The image files in this folder are left exactly as the notebooks produced them. The versions
embedded in the thesis carry the Chapter 5 numbers instead, so that the figure and its caption
agree on the page; nothing else about them differs.

---

## 3. Reproducing the results

Fastest route — one notebook:

```
15_full_reproduction_30seed_5horizon.ipynb
```

It regenerates every number in Chapter 5 across 30 seeds × 5 horizons and short-circuits
through `metrics/full_reproduction_checkpoint.pkl`, so it finishes in minutes. Delete that
checkpoint to force a full retrain.

**Seeds.** `SEEDS = [42 + 7*i for i in range(30)]` → 42, 49, 56, … 245. Fixed on
2026-07-20; every reported figure is a mean over those 30 runs. The `patch_nb*.py` scripts
are the ones that applied the seed fixes, kept here so the correction is auditable rather
than merely asserted.

**Determinism.** Runs reproduce on a given machine and library stack. Exact bit-level
agreement across different TensorFlow builds or CPU/GPU backends is not guaranteed, which is
normal for floating-point reductions. Aggregate figures should agree to reported precision;
an individual seed's fourth decimal place may not.

---

## 4. Two things deliberately not included

**Trained model files.** `models/sarima_model.pkl` (55 MB) and `models/rf_model.pkl` (11 MB)
are excluded — pickled model objects are not portable across library versions, and both
regenerate quickly:

- `sarima_model.pkl` → run notebook `00` (~1–2 minutes)
- `rf_model.pkl` → run notebook `03`, but read §5 first

The two Keras checkpoints (`models/tmp_ILT_proposed.keras`,
`models/tmp_Standalone_LSTM.keras`, ~0.5 MB each) **are** included, so the proposed model can
be loaded and inspected without training anything.

**Working plots.** The project generated 97 figures during development. Only the 12 that
appear in the thesis are included here (9 in `plots/`, 3 in `revision/plots/`). The rest were
superseded — 62 of them predate the 2026-07-20 seed fix and show earlier numbers. Every
figure is in any case embedded in the notebook that produced it.

---

## 5. Known issue — notebooks 02 to 07 will not run as-is

`02_train_sarima`, `03_train_random_forest_PATCHED`, `04_train_vanilla_lstm`,
`05_train_vanilla_transformer`, `06_train_lstm_transformer`,
`06b_train_rf_enhanced_lstm_transformer` and `07_comparison_and_evaluation` all open:

```python
with open('processed_data/all_data.pkl', 'rb') as f:
```

`all_data.pkl` was the earlier 53-feature preprocessing artefact. It was superseded by
`canonical_data.pkl` when the feature set was corrected — the old one was missing the
weather-lag family (`temperature_lag14/21`, `rainfall_lag14/21/28`, `humidity_lag14`) that
carries the rain → breeding → transmission argument, and it encoded `day_of_week` as a single
integer. Nothing here regenerates it, so **these seven notebooks raise `FileNotFoundError`.**

**No reported result is affected.** Every figure and table in the thesis comes from the
canonical path `00`/`01` → `08`–`15`, which is self-contained. Notebooks 02–07 are retained
because their saved outputs are the record of the single-run baseline stage the project passed
through, and because notebook 03 documents the feature-leakage fix that prompted the whole
correction — an earlier Random Forest that could see case-history columns scored R² = 0.93;
after the fix it scores R² = −0.35, which is the figure reported in the thesis.

They are read-only history, not a live entry point. Each of the seven opens with a banner cell
saying so, so the point is visible on the page as well as here.

---

## 6. Folder contents

```
├── *.ipynb                      22 notebooks, outputs saved
├── canonical_pipeline.py        shared model, data and seeding definitions
├── viva_revision_common.py      helpers for the revision analyses
├── 16_rf_feature_parity_*.py    fair-feature Random Forest
├── patch_nb*.py                 the seed / memory fixes that were applied
├── singapore_dengue_weather_2013_2022.csv    benchmark dataset (3,652 days)
├── singapore_dengue_weather_2023_2025.csv    real-world holdout
├── singapore_dengue_weather_2013_2025_combined.csv
├── DengAI_merged 1990-2010.csv  secondary dataset (exploratory only)
├── moh_2023/2024/2025.xls*      raw MOH weekly bulletins
├── processed_data/              canonical_data.pkl (written by notebook 00/01)
├── models/                      .keras checkpoints (.pkl excluded, see §4)
├── metrics/                     71 result files behind every reported number
├── plots/                       the 9 thesis figures generated by the notebooks
├── revision/                    hypothesis tests, post-hoc comparisons, 3 thesis figures
└── CANONICAL_METRICS_RECONCILIATION.md
```

---

## 7. Data provenance

Daily dengue counts from Singapore's National Environment Agency; weather from Open-Meteo
(ERA5 reanalysis); population density from the Singapore Department of Statistics. All public
and aggregate — **no individual patient records are involved at any stage**, so no ethics
approval was required. Full provenance is in Section 4.2.1 of the thesis.

**The two datasets have different provenance evidence, and this is stated plainly:**

- **2023–2025 holdout** — fully constructed by
  `Singapore_Dengue_Dataset_FIXED_v7.ipynb` and
  `Singapore_Dengue_Dataset_2023_2025_UPDATED.ipynb`, from the committed MOH bulletins and
  Open-Meteo.
- **2013–2022 benchmark** — assembled early in the project; **the original ingestion script
  was not retained**. `00b_benchmark_dataset_2013_2022_verification.ipynb` closes as much of
  that gap as the surviving materials allow: it re-fetches the meteorological columns from
  Open-Meteo ERA5 and compares them, recomputes every derived and calendar column from the
  raw columns (all reproduce exactly), and confirms the weekly step structure of
  `dengue_cases`. The case counts themselves cannot be re-derived, because the 2013–2022 NEA
  weekly bulletins were not archived; the notebook reports the annual totals so they can be
  checked against the published figures.

  The re-fetch was run and the meteorological columns match their source: mean absolute
  difference 0.001–0.042 across the seven variables, correlation 0.9912–0.9995 over 3,652
  days. The largest single-day differences (rainfall 68 mm, wind gust 8.8 m/s) are isolated
  and arise from ERA5 reanalysis revisions since the original download. They do not reach the
  model: preprocessing caps each meteorological variable at its 99th percentile
  (rainfall 41.15 mm), so both the original and the revised value map to the same capped
  input.

That notebook also records one wording correction for Section 4.2.1: `population_density` is
an **annual step series** (one constant value per calendar year, 10 distinct values across
3,652 days), not "interpolated to daily" as the text currently states. The column is a weak,
near-constant predictor either way, so no reported result changes.
