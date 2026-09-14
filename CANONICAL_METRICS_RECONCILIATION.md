# Canonical 数字清单（数据溯源核对，2026-06-19）

结论先行：**文件新旧（mtime）不代表对错**。最新生成的 `model_comparison.csv`（06-19）整张表必须废弃——它把"已修好"的 SARIMA/RF 和"从未修好、架构都不对"的 DL 结果拼在一起。下面逐项给出可信数字、来源、和必须弃用的文件。

## 1. 各模型最终数字

| 模型 | RMSE | MAE | R² | 来源（可信） |
|---|---|---|---|---|
| SARIMA | 62.52 | 37.71 | -0.2567 | `02_train_sarima.ipynb` → `metrics/sarima_results.pkl`（06-18，多份文件一致） |
| Random Forest（weather-only，正确） | 64.91 | 46.28 | -0.3533 | `03_train_random_forest_PATCHED.ipynb` → `metrics/rf_results.pkl`（06-19，与 Notebook 08 内嵌的 RF 检查值 R²≈-0.27~-0.38 互相印证） |
| Standalone LSTM（h=1d，30-seed均值） | 30.56±1.30 | 16.83 | 0.7077 | `08_..._30run_RESULT_REVIEWED.ipynb`，30-seed 均值与 `15_full_reproduction...ipynb` 几乎一致（30.50/0.7086） |
| LSTM-Transformer / ILT（h=1d，30-seed均值） | 30.08±1.01 | 16.51 | 0.7169 | 同上，Notebook 15 给出 29.73/0.7235（两者差<1 RMSE，可视为互相验证） |

5 个 horizon 的完整对照表用 **Notebook 08（RESULT_REVIEWED）的输出**（已手动修过 `cases_7d_avg` 泄漏）：

| h | SARIMA | RF(weather) | Std LSTM | ILT |
|---|---|---|---|---|
| 1d | 62.56 | 64.06 | 30.56 | 30.08 |
| 7d | 62.82 | 64.06 | 32.08 | 31.96 |
| 14d | 63.13 | 65.14 | 33.93 | 33.05 |
| 21d | 63.44 | 65.86 | 34.85 | 34.77 |
| 28d | 63.76 | 66.53 | 36.53 | 36.31 |

**Vanilla Transformer（ablation 用）：目前没有可信数字。** 唯一存在的值（RMSE 28.36, R²=0.7484）来自 2026-05-29、修复前的旧架构（53 特征，无天气滞后项），从未在 canonical pipeline 下重跑过。写 Chapter 5.4 前必须补跑一次，否则这一格只能留空或注明"待重跑"。

**RF-Enhanced LSTM-Transformer（Notebook 06b）：不是更优变体，是被项目自己废弃的分支。** `canonical_pipeline.py` 文档原文明确写"Do NOT import anything from Notebooks 01, 04, 05, 06, 06b, 07 ... must not be used for any reported result"。它在 `model_comparison.csv` 里数字好看（26.87, R²=0.7742）纯粹因为同样是旧架构、同样没有修复任何泄漏。**结论：整段都不要引用，不只是数字，连这个模型本身都不该出现在 Chapter 5。**

## 2. H2 显著性检验

唯一方法论正确的检验：`metrics/full_reproduction_h2_test.csv`（Notebook 15，30-seed 配对 t-test/Wilcoxon）：

| h | t检验 p | Wilcoxon p | 结论 |
|---|---|---|---|
| 1d | 0.0459 | 0.1094 | ✅ 显著（仅此一个） |
| 7d | 0.8358 | 0.9677 | ❌ 不显著 |
| 14d | 0.8571 | 0.7303 | ❌ 不显著 |
| 21d | 0.2590 | 0.2711 | ❌ 不显著 |
| 28d | 0.9991 | 0.8872 | ❌ 不显著 |

**作废**：`significance_tests.csv`（05-29，p=0.0102/supported=True）和 `ch5_final_significance_tests.csv`（06-02，p=0.997）——两者互相矛盾，且都只看单一 horizon、方法不如 Notebook 15 严谨，统一用上表替换。

**特别提醒**：`full_reproduction_ensemble_test.csv` 显示 5/5 horizon "全部显著"，**但 Notebook 08 自己的 audit note 已明确指出这个检验把逐日预测误差当独立样本配对，违反独立性假设（误差跨天自相关），属于 n 虚高导致 p 值虚低的经典误用**（例如 h=7d 差距只有 -0.11% 却显示"显著"）。这个文件只能当 sanity check，**不能**用来加强 H2 的结论，Notebook 15 自己也把它标注为"bonus check"而非证据。

## 3. H3 / H4 / 泛化

- **H3**（领域验证）：Top-10 命中率 100%（`ch5_final_h3_domain_validation.csv`，来自 `09_h3_domain_validation_PATCHED.ipynb`）。泄漏 bug 只影响排名第 13 的特征标签，不影响 Top-10，这个数字稳健，可直接用。
- **H4**（同行验证）：n=8，识别准确率 85.4%，Likert 均值 3.85/5（`h4_validation_results.csv` / `ch5_final_h4_summary.csv`）。人工问卷数据，不受 pipeline bug 影响，可直接用。
- **真实世界泛化（2023–2025 OOS）**：用较新的 `realworld_generalisation.csv`（06-03）而非 `ch5_final_realworld_generalisation.csv`（06-02）。两者数字略有差异但结论一致：OOS R² 在 h=1/7/14 为正（0.59/0.23/0.14），h=21/28 为负——"3/5 horizon 为正"这个 Chapter 5 现有表述是对的，不用改。

## 4. 一个新发现：canonical_pipeline.py 自身有一个未修复的泄漏

`canonical_pipeline.py` 的 `get_feature_sets()` 排除清单（`case_pfx`）里没有 `cases_7d_avg`，而原始 CSV 确实带有这一列（与当天病例数 r≈0.96–0.99）。这意味着所有依赖 `canonical_data.pkl` 的产物（Notebook 00/01/11/12/13/14/15）的"公平、仅气象"特征分支里其实偷偷混了一个病例特征——这正是 Notebook 00/01 自己的 audit note 怀疑但没人去 `canonical_pipeline.py` 里实际确认的问题，现已确认为真。

实测影响分两种：
- **对 LSTM-Transformer / Standalone LSTM 几乎无影响**（Notebook 08 手动patch版 vs Notebook 15 未patch版，h=1d 仅差 0.3~0.4 RMSE）——因为完整病例历史本来就通过序列分支进入模型，feature分支多一列冗余信息边际作用很小。
- **对单独用这 45 个特征的 Random Forest 影响巨大**：`metrics/rf_results_fair.pkl`（R²=0.5234）就是这个泄漏的直接产物——同一套 RF，一旦混进这一列，R² 从该有的 -0.35 "变成" 0.52。**这个文件不能用**，正确的 RF 数字仍是上面第 1 节的 -0.3533（来自完全独立、手动排除了这 16 列的 `03_train_random_forest_PATCHED.ipynb`）。

## 5. 文件归档速查

**可信（直接引用）**
`sarima_results.pkl`、`rf_results.pkl`（06-19）、`08_..._30run_RESULT_REVIEWED.ipynb` 内嵌输出、`full_reproduction_rmse/mae/r2_30run.csv`、`full_reproduction_h2_test.csv`、`ch5_final_h3_domain_validation.csv`、`h4_validation_results.csv` / `ch5_final_h4_summary.csv`、`realworld_generalisation.csv`（06-03）。

**作废（不要引用）**
`model_comparison.csv`（整张表，新旧拼接）、`ch5_table51_overall_performance.csv`（RF R²=0.9375，确认泄漏）、`ch5_final_model_results.csv` / `ch5_final_model_summary.csv`（中间版本，已被 Notebook 15 取代）、`ch5_final_significance_tests.csv` 与 `significance_tests.csv`（互相矛盾、方法不严谨）、`rf_results_fair.pkl`（cases_7d_avg 泄漏）、`lstm_transformer_results.pkl`（Notebook 06 的旧架构单次跑，模型本身就不是 ILT）、`rf_enhanced_lstm_transformer_results.pkl`、`vanilla_lstm_results.pkl`、`vanilla_transformer_results.pkl`（三者均为 05-29 旧架构、从未重跑）、`full_reproduction_ensemble_test.csv`（仅作 sanity check，不可作为 H2 证据）。

**待补（目前没有可信数字）**
Vanilla Transformer 在 canonical pipeline 下的 ablation 结果——需要重新跑一次才能完整支撑 Chapter 5.4。
