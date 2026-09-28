# Chapter 5.1 — 95% Confidence Intervals (supplementary computation)

Computed post-hoc from the existing 30-seed distributions (H1/H2) and the n=8 practitioner survey (H4). Method: two-sided 95% CI of the mean, t-distribution, CI = mean ± t(0.975, n−1) × (s/√n). No retraining performed. Source: `metrics/ch5_confidence_intervals.csv`.

**Note on n:** DL/RF models use n=30 seeds (t≈2.045). SARIMA is deterministic (single fit, no seed variation) so no seed-based CI applies — reported as a point value. H4 uses n=8 respondents (t≈2.365), giving intentionally wide intervals.


## R² — mean [95% CI] by horizon

| Horizon | SARIMA | Random Forest | Standalone LSTM | ILT (LSTM-Transformer) |
|---|---|---|---|---|
| 1d | -0.257 (deterministic, n=1) | -0.311 [-0.317, -0.306] | 0.703 [0.693, 0.713] | 0.717 [0.704, 0.730] |
| 7d | -0.260 (deterministic, n=1) | -0.322 [-0.327, -0.317] | 0.678 [0.669, 0.687] | 0.677 [0.661, 0.693] |
| 14d | -0.263 (deterministic, n=1) | -0.341 [-0.346, -0.335] | 0.662 [0.647, 0.677] | 0.656 [0.642, 0.670] |
| 21d | -0.267 (deterministic, n=1) | -0.366 [-0.372, -0.360] | 0.628 [0.616, 0.640] | 0.624 [0.613, 0.635] |
| 28d | -0.271 (deterministic, n=1) | -0.370 [-0.376, -0.365] | 0.583 [0.571, 0.596] | 0.588 [0.572, 0.604] |

## RMSE — mean [95% CI] by horizon

| Horizon | SARIMA | Random Forest | Standalone LSTM | ILT (LSTM-Transformer) |
|---|---|---|---|---|
| 1d | 62.562 (deterministic, n=1) | 63.891 [63.758, 64.024] | 30.798 [30.270, 31.327] | 30.047 [29.345, 30.749] |
| 7d | 62.823 (deterministic, n=1) | 64.365 [64.248, 64.482] | 32.188 [31.747, 32.630] | 32.183 [31.397, 32.968] |
| 14d | 63.131 (deterministic, n=1) | 65.040 [64.909, 65.172] | 33.037 [32.291, 33.782] | 33.340 [32.655, 34.025] |
| 21d | 63.442 (deterministic, n=1) | 65.875 [65.736, 66.014] | 34.812 [34.262, 35.363] | 34.992 [34.469, 35.515] |
| 28d | 63.758 (deterministic, n=1) | 66.200 [66.073, 66.327] | 36.987 [36.424, 37.549] | 36.761 [36.037, 37.486] |

## MAE — mean [95% CI] by horizon

| Horizon | SARIMA | Random Forest | Standalone LSTM | ILT (LSTM-Transformer) |
|---|---|---|---|---|
| 1d | 37.760 (deterministic, n=1) | 45.394 [45.270, 45.518] | 17.007 [16.603, 17.411] | 16.581 [16.050, 17.112] |
| 7d | 38.053 (deterministic, n=1) | 45.972 [45.861, 46.083] | 18.680 [18.248, 19.111] | 18.539 [17.980, 19.098] |
| 14d | 38.386 (deterministic, n=1) | 46.575 [46.453, 46.696] | 19.715 [19.338, 20.092] | 19.866 [19.389, 20.343] |
| 21d | 38.708 (deterministic, n=1) | 47.410 [47.291, 47.531] | 21.202 [20.714, 21.691] | 21.383 [21.042, 21.724] |
| 28d | 39.041 (deterministic, n=1) | 48.092 [47.964, 48.219] | 23.192 [22.844, 23.541] | 23.010 [22.547, 23.473] |

## H4 practitioner survey (n=8)

| Metric | Mean | 95% CI |
|---|---|---|
| H4 combined accuracy | 0.854 | [0.716, 0.992] |
| H4 Likert mean | 3.850 | [3.557, 4.143] |

## Example prose for Chapter 5.1

> Across 30 random seeds, the ILT model achieved a mean test R² of 0.717 (95% CI [0.704, 0.730]) at the 1-day horizon, declining to 0.588 (95% CI [0.572, 0.604]) at 28 days. The narrow intervals (half-width ≤ 0.02 R²) indicate the model's performance is stable and not an artefact of a single favourable initialisation. In the practitioner survey (n=8), interpretation accuracy averaged 85.4% (95% CI [71.6%, 99.2%]) and the Likert usefulness rating averaged 3.85 (95% CI [3.56, 4.14]); the wider intervals here reflect the small sample and should be reported as such.
