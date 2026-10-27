# Reliability layer for an application (V10, out-of-sample)

Inputs a region without official statistics would have: satellite layers, cell count, the model's prediction. Every number below is scored on municipalities the step never saw.

## 1. Error ranges (split conformal, calibrated on each round's validation fold)

| nominal | test coverage | mean radius (log) | as a factor |
| --- | --- | --- | --- |
| 80% | 79.5% | 0.452 | ×/÷ 1.57 |
| 90% | 89.6% | 0.630 | ×/÷ 1.88 |

75% of municipalities are within a factor 1.5 of the official value (median error factor 1.26).

## 2. Satellite-only warnings

| signal | Spearman with abs. error | p |
| --- | --- | --- |
| bare_share | +0.117 | 0.000629 |
| built_share | +0.007 | 0.836 |
| ntl_top_share | -0.012 | 0.728 |
| log_n_cells | -0.084 | 0.0139 |
| pred | +0.206 | 1.36e-09 |
| risk score (cross-fitted, all signals) | +0.185 | 5.29e-08 |

The cross-fitted risk score finds the worst 20% of errors with AUC 0.66 (0.5 = chance).

| risk group | municipalities | mean abs. error | flagged (mining/hydro/top 1%) | 90% range coverage | 90% radius as a factor |
| --- | --- | --- | --- | --- | --- |
| low | 299 | 0.242 | 2 | 89.6% | ×/÷ 1.61 |
| medium | 263 | 0.292 | 5 | 89.7% | ×/÷ 1.87 |
| high | 291 | 0.372 | 27 | 91.1% | ×/÷ 2.42 |

Risk-grouped 90% ranges cover 90.2% of test municipalities overall.

## 3. Economic twins (10 nearest training municipalities)

| space | twin estimate RMSE | twin estimate R² | twin spread vs model error | twin-model disagreement vs model error |
| --- | --- | --- | --- | --- |
| raw GSED | 0.494 | 0.314 | +0.103 | +0.108 |
| Stage B z (GSED) | 0.508 | 0.274 | +0.098 | +0.065 |
| Stage B z (A full) | 0.522 | 0.236 | +0.037 | +0.031 |

For reference, V10 itself: RMSE 0.412. Spearman values are rank correlations with the model's absolute error (higher = better warning).
