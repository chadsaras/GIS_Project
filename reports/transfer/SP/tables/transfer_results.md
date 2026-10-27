# Transfer test: models trained on Minas Gerais applied unchanged to SP

645 municipalities, 231,922 kept 1 km cells. Each prediction is the mean of the 5 Minas Gerais rounds' models; nothing is refitted on the target state.

| variant | R² | RMSE | Spearman (ranks) | mean bias | R² if bias removed | calibration slope | R² in MG (out-of-sample) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | -0.061 | 0.597 | 0.331 | -0.209 | 0.069 | 0.64 | 0.271 |
| V3 | 0.203 | 0.518 | 0.539 | +0.022 | 0.204 | 0.68 | 0.459 |
| V4 | 0.241 | 0.505 | 0.576 | +0.108 | 0.276 | 0.82 | 0.441 |
| V5 | 0.169 | 0.528 | 0.492 | +0.057 | 0.179 | 0.71 | 0.483 |
| V6 | 0.220 | 0.512 | 0.514 | +0.066 | 0.233 | 0.80 | 0.501 |
| V7 | 0.325 | 0.476 | 0.581 | -0.000 | 0.325 | 0.92 | 0.502 |
| V8 | 0.319 | 0.478 | 0.580 | +0.053 | 0.328 | 0.95 | 0.501 |
| V9 | 0.365 | 0.462 | 0.634 | -0.041 | 0.370 | 1.02 | 0.483 |
| V10 | 0.330 | 0.474 | 0.606 | +0.090 | 0.354 | 0.97 | 0.523 |

Minas Gerais 90% range (±0.648 log, ×/÷ 1.91) holds for 84.2% of SP municipalities.

| risk group | municipalities | mean abs. error | flagged (industry > 60% or top 1%) | MG 90% range | holds in target |
| --- | --- | --- | --- | --- | --- |
| low | 11 | 0.265 | 0 | ×/÷ 1.55 | 90.9% |
| medium | 53 | 0.298 | 1 | ×/÷ 1.90 | 86.8% |
| high | 581 | 0.372 | 13 | ×/÷ 2.23 | 92.3% |

1 km activity (Stage B, A(full) input): predicted vs observed log night light, cell R² 0.820 (Spearman 0.558) over 231,922 cells.
