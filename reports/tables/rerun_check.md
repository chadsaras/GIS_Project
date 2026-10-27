# Clean rerun check (scripts/run_all.sh from a fresh clone and data folder, finished 2026-10-27)

Earth Engine rasters, caption tiles and Kaggle captions were reused; every other step was rerun.

| variant | R² frozen | R² rerun | RMSE frozen | RMSE rerun | ΔRMSE |
| --- | --- | --- | --- | --- | --- |
| V1 | 0.271 | 0.271 | 0.510 | 0.510 | +0.000 |
| V2 | 0.123 | 0.123 | 0.559 | 0.559 | +0.000 |
| V3 | 0.459 | 0.469 | 0.439 | 0.435 | -0.004 |
| V4 | 0.441 | 0.489 | 0.446 | 0.427 | -0.020 |
| V5 | 0.483 | 0.483 | 0.429 | 0.429 | -0.000 |
| V6 | 0.501 | 0.509 | 0.421 | 0.418 | -0.004 |
| V7 | 0.502 | 0.519 | 0.421 | 0.414 | -0.007 |
| V8 | 0.501 | 0.516 | 0.422 | 0.415 | -0.006 |
| V9 | 0.483 | 0.489 | 0.429 | 0.426 | -0.003 |
| V10 | 0.523 | 0.522 | 0.412 | 0.412 | +0.000 |
| V11_z_A_full | 0.184 | 0.177 | 0.539 | 0.541 | +0.002 |
| V11_emb_A_raw | 0.337 | 0.337 | 0.486 | 0.486 | +0.000 |
| V11_emb_A_full | 0.397 | 0.397 | 0.463 | 0.463 | +0.000 |
| V11_z_A_facts | 0.031 | -0.147 | 0.587 | 0.639 | +0.052 |
| V11_gsed | 0.345 | 0.345 | 0.483 | 0.483 | +0.000 |
| V11_emb_A_facts | 0.296 | 0.296 | 0.501 | 0.501 | +0.000 |
| V11_z_A_raw | 0.030 | 0.135 | 0.588 | 0.555 | -0.033 |
| V11_z_gsed | 0.149 | 0.150 | 0.551 | 0.550 | -0.000 |
| V12 | 0.386 | 0.386 | 0.468 | 0.468 | +0.000 |

| question | frozen ΔRMSE [95% CI] | rerun ΔRMSE [95% CI] | verdict frozen / rerun |
| --- | --- | --- | --- |
| RQ1 night-light steering (V4 vs V3 / V4 vs V3) | +0.007 [-0.028, +0.041] | -0.008 [-0.043, +0.030] | no clear effect / no clear effect |
| RQ2 text + steering (V6 vs V4 / V6 vs V4) | -0.025 [-0.050, +0.004] | -0.009 [-0.034, +0.014] | no clear effect / no clear effect |
| RQ3 POI/roads on top (V7 vs V6 / V7 vs V6) | -0.001 [-0.017, +0.016] | -0.004 [-0.019, +0.010] | no clear effect / no clear effect |
| RQ4 hybrid vs best single (V10 vs V8 / V10 vs V7) | -0.009 [-0.035, +0.015] | -0.002 [-0.024, +0.019] | no clear effect / no clear effect |

Deterministic steps (V1, V2, V11 on raw GSED and Stage A, V12, caption audit) match exactly. Neural Stage B/C
variants differ by up to 0.02 RMSE (multi-threaded CPU training is not bit-reproducible; V11 ridge on Stage B z
inherits that). All four research-question verdicts are unchanged; the RQ2 text gain shrinks from -0.025 to -0.009.
