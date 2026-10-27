# Twins in Stage C space vs baselines (10 nearest training municipalities, test municipalities)

Official log GDP per capita: sd 0.597. Stage C rows: mean (min) over 5 seeds.

| space | variant | twin R² GDP per capita | twin R² total GDP | twin R² population | median twin spread (pc) | model R² (pc) |
| --- | --- | --- | --- | --- | --- | --- |
| raw GSED mean (cosine) | - | 0.314 | 0.145 | 0.060 | 0.316 |  |
| population only | - | -0.009 | 0.804 | 0.992 | 0.471 |  |
| Stage C pooled | V3 | 0.329 (0.320) | 0.854 | 0.790 | 0.381 | 0.459 |
| Stage C hidden | V3 | 0.313 (0.282) | 0.859 | 0.806 | 0.384 | 0.459 |
| Stage C pooled, size removed | V3 | 0.336 (0.321) | 0.157 | 0.184 | 0.327 | 0.459 |
| Stage C pooled | V6 | 0.259 (0.249) | 0.805 | 0.739 | 0.427 | 0.501 |
| Stage C hidden | V6 | 0.243 (0.228) | 0.856 | 0.776 | 0.402 | 0.501 |
| Stage C pooled, size removed | V6 | 0.284 (0.273) | 0.399 | 0.466 | 0.382 | 0.501 |
| Stage C pooled | V7 | 0.223 (0.213) | 0.861 | 0.778 | 0.419 | 0.502 |
| Stage C hidden | V7 | 0.246 (0.232) | 0.876 | 0.804 | 0.408 | 0.502 |
| Stage C pooled, size removed | V7 | 0.405 (0.396) | 0.196 | 0.190 | 0.346 | 0.502 |
