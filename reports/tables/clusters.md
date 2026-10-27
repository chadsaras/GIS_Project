# Economic types from Stage C vectors: out-of-sample test (Minas Gerais)

Clusters fitted on each round's training municipalities; a test municipality is predicted by its cluster's training mean. Stage C rows: mean (min) over 5 seeds. For reference, the GDP model itself reaches R² 0.50 (V7) and 0.52 (V10).

| space | k | R² income per person | R² population |
| --- | --- | --- | --- |
| Stage C V7 pooled | 6 | 0.051 (0.032) | 0.453 |
| Stage C V7 pooled | 8 | 0.103 (0.083) | 0.535 |
| Stage C V7 pooled, size removed | 6 | 0.065 (0.037) | 0.020 |
| Stage C V7 pooled, size removed | 8 | 0.137 (0.124) | 0.031 |
| raw GSED mean | 6 | 0.233 | -0.049 |
| raw GSED mean | 8 | 0.288 | -0.030 |
| land cover + lights | 6 | 0.167 | 0.171 |
| land cover + lights | 8 | 0.231 | 0.190 |
| population only | 6 | 0.074 | 0.936 |
| population only | 8 | 0.070 | 0.965 |

## Stability (adjusted Rand index; 1 = identical groups, 0 = chance)

| k | Stage C size-removed, seeds of one round | Stage C size-removed, across rounds | raw GSED, two k-means starts |
| --- | --- | --- | --- |
| 6.0 | 0.634 | 0.463 | 0.895 |
| 8.0 | 0.620 | 0.445 | 0.716 |

## Consensus types (k = 6, co-association over 25 Stage C models)

Descriptive only. Types explain 8% of income differences (in-sample).

| type | towns | median people | median GDP per person (R$) | mean log light | built | crop | tree | grass | mean confidence | largest towns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 267 | 6,638 | 23,936 | 0.19 | 0.9% | 8.1% | 41.4% | 47.6% | 0.64 | Juiz de Fora, Divinópolis, Sete Lagoas |
| 2 | 24 | 45,322 | 27,870 | 1.64 | 16.7% | 2.8% | 50.2% | 28.1% | 0.91 | Belo Horizonte, Contagem, Betim |
| 3 | 161 | 7,345 | 22,528 | 0.32 | 1.5% | 4.8% | 44.6% | 48.1% | 0.71 | Pouso Alegre, Varginha, Conselheiro Lafaiete |
| 4 | 219 | 8,226 | 18,112 | 0.08 | 0.4% | 6.7% | 44.1% | 41.7% | 0.55 | Uberlândia, Montes Claros, Governador Valadares |
| 5 | 121 | 8,350 | 22,411 | 0.05 | 0.3% | 13.0% | 35.1% | 36.9% | 0.68 | Uberaba, Patos de Minas, Araguari |
| 6 | 61 | 10,204 | 37,781 | 0.02 | 0.2% | 19.3% | 25.6% | 31.5% | 0.85 | Paracatu, Unaí, Januária |
