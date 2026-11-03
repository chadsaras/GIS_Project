# Final talk plan: the work since the mid-term (`final_talk.tex`)

Audience: the course panel, who saw the mid-term. Focus: what was done after it. No application (ranges, warnings,
website). Same design as the mid-term deck. A chip in the top-right corner marks every slide as
**RECAP · BEFORE MID-TERM** or **NEW SINCE MID-TERM**.

## Work inventory

### Before the mid-term (29 Sep – 6 Oct; 51 commits, ~5,000 lines of code)
1. Plan, configuration, server and Earth Engine setup.
2. Labels: IBGE GDP 2022, Census 2022 population, regions, flags for 853 municipalities.
3. 1 km grid (586,873 cells, 575,166 kept): GSED embeddings, VIIRS night lights, WorldCover, OSM POIs and roads,
   Microsoft + OSM building footprints.
4. Five spatial folds over the 70 immediate regions.
5. Text: 10,000 sampled cells, Esri tiles, Qwen2.5-VL-7B captions on Kaggle (pilot, short prompt), fact sentences,
   cleaning (filler, claim checks, RemoteCLIP): 84% of 60,023 sentences kept.
6. Stage A (image–text alignment), Stage B (night-light steering), Stage C (cell pooling to municipal GDP).
7. Baselines (night lights, land cover, ridge, LightGBM) and the hybrid; all 12 variants over 5 rounds.
8. Stage C fix (validation recalibration, revised grid): RQ1 changed from "steering helps" to "no clear effect".
9. Results and RQ1–RQ4 tests; mid-term deck. Open at the mid-term: caption audit, error analysis (code only).

### After the mid-term (6 Oct – 3 Nov; ~1,150 lines of new analysis code)
1. **Caption audit:** 300 random captions, 1,801 sentences labelled blind; false-claim share raw vs cleaned,
   cleaning precision and recall.
2. **Error analysis (RQ5):** maps, ten largest misses with reasons, error regression, LOWESS, Moran's I,
   seed spread vs error.
3. **Stage diagnostics:** Stage A recall and probe, Stage B accuracy, the embedding paper's neighbour check.
4. **What the text learned:** nearest training captions, rich-vs-poor word contrast.
5. **Clean rerun** of the whole pipeline from a fresh clone and data folder: same answers, text gain shrinks.
6. **Do the learned town vectors carry economic meaning?** Twin test and clustering test.
7. **Transfer to São Paulo:** models trained on Minas Gerais applied unchanged to 645 new municipalities.
8. Final report (12 pages), results frozen (tag v1.0-results), one-command rerun script, data release.
(Excluded from this talk: error ranges, risk warning, website.)

## Storyline

> We set out to estimate local GDP from free satellite data and test it strictly. At the mid-term the models were
> built and satellites explained half of GDP. Since then we asked whether the result holds up (rerun, caption
> audit), where and why it fails (mines, steel, commuter towns), what each part learned (diagnostics, text,
> town vectors), and whether it works elsewhere (São Paulo: satellites yes, night lights no).

## Slides (19 main + 3 backup)

| # | Route | Phase | Title (message) | Content |
| --- | --- | --- | --- | --- |
| 1 | – | – | Reading the economy from space | title |
| 2 | Objective | – | The question: can free satellite data estimate local GDP? | GDP map, 49× gap, goal, RQ1–RQ5 |
| 3 | Recap | recap | Recap: three learning stages, from pixels to official GDP | pipeline |
| 4 | Recap | recap | Recap: six free layers and 10,000 AI captions | layers, captions and cleaning |
| 5 | Recap | recap | Recap: every town is tested in a region never seen | fold map, rotation |
| 6 | Recap | recap | Recap: at the mid-term, satellites explained half of GDP | results chart, RQ status then, open items |
| 7 | Recap | – | Since the mid-term: seven questions | question → work → answer table |
| 8 | Robustness | new | A full rerun from scratch gives the same answers | frozen vs rerun tables |
| 9 | Robustness | new | Blind audit: a third of AI captions hold a false claim | method, bars, precision/recall |
| 10 | Errors | new | Predictions follow the map, but are pulled to the middle | official / predicted / error maps |
| 11 | Errors | new | The biggest misses are mines and steel mills | scatter, ten largest gaps |
| 12 | Errors | new | Errors grow with income, half as fast as night lights | LOWESS, slopes, Moran's I, seeds |
| 13 | Embeddings | new | Each stage did its job, yet steering does not reach GDP | stage diagnostics, neighbour check |
| 14 | Embeddings | new | The text space speaks the state's economic geography | word contrast, nearest captions |
| – | – | – | (removed: town-vector twins and clusters) | – |
| 16 | Transfer | new | Trained in Minas Gerais, it still works in São Paulo | SP maps, R² bars |
| 17 | Wrap-up | – | What did not work, and what it taught us | lessons table |
| 18 | Wrap-up | – | Answers to the five questions, and the limits | RQ chips, limits |
| 19 | – | – | Take-away | key numbers |
| B1 | backup | recap | Two recent ideas, never tested together | literature |
| B2 | backup | recap | Calibration mattered more than steering | before/after |
| B3 | backup | new | What predicts a large error? | error regression table |

Timing (about 15 minutes): 1–2 1:00 · 3–7 recap 3:00 · 8–16 new work 9:00 · 17–19 2:00.
