# Mid-term presentation plan

**Project:** Text- and Night-Light-Guided Satellite Embeddings for Regional Economic Estimation (Minas Gerais, Brazil)
**Status:** plan only. Nothing is built until approved.
**Assumed format:** about 10 minutes plus questions, 10 main slides plus 3 backup slides. Adjust when the time limit is confirmed (see "Open questions").

## Design rules (from the brief)

1. **One message per slide.** The slide title is the message, written as a full sentence ("Text alone ..."), not a topic ("Results").
2. **Visual first, text second.** Each slide has one main visual: a map, a diagram, a chart or an example image. Text is at most 3 short bullets or about 30 words.
3. **Only visuals that carry the point.** Maps for anything spatial, a diagram for the pipeline, a bar chart for numbers that are compared, real tiles and captions for the text part. No decorative icons or stock images.
4. **Progress shown honestly.** Finished work is marked done; running or pending work is shown as such (status chips: Done, Running, Next). Incomplete is fine; the concept and progress are the focus.
5. **Consistent look.** One accent colour for "our method", grey for baselines, green / amber for done / pending. The same map extent and colour scale wherever Minas Gerais appears.

## Storyline (what the audience should leave with)

> Satellites can estimate municipal GDP where statistics are weak. We combine two recent ideas (text descriptions of satellite tiles, and night-light steering of Google's satellite embeddings) and test them honestly on 853 Brazilian municipalities. The data pipeline is complete, first results are in (best model R² 0.53, explaining about half of the variation), and the text experiments are running.

Arc: **Why → Idea → How → Is it honest? → What we found → What's next.**

## Slide-by-slide plan

| # | Title (the message) | Main visual | Supporting text (max) | Time |
|---|---|---|---|---|
| 1 | Estimating municipal GDP from satellite embeddings, text and night lights | Hero image: GSED principal-component colour map of Minas Gerais, with one 1 km tile zoomed in as an inset | Name, course, date | 0:20 |
| 2 | Official GDP is missing or unreliable below the national level; satellites see everywhere | Map of 2022 official log GDP per capita across the 853 municipalities (rich south-west, poor north-east) | 2 bullets: the problem; satellites as an independent, consistent signal | 0:50 |
| 3 | We combine two recent ideas that have never been tested together | Two-box diagram: UrbanCLIP (images + generated text) and Embedding paper (GSED + night-light steering), merging into "our pipeline" | One line per paper: what we take from it; one line on the gap (never combined, weak evaluation) | 1:00 |
| 4 | **The pipeline:** three learning stages, from no labels to official GDP | Pipeline diagram: 4 inputs (GSED, captions, night lights, GDP) feeding Stage A (image-text alignment), Stage B (night-light steering), Stage C (sum-pooling to municipalities) | Label under each stage: "no labels", "cheap proxy label", "real label" | 1:15 |
| 5 | We built a 1 km dataset for all 853 municipalities of Minas Gerais | Data layers stacked on the grid (GSED, VIIRS, WorldCover, OSM + building footprints, Esri tiles, IBGE GDP), each with a small map thumbnail | Big numbers: 853 municipalities · 575k cells · 10,000 captioned tiles | 1:00 |
| 6 | A vision-language model describes each tile, and data checks remove false claims | Example card: satellite tile + its Qwen2.5-VL caption, with dropped sentences struck through and colour-coded by reason (filler / unsupported claim) | Small stat strip: 10,000 captions · 84% of sentences kept · checks against land cover, roads and buildings | 1:15 |
| 7 | Every municipality is tested in a region the model never saw | Fold map: 5 colours over the 70 immediate regions | 2 bullets: whole regions held out (no neighbour leakage, the flaw in UrbanCLIP); all stages retrained each round | 0:50 |
| 8 | First results: the hybrid explains about half of GDP variation | Horizontal bar chart of R² (or RMSE) by model: NTL only, land cover, ridge, LightGBM, raw GSED + Stage C, steered + Stage C, hybrid (accent colour) | RQ verdict chips: RQ1 steering → no clear effect · RQ4 hybrid → better (95% CI excludes 0) | 1:30 |
| 9 | Lessons so far: calibration mattered more than steering | Small before/after chart: Stage C RMSE 0.555 → 0.439 after validation calibration, with the RQ1 verdict flipping | 3 short lessons: over-dispersed predictions; honest re-testing changes conclusions; caption hallucinations need data checks | 0:50 |
| 10 | 80% done: text experiments are running, results this week | Progress bar by phase (Done / Running / Next) plus a short timeline: V5-V9 running → caption audit → error analysis and maps → report | RQ status table: RQ1 done, RQ2 and RQ3 running, RQ4 done, RQ5 next | 0:50 |
| **Total** | | | | **~9:40** |

### Backup slides (shown only if asked)

| # | Content | Visual |
|---|---|---|
| B1 | Data sources and licences (GSED, VIIRS, WorldCover, OSM, Microsoft buildings, Esri, IBGE) | Compact table |
| B2 | Stage A works: recall@10 about 52% against 4% chance | Small bar: model vs chance |
| B3 | Caption examples: forest, farmland, village, town | 4 tiles with one-line captions |

## Visual assets

| Asset | Slide | Source | Status |
|---|---|---|---|
| GSED colour map + zoomed tile | 1 | reports/figures/cells_qc.png (panel 3) + one Esri tile | Exists, needs cropping |
| Official GDP per capita map | 2 | reports/figures/labels_overview.png (map panel) | Exists, needs cropping / restyle |
| Two-papers-merge diagram | 3 | New, drawn | To make |
| Pipeline diagram | 4 | New, drawn (from the proposal figure) | To make |
| Data layer stack with thumbnails | 5 | New: thumbnails from cells_qc.png + OSM / buildings | To make |
| Caption example card | 6 | Pilot / full captions + tile (cell 883143 town or 731134 farmland) | To make |
| Fold map | 7 | reports/figures/folds_map.png | Exists |
| Results bar chart | 8 | reports/tables/main_results.csv | To make |
| Calibration before/after | 9 | First and second results tables (git history) | To make |
| Progress bar + RQ status | 10 | progress.txt | To make |

## Speaker notes (key points per slide)

1. **Title.** One sentence on what the project does.
2. **Problem.** GDP below national level is patchy and sometimes estimated; satellites give an independent, uniform measurement every year. Point at the north-south contrast on the map.
3. **Idea.** UrbanCLIP showed text helps describe places, but its GDP gain was tiny and its test leaked between neighbours. The embedding paper showed night-light steering helps at country level. Nobody combined them or tested them at municipal level.
4. **Pipeline.** Walk left to right: Stage A aligns embeddings with captions (no labels), Stage B steers them with night lights (cheap proxy), Stage C learns municipal GDP by adding up cell contributions (real label). Only the image side is needed at prediction time.
5. **Data.** Everything is free and aligned to one 1 km grid. GDP comes from IBGE, not from night lights, so the labels are not circular.
6. **Text.** The model describes what it sees; about 1 in 6 sentences is removed by checks against land cover, roads and building footprints. A blind hand audit will measure the remaining error rate.
7. **Evaluation.** Whole regions are held out, so a test municipality never has a near-identical neighbour in training. This is the main methodological fix over UrbanCLIP.
8. **Results.** The hybrid (our model blended with night lights) is best, R² 0.53; it beats night lights alone (0.27) and simple baselines. Steering did not add a clear gain once the model was calibrated.
9. **Lessons.** Fixing over-dispersed predictions with a validation-fitted line improved error by about 20% and changed the RQ1 conclusion, which is why honest, fixed evaluation matters.
10. **Next.** The text variants (RQ2, RQ3) finish today; then the caption audit, error analysis (where and why the model fails, RQ5), 1 km maps, and the report.

## Format and production (when approved)

- **Deck format:** editable slide deck (PowerPoint .pptx, or a shareable slides artifact). To be confirmed.
- **Figures:** produced by scripts from the real result files and saved to `presentation/midterm/assets/`, so numbers on slides always match the results tables.
- **Numbers to refresh on the day:** slides 8–10 use the latest `main_results.csv`; if V5-V9 have finished, slide 8 gains the text variants and slide 10 marks RQ2 and RQ3 as done.

## Open questions (needed before building)

1. **Time limit:** how many minutes for the talk (and is Q&A separate)? The plan assumes about 10 minutes.
2. **Format:** PowerPoint (.pptx), Google Slides, or a web slide deck? Any institute or course template to follow?
3. **Presenters and date:** your name only, or team members too? Presentation date (sets the "status as of" line and whether V5-V9 results can be included)?
