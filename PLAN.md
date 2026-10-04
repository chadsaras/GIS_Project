# Execution Plan — Text- and Night-Light-Guided Satellite Embeddings (Minas Gerais, Brazil)

Oct 1, 2026 · @Hemant · Shared version (with timeline diagram): https://claude.ai/code/artifact/b49dfdb9-db04-450f-917a-b0ed3f79657c

## Overview and key decisions

We will estimate 2022 log GDP per capita for the 853 municipalities of Minas Gerais (MG), Brazil, from about 0.5 million 1 km cells. The work runs in 12 phases (0–11) over 12 weeks. Each phase is a numbered list of small steps; every step says what to do, how, and which file it produces.

| Decision | Choice | Why |
| --- | --- | --- |
| Study area | Minas Gerais state, all 853 municipalities | Official municipal GDP from IBGE; a wide income range, from the rich Triângulo and Belo Horizonte metro to the poor Jequitinhonha valley; far above the 150-unit minimum |
| Unit with labels | Município (7-digit IBGE code) | GDP, population and boundaries all come from IBGE with the same codes |
| Main year | 2022 | Population is the 2022 Census count, not an estimate; GSED, VIIRS and IBGE GDP all cover 2022 |
| Grid | 1 km × 1 km raster cells in SIRGAS 2000 / UTM 23S (EPSG:31983) | MG sits mostly in UTM zone 23; scale distortion at the edges stays under about 1% |
| Target | log(GDP ÷ population) per municipality, GDP in current BRL | Official IBGE units; on the log scale a 0.1 error ≈ 10% |
| Evaluation | 5-fold spatial cross-validation over IBGE "Regiões Geográficas Imediatas" (70 in MG) | Official, compact groups of neighbouring municipalities, used as spatial blocks |
| Compute | Institute server gpu77 (48 CPU cores, 125 GB RAM) for all work; Kaggle GPU (2× T4) only for captioning | Stages A–C are small and train on CPU; the server's K40m GPU is too old for current PyTorch |
| Code home | GitHub repo GIS\_Project; data kept outside git | Reproducibility deliverable |

Golden rule for every phase: test municipalities never influence any training stage (A, B or C), any threshold, or any hyperparameter.

## Datasets

All data are free. The first eight rows are required; MapBiomas and the growth-year tables are optional. Catalogue IDs and table numbers are from memory, so Step 0.7 checks each one before any export.

| Dataset | Exact source / ID | Year used | Resolution | Role in the pipeline | How we get it |
| --- | --- | --- | --- | --- | --- |
| Google Satellite Embeddings (GSED) | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, bands A00–A63 | 2022 | 10 m | Main image input (64 numbers per cell) | Earth Engine export, averaged to 1 km |
| VIIRS nighttime lights | Earth Engine `NOAA/VIIRS/DNB/ANNUAL_V22`, band `average_masked` (covers 2012–2025) | 2022 | \~460 m | Stage B proxy label; NTL-only benchmark | Earth Engine export, averaged to 1 km |
| ESA WorldCover | Earth Engine `ESA/WorldCover/v200` | 2021 (closest to 2022) | 10 m | Cell masking, land-cover shares, caption fact-check | Earth Engine export of class fractions per 1 km |
| OpenStreetMap | Geofabrik extract `south-america/brazil/sudeste`, dated snapshot closest to 2023-01-01 (`.osm.pbf`) | End of 2022 | Vector | POI counts, road lengths, waterways (caption fact-check) | Download + parse with pyrosm / osmium |
| High-resolution imagery for captions | Esri World Imagery tiles (\~0.5–1 m); fallback: Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED` 2022 cloud-free true colour (10 m) | 2020–2023 | \~1 m (fallback 10 m) | Input to the vision-language model only | Tile download for sampled cells; licence check at Step 4.1 |
| Municipal boundaries | IBGE Malha Municipal 2022, `MG_Municipios_2022` shapefile (SIRGAS 2000, EPSG:4674) | 2022 | Vector | Assign cells to municipalities; maps | IBGE geoftp download |
| Immediate geographic regions | IBGE "Divisão Regional do Brasil" 2017 (municipality → Região Geográfica Imediata code) | 2017 division | Table | Spatial blocks for cross-validation | IBGE download |
| Municipal GDP | IBGE "PIB dos Municípios", SIDRA table 5938 (GDP and gross value added by sector: agriculture, industry, services, public administration) | 2022 (2018–2021 for growth) | Municipality | Stage C target; sector shares for error analysis | SIDRA API via `sidrapy` |
| Population | IBGE Censo 2022 resident population, SIDRA (table 4709 or current equivalent); IBGE annual estimates (table 6579) for other years | 2022 | Municipality | Convert GDP to per capita | SIDRA API via `sidrapy` |
| MapBiomas Land Use (optional) | MapBiomas Brasil, latest collection, Earth Engine public asset | 2022 | 30 m | Mining and pasture classes for fact sentences and error analysis | Earth Engine export |

We do not use WorldPop, gridded GDP products, or any NTL-derived population as a target: they are partly built from night lights (circular labels, Concept 3.10).

**Models used (all open weights):**

| Purpose | Model | Notes |
| --- | --- | --- |
| Captioning (Step 4.4) | `Qwen/Qwen2.5-VL-7B-Instruct` run on Kaggle (2× T4) | Fallback `Qwen2.5-VL-3B` or Florence-2-large |
| Text embeddings (Stage A) | `sentence-transformers/all-mpnet-base-v2` (768 numbers) | Frozen |
| Image–caption match score (Step 4.8) | RemoteCLIP ViT-B/32 (remote-sensing CLIP) via open\_clip | Only used to drop the worst-matching captions |

## Phase 0 — Setup (week 1)

This phase ends with Earth Engine working, the cluster environment installed, and an empty but runnable repo skeleton.

1. **Confirm the study area with the instructor.** Send one paragraph: Minas Gerais, 853 municipalities, IBGE 2022 GDP and Census 2022 population, spatial CV over immediate regions. *Output:* written approval (email) noted in the README.
2. **Create a Google Cloud project for Earth Engine.** At console.cloud.google.com, create a project (e.g. `gis-econ-mg`), enable the "Google Earth Engine API", then register it at code.earthengine.google.com/register as *Unpaid usage → Academia & Research*. *Output:* project ID written into `configs/config.yaml`.
3. **Authenticate Earth Engine on the cluster.** Run `earthengine authenticate --auth_mode=notebook` once, then `ee.Initialize(project='gis-econ-mg')` in Python. If the cluster has no browser, authenticate locally and copy the credentials file, or use a service account. *Output:* a 3-line test script that prints the size of the GSED collection.
4. **Create the Python environment.** Use conda/mamba with Python 3.11. Core packages: `earthengine-api geemap geopandas rasterio rioxarray xarray pyarrow shapely pyproj pyrosm osmium sidrapy torch sentence-transformers transformers vllm open_clip_torch scikit-learn statsmodels lightgbm matplotlib contextily pyyaml tqdm pytest`. Export it as `environment.yml`. *Output:* `environment.yml` in the repo.
5. **Create the repo skeleton.** Folders: `configs/`, `src/gisecon/{data,text,models,eval,viz}/`, `scripts/` (numbered, e.g. `02_export_gsed.py`), `logs/`, `notebooks/` (exploration only), `reports/{figures,tables}/`, `tests/`. Keep `data/{raw,interim,processed}/` outside git (listed in `.gitignore`) and point to it with `DATA_DIR` in the config. *Output:* first commit with the skeleton.
6. **Write the central config.** One `configs/config.yaml` holds: state code (31 = MG), year (2022), CRS (EPSG:31983), cell size (1000 m), random seeds, paths, and one sub-section per stage. Every script reads only this file, so reruns are reproducible. *Output:* `configs/config.yaml`.
7. **Verify every dataset ID.** For each row of the Datasets table, open its catalogue page or run a tiny query (one image, one SIDRA row) and confirm: ID, band names, that 2022 exists, and the licence. Fix the config wherever something differs. *Output:* `docs/data_sources.md` with the checked IDs and access dates.
8. **Set up job running.** The server has no job scheduler, so long jobs run in tmux or with nohup, logging to logs/. Create a phone-verified Kaggle account (needed for GPU and internet) before Phase 4. *Output:* `s``cripts/run_bg.sh, Kaggle account ready`.

## Phase 1 — Municipal labels and boundaries (week 1)

This phase produces one clean table with one row per MG municipality: code, name, GDP, population, log GDP per capita, sector shares and region code.

1. **Download municipal boundaries.** Get `MG_Municipios_2022` from IBGE, keep columns `CD_MUN` (7-digit code), `NM_MUN`, `AREA_KM2`, and reproject to EPSG:31983. Check: 853 polygons, no invalid geometries (`make_valid`). *Output:* `interim/mg_municipios_2022.gpkg`.
2. **Download 2022 GDP.** Use `sidrapy` to pull SIDRA table 5938 for all MG municipalities (territorial level 6, state 31), year 2022: GDP at current prices and gross value added by activity (agriculture, industry, services, public administration). Values are in thousands of BRL; convert to BRL. IBGE publishes only total GDP for 2022 at municipal level, so the sector split is taken from 2021, the latest year that has it, and used only for flags and the error analysis. *Output:* `raw/ibge/pib_2022.csv`.
3. **Download 2022 population.** Pull Census 2022 resident population per municipality. *Output:* `raw/ibge/pop_2022.csv`.
4. **Download the immediate-region lookup.** Get the IBGE 2017 regional division table and keep municipality code → Região Geográfica Imediata code and name (70 regions in MG) and intermediate region (13). *Output:* `raw/ibge/regioes_2017.csv`.
5. **Join and derive the targets.** Join steps 1–4 on the 7-digit code. Compute `gdp_pc = gdp / pop`, `log_gdp_pc`, `log_gdp`, `log_pop`, and the four sector shares of value added. Assert that all 853 codes match in every table (the script fails loudly otherwise). *Output:* `processed/municipal_labels.parquet`.
6. **Flag special municipalities.** Mark municipalities whose GDP is dominated by one large activity: industry share > 60% (usually mining or hydroelectric plants) or top 1% of GDP per capita. They are kept in training but reported separately in the error analysis, because satellites cannot see a dam's revenue. *Output:* `flag_extreme` column in the same table.
7. **Describe the labels.** Produce a histogram of log GDP per capita, a choropleth map, and a table of min/median/max. This is the first figure of the report and checks for unit mistakes (for example a 1,000× error). *Output:* `reports/figures/labels_overview.png`.

## Phase 2 — 1 km grid and cell feature table (weeks 2–3)

This phase produces `processed/cells.parquet`: one row per 1 km cell (about 0.5 million after masking), with 64 GSED numbers, night lights, land cover, POI counts, road lengths and the municipality code. The grid is a raster, so every layer is exported onto the exact same pixel lattice and no polygon overlay is needed for the image layers.

1. **Define the grid template.** Take the MG bounding box in EPSG:31983 and snap it outward to whole kilometres. Save the affine transform `[1000, 0, xmin, 0, -1000, ymax]` plus width and height, and define `cell_id = row × ncols + col`. Every later export reuses this exact transform. *Output:* `interim/grid.json` and an empty template GeoTIFF.
2. **Assign cells to municipalities.** Rasterize the municipal polygons onto the template: a cell belongs to the municipality containing its centre. Drop cells whose centre lies outside MG. Very small municipalities (MG's smallest is about 4 km²) may get no centre, so give each of those the cell with the largest overlap. Assert that every one of the 853 municipalities has at least one cell. *Output:* `interim/cell_muni.parquet`.
3. **Export GSED at 1 km (Earth Engine).** Filter `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` to 2022 and MG, `mosaic()`, then `reduceResolution(ee.Reducer.mean(), maxPixels=65536)` and `reproject` onto the grid CRS and transform. Export all 64 bands as float32 GeoTIFF to Google Drive, split into 4 tiles if the job times out. Also export the count of valid 10 m pixels per cell. *Output:* `raw/gee/gsed_2022_*.tif` (about 300 MB).
4. **Export night lights at 1 km.** Take VIIRS `average_masked` for 2022 and apply the same `reduceResolution` + `reproject` onto the grid. Later compute `log_ntl = log(1 + radiance)`. *Output:* `raw/gee/viirs_2022.tif`.
5. **Export land-cover fractions.** Turn WorldCover into one 0/1 band per class (tree, shrub, grass, crop, built-up, bare, water, wetland, and so on), then take the mean at 1 km, which gives the share of each class in the cell. Optionally do the same with MapBiomas 2022 (mining, pasture, sugar cane, urban). *Output:* `raw/gee/worldcover_frac.tif` (and `mapbiomas_frac.tif`).
6. **Download and check alignment.** Copy the exports from Drive to the cluster (`rclone`). Assert that every raster has the same width, height, transform and CRS as the template. *Output:* rasters in `raw/gee/` and a passing check.
7. **Prepare OpenStreetMap.** Download the dated Geofabrik `sudeste` extract, cut it to the MG boundary with `osmium extract -p`, and load it with pyrosm. *Output:* `raw/osm/mg_2023-01-01.osm.pbf`.
8. **Count POIs per cell.** Map OSM tags to 10 categories: education (school, university, college), health (hospital, clinic, doctors, pharmacy), finance (bank, ATM), retail (`shop=*`), food (restaurant, café, fast food, bar), industry (`landuse=industrial`, `man_made=works`), office (`office=*`), tourism, fuel, and public services (town hall, police, courthouse). Use points plus polygon centroids, project them to EPSG:31983, compute the cell index by integer division, and count. *Output:* `interim/poi_counts.parquet`.
9. **Measure road and waterway length per cell.** Group `highway` values into major (motorway, trunk, primary and their links), medium (secondary, tertiary), minor (residential, unclassified, service, living\_street) and track. Densify each line to one point every 10 m, assign the points to cells, and multiply the count by 10 m. This gives km per cell without heavy geometry overlays. Do the same for waterways (river, stream, canal), which the caption fact-check needs. *Output:* `interim/road_water_km.parquet`.
10. **Mask cells with no economic relevance.** Drop a cell if water ≥ 90%, or if GSED covers less than 50% of it. Also drop it if it is ≥ 95% tree cover or bare rock, **and** it has zero built-up, zero lights, no POI and no roads. Log how many cells each rule drops. Masked cells are simply left out of the district sums. *Output:* `keep` flag.
11. **Assemble the cell table.** Join everything on `cell_id`. Columns: ids, municipality code, x/y centre, `gsed_00…gsed_63`, `gsed_norm` (length of the mean vector, a measure of how mixed the cell is), `ntl`, `log_ntl`, `lc_*` shares, `poi_*` counts, `road_*_km`, `water_km`, `keep`. *Output:* `processed/cells.parquet`.
12. **Quality checks and figures.** Map `log_ntl`, the built-up share, and an RGB of the first three principal components of GSED, and check by eye that cities and borders line up. Also check the OSM completeness indicator per municipality (road km against built-up share), which is needed later for RQ3. *Output:* `reports/figures/cells_qc_*.png`.

## Phase 3 — Spatial blocks and 5 folds (week 3)

This phase fixes, once and for all, which municipalities train, validate and test in each of the 5 rounds. Every later stage reads this one file.

1. **Use immediate regions as blocks.** Each of the 70 Regiões Geográficas Imediatas in MG is one block, so all its municipalities always move together. This replaces the proposal's generic "about 25 blocks" with official, compact, contiguous groups. *Output:* `block_id` column.
2. **Describe each block.** For each block record: number of municipalities, number of kept cells, mean and spread of log GDP per capita, and the share of "urban" municipalities (top third by built-up share). *Output:* `interim/blocks.parquet`.
3. **Assign blocks to 5 balanced folds.** Try 10,000 random assignments with a fixed seed. Keep the one whose folds are most alike in municipality count, mean and standard deviation of log GDP per capita, and urban share (smallest sum of standardized differences). *Output:* `processed/folds.parquet` (municipality code → fold 0–4).
4. **Define the rotation.** In round *r*, fold *r* is the test fold, fold (*r*+1) mod 5 is validation, and the other three train. Every municipality is tested exactly once. *Output:* `get_split(round)` in `src/gisecon/eval/folds.py`.
5. **Unit-test the split.** A pytest asserts that the train, validation and test sets never overlap, that their union is all 853 municipalities, and that each municipality appears in test once across the 5 rounds. *Output:* `tests/test_folds.py` passing.
6. **Map the folds.** Draw a map of MG coloured by fold, with block borders, for the report's methods section. *Output:* `reports/figures/folds_map.png`.

## Phase 4 — Text: tiles, captions, fact sentences, cleaning, audit (weeks 3–5)

This phase produces `processed/texts.parquet` with three text versions for about 10,000 cells: raw caption, cleaned caption + fact sentences, and fact sentences only. It also produces the 300-caption audit.

1. **Decide the imagery source (gate).** Read the Esri World Imagery terms for academic, non-commercial model use. If they allow it, use Esri; if not, use a 2022 Sentinel-2 true-colour composite (clearly lower quality, so noted as a limitation). Planet NICFI is a third option if access exists. Download 50 test tiles from the chosen source and view them. *Output:* the decision and its reason in `docs/data_sources.md`.
2. **Sample 10,000 cells.** Sample from kept cells in **all** folds: 2,000 per fold, stratified by built-up tercile and night-light decile, with urban cells oversampled so that rural pasture does not dominate. In each round only the captions in training municipalities are used (about 6,000), so no test cell's text is ever seen. This corrects the proposal's "sample only from training districts", which does not work when the folds rotate. Store the sampling weights. *Output:* `interim/caption_sample.parquet`.
3. **Download one tile image per sampled cell.** Convert the cell's bounds to Web Mercator, fetch zoom-17 tiles (about 1.1 m per pixel at MG's latitude), stitch them, crop to the exact 1 km cell, and save a 768 × 768 PNG. Cache every file, retry failures, and keep to the provider's rate limits. *Output:* `raw/tiles/{cell_id}.png` (about 10,000 files, about 5 GB).
4. **Pilot the prompt on 200 tiles.** Compare 2–3 prompt versions with Qwen2.5-VL-7B. The base prompt: *"Describe this 1 km × 1 km satellite image in 4–6 sentences. Cover buildings (density, size, roof types), roads, green areas, water, farmland, industry or mining, and signs of human activity. Only describe what is clearly visible; do not guess place names."* Use temperature 0.2 and at most 200 tokens. Pick the prompt with the fewest false claims in a quick look at 30 outputs, and record the time per image. *Output:* `configs/prompts.yaml` and pilot notes.
5. **Caption all 10,000 tiles.** Upload the tiles to Kaggle as a private dataset. In a Kaggle notebook, run Qwen2.5-VL-7B on 2× T4 (fp16, split over both GPUs) and write results to JSONL every 500 images, so a session timeout (12-hour limit) loses little. Store the model name, prompt id and seed with each caption. Expected time: roughly 6–10 GPU-hours, within Kaggle's 30-hour weekly quota; fall back to Qwen2.5-VL-3B if 7B is too slow. *Output:* `interim/captions_raw.jsonl`.
6. **Write fact sentences from data.** Turn each cell's numbers into 2–4 template sentences that use both words and numbers, for example: *"Mostly built-up area: 65% built-up, 20% grassland, 10% trees. 12 schools, 4 health facilities and 8 shops are mapped. 8.2 km of major roads and 15 km of minor roads. No mapped rivers."* Night lights are deliberately left out, so Stage A never sees the Stage B label and the effects of text and of night-light steering stay separable. *Output:* `interim/fact_sentences.parquet`.
7. **Cleaning pass 1 — remove filler.** Split captions into sentences. Drop sentences that match a configurable list of vague patterns ("comprehensive view", "overall scene", "gives a sense of", "the image shows a satellite view") and pure boilerplate. *Output:* per-sentence `drop_reason` column.
8. **Cleaning pass 2 — check claims against data.** A small keyword lexicon maps words to checks. Water words (river, lake, pond, reservoir) need water ≥ 1% or waterway length > 0. Forest words need tree cover ≥ 10%. Farm words (crop, field, plantation, pasture) need crop + grass ≥ 10%. Building words need built-up ≥ 2% or at least one POI. Industry or mining words need an industry POI, built-up ≥ 5%, or the MapBiomas mining class. Delete sentences that fail, and log counts per rule. *Output:* cleaned captions.
9. **Cleaning pass 3 — image–text match.** Score each tile against its cleaned caption with RemoteCLIP and drop the worst 5% of captions. Fix the cut-off on the pilot set, never on test data. Keep this step light: UrbanCLIP found fully automatic filtering unstable. *Output:* `clip_score` and `keep_caption` columns.
10. **Audit 300 captions by hand.** Draw 300 captions at random. For each, show the tile beside its raw caption, one numbered sentence per line, without showing which sentences the cleaning dropped; raw and cleaned results are both computed from these labels. Mark every sentence as *correct*, *false claim* or *vague*. Report the share of captions with at least one false claim before and after cleaning, with 95% Wilson intervals. If a second person can label 50 of them, report their agreement (Cohen's kappa). *Output:* `reports/tables/caption_audit.csv` and a summary.
11. **Assemble the final text table.** For each sampled cell store `text_raw` (raw caption), `text_full` (cleaned caption + fact sentences) and `text_facts` (fact sentences only). These three feed the text ablations. *Output:* `processed/texts.parquet`.

## Phase 5 — Stage A: image–text alignment (weeks 5–6)

This phase trains, for each of the 5 rounds, a small projection that moves each cell's GSED vector close to its own description. The result is a 128-number embedding for every kept cell, captioned or not.

One limit to state in the report: GSED is frozen, so Stage A can only reshape the 64 numbers GSED already contains; it cannot add new information. Any gain from text comes from a better-organised space that helps Stage C learn from few municipalities.

1. **Encode the texts once.** Run `all-mpnet-base-v2` (frozen) on `text_full`, `text_raw` and `text_facts` for all 10,000 sampled cells and cache the 768-number vectors. *Output:* `interim/text_emb_{variant}.npy`.
2. **Standardize GSED per round.** For round *r*, compute the mean and standard deviation of each GSED number on **training cells only**, and apply them to all cells. Save the scaler with the round. *Output:* `models/round{r}/gsed_scaler.json`.
3. **Build the model.** Image side: 64 → 256 → 128 (Linear, GELU, dropout 0.1, Linear). Text side: 768 → 256 → 128. Both outputs are L2-normalised. A learnable temperature starts at 0.07. Loss: symmetric InfoNCE (image→text + text→image cross-entropy over the batch), as in CLIP and UrbanCLIP. *Output:* `src/gisecon/models/stage_a.py`.
4. **Choose the training and validation pairs.** Train on captioned cells in the training municipalities of round *r* (about 6,000 pairs). Validate on captioned cells of the validation fold (about 2,000). Test-fold captions are never loaded. *Output:* a pair-index function.
5. **Train.** AdamW with learning rate 1e-3, weight decay 1e-4, batch 256, up to 200 epochs, and early stopping when validation loss has not improved for 20 epochs. A few minutes per round on CPU. *Output:* `models/round{r}/stage_a_{variant}.pt` and a loss log.
6. **Check that alignment happened.** On validation pairs, report image→text recall@1/5/10. Chance level for recall@10 at batch 256 is about 4%. Also fit a quick linear probe from the 128-number embedding to built-up share, to confirm the space still carries basic land information. *Output:* `reports/tables/stage_a_diagnostics.csv`.
7. **Embed every cell.** Pass every kept cell's standardized GSED through the trained image projector. No text is needed at this point. *Output:* `interim/round{r}/emb_A_{variant}.npy` (about 0.5 M × 128, float16).
8. **Train the three text variants.** Repeat steps 5–7 with `text_full` (main model), `text_raw` (the "uncleaned captions" ablation) and `text_facts` (the "no VLM" ablation): 3 variants × 5 rounds = 15 short runs. *Output:* all Stage A embeddings.

Optional if time allows: add UrbanCLIP's other idea cheaply, a small text decoder or "caption reconstruction" loss. UrbanCLIP's ablation found the language-modelling loss alone beat the contrastive loss alone. It is listed as out of scope in the proposal, so we only add it after the main results are in.

## Phase 6 — Stage B: night-light steering (week 6)

This phase trains, per round, a network that predicts each cell's night-light brightness from its embedding. Its 32-number middle layer becomes the economy-aware embedding.

1. **Prepare the target.** Use `log_ntl` per kept cell, standardized with training-cell statistics of the round. Report the share of cells with zero light (dark rural cells), since it shapes the loss. *Output:* target arrays per round.
2. **Build the NTL mapper.** Input *d* (128 for Stage A embeddings, 64 for raw GSED) → 128 → 64 → 32 → 1, with ReLU after each hidden layer. This follows the embedding paper's shape (three hidden layers, then one output); its exact widths are not published, so these are our choice. The 32-number output after the third ReLU is **z**. *Output:* `src/gisecon/models/stage_b.py`.
3. **Train on training cells only.** Use all kept cells in the round's training municipalities (about 300,000), and validation-fold cells for early stopping. Adam with learning rate 1e-3, batch 1024, up to 30 epochs, patience 5. The paper used batch 32 for 10 epochs on far fewer cells; with more cells, larger batches are faster and stable. *Output:* `models/round{r}/stage_b_{input}.pt`.
4. **Check accuracy.** Report validation R² and RMSE of log NTL. *Output:* a row in `reports/tables/stage_b_diagnostics.csv`.
5. **Repeat the paper's "neighbour" check.** For 5 reference cells (a rural pasture cell, a small town, an industrial area, a mining area, a Belo Horizonte suburb), find the 100 nearest cells by cosine similarity, first in raw GSED and then in **z**. Compare the mean geographic distance of the neighbours and the spread of their log NTL. We expect **z** neighbours to be farther away but more similar economically, as in the paper's Fig. 2. *Output:* `reports/figures/neighbours_raw_vs_z.png`.
6. **Extract outputs for all cells.** Save **z** (32 numbers) and the predicted log NTL for every kept cell. The predicted NTL becomes the 1 km activity map in Phase 9. *Output:* `interim/round{r}/z_{input}.npy`, `ntl_pred_{input}.npy`.
7. **Train all input variants.** Inputs: raw GSED (RQ1, reproduces the paper), Stage A `full` (main model), Stage A `raw` and Stage A `facts` (ablations). That is 4 inputs × 5 rounds. *Output:* all Stage B embeddings.

## Phase 7 — Stage C: municipal GDP from cells (weeks 7–8)

This phase learns cell-level patterns from municipal totals by sum-pooling, and produces a test prediction for every municipality in every variant. A ridge regression gives the simple check.

1. **Prepare cell inputs.** The cell input is the chosen embedding (**z** with 32 numbers, the Stage A 128, or raw GSED 64). For the "+POI/roads" variant, append `log(1 + count)` for the 10 POI categories and road km for the 4 road classes. Standardize with training-cell statistics. *Output:* input arrays per round and variant.
2. **Build the model.** Per-cell network *p*: input → 32 → 16 (Linear, ReLU, dropout, Linear, softplus, so each cell adds a non-negative amount). Pool: sum the 16-number outputs over the municipality's cells. Municipal network *h*: 16 → 16 → 1, giving predicted log total GDP. Subtract log population to get predicted log GDP per capita. *Output:* `src/gisecon/models/stage_c.py`.
3. **Handle the size problem.** MG municipalities range from 1 to about 10,000 cells, so raw sums differ by 10,000×. Implement two pooling options: plain sum (as in the paper) and `log(1 + sum)`. The validation fold picks one per round. *Output:* `pooling` option in the config.
4. **Train full-batch.** All kept cells fit in memory at once. Use `index_add_` with each cell's municipality index to sum in one step, so one epoch is one forward pass over about 500 training municipalities. Loss: MSE on log GDP per capita. Adam with learning rate 1e-3, reduced to 1e-4 on a plateau, and weight decay. Early stopping on validation-fold MSE, using a 100-epoch rolling mean as in the paper, with a limit of 3,000 epochs. *Output:* `models/round{r}/stage_c_{variant}.pt`.
5. **Tune a small grid on validation only.** Hidden size {16, 32}, weight decay {1e-4, 1e-3}, dropout {0.1, 0.3}, pooling {sum, log-sum}: 16 settings. The best by validation MSE is used on test once. *Output:* `reports/tables/stage_c_tuning.csv`.
6. **Train a 5-seed ensemble.** Retrain the chosen setting with 5 random seeds. The ensemble mean is the prediction; the min–max across seeds is the uncertainty range (Phase 9). *Output:* 5 models per round and variant.
7. **Fit the ridge baseline.** Municipal features: mean of the cell embeddings, log number of kept cells, and log population. Fit `Ridge` with alpha chosen on the validation fold. Population is an allowed input: it is an official Census count, not GDP. *Output:* ridge predictions.
8. **Optional extra baseline: LightGBM.** Use the same municipal features with default settings plus early stopping on validation. It answers "does the deep pipeline beat a strong off-the-shelf model?" *Output:* LightGBM predictions.
9. **Save predictions in one format.** One table: `variant, round, muni_code, y_true, y_pred, ens_min, ens_max`, holding test-fold rows only for final scoring and validation rows separately for the hybrid weight. *Output:* `processed/predictions.parquet`.

## Phase 8 — Benchmark, hybrid, baselines and ablations (weeks 8–9)

This phase runs every variant on the same 5 rounds and produces the main results table with confidence intervals for each research question.

1. **NTL-only benchmark.** Per municipality, compute `log(total NTL ÷ population)` over all its cells. Fit linear and quadratic OLS of log GDP per capita on it, using training municipalities. Keep the form with lower validation MSE, then predict test. This follows Hu and Yao (2022) and the embedding paper. *Output:* V1 predictions.
2. **Land-cover-only baseline.** Municipal mean land-cover shares + log cells + log population → ridge. *Output:* V2 predictions.
3. **Hybrid blend.** For each round, compute `α × model + (1 − α) × NTL-only` for α ∈ {0, 0.25, 0.5, 0.75, 1}. Pick α on validation-fold predictions, then apply it once to test. Report the chosen α per round. *Output:* V10 predictions.
4. **Fix the variant list.** Each row is one config file:

| ID | Pipeline | Answers |
| --- | --- | --- |
| V1 | NTL only (quadratic or linear) | Benchmark to beat |
| V2 | Land-cover shares → ridge | Is a simple model enough? |
| V3 | Raw GSED → C | Base value of embeddings |
| V4 | Raw GSED → B → C | RQ1: does steering help? (reproduces the paper) |
| V5 | A(full) → C | Text without steering |
| V6 | A(full) → B → C | RQ2: text + steering |
| V7 | A(full) → B → C + POI/roads | RQ3: full model |
| V8 | V7 with uncleaned captions | Value of cleaning |
| V9 | V7 with fact sentences only | Is the VLM needed at all? |
| V10 | Best of V3–V9 blended with V1 | RQ4: hybrid |
| V11 | Ridge on mean embeddings (GSED, z, A) | Simple check of each embedding |
| V12 | LightGBM on municipal features (optional) | Strong off-the-shelf baseline |

5. **Note on RQ3.** Fact sentences already mention POIs and roads, so V6 carries some of that information through text. V7 therefore measures what POIs and roads add *as direct numbers on top of text*. If time allows, add V6b: Stage A trained on fact sentences without POI/road sentences.
6. **Write one experiment runner.** `scripts/run_experiment.py --variant V7 --round 2` reads the config, loads cached stage outputs when they exist, trains what is missing, and appends to `predictions.parquet`. Run the 12 variants × 5 rounds in parallel on the server's 48 cores. *Output:* the runner and a completed run log.
7. **Compute metrics.** For each variant: R², RMSE and MAE pooled over all 853 test predictions, plus mean ± standard deviation across the 5 rounds. Add the validation regression of official on predicted (slope near 1, intercept near 0 means no systematic bias). *Output:* `reports/tables/main_results.csv`.
8. **Test the differences.** For the four key comparisons (V4 vs V3 for RQ1, V6 vs V4 for RQ2, V7 vs V6 for RQ3, V10 vs best single model for RQ4), compute the RMSE difference with a 95% interval. Use a block bootstrap: resample the 70 immediate regions with replacement 2,000 times. A difference whose interval includes 0 is reported as "no clear effect". *Output:* `reports/tables/rq_tests.csv`.

## Phase 9 — Error analysis, uncertainty, maps and text explanations (weeks 9–10)

This phase explains where the best model works and fails (RQ5) and produces the maps and the text explanations. Every number here uses test-fold predictions only.

1. **Predicted vs official scatter.** Plot the best model and NTL-only side by side against official log GDP per capita, with the 45° line. Colour the flagged mining and hydro municipalities. *Output:* `reports/figures/pred_vs_official.png`.
2. **Regress absolute error on municipality traits.** For the best model and for NTL-only: |error| \~ official log GDP per capita + built-up share + services share + industry share + extreme flag + OSM completeness. Use standard errors clustered by immediate region. This mirrors the embedding paper's Table 5. IBGE publishes no per-municipality data-quality score, so the report says so plainly and analyses income and economic structure instead. *Output:* `reports/tables/error_regression.csv`.
3. **LOWESS plot.** Plot |error| against official log GDP per capita, with a smooth curve for each model. We expect errors to rise with income, and to rise less for the embedding model than for NTL. *Output:* `reports/figures/error_vs_income.png`.
4. **Check spatial pattern in the errors.** Compute Moran's I of test residuals over municipal neighbours (PySAL `esda`). A clearly positive value means errors cluster in space, pointing to a missing regional factor. *Output:* one number with its p-value.
5. **Inspect the 10 largest gaps.** For each: official vs predicted, sector shares, flag, three of its tiles and their captions. Write one line suggesting a reason (mining, dam, agribusiness, commuter town). As in the embedding paper, these are "candidates", not proof. *Output:* `reports/tables/top10_gaps.md`.
6. **Uncertainty check.** For each municipality, compare the 5-seed min–max range with its absolute error (Spearman correlation plus a binned plot). Do wider ranges go with larger errors? *Output:* `reports/figures/uncertainty_vs_error.png`.
7. **Municipal maps.** Three maps with the same colour scale: official, predicted, and gap (predicted − official, diverging colours). *Output:* `reports/figures/map_{official,predicted,gap}.png`.
8. **1 km activity map.** Every cell's municipality is in the test fold in exactly one round. Take the Stage B predicted log NTL from that round, so the whole map is out-of-sample. Show it as a relative activity index, for display only. *Output:* `reports/figures/activity_1km.png` and a GeoTIFF.
9. **Text explanations.** Pick 3–5 municipalities: Belo Horizonte metro, a Triângulo agribusiness town, a mining town, a Jequitinhonha town, and one large miss. For each, list the 5 training captions nearest (cosine, Stage A space) to its mean cell embedding. *Output:* `reports/tables/explanations.md`.
10. **Words that separate rich from poor.** Compare caption words in the top-20% and bottom-20% predicted municipalities with the log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008). List the 15 strongest words on each side, for example "industrial" or "irrigated" against "scattered houses". *Output:* `reports/figures/word_contrast.png`.

## Phase 10 — Optional: growth extension (week 11, only if Gate 4 is on time)

This phase tests whether predicted changes match official changes. The embedding paper expects NTL to win here.

1. **Add years 2018–2021.** Export GSED and VIIRS for each year; VIIRS V22 covers 2012–2025, so one version serves every year. Download IBGE GDP per year and IBGE population estimates for the non-Census years. *Output:* yearly cell tables.
2. **Reuse the trained models.** Apply each round's frozen Stage A and B to every year, and predict with Stage C. No retraining on other years. *Output:* a municipality × year prediction panel.
3. **Fixed-effects regression.** Regress official log GDP per capita on predicted, with municipality and year fixed effects, separately for the embedding model and NTL (linear and quadratic). Compare MSE, as in the paper's Table 4. *Output:* `reports/tables/growth_results.csv`.
4. **Mind 2020.** The COVID year is a shock that satellites may miss. Report results with and without 2020. *Output:* a robustness row.

## Phase 11 — Report and deliverables (weeks 11–12)

This phase turns the results into the six deliverables promised in the proposal (§12.2).

1. **Freeze results.** Tag the repo (`v1.0-results`), and save the config and environment hashes next to `main_results.csv`. After this, no numbers change without a new tag.
2. **Full rerun test.** On a clean checkout, run `make all` (or a short driver script) from the raw downloads through to the tables, and confirm the numbers match. *Output:* a README section "Reproduce everything".
3. **Publish the data products.** Include the cell feature table (without licence-restricted tiles), the cleaned caption set, the 300-caption audit, predictions and the 1 km map. *Output:* a `release/` folder or a Zenodo draft.
4. **Make the figures.** Pipeline diagram, folds map, predicted vs official, error vs income, gap map, 1 km activity map, neighbour check, word contrast. *Output:* `reports/figures/` in final form.
5. **Write the report (12–15 pages).** Sections: introduction, data (MG choice), methods (Stages A–C, spatial CV), results (main table + RQ tests), error analysis, explanations, limitations, conclusion. The limitations must cover the frozen-GSED ceiling on Stage A, the dropped language-modelling loss, the missing data-quality indicator, the OSM rural gap, and the 2021 WorldCover year. *Output:* `reports/final_report.pdf`.
6. **Prepare the presentation.** 12–15 slides: question, pipeline, data, one results table, three figures, what failed, and what's next. *Output:* the slide deck.

## Timeline and gates

| Week (from Mon 5 Oct 2026) | Phases | Gate at end of week |
| --- | --- | --- |
| W1 | P0 Setup, P1 Labels and boundaries | G1: study area approved, Earth Engine working, label table complete |
| W2 | P2 Grid and cell features | |
| W3 | P2 (end), P3 Spatial folds, P4 starts | G2: cells.parquet and the 5 folds frozen |
| W4 | P4 Captions and cleaning | |
| W5 | P4 (end), P5 Stage A starts | G3: captions cleaned, 300-caption audit done |
| W6 | P5 Stage A, P6 Stage B | |
| W7 | P7 Stage C | |
| W8 | P7 (end), P8 Baselines and ablations | **G4: Stages A–C run on all 5 rounds; if late, drop growth, then the POI/roads ablation** |
| W9 | P8, P9 Error analysis and maps | |
| W10 | P9 | |
| W11 | P10 Growth (optional), P11 Report | |
| W12 | P11 Report and slides | G5: report and presentation delivered |

Phases overlap wherever their inputs are ready early. If the term starts on a different date, shift every bar by the same amount. Each gate is a go/no-go check: a gate that is not met is fixed before the next phase begins.

## Risks and points to confirm before execution

| Risk | Early sign | Fallback |
| --- | --- | --- |
| Earth Engine export times out or quota is hit | Step 2.3 task fails | Split MG into 4–8 tiles; export band groups separately |
| Esri licence does not allow model use | Step 4.1 | Sentinel-2 composite (lower quality) or fact sentences only (V9 becomes the main text) |
| Mining and hydro municipalities dominate the errors | Phase 1 flag counts, Phase 9 top-10 | Report metrics with and without flagged municipalities; never drop them silently |
| Stage C overfits (about 500 training municipalities) | Validation loss rises early | Log-sum pooling, stronger weight decay, rely on ridge and LightGBM rows |
| Captions are generic for rural pasture cells | Pilot in Step 4.4 | Oversample urban cells; stronger prompt; a weak RQ2 result is still valid |
| OSM is sparse in rural MG | Completeness check in Step 2.12 | Interpret RQ3 per urban/rural group |
| V21 and V22 night lights disagree (growth only) | Step 10.1 | Drop the growth extension; it is optional |

**Changes from the proposal, for your approval:**

- [ ] Study area fixed to Minas Gerais (853 municipalities), main year 2022, Census 2022 population.
- [ ] Spatial blocks = 70 IBGE immediate regions (instead of about 25 generic blocks).
- [ ] Captions sampled from all folds and filtered per round (the proposal's "training districts only" does not work with rotating folds).
- [ ] Night lights are left out of fact sentences, to keep the text and steering effects separable.
- [ ] Stage C gets a log-sum pooling option, chosen on validation, because municipality sizes differ by 10,000×.
- [ ] Added: block-bootstrap intervals for each RQ, Moran's I of residuals, LightGBM baseline (optional), sector shares in the error analysis.
- [ ] Layer widths for Stages B and C are our choice; the report will say the embedding paper did not publish them.

**Changes made during implementation (accepted 2026-10-02; details in progress.txt):**

| # | Change | Effect on results |
| --- | --- | --- |
| 1 | Fold balance uses population density for "urban", not built-up share | None; folds are frozen and balanced |
| 2 | Earth Engine layers downloaded directly (computePixels tiles), not exported via Drive | None |
| 3 | Tiles over Earth Engine's memory limit are split into quarters; rate-limit errors wait instead of failing | None |
| 4 | Masking drops a cell when tree + bare ≥ 95% (and nothing built, lit or mapped), not tree or bare alone | A few more empty cells dropped |
| 5 | A municipality whose cells are all masked keeps its cells (logged) | Avoids empty sums |
| 6 | NTL-only benchmark uses log(1 + total NTL) − log(pop), so dark towns stay finite | Negligible |
| 7 | Stage C POI and road inputs both enter as log(1 + x) | None expected |
| 8 | Predictions saved one file per variant and round | None |
| 9 | Calibration line reported on validation and test | None |
| 10 | Caption claim check skips sentences with a negation; MapBiomas mining check not used | Measured by the audit |
| 11 | Stage C learning rate halves after 50 epochs without improvement; output bias starts at the training mean | Training converges |
| 12 | Caption tiles crop the cell's Web Mercator bounding box (thin margin of neighbouring land) | Negligible |
| 13 | CLIP cut-off fixed on a seeded 200-caption subset | None |
| 14 | Caption audit labels each raw sentence once, blind to cleaning; raw and cleaned results derived from the same labels | Half the labelling, unbiased |
| 15 | 2022 sector shares unavailable, so 2021 shares are used for flags and the error analysis | None on the target |
