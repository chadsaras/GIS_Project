# GIS_Project — Text- and Night-Light-Guided Satellite Embeddings

Estimating 2022 municipal GDP per capita in Minas Gerais, Brazil, from Google satellite embeddings (GSED), vision-language captions and VIIRS night lights. It combines UrbanCLIP (Yan et al., 2024) and *Estimating Economic Activity from Satellite Embeddings* (Yue, Zhao and Hu, 2026).

- Proposal: `Term Project Proposal ….pdf`
- Execution plan: [PLAN.md](PLAN.md)
- What is done and where to pick up: [progress.txt](progress.txt)
- **Final report:** [reports/final_report.pdf](reports/final_report.pdf) · **Final slides:** [presentation/final/slides.pdf](presentation/final/slides.pdf)

## Results in one table

Out-of-sample over 5 spatial folds (70 IBGE immediate regions), all 853 municipalities, log GDP per capita:

| Model | R² | RMSE |
| --- | --- | --- |
| Night lights only (V1) | 0.27 | 0.51 |
| LightGBM on municipal features (V12) | 0.39 | 0.47 |
| Raw embeddings → Stage C (V3) | 0.46 | 0.44 |
| Text-aligned + steered embeddings → Stage C (V6) | 0.50 | 0.42 |
| Hybrid with night lights (V10, best) | 0.52 | 0.41 |

None of the research-question comparisons (steering, text, POIs and roads, hybrid) is significant under a 95%
block bootstrap; text gives the largest gain. The largest errors are mining and steel towns. Full tables:
[reports/tables/](reports/tables/).

## Layout

| Path | Contents |
| --- | --- |
| `configs/config.yaml` | Every setting (study area, CRS, dataset IDs, model hyperparameters) |
| `src/gisecon/` | Library code: `data/`, `text/`, `models/`, `eval/`, `viz/` |
| `scripts/` | Numbered pipeline steps, run in order |
| `tests/` | pytest checks |
| `reports/` | Figures and tables for the report |
| `notebooks/` | Exploration only |

Data lives outside the repo, in `~/GIS/data` by default (`raw/`, `interim/`, `processed/`, `models/`); override it with `GISECON_DATA_DIR`.

## Setup (server)

```bash
conda env create -f environment.yml
conda activate gisecon
pip install -e .
pytest -q
```

Earth Engine: set `gee.project` in the config, then run `earthengine authenticate` once.

Long jobs: `scripts/run_bg.sh <name> <command…>` runs them with `nohup` and logs to `~/GIS/logs`.

## Reproduce everything

The results are frozen at tag `v1.0-results` (`reports/tables/freeze.txt` records the config and environment
hashes). From a clean checkout on the server:

```bash
scripts/run_all.sh 6        # every step from the raw downloads to reports/, ~a day (Stage C dominates)
python scripts/18_release.py   # data products -> ~/GIS/release
```

`run_all.sh` expects the inputs that need accounts or keys to be in `$GISECON_DATA_DIR` already: the Earth Engine
rasters (`05_export_gee.py`), the Esri caption tiles (`11_download_tiles.py`) and the Kaggle captions
(`17_kaggle_captions_run.py`). IBGE, OSM, building and WorldCover downloads are cached in `raw/` and fetched only
if missing. The report builds with `cd reports && latexmk -pdf final_report.tex`, the slides with
`cd presentation/final && latexmk -xelatex slides.tex` (needs the Lato font).
