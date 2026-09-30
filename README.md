# GIS_Project — Text- and Night-Light-Guided Satellite Embeddings

Estimating 2022 municipal GDP per capita in Minas Gerais, Brazil, from Google satellite embeddings (GSED), vision-language captions and VIIRS night lights. It combines UrbanCLIP (Yan et al., 2024) and *Estimating Economic Activity from Satellite Embeddings* (Yue, Zhao and Hu, 2026).

- Proposal: `Term Project Proposal ….pdf`
- Execution plan: [PLAN.md](PLAN.md)

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
