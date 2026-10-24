"""Phase 11.3: gather the data products into one release folder (LIGHT: a minute). Data stay outside git.

  python scripts/18_release.py [--out ~/GIS/release]
  -> municipal_labels.parquet, folds.parquet, cells.parquet (cell feature table), texts.parquet (raw and cleaned
     captions + fact sentences), caption_sentences.parquet (every sentence with its drop reason),
     caption_audit_labels.csv, predictions.parquet (all variants, rounds, val + test rows), activity_1km.tif,
     the report tables, and a README with sources and licences. Esri tiles are not included (licence).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

from gisecon.config import REPO_ROOT, data_path, load_config

README = """# GIS_Project release: municipal GDP from satellite embeddings, text and night lights (Minas Gerais, 2022)

Code and method: https://github.com/chadsaras/GIS_Project (tag v1.0-results), report: reports/final_report.pdf.

| File | Contents |
| --- | --- |
| municipal_labels.parquet | 853 municipalities: IBGE GDP 2022, Census 2022 population, log GDP per capita, 2021 sector shares, flags |
| folds.parquet | municipality -> immediate region (block) -> fold 0-4 |
| cells.parquet | 1 km cells (EPSG:31983): GSED 64-d mean, log night light, WorldCover shares, OSM POI counts and road km, keep flag |
| texts.parquet | 10,000 captioned cells: text_raw (Qwen2.5-VL-7B caption), text_full (cleaned caption + facts), text_facts |
| caption_sentences.parquet | every caption sentence with its cleaning drop reason |
| caption_audit_labels.csv | 1,801 audit sentences of 300 captions, labelled correct / false / vague by an LLM (Claude); label2 empty |
| predictions.parquet | variant, round, split (val/test), muni_code, y_true, y_pred, ensemble min/max |
| activity_1km.tif | out-of-sample Stage B predicted log(1 + night light), a relative activity index |
| tables/ | the report tables (main results, RQ tests, diagnostics, error analysis, audit) |

Sources and licences: Google Satellite Embedding V1 (CC-BY 4.0), VIIRS annual night lights V2.2 (Earth Observation
Group, Payne Institute, CC-BY 4.0), ESA WorldCover 2021
(CC-BY 4.0), OpenStreetMap (ODbL 1.0, the POI and road columns are derived data), Microsoft Global ML Building
Footprints (ODbL), IBGE statistics and boundaries (public). Captions were generated from Esri World Imagery tiles
(Esri, Vantor, Earthstar Geographics, and the GIS User Community); the tiles themselves are not redistributed.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/GIS/release")
    args = ap.parse_args()
    cfg = load_config()
    out = Path(args.out).expanduser()
    (out / "tables").mkdir(parents=True, exist_ok=True)

    copies = {data_path(cfg, "processed", "municipal_labels.parquet"): "municipal_labels.parquet",
              data_path(cfg, "processed", "folds.parquet"): "folds.parquet",
              data_path(cfg, "processed", "cells.parquet"): "cells.parquet",
              data_path(cfg, "processed", "texts.parquet"): "texts.parquet",
              data_path(cfg, "interim", "caption_sentences.parquet"): "caption_sentences.parquet",
              data_path(cfg, "processed", "activity_1km.tif"): "activity_1km.tif",
              REPO_ROOT / "reports" / "tables" / "caption_audit_labels_claude.csv": "caption_audit_labels.csv"}
    for src, name in copies.items():
        shutil.copy2(src, out / name)
    preds = pd.concat([pd.read_parquet(f) for f in sorted(data_path(cfg, "processed", "predictions").glob("*.parquet"))],
                      ignore_index=True)
    preds.to_parquet(out / "predictions.parquet", index=False)
    for f in (REPO_ROOT / "reports" / "tables").glob("*.csv"):
        shutil.copy2(f, out / "tables" / f.name)
    (out / "README.md").write_text(README)

    sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}"
            for f in sorted(out.rglob("*")) if f.is_file() and f.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {len(sums)} files ({size:.0f} MB) to {out}")


if __name__ == "__main__":
    main()
