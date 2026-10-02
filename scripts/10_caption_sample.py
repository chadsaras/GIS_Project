"""Phase 4.2: choose the cells that get a caption (2,000 per fold, stratified; LIGHT: seconds).

Run after 06_build_cells.py. Output: interim/caption_sample.parquet (cell_id, muni_code, fold, stratum, weight).
"""
from __future__ import annotations

import pandas as pd

from gisecon.config import data_path, load_config
from gisecon.text.sample import caption_sample


def main() -> None:
    cfg = load_config()
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"),
                            columns=["cell_id", "muni_code", "keep", "lc_built", "ntl"])
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet")).set_index("muni_code")["fold"]
    s = caption_sample(cells[cells["keep"]].reset_index(drop=True), folds, cfg["text"]["per_fold"],
                       cfg["project"]["seed"])
    s.to_parquet(data_path(cfg, "interim", "caption_sample.parquet"), index=False)
    print(f"{len(s):,} cells sampled; per fold: {s.groupby('fold').size().to_dict()}; "
          f"weights {s.weight.min():.1f}-{s.weight.max():.1f}")


if __name__ == "__main__":
    main()
