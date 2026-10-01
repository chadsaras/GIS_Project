"""Phase 2, steps 2.3-2.6: GSED, VIIRS and WorldCover averaged onto the 1 km grid.

Usage:
  python scripts/05_export_gee.py --check              # print band names and one test tile per layer
  python scripts/05_export_gee.py [--layers gsed viirs worldcover] [--year 2022] [--tile 32] [--workers 8]
Needs gee.project in configs/config.yaml and a one-time `earthengine authenticate`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gisecon.config import data_path, ensure_dirs, load_config
from gisecon.data import gee
from gisecon.data.grid import Grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["gsed", "viirs", "worldcover"])
    ap.add_argument("--year", type=int)
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    g = cfg["gee"]
    assert g["project"], "set gee.project in configs/config.yaml (Step 0.2)"
    gee.init(g["project"])
    year = args.year or cfg["study"]["year"]
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))

    layers = {
        "gsed": (lambda: gee.gsed_image(g["gsed"], year, grid.crs),
                 [f"gsed_{i:02d}" for i in range(64)] + ["gsed_coverage"], f"gsed_{year}"),
        "viirs": (lambda: gee.viirs_image(g["viirs"], g["viirs_band"], year), ["ntl"], f"viirs_{year}"),
        "worldcover": (lambda: gee.worldcover_image(g["worldcover"]),
                       [f"lc_{n}" for n in gee.WORLDCOVER_CLASSES.values()], "worldcover_2021"),
    }

    cells = pd.read_parquet(data_path(cfg, "interim", "cell_muni.parquet"), columns=["row", "col"])
    tiles = gee.tiles_needed(grid, cells.row.to_numpy(), cells.col.to_numpy(), args.tile)
    print(f"{len(tiles)} tiles of {args.tile} x {args.tile} cells cover the state", flush=True)

    for name in args.layers:
        make, bands, stem = layers[name]
        img = make()
        if args.check:
            print(name, img.bandNames().getInfo()[:5], "...")
            # a 4 x 4 tile over central Belo Horizonte
            cid = int(grid.cell_index(np.array([610000.0]), np.array([7797000.0]))[0])
            r0, c0 = divmod(cid, grid.width)
            a = gee.fetch_tile(img, grid, r0, c0, 4, 4)
            print(f"  test tile shape {a.shape}, band means {np.nanmean(a, axis=(0, 1))[:4]}")
            continue
        out = data_path(cfg, "raw", "gee", f"{stem}.tif")
        print(f"downloading {name} -> {out}", flush=True)
        gee.download_layer(img, bands, grid, tiles, args.tile, out,
                           data_path(cfg, "raw", "gee", "tiles", stem), args.workers)


if __name__ == "__main__":
    main()
