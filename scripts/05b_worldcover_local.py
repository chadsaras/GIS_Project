"""Phase 2.5 without Earth Engine: WorldCover 2021 class shares per 1 km cell, computed on the server.

Same data as ESA/WorldCover/v200 in Earth Engine, read from ESA's public 3 x 3 degree GeoTIFFs. Every 10 m
pixel is assigned to the grid cell that contains its centre, so a share = class pixels / valid pixels in the
cell (the same mean Earth Engine's reduceResolution computes). Uses no Earth Engine quota.

Usage: python scripts/05b_worldcover_local.py [--workers 12]
Output: raw/gee/worldcover_2021.tif (11 bands lc_tree ... lc_moss, same grid and names as 05_export_gee.py)
"""
from __future__ import annotations

import argparse
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window, from_bounds

from gisecon.config import data_path, ensure_dirs, load_config
from gisecon.data.gee import WORLDCOVER_CLASSES
from gisecon.data.grid import Grid

URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
CODES = list(WORLDCOVER_CLASSES)  # 10, 20, ..., 100
LOOKUP = np.full(256, -1, dtype=np.int64)
LOOKUP[CODES] = np.arange(len(CODES))
BLOCK_ROWS = 512


def tile_names(west, south, east, north) -> list[str]:
    """ESA tiles are named by their south-west corner on a 3-degree lattice, e.g. S24W051."""
    names = []
    for lat in range(math.floor(south / 3) * 3, math.ceil(north / 3) * 3, 3):
        for lon in range(math.floor(west / 3) * 3, math.ceil(east / 3) * 3, 3):
            names.append(f"{'S' if lat < 0 else 'N'}{abs(lat):02d}{'W' if lon < 0 else 'E'}{abs(lon):03d}")
    return names


def download(name: str, dest: Path) -> Path | None:
    f = dest / f"{name}.tif"
    if f.exists():
        return f
    try:
        urllib.request.urlretrieve(URL.format(name), f.with_suffix(".part"))
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # no tile there (ocean)
            return None
        raise
    f.with_suffix(".part").rename(f)
    return f


def count_tile(args) -> np.ndarray:
    """(n_cells, n_classes) pixel counts from one ESA tile, restricted to the study bounding box."""
    path, grid_dict, bounds = args
    grid = Grid(**grid_dict)
    to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    acc = np.zeros(grid.n_cells * len(CODES), dtype=np.int64)
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        t = src.transform
        for r0 in range(int(win.row_off), int(win.row_off + win.height), BLOCK_ROWS):
            h = min(BLOCK_ROWS, int(win.row_off + win.height) - r0)
            cls = src.read(1, window=Window(win.col_off, r0, win.width, h))
            cols = win.col_off + np.arange(cls.shape[1]) + 0.5
            rows = r0 + np.arange(h) + 0.5
            lon = t.c + cols * t.a
            lat = t.f + rows * t.e
            lon2, lat2 = np.meshgrid(lon, lat)
            k = LOOKUP[cls.ravel()]
            ok = k >= 0
            x, y = to_grid.transform(lon2.ravel()[ok], lat2.ravel()[ok])
            cid = grid.cell_index(x, y)
            inside = cid >= 0
            keys, n = np.unique(cid[inside] * len(CODES) + k[ok][inside], return_counts=True)
            acc[keys] += n
    return acc.reshape(grid.n_cells, len(CODES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config()
    ensure_dirs(cfg)
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg"))
    west, south, east, north = munis.to_crs(4326).total_bounds + np.array([-0.05, -0.05, 0.05, 0.05])

    dest = data_path(cfg, "raw", "worldcover")
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    paths = [p for p in (download(n, dest) for n in tile_names(west, south, east, north)) if p]
    print(f"{len(paths)} ESA tiles ready ({time.time() - t0:.0f} s)", flush=True)

    jobs = [(p, grid.__dict__, (west, south, east, north)) for p in paths]
    total = np.zeros((grid.n_cells, len(CODES)), dtype=np.int64)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, c in enumerate(ex.map(count_tile, jobs), 1):
            total += c
            print(f"  {i}/{len(jobs)} tiles counted ({time.time() - t0:.0f} s)", flush=True)

    valid = total.sum(1)
    shares = np.where(valid[:, None] > 0, total / np.maximum(valid, 1)[:, None], np.nan).astype(np.float32)
    out = data_path(cfg, "raw", "gee", "worldcover_2021.tif")
    out.parent.mkdir(parents=True, exist_ok=True)
    bands = [f"lc_{n}" for n in WORLDCOVER_CLASSES.values()]
    with rasterio.open(out, "w", driver="GTiff", width=grid.width, height=grid.height, count=len(bands),
                       dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
                       compress="deflate", tiled=True) as dst:
        dst.descriptions = tuple(bands)
        dst.write(np.moveaxis(shares.reshape(grid.height, grid.width, len(bands)), -1, 0))
    print(f"wrote {out}: {int((valid > 0).sum()):,} cells with data, "
          f"median {int(np.median(valid[valid > 0])):,} pixels per cell ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
