"""Building counts per 1 km cell, used to check caption claims (PLAN 4.8, refined 2026-10-04).

Two sources, both independent of night lights:
  ms_buildings   Microsoft Global ML Building Footprints (imagery-detected, ODbL), Brazil quadkey files
  osm_buildings  OpenStreetMap building=* ways/relations from the same 2023-01-01 extract as 03
A building is counted in the cell that contains its centroid (MS: mean of the outer-ring vertices).

Usage: python scripts/03b_buildings.py [--workers 12]
Output: interim/buildings.parquet (cell_id, ms_buildings, osm_buildings), cells with at least one building only.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from gisecon.config import data_path, ensure_dirs, load_config
from gisecon.data.grid import Grid

LINKS = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


def quadkey_bounds(qk: str) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a Bing quadkey tile in degrees."""
    x = y = 0
    z = len(qk)
    for i, ch in enumerate(qk):
        m = 1 << (z - 1 - i)
        d = int(ch)
        x |= m if d & 1 else 0
        y |= m if d & 2 else 0
    n = 2 ** z
    lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))
    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def count_ms_file(args) -> pd.Series:
    """Building centroids of one gzipped GeoJSON-lines file -> counts per cell_id."""
    path, grid_dict = args
    grid = Grid(**grid_dict)
    lon, lat = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            geom = json.loads(line)["geometry"]
            ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
            pts = np.asarray(ring[:-1] if len(ring) > 1 else ring, dtype=float)
            lon.append(pts[:, 0].mean())
            lat.append(pts[:, 1].mean())
    if not lon:
        return pd.Series(dtype="int64")
    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(np.array(lon), np.array(lat))
    cid = grid.cell_index(x, y)
    cid = cid[cid >= 0]
    return pd.Series(cid).value_counts()


def osm_counts(pbf: Path, grid: Grid) -> pd.Series:
    from pyrosm import OSM
    b = OSM(str(pbf)).get_buildings()
    pts = b.to_crs(grid.crs).geometry.representative_point()
    cid = grid.cell_index(pts.x.to_numpy(), pts.y.to_numpy())
    return pd.Series(cid[cid >= 0]).value_counts()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config()
    ensure_dirs(cfg)
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    t0 = time.time()

    # Microsoft footprints: quadkey files that overlap the grid's extent
    links = pd.read_csv(io.StringIO(urllib.request.urlopen(LINKS, timeout=120).read().decode()))
    links = links[links["Location"] == "Brazil"]
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform([grid.xmin, grid.xmin + grid.width * grid.cell],
                               [grid.ymax - grid.height * grid.cell, grid.ymax])
    W, E, S, N = min(lon), max(lon), min(lat), max(lat)
    keep = [r for r in links.itertuples() if (lambda b: b[0] < E and b[2] > W and b[1] < N and b[3] > S)(
        quadkey_bounds(str(r.QuadKey)))]
    dest = data_path(cfg, "raw", "ms_buildings")
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        files = list(ex.map(lambda r: download(r.Url, dest / f"{r.QuadKey}_{Path(r.Url).name}"), keep))
    print(f"{len(files)} Microsoft quadkey files ready ({time.time() - t0:.0f} s)", flush=True)

    total = pd.Series(dtype="int64")
    with ProcessPoolExecutor(args.workers) as ex:
        for i, s in enumerate(ex.map(count_ms_file, [(f, grid.__dict__) for f in files]), 1):
            total = total.add(s, fill_value=0)
            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files counted ({time.time() - t0:.0f} s)", flush=True)
    ms = total.astype("int64").rename("ms_buildings")

    pbf = data_path(cfg, "raw", "osm") / "mg_buildings.osm.pbf"
    if not pbf.exists():
        import shutil
        import subprocess
        import sys
        exe = shutil.which("osmium") or str(Path(sys.prefix) / "bin" / "osmium")
        src = data_path(cfg, "raw", "osm") / f"{cfg['study']['state_abbr'].lower()}_230101.osm.pbf"
        subprocess.run([exe, "tags-filter", "-o", str(pbf), "--overwrite", str(src), "w/building", "r/building"],
                       check=True)
    osm = osm_counts(pbf, grid).rename("osm_buildings")

    out = pd.concat([ms, osm], axis=1).fillna(0).astype("int64")
    out.index.name = "cell_id"
    out = out.reset_index()
    cells = pd.read_parquet(data_path(cfg, "interim", "cell_muni.parquet"), columns=["cell_id"])
    out = out[out["cell_id"].isin(cells["cell_id"])]
    out.to_parquet(data_path(cfg, "interim", "buildings.parquet"), index=False)
    print(f"wrote interim/buildings.parquet: {len(out):,} cells with buildings; "
          f"MS {int(out.ms_buildings.sum()):,} buildings, OSM {int(out.osm_buildings.sum()):,} "
          f"({time.time() - t0:.0f} s)")
    share = len(out) / len(cells)
    print(f"share of state cells with at least one building: {share:.1%}")


if __name__ == "__main__":
    main()
