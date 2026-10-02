"""Phase 4.3: one 768 x 768 PNG per sampled cell from an XYZ imagery service (config text.tile_url).

Source (PLAN 4.1): Esri World Imagery via the ArcGIS Static Basemap Tiles service, imagery style without
labels, 512 px tiles at zoom 16 (~5 tiles per cell, ~50-60k for 10,000 cells, within the 2M/month free tier).
The API key comes from $ESRI_API_KEY or the file in text.tile_key_file; it is never written to the repo.
LONG but light (hours, network-bound, ~5 GB). Resumable: finished PNGs are skipped.

Usage: python scripts/11_download_tiles.py [--limit 50] [--workers 4]
       (--limit 50 = the PLAN 4.1 trial: download 50 tiles and look at them)
Output: raw/tiles/{cell_id}.png
"""
from __future__ import annotations

import argparse
import io
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

from gisecon.config import data_path, load_config
from gisecon.data.grid import Grid

HALF = math.pi * 6378137  # half the Web Mercator world width, m


def tile_range(x0, y0, x1, y1, z):
    """Tile columns/rows (inclusive) covering a Web Mercator box, and metres per tile (any tile pixel size)."""
    tm = 2 * HALF / 2 ** z
    return (int((x0 + HALF) // tm), int((x1 + HALF) // tm), int((HALF - y1) // tm), int((HALF - y0) // tm)), tm


def api_key(cfg_text: dict) -> str | None:
    key = os.environ.get("ESRI_API_KEY")
    f = cfg_text.get("tile_key_file")
    if not key and f and os.path.exists(os.path.expanduser(f)):
        key = open(os.path.expanduser(f)).read().strip()
    return key or None


def fetch(url: str, retries: int = 5) -> Image.Image:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gisecon-research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) or attempt == retries - 1:  # bad or missing key: retrying will not help
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def cell_image(box3857, z: int, url: str, px: int, ts: int = 256) -> Image.Image:
    # ponytail: crops the cell's bounding box in Web Mercator; the 1 km UTM square is rotated by up
    # to ~2 degrees there, so the image includes a thin margin of neighbouring cells
    x0, y0, x1, y1 = box3857
    (c0, c1, r0, r1), tm = tile_range(x0, y0, x1, y1, z)
    canvas = Image.new("RGB", ((c1 - c0 + 1) * ts, (r1 - r0 + 1) * ts))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            tile = fetch(url.format(z=z, x=c, y=r))
            if tile.size != (ts, ts):
                tile = tile.resize((ts, ts), Image.LANCZOS)
            canvas.paste(tile, ((c - c0) * ts, (r - r0) * ts))
    res = tm / ts
    left, top = (x0 + HALF) / res - c0 * ts, (HALF - y1) / res - r0 * ts
    crop = canvas.crop((round(left), round(top), round(left + (x1 - x0) / res), round(top + (y1 - y0) / res)))
    return crop.resize((px, px), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    t = cfg["text"]
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    sample = pd.read_parquet(data_path(cfg, "interim", "caption_sample.parquet"))
    ids = sample["cell_id"].to_numpy()[: args.limit]
    out = data_path(cfg, "raw", "tiles")
    out.mkdir(parents=True, exist_ok=True)
    to3857 = Transformer.from_crs(grid.crs, "EPSG:3857", always_xy=True)
    url = t["tile_url"]
    if "arcgis.com" in url:
        key = api_key(t)
        assert key, "no Esri API key: set $ESRI_API_KEY or write it to text.tile_key_file (see PLAN 4.1)"
        url = f"{url}?token={key}"

    def job(cid):
        f = out / f"{cid}.png"
        if f.exists():
            return
        bx = grid.cell_box(cid).bounds
        xs, ys = to3857.transform(np.array([bx[0], bx[2], bx[0], bx[2]]), np.array([bx[1], bx[1], bx[3], bx[3]]))
        cell_image((xs.min(), ys.min(), xs.max(), ys.max()), t["tile_zoom"], url, t["tile_px"],
                   t.get("tile_size", 256)).save(f)

    t0, failed = time.time(), []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(job, int(c)): int(c) for c in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.exception():
                if not failed:
                    print(f"first failure (cell {futs[fut]}): {re.sub(r"token=[^&\s'\"]+", "token=***", str(fut.exception()))[:200]}")
                failed.append(futs[fut])
            if i % 200 == 0 or i == len(ids):
                print(f"{i}/{len(ids)} cells ({time.time() - t0:.0f} s), {len(failed)} failed", flush=True)
    if failed:
        print(f"failed cells (rerun to retry): {failed[:20]}{' ...' if len(failed) > 20 else ''}")
    print(f"imagery: {t.get('tile_attribution', t['tile_url'])} (attribution required in the report)")


if __name__ == "__main__":
    main()
