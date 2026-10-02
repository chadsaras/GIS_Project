"""Phase 4.3: one 768 x 768 PNG per sampled cell from an XYZ imagery service (config text.tile_url).

LONG but light: ~20 zoom-17 tiles per cell, ~200k requests for 10,000 cells (hours, network-bound,
~5 GB). Resumable: finished PNGs are skipped. Gate: run only after the imagery licence check (PLAN 4.1).

Usage: python scripts/11_download_tiles.py [--limit 50] [--workers 4]
       (--limit 50 = the PLAN 4.1 trial: download 50 tiles and look at them)
Output: raw/tiles/{cell_id}.png
"""
from __future__ import annotations

import argparse
import io
import math
import time
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
    """Tile columns/rows (inclusive) covering a Web Mercator box, and metres per tile."""
    tm = 2 * HALF / 2 ** z
    return (int((x0 + HALF) // tm), int((x1 + HALF) // tm), int((HALF - y1) // tm), int((HALF - y0) // tm)), tm


def fetch(url: str, retries: int = 5) -> Image.Image:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gisecon-research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def cell_image(box3857, z: int, url: str, px: int) -> Image.Image:
    # ponytail: crops the cell's bounding box in Web Mercator; the 1 km UTM square is rotated by up
    # to ~2 degrees there, so the image includes a thin margin of neighbouring cells
    x0, y0, x1, y1 = box3857
    (c0, c1, r0, r1), tm = tile_range(x0, y0, x1, y1, z)
    canvas = Image.new("RGB", ((c1 - c0 + 1) * 256, (r1 - r0 + 1) * 256))
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            canvas.paste(fetch(url.format(z=z, x=c, y=r)), ((c - c0) * 256, (r - r0) * 256))
    res = tm / 256
    left, top = (x0 + HALF) / res - c0 * 256, (HALF - y1) / res - r0 * 256
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

    def job(cid):
        f = out / f"{cid}.png"
        if f.exists():
            return
        bx = grid.cell_box(cid).bounds
        xs, ys = to3857.transform(np.array([bx[0], bx[2], bx[0], bx[2]]), np.array([bx[1], bx[1], bx[3], bx[3]]))
        cell_image((xs.min(), ys.min(), xs.max(), ys.max()), t["tile_zoom"], t["tile_url"], t["tile_px"]).save(f)

    t0, failed = time.time(), []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(job, int(c)): int(c) for c in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.exception():
                failed.append(futs[fut])
            if i % 200 == 0 or i == len(ids):
                print(f"{i}/{len(ids)} cells ({time.time() - t0:.0f} s), {len(failed)} failed", flush=True)
    if failed:
        print(f"failed cells (rerun to retry): {failed[:20]}{' ...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
