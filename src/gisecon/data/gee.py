"""Earth Engine layers averaged onto the 1 km grid, downloaded tile by tile with computePixels.

Each layer is reduced from its native resolution (10 m GSED/WorldCover, ~460 m VIIRS) to the grid
with a mean reducer, so every value is the average over the exact 1 km cell.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio

from gisecon.data.grid import Grid

WORLDCOVER_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
                      70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


def init(project: str) -> None:
    import ee
    ee.Initialize(project=project)


def gsed_image(collection: str, year: int, crs: str):
    """64 embedding bands averaged to the grid, plus 'coverage' = share of 10 m pixels with data."""
    import ee
    col = ee.ImageCollection(collection).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    native = col.first().projection()
    img = col.mosaic().setDefaultProjection(native)
    cov = img.select(0).mask().rename("coverage")
    out = img.addBands(cov.toFloat()).setDefaultProjection(native)
    # EE needs ~38k input pixels per 1 km cell here (native UTM tiles are skewed vs EPSG:31983); 65536 is its cap
    return out.reduceResolution(ee.Reducer.mean(), maxPixels=65536)


def viirs_image(collection: str, band: str, year: int):
    import ee
    src = ee.ImageCollection(collection).filterDate(f"{year}-01-01", f"{year + 1}-01-01").first().select(band)
    img = src.unmask(0).setDefaultProjection(src.projection()).rename("ntl")
    return img.reduceResolution(ee.Reducer.mean(), maxPixels=64)


def worldcover_image(collection: str):
    import ee
    m = ee.ImageCollection(collection).first().select("Map")
    bands = [m.eq(c).rename(f"lc_{name}") for c, name in WORLDCOVER_CLASSES.items()]
    img = ee.Image.cat(bands).toFloat().setDefaultProjection(m.projection())
    return img.reduceResolution(ee.Reducer.mean(), maxPixels=65536)


def _grid_request(grid: Grid, row0: int, col0: int, h: int, w: int) -> dict:
    return {
        "dimensions": {"width": w, "height": h},
        "affineTransform": {"scaleX": grid.cell, "shearX": 0, "translateX": grid.xmin + col0 * grid.cell,
                            "shearY": 0, "scaleY": -grid.cell, "translateY": grid.ymax - row0 * grid.cell},
        "crsCode": grid.crs,
    }


def fetch_tile(image, grid: Grid, row0: int, col0: int, h: int, w: int, retries: int = 5) -> np.ndarray:
    """(h, w, bands) float32 array for one tile."""
    import ee
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY",
                                         "grid": _grid_request(grid, row0, col0, h, w)})
            return np.stack([arr[n].astype(np.float32) for n in arr.dtype.names], axis=-1)
        except ee.EEException as e:  # quota / timeout: back off and retry
            print(f"  tile r{row0} c{col0} attempt {attempt + 1}/{retries} failed: {str(e)[:200]}", flush=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 5)


def tiles_needed(grid: Grid, cell_rows: np.ndarray, cell_cols: np.ndarray, tile: int) -> list[tuple[int, int]]:
    """Top-left (row, col) of every tile containing at least one study cell."""
    t = np.unique(np.stack([cell_rows // tile, cell_cols // tile], axis=1), axis=0)
    return [(int(r) * tile, int(c) * tile) for r, c in t]


def download_layer(image, band_names: list[str], grid: Grid, tiles: list[tuple[int, int]], tile: int,
                   out_tif: Path, tile_dir: Path, workers: int = 8) -> None:
    """Fetch tiles in parallel (resumable: finished tiles are cached as .npy), then mosaic to one GeoTIFF."""
    tile_dir.mkdir(parents=True, exist_ok=True)

    def job(rc):
        r0, c0 = rc
        f = tile_dir / f"r{r0:05d}_c{c0:05d}.npy"
        if not f.exists():
            h, w = min(tile, grid.height - r0), min(tile, grid.width - c0)
            np.save(f, fetch_tile(image, grid, r0, c0, h, w))
        return f

    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        try:
            for fut in as_completed([ex.submit(job, rc) for rc in tiles]):
                fut.result()
                done += 1
                if done % 50 == 0 or done == len(tiles):
                    print(f"  {out_tif.name}: {done}/{len(tiles)} tiles ({time.time() - t0:.0f} s)", flush=True)
        except BaseException:  # stop at the first failed tile instead of running every queued one
            ex.shutdown(wait=False, cancel_futures=True)
            raise

    nb = len(band_names)
    with rasterio.open(out_tif, "w", driver="GTiff", width=grid.width, height=grid.height, count=nb,
                       dtype="float32", crs=grid.crs, transform=grid.transform, nodata=np.nan,
                       compress="deflate", tiled=True, BIGTIFF="IF_SAFER") as dst:
        dst.descriptions = tuple(band_names)
        for r0, c0 in tiles:
            a = np.load(tile_dir / f"r{r0:05d}_c{c0:05d}.npy")
            assert a.shape[-1] == nb, f"tile has {a.shape[-1]} bands, expected {nb}"
            win = rasterio.windows.Window(c0, r0, a.shape[1], a.shape[0])
            dst.write(np.moveaxis(a, -1, 0), window=win)
