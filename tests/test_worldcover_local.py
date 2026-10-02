"""Local WorldCover shares: a synthetic 10 m class raster in lon/lat, counted onto a small UTM grid."""
import importlib.util

import numpy as np
import rasterio
from rasterio.transform import from_origin

from gisecon.config import REPO_ROOT
from gisecon.data.grid import make_grid

spec = importlib.util.spec_from_file_location("wc", REPO_ROOT / "scripts" / "05b_worldcover_local.py")
wc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wc)


def test_tile_names_cover_bounds():
    assert wc.tile_names(-51.05, -22.9, -39.9, -14.2) == [
        f"{lat}{lon}" for lat in ("S24", "S21", "S18", "S15") for lon in ("W054", "W051", "W048", "W045", "W042")]


def test_count_tile_shares(tmp_path):
    # 0.02 x 0.02 degree raster near Belo Horizonte: left half trees (10), right half built-up (50)
    res = 0.0001
    w = h = 200
    west, north = -44.0, -19.9
    cls = np.full((h, w), 10, dtype=np.uint8)
    cls[:, w // 2:] = 50
    cls[0, 0] = 0  # no-data pixel is ignored
    path = tmp_path / "t.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(west, north, res, res)) as dst:
        dst.write(cls, 1)

    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:31983", always_xy=True)
    xs, ys = tr.transform([west, west + w * res], [north - h * res, north])
    grid = make_grid((min(xs) - 2000, min(ys) - 2000, max(xs) + 2000, max(ys) + 2000), "EPSG:31983", 1000)

    counts = wc.count_tile((path, grid.__dict__, (west, north - h * res, west + w * res, north)))
    assert counts.sum() == h * w - 1  # every valid pixel lands in exactly one cell
    tree, built = wc.CODES.index(10), wc.CODES.index(50)
    assert counts[:, tree].sum() == h * (w // 2) - 1 and counts[:, built].sum() == h * (w // 2)
    assert counts[:, [i for i in range(len(wc.CODES)) if i not in (tree, built)]].sum() == 0
