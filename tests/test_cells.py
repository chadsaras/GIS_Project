import numpy as np
import pandas as pd
import pytest
import rasterio

from gisecon.data.cells import GSED, keep_mask, read_layer, rescue_empty_munis
from gisecon.data.grid import make_grid


def _cells(**cols):
    n = len(next(iter(cols.values())))
    base = {"muni_code": [1] * n, "gsed_coverage": [1.0] * n, "lc_water": [0.0] * n, "lc_tree": [0.0] * n,
            "lc_bare": [0.0] * n, "lc_built": [0.0] * n, "ntl": [0.0] * n, "poi_retail": [0] * n,
            "road_minor_km": [0.0] * n}
    df = pd.DataFrame({**base, **cols})
    return pd.concat([df, pd.DataFrame(np.full((n, 64), 0.1), columns=GSED)], axis=1)


def test_keep_mask_rules():
    df = _cells(gsed_coverage=[1.0, 0.3, 1.0, 1.0, 1.0, 1.0],
                lc_water=[0.0, 0.0, 0.95, 0.0, 0.0, 0.0],
                lc_tree=[0.2, 0.0, 0.0, 0.97, 0.97, 0.97],
                road_minor_km=[0.0, 0.0, 0.0, 0.0, 1.5, 0.0],
                ntl=[0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    keep, drops = keep_mask(df, water_max=0.9, coverage_min=0.5, empty_natural_min=0.95)
    # normal | low coverage | water | empty forest | forest with a road | forest with lights
    assert keep.tolist() == [True, False, False, False, True, True]
    assert drops == {"low_gsed_coverage": 1, "water": 1, "empty_natural": 1}


def test_keep_mask_drops_missing_gsed():
    df = _cells(lc_built=[0.5, 0.5])
    df.loc[1, "gsed_07"] = np.nan
    assert keep_mask(df, 0.9, 0.5, 0.95)[0].tolist() == [True, False]


def test_rescue_municipality_with_no_kept_cell():
    df = _cells(muni_code=[1, 1, 2, 2])
    df.loc[3, "gsed_00"] = np.nan
    keep, lost = rescue_empty_munis(df, np.array([True, False, False, False]))
    assert lost == [2]
    assert keep.tolist() == [True, False, True, False]  # muni 2 keeps its cell that has GSED


def test_read_layer_checks_alignment(tmp_path):
    g = make_grid((0, 0, 4000, 3000), "EPSG:31983", 1000)
    data = np.arange(2 * g.height * g.width, dtype=np.float32).reshape(2, g.height, g.width)

    def write(path, transform):
        with rasterio.open(path, "w", driver="GTiff", width=g.width, height=g.height, count=2, dtype="float32",
                           crs=g.crs, transform=transform) as dst:
            dst.descriptions = ("a", "b")
            dst.write(data)

    write(tmp_path / "ok.tif", g.transform)
    out = read_layer(tmp_path / "ok.tif", g, np.array([0, 2]), np.array([1, 3]))
    assert out.columns.tolist() == ["a", "b"]
    assert out["a"].tolist() == [data[0, 0, 1], data[0, 2, 3]]

    write(tmp_path / "shifted.tif", g.transform @ rasterio.Affine.translation(0.5, 0))
    with pytest.raises(AssertionError, match="transform"):
        read_layer(tmp_path / "shifted.tif", g, np.array([0]), np.array([0]))
