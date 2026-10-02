import numpy as np
import pytest

ee = pytest.importorskip("ee")

from gisecon.data import gee
from gisecon.data.grid import make_grid


def test_fetch_split_reassembles_quarters(monkeypatch):
    grid = make_grid((0, 0, 32_000, 32_000), "EPSG:31983", 1000)
    calls = []

    def fake_fetch(image, g, r0, c0, h, w, retries=5):
        calls.append((h, w))
        if h * w > 64:  # Earth Engine refuses anything bigger than 8 x 8 cells
            raise ee.EEException("User memory limit exceeded.")
        rows, cols = np.mgrid[r0:r0 + h, c0:c0 + w]
        return np.stack([rows, cols], axis=-1).astype(np.float32)  # each pixel knows its own position

    monkeypatch.setattr(gee, "fetch_tile", fake_fetch)
    a = gee.fetch_split(None, grid, 3, 5, 27, 19)  # odd sizes exercise uneven halves
    rows, cols = np.mgrid[3:30, 5:24]
    assert a.shape == (27, 19, 2)
    assert np.array_equal(a[..., 0], rows) and np.array_equal(a[..., 1], cols)
    assert max(h * w for h, w in calls if h * w <= 64) <= 64


def test_fetch_split_raises_other_errors(monkeypatch):
    grid = make_grid((0, 0, 8000, 8000), "EPSG:31983", 1000)

    def fake_fetch(*a, **k):
        raise ee.EEException("Permission denied.")

    monkeypatch.setattr(gee, "fetch_tile", fake_fetch)
    with pytest.raises(ee.EEException, match="Permission"):
        gee.fetch_split(None, grid, 0, 0, 8, 8)
