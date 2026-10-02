import numpy as np
import pytest

ee = pytest.importorskip("ee")

from gisecon.data import gee
from gisecon.data.grid import make_grid


def test_busy_errors_wait_instead_of_failing(monkeypatch):
    grid = make_grid((0, 0, 4000, 4000), "EPSG:31983", 1000)
    replies = iter([ee.EEException("Too Many Requests: Exceeded Earth Engine concurrency limit.")] * 7
                   + [np.zeros((2, 2), dtype=[("b", "f4")])])

    def fake_compute(req):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(ee.data, "computePixels", fake_compute)
    monkeypatch.setattr(gee.time, "sleep", lambda s: None)
    a = gee.fetch_tile(None, grid, 0, 0, 2, 2, retries=2)  # 7 busy replies > retries, still succeeds
    assert a.shape == (2, 2, 1)


def test_memory_errors_raise_immediately(monkeypatch):
    grid = make_grid((0, 0, 4000, 4000), "EPSG:31983", 1000)
    calls = []

    def fake_compute(req):
        calls.append(1)
        raise ee.EEException("User memory limit exceeded.")

    monkeypatch.setattr(ee.data, "computePixels", fake_compute)
    monkeypatch.setattr(gee.time, "sleep", lambda s: None)
    with pytest.raises(ee.EEException):
        gee.fetch_tile(None, grid, 0, 0, 2, 2)
    assert len(calls) == 1
