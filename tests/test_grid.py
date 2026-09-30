import geopandas as gpd
import numpy as np
from shapely.geometry import box

from gisecon.data.grid import assign_cells, make_grid


def test_make_grid_snaps_outward():
    g = make_grid((-134427.8, 7464578.9, 1050593.3, 8426342.3), "EPSG:31983", 1000)
    assert (g.xmin, g.ymax) == (-135000, 8427000)
    assert (g.width, g.height) == (1186, 963)


def test_cell_index_roundtrip():
    g = make_grid((0, 0, 10_000, 5_000), "EPSG:31983", 1000)
    rows, cols, x, y = g.centres()
    ids = g.cell_index(x, y)
    assert np.array_equal(ids, np.arange(g.n_cells))
    assert g.cell_index(np.array([-1.0]), np.array([100.0]))[0] == -1
    # row 0 is the northern edge
    assert g.cell_index(np.array([500.0]), np.array([4_999.0]))[0] == 0


def test_assign_cells_rescues_tiny_polygon():
    g = make_grid((0, 0, 4000, 4000), "EPSG:31983", 1000)
    big = box(0, 0, 4000, 4000).difference(box(1100, 1100, 1400, 1400))
    tiny = box(1100, 1100, 1400, 1400)  # contains no cell centre (centres at x.5 km)
    munis = gpd.GeoDataFrame({"muni_code": [1, 2]}, geometry=[big, tiny], crs="EPSG:31983")
    cells, rescued = assign_cells(g, munis)
    assert set(cells.muni_code) == {1, 2}
    assert len(rescued) == 1 and rescued.muni_code.iloc[0] == 2
    assert len(cells) == 16
