import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from gisecon.data.grid import make_grid
from gisecon.data.osm import classify_pois, line_length_per_cell, poi_counts


def test_classify_priority_and_any_value():
    df = pd.DataFrame({
        "amenity": ["school", "bank", None, "fuel", None],
        "shop": [None, None, "bakery", "convenience", None],
        "office": [None, None, None, None, "government"],
    })
    assert classify_pois(df).tolist() == ["education", "finance", "retail", "fuel", "public"]


def test_poi_counts_per_cell():
    g = make_grid((0, 0, 2000, 1000), "EPSG:31983", 1000)
    pois = gpd.GeoDataFrame({"amenity": ["school", "school", "bank"], "shop": [None, None, None]},
                            geometry=[Point(100, 100), Point(900, 900), Point(1500, 500)], crs="EPSG:31983")
    c = poi_counts(pois, g).set_index("cell_id")
    assert c.loc[0, "poi_education"] == 2 and c.loc[1, "poi_finance"] == 1


def test_line_length_split_across_cells():
    g = make_grid((0, 0, 3000, 1000), "EPSG:31983", 1000)
    line = gpd.GeoSeries([LineString([(500, 500), (2500, 500)])], crs="EPSG:31983")
    km = line_length_per_cell(line, g)
    assert km.sum() == pytest.approx(2.0, abs=1e-9)
    assert km.loc[0] == pytest.approx(0.5, abs=0.011)
    assert km.loc[1] == pytest.approx(1.0, abs=0.011)
