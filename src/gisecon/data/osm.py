"""OpenStreetMap features per grid cell: POI counts, road and waterway lengths."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from gisecon.data.grid import Grid

# Category -> list of (tag key, values or None for "any value"). First match wins, in this order.
POI_CATEGORIES: dict[str, list[tuple[str, list[str] | None]]] = {
    "education": [("amenity", ["school", "university", "college", "kindergarten"])],
    "health": [("amenity", ["hospital", "clinic", "doctors", "pharmacy", "dentist"]), ("healthcare", None)],
    "finance": [("amenity", ["bank", "atm", "bureau_de_change"])],
    "fuel": [("amenity", ["fuel"])],
    "food": [("amenity", ["restaurant", "cafe", "fast_food", "bar", "pub", "food_court"])],
    "public": [("amenity", ["townhall", "police", "courthouse", "post_office", "fire_station"]),
               ("office", ["government"])],
    "industry": [("landuse", ["industrial"]), ("man_made", ["works"]), ("industrial", None)],
    "office": [("office", None)],
    "retail": [("shop", None)],
    "tourism": [("tourism", None)],
}
POI_KEYS = sorted({k for rules in POI_CATEGORIES.values() for k, _ in rules})

ROAD_CLASSES = {
    "major": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"],
    "medium": ["secondary", "secondary_link", "tertiary", "tertiary_link"],
    "minor": ["residential", "unclassified", "living_street", "service", "road"],
    "track": ["track"],
}
WATERWAYS = ["river", "stream", "canal"]


def clip_pbf(src: Path, polygon_geojson: Path, dst: Path) -> None:
    """Cut an OSM extract to a polygon with osmium (complete ways and relations kept)."""
    exe = shutil.which("osmium") or str(Path(sys.prefix) / "bin" / "osmium")
    subprocess.run([exe, "extract", "-p", str(polygon_geojson), "-s", "complete_ways",
                    "--overwrite", "-o", str(dst), str(src)], check=True)


def classify_pois(df: pd.DataFrame) -> pd.Series:
    """Category per feature (NaN when no rule matches), using POI_CATEGORIES priority order."""
    cat = pd.Series(np.nan, index=df.index, dtype="object")
    for name, rules in POI_CATEGORIES.items():
        hit = pd.Series(False, index=df.index)
        for key, values in rules:
            if key not in df:
                continue
            col = df[key]
            hit |= col.notna() if values is None else col.isin(values)
        cat = cat.where(cat.notna() | ~hit, name)
    return cat


def poi_counts(pois: gpd.GeoDataFrame, grid: Grid) -> pd.DataFrame:
    """Count POIs per cell and category. Polygons are represented by a point inside them."""
    pois = pois.to_crs(grid.crs)
    pois = pois.assign(category=classify_pois(pois)).dropna(subset=["category"])
    pts = pois.geometry.representative_point()
    pois["cell_id"] = grid.cell_index(pts.x.to_numpy(), pts.y.to_numpy())
    pois = pois[pois.cell_id >= 0]
    counts = pois.groupby(["cell_id", "category"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(POI_CATEGORIES), fill_value=0)
    return counts.add_prefix("poi_").reset_index()


def line_length_per_cell(lines: gpd.GeoSeries, grid: Grid, max_seg: float = 10.0) -> pd.Series:
    """Line length (km) per cell: split lines into <= max_seg m pieces, credit each piece to its midpoint's cell."""
    geoms = shapely.segmentize(lines.to_crs(grid.crs).explode(index_parts=False).to_numpy(), max_seg)
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    same = idx[1:] == idx[:-1]  # consecutive vertices of the same line form a segment
    a, b = coords[:-1][same], coords[1:][same]
    length = np.hypot(*(b - a).T)
    mid = (a + b) / 2
    cid = grid.cell_index(mid[:, 0], mid[:, 1])
    ok = cid >= 0
    return pd.Series(length[ok], index=cid[ok]).groupby(level=0).sum() / 1000.0


def road_water_lengths(roads: gpd.GeoDataFrame, water: gpd.GeoDataFrame, grid: Grid) -> pd.DataFrame:
    parts = {}
    for cls, values in ROAD_CLASSES.items():
        sel = roads[roads["highway"].isin(values)]
        parts[f"road_{cls}_km"] = line_length_per_cell(sel.geometry, grid)
    sel = water[water["waterway"].isin(WATERWAYS)]
    parts["water_km"] = line_length_per_cell(sel.geometry, grid)
    out = pd.DataFrame(parts).fillna(0.0)
    out.index.name = "cell_id"
    return out.reset_index()
