"""Assemble the per-cell feature table from the grid rasters and OSM tables (PLAN 2.6, 2.10-2.11)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from gisecon.data.grid import Grid

GSED = [f"gsed_{i:02d}" for i in range(64)]


def read_layer(path: Path, grid: Grid, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """Band values at the given cells, one column per band (named from the band descriptions).

    Fails unless the raster sits on exactly the grid lattice (PLAN 2.6).
    """
    with rasterio.open(path) as src:
        assert (src.width, src.height) == (grid.width, grid.height), f"{path.name}: size differs from the grid"
        assert src.transform.almost_equals(grid.transform), f"{path.name}: transform differs from the grid"
        assert src.crs.to_string() == grid.crs, f"{path.name}: CRS {src.crs} differs from {grid.crs}"
        names = list(src.descriptions)
        assert all(names), f"{path.name}: unnamed bands"
        data = src.read()
    return pd.DataFrame(data[:, rows, cols].T, columns=names)


def keep_mask(df: pd.DataFrame, water_max: float, coverage_min: float, empty_natural_min: float
              ) -> tuple[np.ndarray, dict[str, int]]:
    """PLAN 2.10: which cells carry economic signal. Returns (keep, cells dropped per rule)."""
    no_gsed = (df["gsed_coverage"].fillna(0) < coverage_min) | df[GSED].isna().any(axis=1)
    water = df["lc_water"] >= water_max
    poi = df.filter(like="poi_").sum(axis=1)
    roads = df.filter(regex=r"^road_.*_km$").sum(axis=1)
    empty = ((df["lc_tree"] + df["lc_bare"] >= empty_natural_min) & (df["lc_built"] == 0)
             & (df["ntl"] == 0) & (poi == 0) & (roads == 0))
    drops = {"low_gsed_coverage": no_gsed, "water": water & ~no_gsed, "empty_natural": empty & ~no_gsed & ~water}
    keep = ~(no_gsed | water | empty)
    return keep.to_numpy(), {k: int(v.sum()) for k, v in drops.items()}


def rescue_empty_munis(df: pd.DataFrame, keep: np.ndarray) -> tuple[np.ndarray, list]:
    """A municipality with no kept cell would get an empty sum in Stage C: keep its cells that have GSED."""
    has_gsed = df[GSED].notna().all(axis=1).to_numpy()
    kept = set(df.loc[keep, "muni_code"])
    lost = sorted(set(df["muni_code"]) - kept)
    keep = keep | (df["muni_code"].isin(lost).to_numpy() & has_gsed)
    return keep, lost


def osm_completeness(cells: pd.DataFrame) -> pd.DataFrame:
    """Per municipality: road km per cell vs mean built-up share (PLAN 2.12, used in the RQ3/RQ5 analysis).

    osm_completeness is the residual of log(1 + road km per cell) on built-up share across
    municipalities: positive = more mapped roads than its urbanness predicts.
    """
    m = cells.assign(road_km=cells.filter(regex=r"^road_.*_km$").sum(axis=1),
                     poi=cells.filter(like="poi_").sum(axis=1)).groupby("muni_code").agg(
        road_km_per_cell=("road_km", "mean"), built_share=("lc_built", "mean"), poi_per_cell=("poi", "mean"))
    y = np.log1p(m["road_km_per_cell"].to_numpy())
    X = np.column_stack([np.ones(len(m)), m["built_share"].to_numpy()])
    m["osm_completeness"] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return m.reset_index()
