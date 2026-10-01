"""Phase 2, steps 2.6 and 2.10-2.12: one row per 1 km cell with every feature and a keep flag.

Run after 02_build_grid.py, 03_osm_features.py and 05_export_gee.py.
Inputs:  interim/grid.json, interim/cell_muni.parquet, raw/gee/{gsed_<year>,viirs_<year>,worldcover_2021}.tif,
         interim/poi_counts.parquet, interim/road_water_km.parquet
Outputs: processed/cells.parquet, interim/osm_completeness.parquet,
         reports/tables/cells_mask_summary.csv, reports/figures/cells_qc.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, ensure_dirs, load_config
from gisecon.data.cells import GSED, keep_mask, osm_completeness, read_layer, rescue_empty_munis
from gisecon.data.grid import Grid


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    year, mk = cfg["study"]["year"], cfg["masking"]
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    cells = pd.read_parquet(data_path(cfg, "interim", "cell_muni.parquet"))
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()

    # 2.6 + 2.11: raster layers on the grid lattice (read_layer asserts the alignment)
    gee = data_path(cfg, "raw", "gee")
    layers = [read_layer(gee / f, grid, rows, cols) for f in (f"gsed_{year}.tif", f"viirs_{year}.tif", "worldcover_2021.tif")]
    df = pd.concat([cells.reset_index(drop=True), *layers], axis=1)
    lc = [c for c in df if c.startswith("lc_")]
    df[lc] = df[lc].fillna(0.0)
    df["ntl"] = df["ntl"].fillna(0.0).clip(lower=0)  # VIIRS can dip slightly below 0 after background removal
    df["log_ntl"] = np.log1p(df["ntl"])
    df["gsed_norm"] = np.sqrt((df[GSED] ** 2).sum(axis=1, min_count=64))  # length of the mean 10 m vector; < 1 = mixed cell

    # OSM tables are sparse (only cells with something mapped)
    for f in ("poi_counts.parquet", "road_water_km.parquet"):
        osm = pd.read_parquet(data_path(cfg, "interim", f))
        df = df.merge(osm, on="cell_id", how="left")
        df[osm.columns.drop("cell_id")] = df[osm.columns.drop("cell_id")].fillna(0)

    # 2.10 mask
    keep, drops = keep_mask(df, mk["water_max"], mk["gsed_coverage_min"], mk["empty_natural_min"])
    keep, rescued = rescue_empty_munis(df, keep)
    df["keep"] = keep
    float_cols = df.select_dtypes("float64").columns.drop(["x", "y"])
    df[float_cols] = df[float_cols].astype(np.float32)
    df.to_parquet(data_path(cfg, "processed", "cells.parquet"), index=False)

    summary = pd.Series({"cells": len(df), **{f"dropped_{k}": v for k, v in drops.items()},
                         "rescued_municipalities": len(rescued), "kept": int(keep.sum())})
    summary.to_csv(REPO_ROOT / cfg["paths"]["reports"] / "tables" / "cells_mask_summary.csv", header=["n"])
    per_muni = df[df.keep].groupby("muni_code").size()
    print(summary.to_string())
    print(f"kept cells per municipality: min {per_muni.min()}, median {per_muni.median():.0f}, max {per_muni.max()}")
    if rescued:
        print(f"municipalities whose cells were all masked, kept anyway: {rescued}")

    # 2.12 OSM completeness per municipality and QC maps
    osm_completeness(df).to_parquet(data_path(cfg, "interim", "osm_completeness.parquet"), index=False)
    qc_figure(df[df.keep], grid, REPO_ROOT / cfg["paths"]["reports"] / "figures" / "cells_qc.png")


def qc_figure(kept: pd.DataFrame, grid: Grid, out) -> None:
    """log NTL, built-up share and the first 3 GSED principal components as RGB; cities and borders should line up."""
    def to_raster(values: np.ndarray) -> np.ndarray:
        a = np.full((grid.height, grid.width) + values.shape[1:], np.nan, dtype=np.float32)
        a[kept["row"].to_numpy(), kept["col"].to_numpy()] = values
        return a

    g = kept[GSED].to_numpy(np.float64)
    g -= g.mean(0)
    pcs = g @ np.linalg.svd(g[:: max(1, len(g) // 50_000)], full_matrices=False)[2][:3].T
    lo, hi = np.percentile(pcs, [2, 98], axis=0)
    rgb = to_raster(np.clip((pcs - lo) / (hi - lo), 0, 1))
    rgb[np.isnan(rgb)] = 1.0  # masked cells in white

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for a, (title, img, kw) in zip(ax, [
            ("log(1 + night lights)", to_raster(kept["log_ntl"].to_numpy()), {"cmap": "magma"}),
            ("built-up share (WorldCover)", to_raster(kept["lc_built"].to_numpy()), {"cmap": "viridis", "vmax": 0.3}),
            ("GSED principal components 1-3 as RGB", rgb, {})]):
        im = a.imshow(img, interpolation="nearest", **kw)
        if kw:
            fig.colorbar(im, ax=a, shrink=0.6)
        a.set_title(title)
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
