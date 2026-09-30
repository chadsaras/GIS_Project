"""Phase 3: spatial blocks (IBGE immediate regions) and 5 balanced folds.

Urban share uses population density (top third of municipalities = urban). It needs only
the label table, so the folds are frozen before any satellite data arrives.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, ensure_dirs, load_config
from gisecon.eval.folds import assign_folds, get_split


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    cv = cfg["cv"]
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet"))
    cells = pd.read_parquet(data_path(cfg, "interim", "cell_muni.parquet"))

    # 3.1-3.2 blocks and their profiles
    lab["density"] = lab["pop"] / lab["area_km2"]
    lab["urban"] = (lab["density"] >= lab["density"].quantile(2 / 3)).astype(float)
    lab["n_cells"] = lab["muni_code"].map(cells.groupby("muni_code").size())
    blocks = lab.groupby("rgi_code").agg(
        rgi_name=("rgi_name", "first"), n=("muni_code", "size"), n_cells=("n_cells", "sum"),
        mean_log_gdp_pc=("log_gdp_pc", "mean"), sd_log_gdp_pc=("log_gdp_pc", "std"),
        urban_share=("urban", "mean"))
    blocks["sd_log_gdp_pc"] = blocks["sd_log_gdp_pc"].fillna(0.0)
    blocks.to_parquet(data_path(cfg, "interim", "blocks.parquet"))
    print(f"{len(blocks)} blocks; municipalities per block: min {blocks.n.min()}, "
          f"median {blocks.n.median():.0f}, max {blocks.n.max()}")

    # 3.3 balanced assignment
    fold, score = assign_folds(
        blocks, cv["n_folds"], cv["n_assignment_tries"], cfg["project"]["seed"],
        balance_cols={"n": "sum", "n_cells": "sum", "mean_log_gdp_pc": "mean",
                      "sd_log_gdp_pc": "mean", "urban_share": "mean"})
    lab["fold"] = lab["rgi_code"].map(fold)
    folds = lab[["muni_code", "muni_name", "rgi_code", "fold"]].rename(columns={"rgi_code": "block_id"})
    folds.to_parquet(data_path(cfg, "processed", "folds.parquet"), index=False)
    folds.to_csv(REPO_ROOT / cfg["paths"]["reports"] / "tables" / "folds.csv", index=False)  # tracked copy

    prof = lab.groupby("fold").agg(blocks=("rgi_code", "nunique"), municipalities=("muni_code", "size"),
                                   cells=("n_cells", "sum"), mean_log_gdp_pc=("log_gdp_pc", "mean"),
                                   sd_log_gdp_pc=("log_gdp_pc", "std"), urban_share=("urban", "mean"),
                                   flagged=("flag_extreme", "sum"))
    prof.round(3).to_csv(REPO_ROOT / cfg["paths"]["reports"] / "tables" / "fold_profiles.csv")
    print(f"best imbalance score {score:.4f}\n{prof.round(3).to_string()}")

    # 3.4 rotation check
    for r in range(cv["n_folds"]):
        s = get_split(folds, r)
        print(f"round {r}: train {len(s.train)}, val {len(s.val)}, test {len(s.test)}")

    # 3.6 map
    munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg")).merge(folds, on="muni_code")
    fig, ax = plt.subplots(figsize=(8, 7))
    munis.plot(column="fold", categorical=True, cmap="Set2", legend=True, ax=ax,
               legend_kwds={"title": "fold", "loc": "lower left"})
    munis.dissolve("block_id").boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(f"5 spatial folds over {len(blocks)} immediate regions")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(REPO_ROOT / cfg["paths"]["reports"] / "figures" / "folds_map.png", dpi=150)
    print("wrote reports/figures/folds_map.png")


if __name__ == "__main__":
    main()
