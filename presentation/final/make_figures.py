"""Figures for the final deck that the mid-term deck does not have (run on the server, which has the Lato font).

  python presentation/final/make_figures.py
Writes presentation/final/assets/sp_official.png and sp_predicted.png: São Paulo municipalities coloured by official
and predicted (V10, models trained only on Minas Gerais) GDP per capita, with the mid-term maps' colour map and scale.
Every other asset is reused from presentation/midterm/assets or reports/figures.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from gisecon.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("mid", REPO_ROOT / "presentation" / "midterm" / "make_figures.py")
mid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mid)  # palette, colour maps and muni_map, so both decks look the same

OUT = REPO_ROOT / "presentation" / "final" / "assets"
SP_DATA = Path("~/GIS/data_sp").expanduser()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(REPO_ROOT / "reports" / "transfer" / "SP" / "tables" / "transfer_municipalities.csv")
    munis = gpd.read_file(SP_DATA / "interim" / "municipios.gpkg").to_crs(31983)
    m = munis.merge(t[["muni_code", "official", "pred_V10"]], on="muni_code")
    m["off"], m["pre"] = np.exp(m.official), np.exp(m.pred_V10)
    gn = LogNorm(*mid.GDP_NORM)
    for col, name in (("off", "sp_official"), ("pre", "sp_predicted")):
        mid.muni_map(m, col, OUT / f"{name}.png", mid.CMAP_GDP, gn, size=(4.6, 3.0), scale=False, edge=0.03)
    print(f"wrote {OUT}/sp_official.png, sp_predicted.png ({len(m)} municipalities)")


if __name__ == "__main__":
    main()
