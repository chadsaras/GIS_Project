"""São Paulo maps for the final deck, drawn without geopandas from the website's data file (app/data/data.js).

  python presentation/final/make_maps_local.py
Use when the server is out of reach; make_figures.py draws the same maps from the full-resolution polygons.
Same colour map and scale as the mid-term maps (CMAP_GDP, GDP_NORM in presentation/midterm/make_figures.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "presentation" / "final" / "assets"
CMAP_GDP = LinearSegmentedColormap.from_list("gdp", ["#FBF1DD", "#F2B134", "#D9632E", "#8E2C3A", "#3B1530"])
GDP_NORM = (8000, 120000)


def rings(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [np.array(p[0]) for p in polys]  # outer rings only


def main() -> None:
    text = (ROOT / "app" / "data" / "data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"):text.rstrip().rstrip(";").rindex("}") + 1])
    sp = data["states"]["SP"]
    rows = {r["code"]: r for r in sp["rows"]}
    lat0 = np.deg2rad(-22.3)  # equirectangular with the state's mean latitude: close to the UTM look
    for key, name in (("official", "sp_official"), ("est", "sp_predicted")):
        verts, vals = [], []
        for f in sp["shapes"]["features"]:
            r = rows.get(f["properties"]["muni_code"])
            if r is None:
                continue
            for ring in rings(f["geometry"]):
                verts.append(np.c_[ring[:, 0] * np.cos(lat0), ring[:, 1]])
                vals.append(r[key])
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        pc = PolyCollection(verts, array=np.array(vals, float), cmap=CMAP_GDP, norm=LogNorm(*GDP_NORM),
                            edgecolors="white", linewidths=0.08)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
    print(f"wrote {OUT}/sp_official.png, sp_predicted.png ({len(rows)} municipalities)")


if __name__ == "__main__":
    main()
