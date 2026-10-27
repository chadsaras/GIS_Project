"""Data file for the demo website in app/ (LIGHT: seconds).

  python scripts/22_app_data.py
Writes app/data/data.js (window.APP_DATA = {...}) so the page also works opened from disk, offline. Per state:
simplified municipal polygons (WGS84) with the official value, the satellite estimate (V10) and its 90% range by
risk group, the night-light estimate (V1), the risk group and, for the largest misses, a one-line likely reason.
Minas Gerais values are out-of-sample (reports/tables/app_municipalities.csv); a transfer state is added when its
reports/transfer/<STATE>/tables/transfer_municipalities.csv exists (scripts/21_transfer.py).
"""
from __future__ import annotations

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, load_config

SIMPLIFY_M = 400  # polygon simplification tolerance in metres (grid CRS); keeps the file small


def reasons() -> dict[int, str]:
    """'Likely reason' lines of reports/tables/top10_gaps.md, keyed by municipality code."""
    text = (REPO_ROOT / "reports" / "tables" / "top10_gaps.md").read_text(encoding="utf-8")
    out = {}
    for code, body in re.findall(r"^## .*?\((\d{7})\)\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M):
        m = re.search(r"Likely reason: (.*?)\((candidate)\): (.*)", body)
        if m:
            out[int(code)] = m.group(1).strip() + ": " + m.group(3).strip()
    return out


def shapes(path, codes) -> dict:
    g = gpd.read_file(path)[["muni_code", "geometry"]]
    g = g[g.muni_code.isin(codes)].copy()
    g["geometry"] = g.geometry.simplify(SIMPLIFY_M, preserve_topology=True)
    g = g.to_crs(4326)
    gj = json.loads(g.to_json(drop_id=True))
    for f in gj["features"]:  # 4 decimals ~ 10 m: plenty for a state map
        f["geometry"] = json.loads(json.dumps(f["geometry"]), parse_float=lambda s: round(float(s), 4))
    return gj


def rs(x) -> int:
    return int(round(float(np.exp(x))))


def main() -> None:
    cfg = load_config()
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    app = pd.read_csv(REPO_ROOT / "reports" / "tables" / "app_municipalities.csv").set_index("muni_code")
    v1 = pd.concat([pd.read_parquet(data_path(cfg, "processed", "predictions", f"V1_r{r}.parquet"))
                    for r in range(cfg["cv"]["n_folds"])])
    v1 = v1[v1.split == "test"].set_index("muni_code").y_pred
    why = reasons()
    rows = [{"code": int(c), "name": r.muni_name, "pop": int(lab.at[c, "pop"]), "official": rs(r.y_true),
             "est": rs(r.y_pred), "lo": rs(r.glo90), "hi": rs(r.ghi90), "ntl": rs(v1[c]), "risk": r.risk_group,
             "why": why.get(int(c), "")} for c, r in app.iterrows()]
    states = {"MG": {"name": "Minas Gerais", "kind": "tested", "rows": rows,
                     "shapes": shapes(data_path(cfg, "interim", "municipios.gpkg"), app.index)}}

    for f in sorted((REPO_ROOT / "reports" / "transfer").glob("*/tables/transfer_municipalities.csv")):
        st = f.parts[-3]
        t = pd.read_csv(f).set_index("muni_code")
        tcfg_dir = data_path(cfg, "raw").parents[1] / f"data_{st.lower()}"  # ~/GIS/data_<state>
        rows = [{"code": int(c), "name": r.muni_name, "pop": int(r["pop"]), "official": rs(r.official),
                 "est": rs(r.pred_V10), "lo": rs(r.glo90), "hi": rs(r.ghi90), "ntl": rs(r.pred_V1),
                 "risk": r.risk_group, "why": ""} for c, r in t.iterrows()]
        states[st] = {"name": {"SP": "São Paulo"}.get(st, st), "kind": "transfer", "rows": rows,
                      "shapes": shapes(tcfg_dir / "interim" / "municipios.gpkg", t.index)}

    out = REPO_ROOT / "app" / "data" / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.APP_DATA = " + json.dumps({"states": states}, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): " + ", ".join(f"{k} {len(v['rows'])}" for k, v in states.items()))


if __name__ == "__main__":
    main()
