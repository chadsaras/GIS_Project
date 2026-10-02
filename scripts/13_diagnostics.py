"""Phases 5.6 and 6.5: Stage A / B diagnostics tables and the neighbour check (LIGHT: about a minute).

Run after 07_run_experiment.py.
  reports/tables/stage_a_diagnostics.csv   recall@k per round/text + linear probe R2 (embedding -> built-up share)
  reports/tables/stage_b_diagnostics.csv   val R2 / RMSE of log NTL per round/input
  reports/tables/neighbours.csv, reports/figures/neighbours_raw_vs_z.png
      for 5 reference cells: mean distance (km) and log-NTL spread of the 100 nearest cells by
      cosine similarity, in raw GSED vs Stage B z (expected: z neighbours farther away but more alike)
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.eval.folds import get_split


def reference_cells(c: pd.DataFrame) -> dict[str, int]:
    """Row positions of 5 recognisable cell types, picked by simple rules."""
    def best(score: pd.Series, where) -> int:  # highest score among `where`, else over all cells
        return int(score[where].idxmax() if where.any() else score.idxmax())

    poi = c.filter(like="poi_").sum(axis=1)
    bh = np.hypot(c.x - 610000, c.y - 7797000)  # distance to Belo Horizonte centre, m
    return {
        "rural pasture": best(c.lc_grass, c.ntl == 0),
        "small town": best(-(c.lc_built - 0.3).abs(), poi.between(1, 20)),
        "industrial area": best(c.get("poi_industry", c.lc_built), c.lc_built > 0),
        "mining / bare ground": best(c.lc_bare, c.lc_built > 0),
        "Belo Horizonte suburb": best(c.lc_built, bh.between(10_000, 20_000)),
    }


def neighbours(emb: np.ndarray, i: int, k: int = 100) -> np.ndarray:
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    sim = e @ e[i]
    sim[i] = -np.inf
    return np.argpartition(-sim, k)[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0, help="round used for the neighbour check")
    args = ap.parse_args()
    cfg = load_config()
    reports = REPO_ROOT / cfg["paths"]["reports"]
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"))
    cells = cells[cells["keep"]].reset_index(drop=True)
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))

    a_rows, b_rows = [], []
    for r in range(cfg["cv"]["n_folds"]):
        mdir, idir = data_path(cfg, "models", f"round{r}"), data_path(cfg, "interim", f"round{r}")
        s = get_split(folds, r)
        tr, va = cells.muni_code.isin(s.train).to_numpy(), cells.muni_code.isin(s.val).to_numpy()
        for f in sorted(mdir.glob("stage_a_*.json")):
            text = f.stem.removeprefix("stage_a_")
            d = {k: v for k, v in json.loads(f.read_text()).items() if k != "log"}
            emb = np.load(idir / f"emb_A_{text}.npy").astype(np.float32)
            probe = Ridge(1.0).fit(emb[tr], cells.lc_built[tr])
            a_rows.append({"round": r, "text": text, **d, "probe_built_r2_val": probe.score(emb[va], cells.lc_built[va])})
        for f in sorted(mdir.glob("stage_b_*.json")):
            d = {k: v for k, v in json.loads(f.read_text()).items() if k != "log"}
            b_rows.append({"round": r, "input": f.stem.removeprefix("stage_b_"), **d})
    if a_rows:
        pd.DataFrame(a_rows).round(4).to_csv(reports / "tables" / "stage_a_diagnostics.csv", index=False)
        print(pd.DataFrame(a_rows).round(3).to_string(index=False))
    if b_rows:
        pd.DataFrame(b_rows).round(4).to_csv(reports / "tables" / "stage_b_diagnostics.csv", index=False)
        print(pd.DataFrame(b_rows).round(3).to_string(index=False))

    zf = data_path(cfg, "interim", f"round{args.round}", "z_gsed.npy")
    if not zf.exists():
        print(f"neighbour check skipped: {zf.name} not found (run V4 for round {args.round})")
        return
    spaces = {"raw GSED": cells[GSED].to_numpy(np.float32), "Stage B z": np.load(zf).astype(np.float32)}
    xy, ntl = cells[["x", "y"]].to_numpy(), cells["log_ntl"].to_numpy()
    rows = []
    for name, i in reference_cells(cells).items():
        for space, emb in spaces.items():
            nb = neighbours(emb, i)
            rows.append({"reference": name, "cell_id": int(cells.cell_id[i]), "space": space,
                         "mean_distance_km": np.hypot(*(xy[nb] - xy[i]).T).mean() / 1000,
                         "log_ntl_sd": ntl[nb].std(), "log_ntl_abs_gap": np.abs(ntl[nb] - ntl[i]).mean()})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(reports / "tables" / "neighbours.csv", index=False)
    print(t.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, col, title in zip(ax, ["mean_distance_km", "log_ntl_abs_gap"],
                             ["mean distance to the 100 nearest (km)", "mean |log NTL gap| to the 100 nearest"]):
        t.pivot(index="reference", columns="space", values=col).plot.barh(ax=a)
        a.set_title(title)
        a.set_ylabel("")
    fig.tight_layout()
    fig.savefig(reports / "figures" / "neighbours_raw_vs_z.png", dpi=150)
    print("wrote reports/figures/neighbours_raw_vs_z.png")


if __name__ == "__main__":
    main()
