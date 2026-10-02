"""Phase 7.7-7.8 and 8.1-8.3: municipal baselines and the hybrid, per round.

  V1   NTL only: log GDP pc ~ log(total NTL / pop), linear or quadratic (chosen on validation)
  V2   ridge on mean land-cover shares + log kept cells + log pop
  V11  ridge on mean cell embeddings + log kept cells + log pop: raw GSED (V11_gsed) and every
       cached Stage A / Stage B embedding of the round (V11_emb_A_full, V11_z_gsed, ...)
  V12  LightGBM on the V11_gsed features: mean GSED + log kept cells + log pop (skipped if lightgbm is missing)
  V10  hybrid: the V3-V9 run with the lowest validation MSE blended with V1 (weight chosen on validation)

Usage: python scripts/08_baselines.py --round 0      (run after 07_run_experiment.py for V10)
Output: processed/predictions/{variant}_r{round}.parquet, same columns as 07_run_experiment.py plus `note`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gisecon.config import data_path, load_config
from gisecon.data.cells import GSED
from gisecon.eval.folds import get_split
from gisecon.models import baselines as bl

SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    r, seed = args.round, cfg["project"]["seed"]

    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"))
    kept = cells[cells["keep"]].reset_index(drop=True)  # same row order as the cached embeddings
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    split = get_split(pd.read_parquet(data_path(cfg, "processed", "folds.parquet")), r)
    codes = {k: getattr(split, k) for k in SPLITS}

    m = lab[["log_pop", "log_gdp_pc"]].copy()
    ntl_total = cells.groupby("muni_code")["ntl"].sum().reindex(m.index, fill_value=0)
    m["ntl_x"] = np.log1p(ntl_total) - m["log_pop"]  # log(1 + total) so a fully dark municipality stays finite
    m["log_cells"] = np.log(kept.groupby("muni_code").size().reindex(m.index, fill_value=0) + 1)
    lc = [c for c in cells if c.startswith("lc_")]
    by_muni = kept.groupby("muni_code")
    m = m.join(by_muni[lc].mean()).join(by_muni[GSED].mean())
    y = {k: m.loc[c, "log_gdp_pc"].to_numpy() for k, c in codes.items()}

    def feats(cols):
        return {k: m.loc[c, cols].fillna(0).to_numpy() for k, c in codes.items()}

    out_dir = data_path(cfg, "processed", "predictions")
    out_dir.mkdir(exist_ok=True)
    results = {}

    def save(variant, pred, note):
        results[variant] = pred
        df = pd.concat([pd.DataFrame({"variant": variant, "round": r, "split": k, "muni_code": codes[k],
                                      "y_true": y[k], "y_pred": pred[k], "ens_min": pred[k], "ens_max": pred[k],
                                      "note": note}) for k in ("val", "test")], ignore_index=True)
        df.to_parquet(out_dir / f"{variant}_r{r}.parquet", index=False)
        rmse = np.sqrt(np.mean((pred["test"] - y["test"]) ** 2))
        print(f"{variant:<18} test RMSE {rmse:.3f}  ({note})")

    base = ["log_cells", "log_pop"]
    pred, form = bl.ntl_only({k: m.loc[c, "ntl_x"].to_numpy() for k, c in codes.items()}, y)
    save("V1", pred, form)
    pred, a = bl.ridge(feats(lc + base), y)
    save("V2", pred, f"alpha={a:g}")
    pred, a = bl.ridge(feats(GSED + base), y)
    save("V11_gsed", pred, f"alpha={a:g}")

    for f in sorted(data_path(cfg, "interim", f"round{r}").glob("*.npy")):
        if not f.stem.startswith(("emb_A_", "z_")):
            continue
        emb = np.load(f).astype(np.float32)
        cols = [f"{f.stem}_{i}" for i in range(emb.shape[1])]
        means = pd.DataFrame(emb, columns=cols).assign(muni_code=kept["muni_code"]).groupby("muni_code").mean()
        mm = m[base].join(means)
        pred, a = bl.ridge({k: mm.loc[c, cols + base].fillna(0).to_numpy() for k, c in codes.items()}, y)
        save(f"V11_{f.stem}", pred, f"alpha={a:g}")

    try:
        save("V12", bl.lightgbm(feats(GSED + base), y, seed), "lightgbm")
    except ImportError:
        print("V12 skipped: lightgbm not installed")

    # V10: best V3-V9 run of this round by validation MSE, blended with V1
    runs = {}
    for v in [f"V{i}" for i in range(3, 10)]:
        f = out_dir / f"{v}_r{r}.parquet"
        if f.exists():
            p = pd.read_parquet(f).set_index(["split", "muni_code"])["y_pred"]
            runs[v] = {k: p.loc[k].reindex(codes[k]).to_numpy() for k in ("val", "test")}
    if runs:
        best = min(runs, key=lambda v: np.mean((runs[v]["val"] - y["val"]) ** 2))
        pred, a = bl.blend(runs[best], results["V1"], y["val"], cfg["hybrid"]["alphas"])
        save("V10", pred, f"base={best} alpha={a:g}")
    else:
        print("V10 skipped: no V3-V9 predictions for this round yet")


if __name__ == "__main__":
    main()
