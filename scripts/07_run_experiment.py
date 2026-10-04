"""Phases 5-7: run Stage A -> B -> C for one variant and round, caching each stage's output.

Usage:
  python scripts/07_run_experiment.py --variant V7 --round 2
  python scripts/07_run_experiment.py --variant all --round 2      # V3..V9, reusing the round's caches
  for r in 0 1 2 3 4; do
    scripts/run_bg.sh exp_r$r python scripts/07_run_experiment.py --variant all --round $r --threads 8
  done
Run rounds in parallel, not variants of the same round: those share Stage A/B caches.

Inputs (DATA_DIR/processed):
  cells.parquet             Phase 2.11: cell_id, muni_code, keep, gsed_00..gsed_63, log_ntl, poi_*, road_*_km
  texts.parquet             Phase 4.11: cell_id, text_full, text_raw, text_facts (only for V5-V9)
  municipal_labels.parquet  Phase 1: muni_code, log_pop, log_gdp_pc
  folds.parquet             Phase 3
Outputs:
  interim/text_emb_{text}.npy                    frozen sentence embeddings, rows = texts.parquet rows
  interim/round{r}/emb_A_{text}.npy, z_{input}.npy, ntl_pred_{input}.npy   per kept cell, cells.parquet order
  models/round{r}/gsed_scaler.json, stage_{a,b}_*.{pt,json}, stage_c_{variant}_seed{s}.pt
  processed/predictions/{variant}_r{r}.parquet   val + test rows: y_true, y_pred (5-seed mean), ens_min, ens_max
  reports/tables/stage_c_tuning/{variant}_r{r}.csv
Test-fold municipalities never enter training, early stopping or tuning: they are only predicted.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gisecon.config import REPO_ROOT, data_path, ensure_dirs, load_config
from gisecon.eval.folds import get_split
from gisecon.models.stage_a import embed_cells as embed_a, recall_at_k, train_stage_a
from gisecon.models.stage_b import embed_cells as embed_b, train_stage_b
from gisecon.models.stage_c import MuniSet, fit_calibration, predict, train_stage_c

VARIANTS = {  # (Stage A text variant or None for raw GSED, Stage B on, POI/roads appended) - PLAN 8.4
    "V3": (None, False, False),
    "V4": (None, True, False),
    "V5": ("full", False, False),
    "V6": ("full", True, False),
    "V7": ("full", True, True),
    "V8": ("raw", True, True),
    "V9": ("facts", True, True),
}
GSED = [f"gsed_{i:02d}" for i in range(64)]


def standardize(a: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale with training-cell mean and sd only."""
    mu, sd = a[train].mean(0), a[train].std(0)
    sd[sd == 0] = 1.0
    return ((a - mu) / sd).astype(np.float32), mu, sd


def text_embeddings(cfg: dict, texts: pd.DataFrame, text: str) -> np.ndarray:
    f = data_path(cfg, "interim", f"text_emb_{text}.npy")
    if not f.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg["text"]["sentence_model"])
        np.save(f, model.encode(texts[f"text_{text}"].tolist(), batch_size=128, show_progress_bar=True))
    return np.load(f).astype(np.float32)


def stage_a(cfg, r, text, cells, part, x_gsed) -> np.ndarray:
    out = data_path(cfg, "interim", f"round{r}", f"emb_A_{text}.npy")
    if out.exists():
        return np.load(out).astype(np.float32)
    texts = pd.read_parquet(data_path(cfg, "processed", "texts.parquet"))
    temb = text_embeddings(cfg, texts, text)
    pos = texts["cell_id"].map(pd.Series(np.arange(len(cells)), index=cells["cell_id"]))
    ok = pos.notna().to_numpy()  # captioned cells that were later masked are skipped
    pos, temb = pos[ok].astype(int).to_numpy(), torch.from_numpy(temb[ok])
    tr, va = part[pos] == "train", part[pos] == "val"  # test-fold captions are never used
    X = torch.from_numpy(x_gsed)
    torch.manual_seed(cfg["project"]["seed"] + r)
    model, log = train_stage_a(X[pos[tr]], temb[tr], X[pos[va]], temb[va], cfg["stage_a"])
    diag = {"n_train": int(tr.sum()), "n_val": int(va.sum()), **recall_at_k(model, X[pos[va]], temb[va])}
    print(f"  stage A {text}: {diag}")
    mdir = data_path(cfg, "models", f"round{r}")
    torch.save(model.state_dict(), mdir / f"stage_a_{text}.pt")
    (mdir / f"stage_a_{text}.json").write_text(json.dumps({**diag, "log": log}))
    emb = embed_a(model, X).numpy()
    np.save(out, emb.astype(np.float16))
    return emb


def stage_b(cfg, r, name, x, part, log_ntl) -> np.ndarray:
    idir = data_path(cfg, "interim", f"round{r}")
    out = idir / f"z_{name}.npy"
    if out.exists():
        return np.load(out).astype(np.float32)
    y, mu, sd = standardize(log_ntl[:, None], part == "train")
    tr, va = part == "train", part == "val"
    X, Y = torch.from_numpy(x), torch.from_numpy(y[:, 0])
    torch.manual_seed(cfg["project"]["seed"] + r)
    model, log = train_stage_b(X[tr], Y[tr], X[va], Y[va], cfg["stage_b"])
    z, pred = (t.numpy() for t in embed_b(model, X))
    ntl_pred = pred * sd[0] + mu[0]  # back to log NTL
    err = ntl_pred[va] - log_ntl[va]
    diag = {"val_r2": float(1 - (err ** 2).mean() / log_ntl[va].var()), "val_rmse": float(np.sqrt((err ** 2).mean())),
            "zero_light_share_train": float((log_ntl[tr] == 0).mean())}
    print(f"  stage B {name}: {diag}")
    mdir = data_path(cfg, "models", f"round{r}")
    torch.save(model.state_dict(), mdir / f"stage_b_{name}.pt")
    (mdir / f"stage_b_{name}.json").write_text(json.dumps({**diag, "log": log}))
    np.save(idir / f"ntl_pred_{name}.npy", ntl_pred.astype(np.float32))
    np.save(out, z.astype(np.float16))
    return z


def stage_c(cfg, r, variant, x, cells, labels, split, part) -> pd.DataFrame:
    X = torch.from_numpy(standardize(x, part == "train")[0])

    def muniset(codes) -> MuniSet:
        pos = pd.Series(np.arange(len(codes)), index=codes)
        m = cells["muni_code"].isin(codes).to_numpy()
        lab = labels.loc[codes]
        return MuniSet(X[m], torch.from_numpy(cells["muni_code"][m].map(pos).to_numpy()),
                       torch.tensor(lab["log_pop"].to_numpy(), dtype=torch.float32),
                       torch.tensor(lab["log_gdp_pc"].to_numpy(), dtype=torch.float32))

    sets = {k: muniset(getattr(split, k)) for k in ("train", "val", "test")}
    c, seed = cfg["stage_c"], cfg["project"]["seed"]

    # 7.5 tuning grid, chosen on validation MSE only
    rows, t0 = [], time.time()
    n_settings = int(np.prod([len(c["grid"][k]) for k in ("hidden", "weight_decay", "dropout", "pooling")]))
    for hidden, wd, dropout, pooling in product(*(c["grid"][k] for k in ("hidden", "weight_decay", "dropout", "pooling"))):
        torch.manual_seed(seed + r)
        m, log = train_stage_c(sets["train"], sets["val"], c, hidden, wd, dropout, pooling)
        mse = F.mse_loss(predict(m, sets["val"]), sets["val"].y).item()
        rows.append({"hidden": hidden, "weight_decay": wd, "dropout": dropout, "pooling": pooling,
                     "epochs": len(log), "val_mse": mse})
        print(f"    setting {len(rows)}/{n_settings}: hidden {hidden}, wd {wd:g}, dropout {dropout}, {pooling}"
              f" -> val MSE {mse:.4f} ({len(log)} epochs, {time.time() - t0:.0f} s)", flush=True)
    tuning = pd.DataFrame(rows)
    tdir = REPO_ROOT / cfg["paths"]["reports"] / "tables" / "stage_c_tuning"
    tdir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(tdir / f"{variant}_r{r}.csv", index=False)
    best = min(rows, key=lambda d: d["val_mse"])
    print(f"  stage C best: {best}")

    # 7.6 5-seed ensemble of the chosen setting
    preds = {k: [] for k in ("val", "test")}
    for s in range(c["n_seeds"]):
        torch.manual_seed(seed + 1000 * (s + 1) + r)
        m, _ = train_stage_c(sets["train"], sets["val"], c, best["hidden"], best["weight_decay"],
                             best["dropout"], best["pooling"])
        torch.save(m.state_dict(), data_path(cfg, "models", f"round{r}", f"stage_c_{variant}_seed{s}.pt"))
        print(f"    seed {s + 1}/{c['n_seeds']} done ({time.time() - t0:.0f} s)", flush=True)
        for k in preds:
            preds[k].append(predict(m, sets[k]).numpy())

    # recalibration a + b * prediction, fitted on validation only (Stage C is over-dispersed)
    pv = np.stack(preds["val"]).mean(0)
    a, b = fit_calibration(pv, sets["val"].y.numpy())
    print(f"  calibration on validation: official = {a:.3f} + {b:.3f} * predicted", flush=True)

    # 7.9 one prediction format; y_pred is calibrated, y_pred_raw is the ensemble mean before calibration
    out = []
    for k, p in preds.items():
        p = np.stack(p)
        lo, hi = a + b * p.min(0), a + b * p.max(0)
        out.append(pd.DataFrame({"variant": variant, "round": r, "split": k, "muni_code": getattr(split, k),
                                 "y_true": sets[k].y.numpy(), "y_pred": a + b * p.mean(0),
                                 "ens_min": np.minimum(lo, hi), "ens_max": np.maximum(lo, hi),
                                 "y_pred_raw": p.mean(0), "calib_a": a, "calib_b": b}))
    return pd.concat(out, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=[*VARIANTS, "all"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--config")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    r = args.round
    for stage in ("interim", "models"):
        data_path(cfg, stage, f"round{r}").mkdir(exist_ok=True)
    data_path(cfg, "processed", "predictions").mkdir(exist_ok=True)

    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"))
    cells = cells[cells["keep"]].reset_index(drop=True)
    labels = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    split = get_split(pd.read_parquet(data_path(cfg, "processed", "folds.parquet")), r)
    roles = {**{m: "train" for m in split.train}, **{m: "val" for m in split.val}, **{m: "test" for m in split.test}}
    part = cells["muni_code"].map(roles).to_numpy()
    assert not pd.isna(part).any(), "cells.parquet has municipalities missing from folds.parquet"
    print(f"round {r}: {len(cells):,} kept cells; municipalities train {len(split.train)}, "
          f"val {len(split.val)}, test {len(split.test)}")

    x_gsed, mu, sd = standardize(cells[GSED].to_numpy(np.float32), part == "train")
    data_path(cfg, "models", f"round{r}", "gsed_scaler.json").write_text(
        json.dumps({"columns": GSED, "mean": mu.tolist(), "sd": sd.tolist()}))
    extra = [c for c in cells if c.startswith("poi_") or (c.startswith("road_") and c.endswith("_km"))]

    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        text, use_b, poi = VARIANTS[variant]
        print(f"{variant} round {r}: Stage A {text or 'off'}, Stage B {'on' if use_b else 'off'}, "
              f"POI/roads {'on' if poi else 'off'}")
        x = x_gsed if text is None else stage_a(cfg, r, text, cells, part, x_gsed)
        if use_b:
            x = stage_b(cfg, r, "gsed" if text is None else f"A_{text}", x, part, cells["log_ntl"].to_numpy(np.float32))
        if poi:
            x = np.hstack([x, np.log1p(cells[extra].to_numpy(np.float32))])
        preds = stage_c(cfg, r, variant, x, cells, labels, split, part)
        out = data_path(cfg, "processed", "predictions", f"{variant}_r{r}.parquet")
        preds.to_parquet(out, index=False)
        test = preds[preds["split"] == "test"]
        print(f"  wrote {out.name}: test RMSE {np.sqrt(((test.y_pred - test.y_true) ** 2).mean()):.3f}")


if __name__ == "__main__":
    main()
