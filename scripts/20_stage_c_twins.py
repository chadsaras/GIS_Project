"""Do Stage C's municipal vectors give economic similarity? Twin test on held-out municipalities (LIGHT: minutes).

Run after 07_run_experiment.py. For each round and seed, the saved Stage C models are re-applied (no training) to get
two 16-number vectors per municipality: the pooled cell contributions (the input of the municipal network) and the
municipal network's hidden layer. For every test municipality, its 10 nearest training municipalities ("twins") are
found in each space, and their official values are compared with its own:
  twin R² of log GDP per capita   - economic similarity, the claim being tested
  twin R² of log population       - is it only similar size? (Stage C predicts total GDP, then subtracts population)
Also "pooled, size removed": the pooled vector minus its linear fit on log population (fitted on training
municipalities; population is a model input anyway), to see whether per-capita similarity survives without size.
Baselines: twins by raw GSED mean (cosine), by log population alone. Reference: the variant's own model R².

Outputs: reports/tables/stage_c_twins.csv, reports/tables/stage_c_twins.md
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.eval.folds import get_split
from gisecon.models.stage_c import StageC

K = 10
INPUTS = {"V3": None, "V6": "z_A_full", "V7": "z_A_full"}  # Stage C input before standardization


def standardize(a: np.ndarray, train: np.ndarray) -> np.ndarray:
    mu, sd = a[train].mean(0), a[train].std(0)
    sd[sd == 0] = 1.0
    return ((a - mu) / sd).astype(np.float32)


@torch.no_grad()
def muni_vectors(model: StageC, x: torch.Tensor, idx: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pooled cell contributions and the municipal network's hidden layer, as in StageC.forward."""
    model.eval()
    c = model.cell(x)
    pooled = torch.zeros(n, c.shape[1]).index_add_(0, idx, c)
    if model.pooling == "logsum":
        pooled = torch.log1p(pooled)
    elif model.pooling == "mean":
        cnt = torch.bincount(idx, minlength=n).clamp(min=1).float()
        pooled = torch.cat([pooled / cnt[:, None], torch.log(cnt)[:, None]], dim=1)
    hidden = model.muni[1](model.muni[0](pooled))
    return pooled.numpy(), hidden.numpy()


def twins(v: np.ndarray, test: np.ndarray, train: np.ndarray, metric: str) -> np.ndarray:
    """Indices (into train) of the K nearest training rows for each test row."""
    if metric == "cosine":
        e = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
        d = -(e[test] @ e[train].T)
    else:  # euclidean on vectors standardized with training-municipality statistics
        mu, sd = v[train].mean(0), v[train].std(0)
        sd[sd == 0] = 1.0
        z = (v - mu) / sd
        d = ((z[test][:, None, :] - z[train][None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :K]


def r2(est: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((est - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(INPUTS))
    args = ap.parse_args()
    cfg = load_config()
    tab = REPO_ROOT / cfg["paths"]["reports"] / "tables"
    n_rounds, n_seeds = cfg["cv"]["n_folds"], cfg["stage_c"]["n_seeds"]
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"),
                            columns=["muni_code", "keep", *GSED,
                                     *[f"poi_{p}" for p in ("education", "health", "finance", "fuel", "food", "public",
                                                            "industry", "office", "retail", "tourism")],
                                     *[f"road_{k}_km" for k in ("major", "medium", "minor", "track")]])
    cells = cells[cells["keep"]].reset_index(drop=True)
    extra = [c for c in cells if c.startswith("poi_") or (c.startswith("road_") and c.endswith("_km"))]
    codes = np.array(sorted(cells.muni_code.unique()))
    pos = pd.Series(np.arange(len(codes)), index=codes)
    idx = torch.from_numpy(cells.muni_code.map(pos).to_numpy())
    y_pc, y_tot, y_pop = (lab.loc[codes, c].to_numpy() for c in ("log_gdp_pc", "log_gdp", "log_pop"))
    gsed_mean = cells.groupby("muni_code")[GSED].mean().loc[codes].to_numpy()

    rows = []

    def score(space: str, variant: str, seed, t: np.ndarray, nb: np.ndarray, model_r2: float = np.nan) -> None:
        """t: test rows; nb: (len(t), K) twin rows."""
        rec = {"space": space, "variant": variant, "seed": seed, "model_r2_pc": model_r2}
        for name, y in (("pc", y_pc), ("total", y_tot), ("pop", y_pop)):
            rec[f"twin_r2_{name}"] = r2(y[nb].mean(1), y[t])
        rec["median_twin_sd_pc"] = float(np.median(y_pc[nb].std(1)))
        rows.append(rec)

    # baselines, independent of Stage C
    base = {"raw GSED mean (cosine)": ([], []), "population only": ([], [])}
    for r in range(n_rounds):
        s = get_split(folds, r)
        te, tr = pos[list(s.test)].to_numpy(), pos[list(s.train)].to_numpy()
        for name, nb in (("raw GSED mean (cosine)", tr[twins(gsed_mean, te, tr, "cosine")]),
                         ("population only", tr[np.argsort(np.abs(y_pop[te][:, None] - y_pop[tr][None, :]), axis=1)[:, :K]])):
            base[name][0].append(te)
            base[name][1].append(nb)
    for name, (ts, nbs) in base.items():
        score(name, "-", "-", np.concatenate(ts), np.concatenate(nbs))

    for v in args.variants:
        preds = pd.concat([pd.read_parquet(data_path(cfg, "processed", "predictions", f"{v}_r{r}.parquet"))
                           for r in range(n_rounds)])
        test = preds[preds.split == "test"]
        model_r2 = r2(test.y_pred.to_numpy(), test.y_true.to_numpy())
        per_seed = {sp: {s: ([], []) for s in range(n_seeds)} for sp in ("pooled", "hidden", "pooled, size removed")}
        for r in range(n_rounds):
            split = get_split(folds, r)
            part = cells.muni_code.map({**{m: "train" for m in split.train}, **{m: "val" for m in split.val},
                                        **{m: "test" for m in split.test}}).to_numpy()
            x = cells[GSED].to_numpy(np.float32)
            x = standardize(x, part == "train")  # 07: GSED scaled on training cells
            if INPUTS[v]:
                x = np.load(data_path(cfg, "interim", f"round{r}", f"{INPUTS[v]}.npy")).astype(np.float32)
            if v == "V7":
                x = np.hstack([x, np.log1p(cells[extra].to_numpy(np.float32))])
            X = torch.from_numpy(standardize(x, part == "train"))  # 07 stage_c: scaled again on training cells
            tune = pd.read_csv(REPO_ROOT / cfg["paths"]["reports"] / "tables" / "stage_c_tuning" / f"{v}_r{r}.csv")
            b = tune.loc[tune.val_mse.idxmin()]
            te, tr = pos[list(split.test)].to_numpy(), pos[list(split.train)].to_numpy()
            for s in range(n_seeds):
                m = StageC(X.shape[1], int(b.hidden), cfg["stage_c"]["cell_out"], cfg["stage_c"]["muni_hidden"],
                           float(b.dropout), b.pooling)
                m.load_state_dict(torch.load(data_path(cfg, "models", f"round{r}", f"stage_c_{v}_seed{s}.pt")))
                pooled, hidden = muni_vectors(m, X, idx, len(codes))
                # size removed: residual of each component on [1, log population], fitted on training municipalities
                A = np.c_[np.ones(len(codes)), y_pop]
                coef = np.linalg.lstsq(A[tr], pooled[tr], rcond=None)[0]
                for sp, vec in (("pooled", pooled), ("hidden", hidden), ("pooled, size removed", pooled - A @ coef)):
                    per_seed[sp][s][0].append(te)
                    per_seed[sp][s][1].append(tr[twins(vec, te, tr, "euclidean")])
            print(f"{v} round {r} done", flush=True)
        for sp in per_seed:
            for s, (ts, nbs) in per_seed[sp].items():
                score(f"Stage C {sp}", v, s, np.concatenate(ts), np.concatenate(nbs), model_r2)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(tab / "stage_c_twins.csv", index=False)
    summ = (res.groupby(["space", "variant"], sort=False)
               .agg(twin_r2_pc=("twin_r2_pc", "mean"), twin_r2_pc_min=("twin_r2_pc", "min"),
                    twin_r2_total=("twin_r2_total", "mean"), twin_r2_pop=("twin_r2_pop", "mean"),
                    median_twin_sd_pc=("median_twin_sd_pc", "mean"), model_r2_pc=("model_r2_pc", "first"))
               .reset_index())
    lines = ["# Twins in Stage C space vs baselines (10 nearest training municipalities, test municipalities)\n",
             f"Official log GDP per capita: sd {y_pc.std():.3f}. Stage C rows: mean (min) over {n_seeds} seeds.\n",
             "| space | variant | twin R² GDP per capita | twin R² total GDP | twin R² population | "
             "median twin spread (pc) | model R² (pc) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in summ.iterrows():
        mr = "" if pd.isna(r.model_r2_pc) else f"{r.model_r2_pc:.3f}"
        lo = "" if r.space.startswith(("raw", "pop")) else f" ({r.twin_r2_pc_min:.3f})"
        lines.append(f"| {r.space} | {r.variant} | {r.twin_r2_pc:.3f}{lo} | {r.twin_r2_total:.3f} | "
                     f"{r.twin_r2_pop:.3f} | {r.median_twin_sd_pc:.3f} | {mr} |")
    (tab / "stage_c_twins.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
