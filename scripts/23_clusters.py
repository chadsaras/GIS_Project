"""Can Stage C's municipal vectors group municipalities into economic types? (LIGHT: minutes, no training)

Run after 07_run_experiment.py. Spaces compared (one vector per municipality):
  Stage C V7 pooled, size removed  - pooled 16-d vector minus its linear fit on log population (fitted on training
                                     municipalities); the best economic space in 20_stage_c_twins.py
  Stage C V7 pooled                - as trained (mostly economy size)
  raw GSED mean                    - mean satellite embedding (baseline)
  land cover + lights              - interpretable: mean WorldCover shares, mean log night light, log cell count
  population only                  - log population (size baseline)

1. Out-of-sample income test, per round and seed: k-means (k = 6, 8) on the round's TRAINING municipalities, test
   municipalities assigned to the nearest centre; a test municipality's prediction is the mean official log GDP per
   capita of the training municipalities in its cluster. R² pooled over the 853 test predictions (and the same for
   log population, to see whether a grouping is only about size).
2. Stability: adjusted Rand index between the clusterings of all 853 municipalities from different seeds of a round
   and from different rounds (Stage C spaces differ per model).
3. Consensus types for the application: co-association over the 25 Stage C models, average-linkage clusters, and a
   profile of each type (descriptive: most vectors there come from models that trained on those municipalities).

Outputs: reports/tables/clusters.md, clusters_scores.csv, cluster_types.csv (municipality -> type)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.eval.folds import get_split
from gisecon.models.stage_c import StageC

KS = (6, 8)
LC = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water"]
EXTRA = [f"poi_{p}" for p in ("education", "health", "finance", "fuel", "food", "public", "industry", "office",
                              "retail", "tourism")] + [f"road_{k}_km" for k in ("major", "medium", "minor", "track")]


def zscore(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd == 0] = 1.0
    return ((a - mu) / sd).astype(np.float32)


@torch.no_grad()
def pooled_vectors(m: StageC, X: torch.Tensor, idx: torch.Tensor, n: int) -> np.ndarray:
    """The pooled vector that Stage C's municipal network reads (StageC.forward)."""
    m.eval()
    c = m.cell(X)
    p = torch.zeros(n, c.shape[1]).index_add_(0, idx, c)
    if m.pooling == "logsum":
        p = torch.log1p(p)
    elif m.pooling == "mean":
        cnt = torch.bincount(idx, minlength=n).clamp(min=1).float()
        p = torch.cat([p / cnt[:, None], torch.log(cnt)[:, None]], dim=1)
    return p.numpy()


def size_removed(v: np.ndarray, log_pop: np.ndarray, train: np.ndarray) -> np.ndarray:
    A = np.c_[np.ones(len(v)), log_pop]
    coef = np.linalg.lstsq(A[train], v[train], rcond=None)[0]
    return v - A @ coef


def r2(p: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cluster_predict(v: np.ndarray, tr: np.ndarray, te: np.ndarray, k: int, seed: int, targets: dict) -> dict:
    """Fit k-means on training rows, assign test rows, predict each target by its cluster's training mean."""
    z = zscore(v, v[tr])
    km = KMeans(k, n_init=10, random_state=seed).fit(z[tr])
    lab_tr, lab_te = km.labels_, km.predict(z[te])
    out = {}
    for name, y in targets.items():
        means = pd.Series(y[tr]).groupby(lab_tr).mean()
        out[name] = means.reindex(lab_te).fillna(y[tr].mean()).to_numpy()
    return out


def main() -> None:
    cfg = load_config()
    tab = REPO_ROOT / cfg["paths"]["reports"] / "tables"
    n_rounds, n_seeds, c = cfg["cv"]["n_folds"], cfg["stage_c"]["n_seeds"], cfg["stage_c"]
    seed = cfg["project"]["seed"]
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"),
                            columns=["muni_code", "keep", "log_ntl", *LC, *GSED, *EXTRA])
    cells = cells[cells["keep"]].reset_index(drop=True)
    codes = np.array(sorted(cells.muni_code.unique()))
    pos = pd.Series(np.arange(len(codes)), index=codes)
    idx = torch.from_numpy(cells.muni_code.map(pos).to_numpy())
    y_pc, y_pop = lab.loc[codes, "log_gdp_pc"].to_numpy(), lab.loc[codes, "log_pop"].to_numpy()
    g = cells.groupby("muni_code")
    fixed = {"raw GSED mean": g[GSED].mean().loc[codes].to_numpy(),
             "land cover + lights": np.c_[g[LC].mean().loc[codes].to_numpy(), g["log_ntl"].mean().loc[codes].to_numpy(),
                                          np.log(g.size().loc[codes].to_numpy())],
             "population only": y_pop[:, None]}
    extra = np.log1p(cells[EXTRA].to_numpy(np.float32))

    preds = {}  # (space, k, seed) -> {"pc": [...], "pop": [...], "rows": [...]}
    labels_all = {k: [] for k in KS}  # consensus input: k-means on all municipalities, per Stage C model
    per_model = {}  # (round, seed) -> size-removed vectors of all municipalities
    for r in range(n_rounds):
        split = get_split(folds, r)
        tr, te = pos[list(split.train)].to_numpy(), pos[list(split.test)].to_numpy()
        part = cells.muni_code.map({**{m: "train" for m in split.train}, **{m: "val" for m in split.val},
                                    **{m: "test" for m in split.test}}).to_numpy()
        z = np.load(data_path(cfg, "interim", f"round{r}", "z_A_full.npy")).astype(np.float32)
        x = np.hstack([z, extra])  # V7 input (07_run_experiment.py)
        X = torch.from_numpy(zscore(x, x[part == "train"]))
        tune = pd.read_csv(REPO_ROOT / "reports" / "tables" / "stage_c_tuning" / f"V7_r{r}.csv")
        b = tune.loc[tune.val_mse.idxmin()]
        spaces = dict(fixed)
        for s in range(n_seeds):
            m = StageC(X.shape[1], int(b.hidden), c["cell_out"], c["muni_hidden"], float(b.dropout), b.pooling)
            m.load_state_dict(torch.load(data_path(cfg, "models", f"round{r}", f"stage_c_V7_seed{s}.pt")))
            pv = pooled_vectors(m, X, idx, len(codes))
            sr = size_removed(pv, y_pop, tr)
            per_model[(r, s)] = sr
            for name, v in (("Stage C V7 pooled", pv), ("Stage C V7 pooled, size removed", sr)):
                for k in KS:
                    p = cluster_predict(v, tr, te, k, seed + s, {"pc": y_pc, "pop": y_pop})
                    d = preds.setdefault((name, k, s), {"pc": [], "pop": [], "rows": []})
                    d["pc"].append(p["pc"]); d["pop"].append(p["pop"]); d["rows"].append(te)
        for name, v in fixed.items():
            for k in KS:
                p = cluster_predict(v, tr, te, k, seed, {"pc": y_pc, "pop": y_pop})
                d = preds.setdefault((name, k, "-"), {"pc": [], "pop": [], "rows": []})
                d["pc"].append(p["pc"]); d["pop"].append(p["pop"]); d["rows"].append(te)
        print(f"round {r} done", flush=True)

    rows = []
    for (name, k, s), d in preds.items():
        t = np.concatenate(d["rows"])
        rows.append({"space": name, "k": k, "seed": s, "r2_income_pc": r2(np.concatenate(d["pc"]), y_pc[t]),
                     "r2_population": r2(np.concatenate(d["pop"]), y_pop[t])})
    sc = pd.DataFrame(rows)

    # stability: k-means on all municipalities per Stage C model (size removed), ARI across seeds and rounds
    stab = []
    for k in KS:
        lab_m = {key: KMeans(k, n_init=10, random_state=seed).fit_predict(zscore(v, v)) for key, v in per_model.items()}
        labels_all[k] = list(lab_m.values())
        same_round = [adjusted_rand_score(lab_m[(r, a)], lab_m[(r, b)]) for r in range(n_rounds)
                      for a, b in itertools.combinations(range(n_seeds), 2)]
        cross_round = [adjusted_rand_score(lab_m[(r1, 0)], lab_m[(r2_, 0)])
                       for r1, r2_ in itertools.combinations(range(n_rounds), 2)]
        gsed_lab = KMeans(k, n_init=10, random_state=seed).fit_predict(zscore(fixed["raw GSED mean"], fixed["raw GSED mean"]))
        gsed_lab2 = KMeans(k, n_init=10, random_state=seed + 1).fit_predict(zscore(fixed["raw GSED mean"], fixed["raw GSED mean"]))
        stab.append({"k": k, "ari_same_round_seeds": float(np.mean(same_round)),
                     "ari_across_rounds": float(np.mean(cross_round)),
                     "ari_raw_gsed_two_inits": adjusted_rand_score(gsed_lab, gsed_lab2)})
    st = pd.DataFrame(stab)

    # consensus economic types (k = 6) from the 25 Stage C models
    k = 6
    L = np.array(labels_all[k])
    co = (L[:, :, None] == L[:, None, :]).mean(0)
    np.fill_diagonal(co, 1.0)
    types = fcluster(linkage(squareform(1 - co, checks=False), "average"), k, criterion="maxclust")
    conf = np.array([co[i, types == types[i]].mean() for i in range(len(codes))])  # how often a municipality sits with its type
    lc_mean = g[LC].mean().loc[codes]
    prof = pd.DataFrame({"type": types, "pop": lab.loc[codes, "pop"].to_numpy(), "gdp_pc": np.exp(y_pc),
                         "log_ntl": g["log_ntl"].mean().loc[codes].to_numpy(), "confidence": conf,
                         "name": lab.loc[codes, "muni_name"].to_numpy(), **{c_: lc_mean[c_].to_numpy() for c_ in LC}},
                        index=codes)
    prof.rename_axis("muni_code").reset_index().round(4).to_csv(tab / "cluster_types.csv", index=False)
    eta2 = 1 - prof.groupby("type")["gdp_pc"].transform(lambda s: np.log(s).mean()).sub(np.log(prof.gdp_pc)).pow(2).sum() \
        / ((np.log(prof.gdp_pc) - np.log(prof.gdp_pc).mean()) ** 2).sum()

    summ = (sc.groupby(["space", "k"], sort=False).agg(r2_income_pc=("r2_income_pc", "mean"),
                                                       r2_income_pc_min=("r2_income_pc", "min"),
                                                       r2_population=("r2_population", "mean")).reset_index())
    sc.round(4).to_csv(tab / "clusters_scores.csv", index=False)
    Lines = ["# Economic types from Stage C vectors: out-of-sample test (Minas Gerais)\n",
             "Clusters fitted on each round's training municipalities; a test municipality is predicted by its cluster's "
             "training mean. Stage C rows: mean (min) over 5 seeds. For reference, the GDP model itself reaches R² 0.50 "
             "(V7) and 0.52 (V10).\n",
             "| space | k | R² income per person | R² population |", "| --- | --- | --- | --- |"]
    for _, r in summ.iterrows():
        mn = f" ({r.r2_income_pc_min:.3f})" if r.space.startswith("Stage C") else ""
        Lines.append(f"| {r.space} | {r.k} | {r.r2_income_pc:.3f}{mn} | {r.r2_population:.3f} |")
    Lines += ["", "## Stability (adjusted Rand index; 1 = identical groups, 0 = chance)\n",
              "| k | Stage C size-removed, seeds of one round | Stage C size-removed, across rounds | raw GSED, two k-means starts |",
              "| --- | --- | --- | --- |"]
    Lines += [f"| {r.k} | {r.ari_same_round_seeds:.3f} | {r.ari_across_rounds:.3f} | {r.ari_raw_gsed_two_inits:.3f} |"
              for _, r in st.iterrows()]
    Lines += ["", f"## Consensus types (k = {k}, co-association over 25 Stage C models)\n",
              f"Descriptive only. Types explain {eta2:.0%} of income differences (in-sample).\n",
              "| type | towns | median people | median GDP per person (R$) | mean log light | built | crop | tree | grass | "
              "mean confidence | largest towns |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t, d in prof.groupby("type"):
        big = ", ".join(d.sort_values("pop", ascending=False).name.head(3))
        Lines.append(f"| {t} | {len(d)} | {d['pop'].median():,.0f} | {d.gdp_pc.median():,.0f} | {d.log_ntl.mean():.2f} | "
                     f"{d.lc_built.mean():.1%} | {d.lc_crop.mean():.1%} | {d.lc_tree.mean():.1%} | {d.lc_grass.mean():.1%} | "
                     f"{d.confidence.mean():.2f} | {big} |")
    (tab / "clusters.md").write_text("\n".join(Lines) + "\n")
    print("\n".join(Lines))


if __name__ == "__main__":
    main()
