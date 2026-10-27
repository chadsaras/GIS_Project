"""Transfer test: apply the models trained on Minas Gerais to another state, unchanged (LIGHT: minutes).

  GISECON_CONFIG=configs/config_sp.yaml python scripts/21_transfer.py
Needs the target state's processed/cells.parquet and municipal_labels.parquet (01, 02, 03, 05, 05b, 06 run with its
config) and the source run's models, caches and predictions in transfer.source_data_dir. Nothing is refitted on the
target state: every scaler, model, calibration line, hybrid weight, error range and risk model comes from Minas Gerais.
Each of the 5 cross-validation rounds gives one set of models; the target prediction is their average.

Outputs (the target config's reports folder):
  tables/transfer_metrics.csv, tables/transfer_results.md   accuracy against the target state's official GDP
  tables/transfer_municipalities.csv                         one row per municipality (official, predictions, ranges)
  processed/transfer_activity_1km.parquet (target data dir)  Stage B predicted vs observed log night light per cell
"""
from __future__ import annotations

import copy
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.eval.folds import get_split
from gisecon.models.stage_a import StageA, embed_cells as embed_a
from gisecon.models.stage_b import StageB, embed_cells as embed_b
from gisecon.models.stage_c import MuniSet, StageC, predict

VARIANTS = {"V3": (None, False, False), "V4": (None, True, False), "V5": ("full", False, False),
            "V6": ("full", True, False), "V7": ("full", True, True), "V8": ("raw", True, True),
            "V9": ("facts", True, True)}  # as in 07_run_experiment.py
EXTRA = [f"poi_{p}" for p in ("education", "health", "finance", "fuel", "food", "public", "industry", "office",
                              "retail", "tourism")] + [f"road_{k}_km" for k in ("major", "medium", "minor", "track")]


def standardize_with(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd == 0] = 1.0
    return ((a - mu) / sd).astype(np.float32)


def as_cached(a: np.ndarray) -> np.ndarray:
    """The source run stored Stage A/B outputs as float16; round-trip the target's the same way."""
    return a.astype(np.float16).astype(np.float32)


def ntl_x(cells: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    """V1 regressor: log(1 + total NTL) - log population (08_baselines.py)."""
    total = cells.groupby("muni_code")["ntl"].sum().reindex(lab.index, fill_value=0)
    return np.log1p(total) - lab["log_pop"]


def signals(cells: pd.DataFrame) -> pd.DataFrame:
    """Satellite-only warning signals (19_app_reliability.py)."""
    g = cells.groupby("muni_code")
    ntl_sum = g["ntl"].sum()
    return pd.DataFrame({"bare_share": g["lc_bare"].mean(), "built_share": g["lc_built"].mean(),
                         "ntl_top_share": (g["ntl"].max() / ntl_sum.where(ntl_sum > 0)).fillna(0.0),
                         "log_n_cells": np.log(g.size())})


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {"r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
            "r2_after_bias": float(1 - ((e - e.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "spearman": float(spearmanr(y, p)[0]), "calib_slope": float(np.polyfit(p, y, 1)[0])}


def main() -> None:
    cfg = load_config()
    src = copy.deepcopy(cfg)
    src["paths"]["data_dir"] = os.path.expanduser(os.path.expandvars(cfg["transfer"]["source_data_dir"]))
    rep = REPO_ROOT / cfg["paths"]["reports"] / "tables"
    rep.mkdir(parents=True, exist_ok=True)
    tuning_dir = REPO_ROOT / "reports" / "tables" / "stage_c_tuning"  # the source run's tuning tables
    n_rounds, n_seeds, c = src["cv"]["n_folds"], src["stage_c"]["n_seeds"], src["stage_c"]
    state = cfg["study"]["state_abbr"]

    cols = ["cell_id", "row", "col", "muni_code", "keep", "ntl", "log_ntl", "lc_bare", "lc_built", *GSED, *EXTRA]
    tc = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"), columns=cols)
    tc = tc[tc["keep"]].reset_index(drop=True)
    tlab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    sc = pd.read_parquet(data_path(src, "processed", "cells.parquet"), columns=cols)
    sc = sc[sc["keep"]].reset_index(drop=True)
    slab = pd.read_parquet(data_path(src, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(src, "processed", "folds.parquet"))

    codes = np.array(sorted(tc.muni_code.unique()))
    pos = pd.Series(np.arange(len(codes)), index=codes)
    tset_idx = torch.from_numpy(tc.muni_code.map(pos).to_numpy())
    t_logpop = torch.tensor(tlab.loc[codes, "log_pop"].to_numpy(), dtype=torch.float32)
    y = tlab.loc[codes, "log_gdp_pc"].to_numpy()
    print(f"{state}: {len(tc):,} kept cells, {len(codes)} municipalities; source: {len(sc):,} cells")

    t_gsed_raw, s_gsed_raw = tc[GSED].to_numpy(np.float32), sc[GSED].to_numpy(np.float32)
    t_extra, s_extra = np.log1p(tc[EXTRA].to_numpy(np.float32)), np.log1p(sc[EXTRA].to_numpy(np.float32))
    t_ntlx, s_ntlx = ntl_x(tc, tlab).loc[codes].to_numpy(), ntl_x(sc, slab)
    preds = {v: [] for v in [*VARIANTS, "V1", "V10"]}
    act = []

    for r in range(n_rounds):
        mdir, idir = data_path(src, "models", f"round{r}"), data_path(src, "interim", f"round{r}")
        split = get_split(folds, r)
        roles = {**{m: "train" for m in split.train}, **{m: "val" for m in split.val}, **{m: "test" for m in split.test}}
        s_train = (sc.muni_code.map(roles) == "train").to_numpy()
        scaler = json.loads((mdir / "gsed_scaler.json").read_text())
        t_gsed = ((t_gsed_raw - np.array(scaler["mean"])) / np.array(scaler["sd"])).astype(np.float32)
        s_gsed = ((s_gsed_raw - np.array(scaler["mean"])) / np.array(scaler["sd"])).astype(np.float32)
        ntl_mu, ntl_sd = sc.log_ntl[s_train].mean(), sc.log_ntl[s_train].std(ddof=0)

        t_in, s_in = {"gsed": t_gsed}, {"gsed": s_gsed}  # Stage A/B outputs for target (fresh) and source (cached)
        for text in ("full", "raw", "facts"):
            a = StageA(64, 768, **{k: src["stage_a"][k] for k in ("proj_hidden", "emb_dim", "dropout", "temperature_init")})
            a.load_state_dict(torch.load(mdir / f"stage_a_{text}.pt"))
            t_in[f"A_{text}"] = as_cached(embed_a(a, torch.from_numpy(t_gsed)).numpy())
            s_in[f"A_{text}"] = np.load(idir / f"emb_A_{text}.npy").astype(np.float32)
        for name in ("gsed", "A_full", "A_raw", "A_facts"):
            b = StageB(t_in[name].shape[1], tuple(src["stage_b"]["hidden"]))
            b.load_state_dict(torch.load(mdir / f"stage_b_{name}.pt"))
            z, p = (t.numpy() for t in embed_b(b, torch.from_numpy(t_in[name])))
            t_in[f"z_{name}"] = as_cached(z)
            s_in[f"z_{name}"] = np.load(idir / f"z_{name}.npy").astype(np.float32)
            if name == "A_full":  # 1 km activity map: Stage B predicted log NTL, back on the source's scale
                act.append(p * ntl_sd + ntl_mu)

        for v, (text, use_b, poi) in VARIANTS.items():
            key = "gsed" if text is None else f"A_{text}"
            key = f"z_{key}" if use_b else key
            xt, xs = t_in[key], s_in[key]
            if poi:
                xt, xs = np.hstack([xt, t_extra]), np.hstack([xs, s_extra])
            X = torch.from_numpy(standardize_with(xt, xs[s_train]))  # Stage C scaler: source training cells
            tset = MuniSet(X, tset_idx, t_logpop)
            tune = pd.read_csv(tuning_dir / f"{v}_r{r}.csv")
            best = tune.loc[tune.val_mse.idxmin()]
            raw = []
            for s in range(n_seeds):
                m = StageC(X.shape[1], int(best.hidden), c["cell_out"], c["muni_hidden"], float(best.dropout), best.pooling)
                m.load_state_dict(torch.load(mdir / f"stage_c_{v}_seed{s}.pt"))
                with torch.no_grad():
                    raw.append(predict(m, tset).numpy())
            sp = pd.read_parquet(data_path(src, "processed", "predictions", f"{v}_r{r}.parquet"))
            preds[v].append(sp.calib_a.iloc[0] + sp.calib_b.iloc[0] * np.mean(raw, axis=0))

        # V1 refitted on the source round's training municipalities exactly as 08_baselines.py (form chosen on val)
        sy = slab["log_gdp_pc"]
        best = None
        for deg in (1, 2):
            coef = np.polyfit(s_ntlx.loc[split.train], sy.loc[split.train], deg)
            mse = float(np.mean((np.polyval(coef, s_ntlx.loc[split.val]) - sy.loc[split.val]) ** 2))
            best = (mse, coef) if best is None or mse < best[0] else best
        preds["V1"].append(np.polyval(best[1], t_ntlx))
        note = pd.read_parquet(data_path(src, "processed", "predictions", f"V10_r{r}.parquet")).note.iloc[0]
        base, alpha = re.match(r"base=(V\d) alpha=([\d.]+)", note).groups()
        preds["V10"].append(float(alpha) * preds[base][-1] + (1 - float(alpha)) * preds["V1"][-1])
        print(f"round {r}: done ({note})", flush=True)

    P = {v: np.mean(p, axis=0) for v, p in preds.items()}
    metrics = pd.DataFrame([{"variant": v, **scores(y, P[v])} for v in ["V1", *VARIANTS, "V10"]])
    metrics.round(4).to_csv(rep / "transfer_metrics.csv", index=False)

    # error ranges and risk groups exactly as set on the source state's out-of-sample V10 errors (19_app_reliability.py)
    s_pred = pd.concat([pd.read_parquet(data_path(src, "processed", "predictions", f"V10_r{r}.parquet"))
                        for r in range(n_rounds)])
    s_test = s_pred[s_pred.split == "test"].set_index("muni_code")
    s_err = (s_test.y_pred - s_test.y_true).abs()
    q90 = float(np.quantile(s_err, 0.9))
    feats = ["bare_share", "built_share", "ntl_top_share", "log_n_cells", "pred"]
    s_sig = signals(sc).loc[s_test.index].assign(pred=s_test.y_pred)
    t_sig = signals(tc).loc[codes].assign(pred=P["V10"])
    scl = StandardScaler().fit(s_sig[feats])
    clf = LogisticRegression(max_iter=1000).fit(scl.transform(s_sig[feats]), (s_err > s_err.quantile(0.8)).astype(int))
    s_risk = clf.predict_proba(scl.transform(s_sig[feats]))[:, 1]
    cuts = np.quantile(s_risk, [1 / 3, 2 / 3])
    s_grp = np.digitize(s_risk, cuts)
    q_grp = [float(np.quantile(s_err.to_numpy()[s_grp == g], 0.9)) for g in range(3)]
    t_risk = clf.predict_proba(scl.transform(t_sig[feats]))[:, 1]
    t_grp = np.digitize(t_risk, cuts)
    t_err = np.abs(P["V10"] - y)
    rad = np.array(q_grp)[t_grp]

    # activity: does Stage B, trained on the source's night lights, predict the target's observed lights?
    a = np.mean(act, axis=0)
    obs = tc.log_ntl.to_numpy()
    act_r2 = float(1 - ((a - obs) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())
    pd.DataFrame({"cell_id": tc.cell_id, "row": tc.row, "col": tc.col, "muni_code": tc.muni_code,
                  "log_ntl_obs": obs, "log_ntl_pred": a}).to_parquet(data_path(cfg, "processed", "transfer_activity_1km.parquet"))

    out = pd.DataFrame({"muni_code": codes, "muni_name": tlab.loc[codes, "muni_name"].to_numpy(),
                        "pop": tlab.loc[codes, "pop"].to_numpy(), "official": y,
                        **{f"pred_{v}": P[v] for v in ("V1", "V3", "V7", "V10")},
                        "pred_V10_round_min": np.min(preds["V10"], axis=0), "pred_V10_round_max": np.max(preds["V10"], axis=0),
                        "lo90": P["V10"] - q90, "hi90": P["V10"] + q90, "risk": t_risk,
                        "risk_group": pd.Series(t_grp).map({0: "low", 1: "medium", 2: "high"}).to_numpy(),
                        "glo90": P["V10"] - rad, "ghi90": P["V10"] + rad,
                        "share_industry": tlab.loc[codes, "share_industry"].to_numpy(),
                        "flag_extreme": tlab.loc[codes, "flag_extreme"].to_numpy()})
    out.round(4).to_csv(rep / "transfer_municipalities.csv", index=False)

    src_res = pd.read_csv(REPO_ROOT / "reports" / "tables" / "main_results.csv").set_index("variant")
    L = [f"# Transfer test: models trained on Minas Gerais applied unchanged to {state}\n",
         f"{len(codes)} municipalities, {len(tc):,} kept 1 km cells. Each prediction is the mean of the 5 Minas Gerais "
         "rounds' models; nothing is refitted on the target state.\n",
         "| variant | R² | RMSE | Spearman (ranks) | mean bias | R² if bias removed | calibration slope | R² in MG (out-of-sample) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, m in metrics.iterrows():
        L.append(f"| {m.variant} | {m.r2:.3f} | {m.rmse:.3f} | {m.spearman:.3f} | {m.bias:+.3f} | {m.r2_after_bias:.3f} | "
                 f"{m.calib_slope:.2f} | {src_res.r2.get(m.variant, np.nan):.3f} |")
    cov = float((t_err <= q90).mean())
    L += ["", f"Minas Gerais 90% range (±{q90:.3f} log, ×/÷ {np.exp(q90):.2f}) holds for {cov:.1%} of {state} municipalities.",
          "", "| risk group | municipalities | mean abs. error | flagged (industry > 60% or top 1%) | MG 90% range | holds in target |",
          "| --- | --- | --- | --- | --- | --- |"]
    for g, name in enumerate(("low", "medium", "high")):
        s = t_grp == g
        L.append(f"| {name} | {int(s.sum())} | {t_err[s].mean():.3f} | {int(out.flag_extreme[s].sum())} | "
                 f"×/÷ {np.exp(q_grp[g]):.2f} | {(t_err[s] <= q_grp[g]).mean():.1%} |")
    L += ["", f"1 km activity (Stage B, A(full) input): predicted vs observed log night light, cell R² {act_r2:.3f} "
          f"(Spearman {spearmanr(a, obs)[0]:.3f}) over {len(tc):,} cells."]
    (rep / "transfer_results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
