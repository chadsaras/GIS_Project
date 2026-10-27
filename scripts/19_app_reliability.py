"""Reliability layer for an application: honest error ranges, satellite-only warnings, economic twins (LIGHT).

Run after 09_evaluate.py. Uses only the out-of-sample predictions already made (no retraining), and only
inputs a region WITHOUT official statistics would have: satellite layers, cell count and the model's own
prediction. Sector shares and flags are used only to describe the result, never as inputs.

  1. Error ranges: split conformal per round, calibrated on that round's validation residuals, scored on test.
  2. Warnings: which satellite-visible signals pick out large errors. A logistic risk score is cross-fitted
     (round r uses the other rounds' test residuals), and ranges are calibrated per risk group.
  3. Twins: for each test municipality, the 10 training municipalities nearest in mean cell embedding
     (raw GSED, Stage B z of raw GSED, Stage B z of text-aligned A(full)); does twin disagreement flag errors?

Outputs: reports/tables/app_reliability.md, app_conformal.csv, app_signals.csv, app_twins.csv,
         app_municipalities.csv (one row per municipality: estimate, ranges, risk, warnings, twins)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.eval.folds import get_split

SIGNALS = ["bare_share", "built_share", "ntl_top_share", "log_n_cells", "pred"]
LEVELS = (0.8, 0.9)
K_TWINS = 10


def conformal_q(abs_res: np.ndarray, level: float) -> float:
    """Split-conformal radius: the ceil((n+1) level)-th smallest absolute residual."""
    n = len(abs_res)
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(abs_res)[k - 1])


def municipal_signals(cells: pd.DataFrame) -> pd.DataFrame:
    g = cells.groupby("muni_code")
    ntl_sum = g["ntl"].sum()
    return pd.DataFrame({
        "bare_share": g["lc_bare"].mean(),  # open-pit mines and quarries show up as bare ground
        "built_share": g["lc_built"].mean(),
        "ntl_top_share": (g["ntl"].max() / ntl_sum.where(ntl_sum > 0)).fillna(0.0),  # activity in one cell
        "log_n_cells": np.log(g.size()),
    })


def twin_stats(emb_df: pd.DataFrame, test: list, train: list, y: pd.Series) -> pd.DataFrame:
    """Mean official value and spread of the K nearest training municipalities (cosine on mean embeddings)."""
    emb = emb_df.to_numpy()
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    idx = {m: i for i, m in enumerate(emb_df.index)}
    tr = np.array([idx[m] for m in train])
    rows = []
    for m in test:
        sim = e[tr] @ e[idx[m]]
        nb = np.array(train)[np.argsort(-sim)[:K_TWINS]]
        rows.append({"muni_code": m, "twin_est": y[nb].mean(), "twin_sd": y[nb].std(), "twins": list(nb)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="V10")
    args = ap.parse_args()
    cfg = load_config()
    tab = REPO_ROOT / cfg["paths"]["reports"] / "tables"
    n_rounds = cfg["cv"]["n_folds"]
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))
    preds = pd.concat([pd.read_parquet(data_path(cfg, "processed", "predictions", f"{args.model}_r{r}.parquet"))
                       for r in range(n_rounds)])
    preds["abs_err"] = (preds.y_pred - preds.y_true).abs()
    test = preds[preds.split == "test"].set_index("muni_code")
    cols = ["muni_code", "ntl", "lc_bare", "lc_built", *GSED]
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"), columns=[*cols, "keep"])
    cells = cells[cells["keep"]].reset_index(drop=True)
    sig = municipal_signals(cells)
    sig["pred"] = test["y_pred"]
    m = test[["round", "y_true", "y_pred", "abs_err"]].join(sig[[c for c in SIGNALS if c != "pred"]])
    m = m.join(lab[["muni_name", "flag_extreme"]])
    lines = [f"# Reliability layer for an application ({args.model}, out-of-sample)\n",
             "Inputs a region without official statistics would have: satellite layers, cell count, the model's "
             "prediction. Every number below is scored on municipalities the step never saw.\n"]

    # 1. split conformal per round: radius from the round's validation residuals, scored on its test fold
    rows = []
    for r in range(n_rounds):
        val = preds[(preds["round"] == r) & (preds.split == "val")]
        for lv in LEVELS:
            q = conformal_q(val.abs_err.to_numpy(), lv)
            t = m[m["round"] == r]
            m.loc[t.index, f"lo{int(lv * 100)}"] = t.y_pred - q
            m.loc[t.index, f"hi{int(lv * 100)}"] = t.y_pred + q
            rows.append({"round": r, "level": lv, "radius_log": q, "coverage": float((t.abs_err <= q).mean())})
    conf = pd.DataFrame(rows)
    conf.round(4).to_csv(tab / "app_conformal.csv", index=False)
    lines.append("## 1. Error ranges (split conformal, calibrated on each round's validation fold)\n")
    lines.append("| nominal | test coverage | mean radius (log) | as a factor |\n| --- | --- | --- | --- |")
    for lv, d in conf.groupby("level"):
        cov = sum(((m["round"] == r) & (m.abs_err <= q)).sum() for r, q in zip(d["round"], d.radius_log)) / len(m)
        lines.append(f"| {lv:.0%} | {cov:.1%} | {d.radius_log.mean():.3f} | ×/÷ {np.exp(d.radius_log.mean()):.2f} |")
    within = (m.abs_err <= np.log(1.5)).mean()
    lines.append(f"\n{within:.0%} of municipalities are within a factor 1.5 of the official value "
                 f"(median error factor {np.exp(m.abs_err.median()):.2f}).\n")

    # 2a. single signals: rank correlation with |error|
    srows = []
    for s in SIGNALS:
        x = m["y_pred"] if s == "pred" else m[s]
        rho, p = spearmanr(x, m.abs_err)
        srows.append({"signal": s, "spearman_abs_err": rho, "p": p})
    # 2b. cross-fitted risk score: round r's model is fitted on the other rounds' test residuals
    X = m[["bare_share", "built_share", "ntl_top_share", "log_n_cells"]].assign(pred=m.y_pred)
    m["risk"] = np.nan
    for r in range(n_rounds):
        tr, te = m["round"] != r, m["round"] == r
        big = (m.abs_err[tr] > m.abs_err[tr].quantile(0.8)).astype(int)
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X[tr]), big)
        m.loc[te, "risk"] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        # risk groups and per-group ranges from the other rounds only
        risk_tr = clf.predict_proba(sc.transform(X[tr]))[:, 1]
        cuts = np.quantile(risk_tr, [1 / 3, 2 / 3])
        g_tr = np.digitize(risk_tr, cuts)
        g_te = np.digitize(m.loc[te, "risk"], cuts)
        m.loc[te, "risk_group"] = g_te
        for g in range(3):
            res = m.abs_err[tr].to_numpy()[g_tr == g]
            for lv in LEVELS:
                q = conformal_q(res, lv)
                sel = m.index[te][g_te == g]
                m.loc[sel, f"glo{int(lv * 100)}"] = m.loc[sel, "y_pred"] - q
                m.loc[sel, f"ghi{int(lv * 100)}"] = m.loc[sel, "y_pred"] + q
    big_all = (m.abs_err > m.abs_err.quantile(0.8)).astype(int)
    auc = roc_auc_score(big_all, m.risk)
    srows.append({"signal": "risk score (cross-fitted, all signals)", "spearman_abs_err": spearmanr(m.risk, m.abs_err)[0],
                  "p": spearmanr(m.risk, m.abs_err)[1], "auc_top20_error": auc})
    pd.DataFrame(srows).round(4).to_csv(tab / "app_signals.csv", index=False)
    lines.append("## 2. Satellite-only warnings\n")
    lines.append("| signal | Spearman with abs. error | p |\n| --- | --- | --- |")
    lines += [f"| {r['signal']} | {r['spearman_abs_err']:+.3f} | {r['p']:.3g} |" for r in srows]
    lines.append(f"\nThe cross-fitted risk score finds the worst 20% of errors with AUC {auc:.2f} (0.5 = chance).\n")
    lines.append("| risk group | municipalities | mean abs. error | flagged (mining/hydro/top 1%) | "
                 "90% range coverage | 90% radius as a factor |\n| --- | --- | --- | --- | --- | --- |")
    for g, d in m.groupby("risk_group"):
        cov = ((d.y_true >= d.glo90) & (d.y_true <= d.ghi90)).mean()
        rad = ((d.ghi90 - d.glo90) / 2).mean()
        lines.append(f"| {['low', 'medium', 'high'][int(g)]} | {len(d)} | {d.abs_err.mean():.3f} | "
                     f"{int(d.flag_extreme.sum())} | {cov:.1%} | ×/÷ {np.exp(rad):.2f} |")
    cov_all = ((m.y_true >= m.glo90) & (m.y_true <= m.ghi90)).mean()
    lines.append(f"\nRisk-grouped 90% ranges cover {cov_all:.1%} of test municipalities overall.\n")

    # 3. twins in three embedding spaces
    y = lab["log_gdp_pc"]
    twins_rows, best = [], None
    spaces = {"raw GSED": None, "Stage B z (GSED)": "z_gsed", "Stage B z (A full)": "z_A_full"}
    for name, zf in spaces.items():
        out = []
        for r in range(n_rounds):
            s = get_split(folds, r)
            if zf is None:
                emb_df = cells[GSED].groupby(cells.muni_code).mean()
            else:
                z = np.load(data_path(cfg, "interim", f"round{r}", f"{zf}.npy")).astype(np.float32)
                emb_df = pd.DataFrame(z).groupby(cells.muni_code.to_numpy()).mean()
            out.append(twin_stats(emb_df, list(s.test), list(s.train), y))
        t = pd.concat(out).set_index("muni_code").join(m[["y_true", "y_pred", "abs_err"]])
        twin_err = t.twin_est - t.y_true
        rows = {"space": name, "twin_rmse": np.sqrt((twin_err ** 2).mean()),
                "twin_r2": 1 - (twin_err ** 2).sum() / ((t.y_true - t.y_true.mean()) ** 2).sum(),
                "spearman_twin_sd_vs_model_err": spearmanr(t.twin_sd, t.abs_err)[0],
                "spearman_disagreement_vs_model_err": spearmanr((t.twin_est - t.y_pred).abs(), t.abs_err)[0]}
        twins_rows.append(rows)
        if zf == "z_A_full":
            best = t
    tw = pd.DataFrame(twins_rows)
    tw.round(4).to_csv(tab / "app_twins.csv", index=False)
    lines.append(f"## 3. Economic twins ({K_TWINS} nearest training municipalities)\n")
    lines.append("| space | twin estimate RMSE | twin estimate R² | twin spread vs model error | "
                 "twin-model disagreement vs model error |\n| --- | --- | --- | --- | --- |")
    lines += [f"| {r['space']} | {r['twin_rmse']:.3f} | {r['twin_r2']:.3f} | {r['spearman_twin_sd_vs_model_err']:+.3f} | "
              f"{r['spearman_disagreement_vs_model_err']:+.3f} |" for r in twins_rows]
    lines.append(f"\nFor reference, {args.model} itself: RMSE {np.sqrt((m.y_pred - m.y_true).pow(2).mean()):.3f}. "
                 "Spearman values are rank correlations with the model's absolute error (higher = better warning).\n")

    # per-municipality table for the application
    m = m.join(best[["twin_est", "twin_sd", "twins"]])
    m["twins"] = m["twins"].map(lambda c: "; ".join(lab.loc[c[:5], "muni_name"]))
    m["risk_group"] = m["risk_group"].map({0: "low", 1: "medium", 2: "high"})
    keep = ["muni_name", "round", "y_true", "y_pred", "lo80", "hi80", "lo90", "hi90", "glo90", "ghi90", "risk",
            "risk_group", "bare_share", "built_share", "ntl_top_share", "log_n_cells", "twin_est", "twin_sd", "twins"]
    m[keep].round(4).sort_index().to_csv(tab / "app_municipalities.csv")
    (tab / "app_reliability.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
