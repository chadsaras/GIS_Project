"""Phase 9.1-9.8: where the best model works and fails, plus the maps (LIGHT: minutes). Test-fold predictions only.

Run after 09_evaluate.py. The "best" model is the complete variant with the lowest *validation* RMSE
(never chosen on test); override with --model. It is always compared with V1 (NTL only).
Outputs (reports/): figures/pred_vs_official.png, tables/error_regression.csv, figures/error_vs_income.png,
tables/moran.csv, tables/top10_gaps.md, figures/uncertainty_vs_error.png, figures/map_{official,predicted,gap}.png,
figures/activity_1km.png (+ processed/activity_1km.tif)
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.grid import Grid
from gisecon.eval.folds import get_split

STAGE_B_INPUT = {"V4": "gsed", "V6": "A_full", "V7": "A_full", "V8": "A_raw", "V9": "A_facts"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    args = ap.parse_args()
    cfg = load_config()
    rep = REPO_ROOT / cfg["paths"]["reports"]
    fig_dir, tab_dir = rep / "figures", rep / "tables"
    n_rounds = cfg["cv"]["n_folds"]

    res = pd.read_csv(tab_dir / "main_results.csv").set_index("variant")
    complete = res[res.n_rounds == n_rounds]
    best = args.model or complete.drop(index="V1", errors="ignore")["val_rmse"].idxmin()
    # uncertainty (9.6) and the activity map (9.8) need a 5-seed ensemble, which only V3-V9 have
    ens = [v for v in complete.index if v in STAGE_B_INPUT or v in ("V3", "V5")]
    best_ens = best if best in ens else complete.loc[ens, "val_rmse"].idxmin()
    print(f"best model by validation RMSE: {best}; best ensemble model: {best_ens}")
    preds = pd.concat([pd.read_parquet(f) for f in data_path(cfg, "processed", "predictions").glob("*.parquet")])
    test = preds[preds.split == "test"]
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet")).set_index("muni_code")
    osm = pd.read_parquet(data_path(cfg, "interim", "osm_completeness.parquet")).set_index("muni_code")
    m = lab.join(osm[["built_share", "osm_completeness"]]).join(folds["block_id"])
    for v in dict.fromkeys((best, best_ens, "V1")):
        p = test[test.variant == v].set_index("muni_code")
        m[f"pred_{v}"], m[f"lo_{v}"], m[f"hi_{v}"] = p.y_pred, p.ens_min, p.ens_max
        m[f"abs_err_{v}"] = (p.y_pred - m.log_gdp_pc).abs()

    # 9.1 predicted vs official
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    lim = [m.log_gdp_pc.min() - 0.2, m.log_gdp_pc.max() + 0.2]
    for a, v in zip(ax, (best, "V1")):
        a.scatter(m[f"pred_{v}"], m.log_gdp_pc, s=8, c=np.where(m.flag_extreme, "crimson", "steelblue"), alpha=0.6)
        a.plot(lim, lim, "k--", lw=1)
        a.set_title(f"{v}: RMSE {res.at[v, 'rmse']:.3f}, R² {res.at[v, 'r2']:.3f}")
        a.set_xlabel("predicted log GDP per capita")
    ax[0].set_ylabel("official log GDP per capita (red: mining/hydro/top-1% flag)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pred_vs_official.png", dpi=150)

    # 9.2 |error| on municipality traits, SEs clustered by immediate region
    rows = []
    for v in (best, "V1"):
        d = m.rename(columns={f"abs_err_{v}": "abs_err"}).dropna(subset=["abs_err"])
        fit = smf.ols("abs_err ~ log_gdp_pc + built_share + share_services + share_industry + flag_extreme"
                      " + osm_completeness", data=d.assign(flag_extreme=d.flag_extreme.astype(float))
                      ).fit(cov_type="cluster", cov_kwds={"groups": d.block_id})
        rows += [{"model": v, "term": t, "coef": fit.params[t], "se": fit.bse[t], "p": fit.pvalues[t]}
                 for t in fit.params.index]
    pd.DataFrame(rows).round(4).to_csv(tab_dir / "error_regression.csv", index=False)

    # 9.3 error vs income, LOWESS
    fig, ax = plt.subplots(figsize=(7, 5))
    for v, col in ((best, "steelblue"), ("V1", "darkorange")):
        ax.scatter(m.log_gdp_pc, m[f"abs_err_{v}"], s=5, alpha=0.3, color=col)
        sm = lowess(m[f"abs_err_{v}"], m.log_gdp_pc, frac=0.4)
        ax.plot(sm[:, 0], sm[:, 1], color=col, lw=2, label=v)
    ax.set_xlabel("official log GDP per capita")
    ax.set_ylabel("absolute error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "error_vs_income.png", dpi=150)

    # 9.4 Moran's I of residuals over neighbouring municipalities
    import esda
    from libpysal.weights import Queen
    munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg"))[["muni_code", "geometry"]]
    munis = munis.set_index("muni_code").join(m)
    w = Queen.from_dataframe(munis, use_index=True)
    w.transform = "r"
    moran = []
    for v in (best, "V1"):
        mi = esda.Moran((munis[f"pred_{v}"] - munis.log_gdp_pc).to_numpy(), w, permutations=999)
        moran.append({"model": v, "moran_I": mi.I, "p_sim": mi.p_sim})
    pd.DataFrame(moran).round(4).to_csv(tab_dir / "moran.csv", index=False)
    print(pd.DataFrame(moran).round(3).to_string(index=False))

    # 9.5 the 10 largest gaps (reasons are written by hand afterwards)
    m["gap"] = m[f"pred_{best}"] - m.log_gdp_pc
    top = m.reindex(m.gap.abs().sort_values(ascending=False).index[:10])
    texts_f = data_path(cfg, "processed", "texts.parquet")
    texts = pd.read_parquet(texts_f) if texts_f.exists() else None
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"), columns=["cell_id", "muni_code"])
    lines = [f"# Top 10 gaps, {best} (predicted - official, log GDP per capita)\n"]
    for code, r in top.iterrows():
        lines.append(f"## {r.muni_name} ({code})\n\nofficial {r.log_gdp_pc:.2f}, predicted {r[f'pred_{best}']:.2f}, "
                     f"gap {r.gap:+.2f}; agro {r.share_agro:.0%}, industry {r.share_industry:.0%}, "
                     f"services {r.share_services:.0%}, public {r.share_public:.0%}; "
                     f"flagged: {bool(r.flag_extreme)}\n")
        if texts is not None:
            t = texts[texts.cell_id.isin(cells.cell_id[cells.muni_code == code])].head(3)
            lines += [f"- cell {c}: {txt}" for c, txt in zip(t.cell_id, t.text_raw)]
        lines.append("\nLikely reason: _(fill in: mining, dam, agribusiness, commuter town, ...)_\n")
    (tab_dir / "top10_gaps.md").write_text("\n".join(lines))

    # 9.6 does ensemble spread track error?
    spread = m[f"hi_{best_ens}"] - m[f"lo_{best_ens}"]
    rho, p = spearmanr(spread, m[f"abs_err_{best_ens}"], nan_policy="omit")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = pd.qcut(spread, 10, duplicates="drop")
    m.groupby(bins, observed=True)[f"abs_err_{best_ens}"].mean().plot(marker="o", ax=ax)
    ax.set_xlabel("5-seed min-max range (decile)")
    ax.set_ylabel("mean absolute error")
    ax.set_title(f"{best_ens}: Spearman rho {rho:.2f} (p={p:.3g})")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "uncertainty_vs_error.png", dpi=150)

    # 9.7 municipal maps on one colour scale
    vmin, vmax = munis.log_gdp_pc.min(), munis.log_gdp_pc.max()
    g = munis[f"pred_{best}"] - munis.log_gdp_pc
    for name, col, kw in (("official", munis.log_gdp_pc, {"cmap": "viridis", "vmin": vmin, "vmax": vmax}),
                          ("predicted", munis[f"pred_{best}"], {"cmap": "viridis", "vmin": vmin, "vmax": vmax}),
                          ("gap", g, {"cmap": "RdBu_r", "vmin": -g.abs().max(), "vmax": g.abs().max()})):
        fig, ax = plt.subplots(figsize=(8, 7))
        munis.assign(v=col).plot(column="v", legend=True, ax=ax, legend_kwds={"shrink": 0.6}, **kw)
        ax.set_title(f"{name} log GDP per capita" + (f" ({best} - official)" if name == "gap" else ""))
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(fig_dir / f"map_{name}.png", dpi=150)
        plt.close(fig)

    # 9.8 out-of-sample 1 km activity map: each cell from the round where its municipality is tested
    inp = STAGE_B_INPUT.get(best_ens) or next((v for k, v in STAGE_B_INPUT.items() if k in complete.index), None)
    if inp is None:
        print("activity map skipped: no Stage B run found")
        return
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    kept = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"), columns=["row", "col", "muni_code", "keep"])
    kept = kept[kept.keep].reset_index(drop=True)
    act = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    for r in range(n_rounds):
        pred = np.load(data_path(cfg, "interim", f"round{r}", f"ntl_pred_{inp}.npy"))
        t = kept.muni_code.isin(get_split(folds.reset_index(), r).test).to_numpy()
        act[kept.row[t], kept.col[t]] = pred[t]
    with rasterio.open(data_path(cfg, "processed", "activity_1km.tif"), "w", driver="GTiff", width=grid.width,
                       height=grid.height, count=1, dtype="float32", crs=grid.crs, transform=grid.transform,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(act, 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(act, cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, label="predicted log(1 + NTL), relative activity index")
    ax.set_title(f"Out-of-sample 1 km activity map (Stage B, input {inp})")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(fig_dir / "activity_1km.png", dpi=150)
    print("wrote all Phase 9.1-9.8 outputs to reports/")


if __name__ == "__main__":
    main()
