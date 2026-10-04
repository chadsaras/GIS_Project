"""Phase 1: build the municipal label table for the study state.

Steps 1.1-1.7 of PLAN.md: boundaries, GDP and sector value added, Census population,
immediate regions -> processed/municipal_labels.parquet, plus an overview figure.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, ensure_dirs, load_config
from gisecon.data import ibge

INDUSTRY_SHARE_FLAG = 0.60
TOP_GDPPC_QUANTILE = 0.99


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    src, study = cfg["sources"], cfg["study"]
    year, uf = study["year"], study["state_code"]

    # 1.1 boundaries
    shp = ibge.fetch_boundaries(src["ibge_boundaries_url"], data_path(cfg, "raw", "ibge", "malha_2022"))
    bnd = ibge.load_boundaries(shp, cfg["grid"]["crs"])
    bnd.to_file(data_path(cfg, "interim", "municipios.gpkg"), driver="GPKG")

    # 1.2 GDP for the main year; value added by sector from the latest year that has it.
    # IBGE publishes only total GDP for 2022-2023 at municipal level, so sector shares
    # (used for flags and error analysis only) come from study.sector_year. Values R$ 1,000 -> R$.
    gv = src["sidra_gdp_vars"]
    sector_year = study["sector_year"]
    gdp_long = ibge.fetch_sidra(src["sidra_gdp_table"], [gv["gdp"]], year, uf)
    gdp_long.to_csv(data_path(cfg, "raw", "ibge", f"pib_{year}.csv"), index=False)
    va_vars = [v for k, v in gv.items() if k != "gdp"]
    va_long = ibge.fetch_sidra(src["sidra_gdp_table"], va_vars, sector_year, uf)
    va_long.to_csv(data_path(cfg, "raw", "ibge", f"va_{sector_year}.csv"), index=False)
    wide = pd.concat([gdp_long, va_long]).pivot(index="muni_code", columns="variable", values="value")
    gdp = wide.rename(columns={v: k for k, v in gv.items()}) * 1000.0

    # 1.3 population (Census 2022)
    pop_long = ibge.fetch_sidra(src["sidra_pop_table"], [src["sidra_pop_var"]], year, uf)
    pop_long.to_csv(data_path(cfg, "raw", "ibge", f"pop_{year}.csv"), index=False)
    pop = pop_long.set_index("muni_code")["value"].rename("pop")

    # 1.4 regions
    reg = ibge.fetch_municipios(src["ibge_municipios_api"])
    reg.to_csv(data_path(cfg, "raw", "ibge", "regioes.csv"), index=False)

    # 1.5 join and derive targets; every table must cover the same municipalities
    n = study["n_municipalities"]
    codes = [set(bnd.muni_code), set(gdp.index), set(pop.index), set(reg.muni_code)]
    for name, c in zip(["boundaries", "gdp", "pop", "regions"], codes):
        assert len(c) == n, f"{name}: {len(c)} municipalities, expected {n}"
    assert all(c == codes[0] for c in codes), "municipality codes differ between sources"

    df = (reg.set_index("muni_code")
          .join(bnd.drop(columns="geometry").set_index("muni_code"))
          .join(gdp).join(pop).reset_index())
    assert df[["gdp", "pop", "va_total"]].notna().all().all(), "missing GDP or population values"

    df["gdp_pc"] = df["gdp"] / df["pop"]
    df["log_gdp"] = np.log(df["gdp"])
    df["log_pop"] = np.log(df["pop"])
    df["log_gdp_pc"] = np.log(df["gdp_pc"])
    for sec in ["agro", "industry", "services", "public"]:
        df[f"share_{sec}"] = df[f"va_{sec}"] / df["va_total"]

    # 1.6 flag municipalities dominated by one large activity (mining, hydro) or extreme GDP pc
    df["flag_industry"] = df["share_industry"] > INDUSTRY_SHARE_FLAG
    df["flag_top_gdppc"] = df["gdp_pc"] >= df["gdp_pc"].quantile(TOP_GDPPC_QUANTILE)
    df["flag_extreme"] = df["flag_industry"] | df["flag_top_gdppc"]

    out = data_path(cfg, "processed", "municipal_labels.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} municipalities, {df.flag_extreme.sum()} flagged)")

    # 1.7 describe the labels
    reports = REPO_ROOT / cfg["paths"]["reports"]
    summary = df[["gdp_pc", "log_gdp_pc", "pop", "share_agro", "share_industry",
                  "share_services", "share_public"]].describe(percentiles=[0.05, 0.5, 0.95]).T
    summary.to_csv(reports / "tables" / "labels_summary.csv")
    print(summary.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].hist(df["log_gdp_pc"], bins=40, color="#4C72B0")
    ax[0].set_xlabel("log GDP per capita (R$, 2022)")
    ax[0].set_ylabel("municipalities")
    ax[0].set_title(f"Minas Gerais, n = {len(df)}")
    m = bnd.merge(df[["muni_code", "log_gdp_pc", "flag_extreme"]], on="muni_code")
    m.plot(column="log_gdp_pc", cmap="viridis", legend=True, ax=ax[1],
           legend_kwds={"label": "log GDP per capita", "shrink": 0.7})
    m[m.flag_extreme].boundary.plot(ax=ax[1], color="red", linewidth=0.6)
    ax[1].set_title("log GDP per capita (red outline: flagged)")
    ax[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(reports / "figures" / "labels_overview.png", dpi=150)
    print("wrote reports/figures/labels_overview.png")


if __name__ == "__main__":
    main()
