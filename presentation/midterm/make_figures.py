"""Figures for the mid-term presentation, generated from the project's real data and results.

Run on the server from the repo root:  python presentation/midterm/make_figures.py
Writes PNG maps / tiles to presentation/midterm/assets/, TikZ charts (.tex, drawn in the deck's fonts and
colours) and small .tex snippets (caption example, backup captions, numbers) that slides.tex reads, so every
number on a slide comes from the results.

Design: one colour per ingredient, the same in maps, charts and slides
  embeddings = teal, text = violet, night lights = amber, official GDP = brick, POI / roads = blue.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.data.grid import Grid

OUT = REPO_ROOT / "presentation" / "midterm" / "assets"
# palette (same hex values as the colour definitions in slides.tex)
NIGHT, NIGHT2, GLOW = "#0B1328", "#16223F", "#F2B134"
PAPER, INK, MUTED, RULE = "#F7F5F0", "#1E2433", "#687082", "#D9D5CC"
C_IMG, C_TXT, C_NTL, C_GDP, C_POI = "#138A8A", "#7057C9", "#E19A1F", "#C4493B", "#4A6FA5"

CMAP_GDP = LinearSegmentedColormap.from_list("gdp", ["#FBF1DD", "#F2B134", "#D9632E", "#8E2C3A", "#3B1530"])
CMAP_NTL = LinearSegmentedColormap.from_list("ntl", [NIGHT2, "#2B3466", "#8A4A6B", C_NTL, "#FFE9A8", "#FFFFFF"])
CMAP_ERR = LinearSegmentedColormap.from_list("err", [C_IMG, "#9FD0CC", "#F4F1EA", "#EBA493", C_GDP])
GDP_NORM = (8000, 120000)

CAPTION_CELL = 236778          # centre-pivot field the model called "a mine pit" (removed by the industry check)
# backup slide: one clear example per land type, picked by rule in main()
BACKUP_TYPES = {"Forest": ("lc_tree", 0.85, r"forest|trees|woodland|vegetation"),
                "Farmland": ("lc_crop", 0.5, r"farm|field|crop|plot|agricultur"),
                "Village edge": ("lc_built", 0.05, r"house|building|residential|roof"),
                "Town centre": ("lc_built", 0.6, r"dense|urban|grid|roof")}
LABELS = {"V1": "Night lights only", "V2": "Land-cover shares", "V11_gsed": "Ridge, mean embedding",
          "V12": "LightGBM, mean embedding", "V3": "Embeddings + Stage C", "V4": "+ night-light steering",
          "V5": "+ text alignment", "V6": "+ text + steering", "V7": "+ text + steering + POI",
          "V8": "Full, uncleaned captions", "V9": "Full, fact sentences only", "V10": "Hybrid with night lights"}
# which ingredients each variant uses: I = embeddings, T = text, L = night lights, P = POI / roads
INGR = {"V1": "L", "V2": "", "V11_gsed": "I", "V12": "I", "V3": "I", "V4": "IL", "V5": "IT", "V6": "ITL",
        "V7": "ITLP", "V8": "ITL", "V9": "ITL", "V10": "IL"}
BASELINES = {"V1", "V2", "V11_gsed", "V12"}

for f in Path("/usr/share/fonts/truetype/lato").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "Lato", "text.color": INK})


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def raster(values, rows, cols, grid, fill=np.nan):
    a = np.full((grid.height, grid.width) + np.shape(values)[1:], fill, dtype=np.float32)
    a[rows, cols] = values
    return a


def crop(a, rows, cols, pad=4):
    r0, r1, c0, c1 = rows.min() - pad, rows.max() + pad, cols.min() - pad, cols.max() + pad
    return a[max(r0, 0):r1, max(c0, 0):c1]


def furniture(ax, unit_km: float, color=INK, km=200):
    """Scale bar (km) in the lower right and a north arrow in the upper right; unit_km = km per data unit."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, abs(y1 - y0)
    up = 1 if y1 > y0 else -1                      # images have y pointing down
    L = km / unit_km
    bx, by = x1 - 0.06 * w - L, y0 + up * 0.06 * h
    for k in range(4):
        ax.add_patch(plt.Rectangle((bx + k * L / 4, by), L / 4, up * 0.012 * h, lw=0.6, ec=color,
                                   fc=color if k % 2 == 0 else "none", zorder=5))
    for xx, t in ((bx, "0"), (bx + L, f"{km} km")):
        ax.text(xx, by + up * 0.03 * h, t, ha="center", va="bottom" if up > 0 else "top", fontsize=7,
                color=color, zorder=5)
    nx, ny = x1 - 0.07 * w, y1 - up * 0.16 * h
    ax.annotate("", xy=(nx, ny + up * 0.08 * h), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.3", color=color, lw=1.2), zorder=5)
    ax.text(nx, ny + up * 0.1 * h, "N", ha="center", va="bottom" if up > 0 else "top", fontsize=8,
            fontweight="bold", color=color, zorder=5)


def save_map(img, path, cmap=None, norm=None, vmin=None, vmax=None, size=(5, 4.2), scale=False, dpi=220,
             color=INK):
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    if scale:
        furniture(ax, 1.0, color=color)            # grid cells are 1 km
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def muni_map(m, column, path, cmap, norm, size=(5.2, 4.6), cbar=None, ticks=None, ticklabels=None,
             scale=True, edge=0.05):
    fig, ax = plt.subplots(figsize=size)
    m.plot(column=column, cmap=cmap, norm=norm, ax=ax, linewidth=edge, edgecolor="white")
    ax.set_axis_off()
    if scale:
        furniture(ax, 0.001)                       # EPSG:31983 is in metres
    if cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.55, pad=0.01, aspect=28)
        cb.outline.set_visible(False)
        if ticks is not None:
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklabels)
        cb.ax.tick_params(labelsize=8, length=0, colors=INK)
        cb.ax.set_xlabel(cbar, fontsize=9, color=MUTED)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ TikZ charts
def tikz_head(extra=""):
    return [r"\begin{tikzpicture}[x=1cm,y=1cm,",
            r"  lab/.style={font=\fontsize{7.5}{9}\selectfont,text=ink,inner sep=0pt},",
            r"  num/.style={font=\bfseries\fontsize{7.5}{9}\selectfont,inner sep=1pt},",
            r"  tick/.style={font=\fontsize{6.5}{7.8}\selectfont,text=muted,inner sep=0pt}" + extra + "]"]


def write_results_chart(r: pd.DataFrame, path: Path) -> None:
    """R2 bars, best first, with a dot matrix of the ingredients each variant uses."""
    X, DY = 9.0, 0.5            # cm per unit of R2, row height
    cols = [("I", "cimg", "image"), ("T", "ctxt", "text"), ("L", "cntl", "lights"), ("P", "cpoi", "POI")]
    lines = tikz_head()
    n = len(r)
    bottom = -DY * (n - 1) - 0.32
    # ingredient column heads
    for k, (_, c, name) in enumerate(cols):
        lines.append(rf"\node[tick,text={c},font=\bfseries\fontsize{{6}}{{7}}\selectfont,rotate=45,anchor=south west]"
                     rf" at ({-1.15 + 0.3 * k:.2f},0.3) {{{name}}};")
    for t in np.arange(0, 0.61, 0.1):
        lines.append(rf"\draw[rule,line width=0.4pt] ({t * X:.2f},0.32) -- ({t * X:.2f},{bottom:.2f});")
        lines.append(rf"\node[tick,anchor=north] at ({t * X:.2f},{bottom - 0.08:.2f}) {{{t:.1f}}};")
    lines.append(rf"\node[tick,anchor=north] at ({0.3 * X:.2f},{bottom - 0.38:.2f}) "
                 rf"{{R\textsuperscript{{2}} on held-out municipalities}};")
    for i, (v, row) in enumerate(r.iterrows()):
        y = -DY * i
        fill = "glow" if v == "V10" else "muted!45" if v in BASELINES else "cimg"
        lines.append(rf"\node[lab,anchor=east] at (-1.4,{y:.2f}) {{{tex_escape(LABELS[v])}}};")
        for k, (key, c, _) in enumerate(cols):
            on = key in INGR.get(v, "")
            style = f"fill={c}" if on else "draw=rule,line width=0.5pt,fill=white"
            lines.append(rf"\path[{style}] ({-1.15 + 0.3 * k:.2f},{y:.2f}) circle (0.075);")
        lines.append(rf"\fill[{fill},rounded corners=1pt] (0,{y - 0.15:.2f}) rectangle ({row.r2 * X:.2f},{y + 0.15:.2f});")
        lines.append(rf"\node[num,anchor=west,text=ink] at ({row.r2 * X + 0.06:.2f},{y:.2f}) {{{row.r2:.2f}}};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_chart(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> None:
    """Dumbbell: RMSE before -> after the validation calibration, for the two Stage C variants."""
    X0, X = 0.40, 26.0

    def ax(v):
        return (v - X0) * X

    lines = tikz_head()
    for t in np.arange(0.40, 0.601, 0.05):
        lines.append(rf"\draw[rule,line width=0.4pt] ({ax(t):.2f},0.5) -- ({ax(t):.2f},-1.75);")
        lines.append(rf"\node[tick,anchor=north] at ({ax(t):.2f},-1.83) {{{t:.2f}}};")
    lines.append(rf"\node[tick,anchor=north] at ({ax(0.50):.2f},-2.15) {{error (RMSE) on held-out municipalities, lower is better}};")
    for i, (v, name) in enumerate([("V3", "Embeddings + C"), ("V4", "+ steering")]):
        y = -1.1 * i
        b, a = before.at[v, "rmse"], after.at[v, "rmse"]
        lines.append(rf"\node[lab,anchor=east] at (-0.2,{y:.2f}) {{{name}}};")
        lines.append(rf"\draw[-{{Stealth[length=2.2mm]}},cimg,line width=1.4pt] ({ax(b) - 0.14:.2f},{y:.2f}) -- "
                     rf"({ax(a) + 0.17:.2f},{y:.2f});")
        lines.append(rf"\node[circle,draw=muted,fill=white,line width=1pt,minimum size=2.6mm,inner sep=0pt] at ({ax(b):.2f},{y:.2f}) {{}};")
        lines.append(rf"\node[circle,fill=cimg,minimum size=2.8mm,inner sep=0pt] at ({ax(a):.2f},{y:.2f}) {{}};")
        lines.append(rf"\node[num,text=muted,anchor=south] at ({ax(b):.2f},{y + 0.14:.2f}) {{{b:.3f}}};")
        lines.append(rf"\node[num,text=cimg,anchor=south] at ({ax(a):.2f},{y + 0.14:.2f}) {{{a:.3f}}};")
        lines.append(rf"\node[num,text=okc,anchor=north] at ({(ax(a) + ax(b)) / 2:.2f},{y - 0.1:.2f}) {{{100 * (a - b) / b:+.0f}\%}};")
    lines.append(r"\node[circle,draw=muted,fill=white,line width=1pt,minimum size=2.4mm,inner sep=0pt] at (0.1,0.95) {};")
    lines.append(r"\node[lab,anchor=west] at (0.25,0.95) {before};")
    lines.append(r"\node[circle,fill=cimg,minimum size=2.4mm,inner sep=0pt] at (1.4,0.95) {};")
    lines.append(r"\node[lab,anchor=west] at (1.55,0.95) {after calibration on validation};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recall_chart(rec: np.ndarray, path: Path) -> None:
    """Grouped bars: Stage A recall@k on validation captions vs chance (batches of 256)."""
    Y = 0.055
    lines = tikz_head()
    for t in range(0, 61, 20):
        lines.append(rf"\draw[rule,line width=0.4pt] (0,{t * Y:.2f}) -- (6.2,{t * Y:.2f});")
        lines.append(rf"\node[tick,anchor=east] at (-0.1,{t * Y:.2f}) {{{t}\%}};")
    for i, (k, v) in enumerate(zip([1, 5, 10], rec * 100)):
        x = 1.0 + 2.0 * i
        ch = k / 256 * 100
        lines.append(rf"\fill[ctxt] ({x - 0.62:.2f},0) rectangle ({x - 0.02:.2f},{v * Y:.2f});")
        lines.append(rf"\node[num,text=ctxt,anchor=south] at ({x - 0.32:.2f},{v * Y:.2f}) {{{v:.0f}\%}};")
        lines.append(rf"\fill[muted!40] ({x + 0.02:.2f},0) rectangle ({x + 0.62:.2f},{max(ch * Y, 0.03):.2f});")
        lines.append(rf"\node[num,text=muted,anchor=south] at ({x + 0.32:.2f},{max(ch * Y, 0.03):.2f}) {{{ch:.0f}\%}};")
        lines.append(rf"\node[lab,anchor=north] at ({x:.2f},-0.1) {{recall@{k}}};")
    lines.append(r"\fill[ctxt] (0.3,3.75) rectangle (0.6,3.95); \node[lab,anchor=west] at (0.65,3.85) {Stage A};")
    lines.append(r"\fill[muted!40] (2.1,3.75) rectangle (2.4,3.95); \node[lab,anchor=west] at (2.45,3.85) {chance};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_drops_chart(vc: pd.Series, path: Path) -> None:
    """Horizontal bars: sentences removed by each cleaning rule."""
    rows = [("Filler, no content", "filler", "muted!55"), ("Industry / mine", "claim_industry", "cgdp"),
            ("Building", "claim_building", "cgdp"), ("Water", "claim_water", "cgdp"), ("Road", "claim_road", "cgdp"),
            ("Farmland", "claim_farm", "cgdp")]
    top = max(int(vc.get(k, 0)) for _, k, _ in rows)
    W = 2.6 / top
    lines = tikz_head()
    for i, (name, key, c) in enumerate(rows):
        y = -0.36 * i
        n = int(vc.get(key, 0))
        lines.append(rf"\node[lab,anchor=east] at (-0.1,{y:.2f}) {{{name}}};")
        lines.append(rf"\fill[{c},rounded corners=0.8pt] (0,{y - 0.11:.2f}) rectangle ({n * W:.2f},{y + 0.11:.2f});")
        lines.append(rf"\node[num,text=ink,anchor=west] at ({n * W + 0.05:.2f},{y:.2f}) {{{n:,}}};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    grid = Grid.load(data_path(cfg, "interim", "grid.json"))
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"))
    kept = cells[cells.keep].reset_index(drop=True)
    rows, cols = kept.row.to_numpy(), kept.col.to_numpy()
    tables = REPO_ROOT / cfg["paths"]["reports"] / "tables"
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet"))

    # display layers use every cell in the state (masked cells too, so maps have no holes), cropped like the rest
    allc = cells.dropna(subset=["log_ntl"])
    arow, acol = allc.row.to_numpy(), allc.col.to_numpy()

    def full(values):
        return crop(raster(values, arow, acol, grid), rows, cols)

    # ---- title slide: night lights glowing on the dark background (every cell drawn, so the state shows)
    ntl = allc.log_ntl.to_numpy()
    lo, hi = np.percentile(ntl, [1, 99.7])
    save_map(full(np.clip((ntl - lo) / (hi - lo), 0, 1)),
             OUT / "hero_ntl.png", cmap=CMAP_NTL, vmin=0, vmax=1, size=(7, 6), dpi=300)

    # ---- GSED principal components as RGB
    g = kept[GSED].to_numpy(np.float64)
    g = g - g.mean(0)
    pcs = g @ np.linalg.svd(g[:: max(1, len(g) // 50_000)], full_matrices=False)[2][:3].T
    lo3, hi3 = np.percentile(pcs, [2, 98], axis=0)
    rgb = raster(np.clip((pcs - lo3) / (hi3 - lo3), 0, 1), rows, cols, grid)
    alpha = (~np.isnan(rgb[..., 0])).astype(np.float32)[..., None]
    rgba = np.concatenate([np.nan_to_num(rgb, nan=1.0), alpha], axis=-1)
    save_map(crop(rgba, rows, cols), OUT / "hero_gsed.png", size=(6, 5))

    # ---- layer stack for the data slide (identical extent for every layer)
    def layer(values, name, **kw):
        save_map(full(values), OUT / f"layer_{name}.png", size=(3, 2.6), **kw)

    save_map(crop(rgba, rows, cols), OUT / "layer_gsed.png", size=(3, 2.6))
    layer(np.clip((ntl - lo) / (hi - lo), 0, 1), "ntl", cmap=CMAP_NTL, vmin=0, vmax=1)
    road = allc.filter(regex=r"^road_.*_km$").sum(axis=1).to_numpy()
    layer(road, "roads", cmap=LinearSegmentedColormap.from_list("r", ["#EEF2F8", C_POI, "#1D2F55"]),
          vmax=np.percentile(road, 99))
    b = pd.read_parquet(data_path(cfg, "interim", "buildings.parquet")).set_index("cell_id")
    nb = allc.cell_id.map(b.ms_buildings).fillna(0).to_numpy() + 1
    layer(nb, "buildings", cmap=LinearSegmentedColormap.from_list("b", ["#F4F1EA", "#B9B3A6", INK]),
          norm=LogNorm(1, np.percentile(nb, 99.5)))
    gdp_cell = allc.muni_code.map(lab.set_index("muni_code").gdp_pc).to_numpy()
    layer(gdp_cell, "gdp", cmap=CMAP_GDP, norm=LogNorm(*GDP_NORM))
    caps = {json.loads(l)["cell_id"]: json.loads(l)["caption"]
            for l in open(data_path(cfg, "interim", "captions_raw_short_v1.jsonl"))}
    capimg = full(np.where(allc.cell_id.isin(list(caps)).to_numpy(), 1.0, 0.0))
    fig, ax = plt.subplots(figsize=(3, 2.6))
    ax.imshow(np.where(np.isnan(capimg), np.nan, 0), cmap=LinearSegmentedColormap.from_list("c", ["#EFEAF9"] * 2),
              interpolation="nearest")
    yy, xx = np.nonzero(capimg == 1)
    ax.scatter(xx, yy, s=0.25, c=C_TXT, linewidths=0)
    ax.set_axis_off()
    fig.savefig(OUT / "layer_captions.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    # ---- municipal maps: official GDP, folds, official vs predicted vs error (V10, test rounds)
    munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg")).to_crs(31983)
    m = munis.merge(lab[["muni_code", "gdp_pc"]], on="muni_code")
    gn = LogNorm(*GDP_NORM)
    gticks, glabels = [10000, 20000, 50000, 100000], ["R$ 10k", "20k", "50k", "100k"]
    muni_map(m, "gdp_pc", OUT / "map_gdp.png", CMAP_GDP, gn, size=(6.2, 5.2),
             cbar="GDP per capita, 2022 (log scale)", ticks=gticks, ticklabels=glabels)

    pred = pd.concat([pd.read_parquet(data_path(cfg, "processed", "predictions", f"V10_r{r}.parquet"))
                      for r in range(cfg["cv"]["n_folds"])])
    pred = pred[pred.split == "test"][["muni_code", "y_true", "y_pred"]]
    mp = munis.merge(pred, on="muni_code")
    mp["off"], mp["pre"], mp["err"] = np.exp(mp.y_true), np.exp(mp.y_pred), mp.y_pred - mp.y_true
    muni_map(mp, "off", OUT / "map_official.png", CMAP_GDP, gn, size=(4, 3.6), scale=False, edge=0.03)
    muni_map(mp, "pre", OUT / "map_predicted.png", CMAP_GDP, gn, size=(4, 3.6), scale=False, edge=0.03)
    muni_map(mp, "err", OUT / "map_error.png", CMAP_ERR, TwoSlopeNorm(0, -1, 1), size=(4, 3.6), scale=False,
             edge=0.03)
    # shared colour bars as separate strips
    for name, cmap, norm, ticks, tl, label in (
            ("cbar_gdp", CMAP_GDP, gn, gticks, glabels, "GDP per capita (log scale)"),
            ("cbar_err", CMAP_ERR, TwoSlopeNorm(0, -1, 1), [-1, -0.5, 0, 0.5, 1],
             ["$\\div$2.7", "", "exact", "", "$\\times$2.7"], "predicted / official")):
        fig, ax = plt.subplots(figsize=(3.2, 0.5))
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.set_ticks(ticks)
        cb.set_ticklabels(tl)
        cb.ax.tick_params(labelsize=8, length=0)
        cb.ax.set_xlabel(label, fontsize=8, color=MUTED)
        fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)

    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))
    mf = munis.merge(folds[["muni_code", "fold", "block_id"]], on="muni_code")
    fold_colors = ["#138A8A", "#E19A1F", "#7057C9", "#C4493B", "#4A6FA5"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for f, c in enumerate(fold_colors):
        mf[mf.fold == f].plot(ax=ax, color=c, linewidth=0, alpha=0.85)
    mf.dissolve("block_id").boundary.plot(ax=ax, color="white", linewidth=0.7)
    ax.set_axis_off()
    furniture(ax, 0.001)
    fig.savefig(OUT / "map_folds.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    # ---- charts as TikZ code
    res = pd.read_csv(tables / "main_results.csv").set_index("variant")
    res = res[res.n_rounds == cfg["cv"]["n_folds"]]
    show = [v for v in LABELS if v in res.index]
    write_results_chart(res.loc[show].sort_values("r2", ascending=False), OUT / "chart_results.tex")
    before = pd.read_csv(io.StringIO(subprocess.run(
        ["git", "show", "c5557ea:reports/tables/main_results.csv"], cwd=REPO_ROOT, capture_output=True,
        text=True, check=True).stdout)).set_index("variant")
    write_calibration_chart(before, res, OUT / "chart_calibration.tex")
    rec = []
    for rd in range(cfg["cv"]["n_folds"]):
        f = data_path(cfg, "models", f"round{rd}", "stage_a_full.json")
        if f.exists():
            d = json.loads(f.read_text())
            rec.append([d["recall@1"], d["recall@5"], d["recall@10"]])
    if rec:
        write_recall_chart(np.array(rec).mean(0), OUT / "chart_recall.tex")
    sents = pd.read_parquet(data_path(cfg, "interim", "caption_sentences.parquet"))
    vc = sents.drop_reason.fillna("kept").value_counts()
    write_drops_chart(vc, OUT / "chart_drops.tex")

    # ---- caption example and backup captions
    tiles = data_path(cfg, "raw", "tiles")
    shutil.copy(tiles / f"{CAPTION_CELL}.png", OUT / "caption_tile.png")
    reason = {"filler": "filler", "truncated": "cut off"}
    lines = []
    for x in sents[sents.cell_id == CAPTION_CELL].itertuples():
        dr = x.drop_reason if isinstance(x.drop_reason, str) else ""
        t = tex_escape(x.sentence)
        if not dr:
            lines.append(r"\keep{" + t + "}")
        else:
            why = reason.get(dr, "no " + dr.replace("claim_", "") + " in the data")
            lines.append(r"\dropped{" + t + "}{" + why + "}")
    (OUT / "caption_example.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    blines = []
    sample_cells = cells.set_index("cell_id").loc[list(caps)]
    kept1 = sents[sents.drop_reason.isna()].groupby("cell_id").sentence.first()
    picks = {}
    for name, (col, thr, pat) in BACKUP_TYPES.items():
        ok = sample_cells[sample_cells[col] >= thr]
        if col != "lc_built":
            ok = ok[ok.lc_built <= 0.01]
        if name == "Village edge":
            ok = ok[ok.lc_built <= 0.15]
        ok = [c for c in ok.index if c in kept1.index and re.search(pat, kept1[c], re.I) and c not in picks.values()]
        picks[name] = sorted(ok)[len(ok) // 2]  # deterministic middle pick
    for i, (name, cid) in enumerate(picks.items(), 1):
        shutil.copy(tiles / f"{cid}.png", OUT / f"backup_tile{i}.png")
        kept_s = sents[(sents.cell_id == cid) & sents.drop_reason.isna()].sentence.tolist()
        first = kept_s[0] if kept_s else re.split(r"(?<=[.!?])\s+", caps[cid].strip())[0]
        first = re.sub(r"^(The|This) satellite image (shows|captures|depicts) ", "", first)
        blines.append(rf"\newcommand{{\bcapname{chr(64 + i)}}}{{{name}}}"
                      rf"\newcommand{{\bcap{chr(64 + i)}}}{{{tex_escape(first[0].upper() + first[1:])}}}")
    (OUT / "backup_captions.tex").write_text("\n".join(blines) + "\n", encoding="utf-8")

    # ---- numbers used in the slide text
    lo_m, hi_m = lab.loc[lab.gdp_pc.idxmin()], lab.loc[lab.gdp_pc.idxmax()]
    n = {"munis": len(lab), "cells": len(kept), "captions": len(caps),
         "kept": 100 * sents.drop_reason.isna().mean(), "msb": int(b.ms_buildings.sum())}
    rq = pd.read_csv(tables / "rq_tests.csv").set_index("question")
    within = (mp.err.abs() < np.log(1.5)).mean() * 100
    nums = [rf"\newcommand{{\nMunis}}{{{n['munis']}}}", rf"\newcommand{{\nCells}}{{{n['cells'] / 1000:.0f}k}}",
            rf"\newcommand{{\nCaptions}}{{{n['captions']:,}}}", rf"\newcommand{{\nKept}}{{{n['kept']:.0f}\%}}",
            rf"\newcommand{{\nHybridR}}{{{res.at['V10', 'r2']:.2f}}}", rf"\newcommand{{\nNtlR}}{{{res.at['V1', 'r2']:.2f}}}",
            rf"\newcommand{{\nBestSingleR}}{{{res.loc[[v for v in res.index if v != 'V10']].r2.max():.2f}}}",
            rf"\newcommand{{\nBuildings}}{{{n['msb'] / 1e6:.0f}M}}",
            rf"\newcommand{{\nHybridRMSE}}{{{res.at['V10', 'rmse']:.2f}}}",
            rf"\newcommand{{\nNtlRMSE}}{{{res.at['V1', 'rmse']:.2f}}}",
            rf"\newcommand{{\nSentences}}{{{len(sents):,}}}",
            rf"\newcommand{{\nGdpRatio}}{{{hi_m.gdp_pc / lo_m.gdp_pc:.0f}}}",
            rf"\newcommand{{\nRichName}}{{{tex_escape(hi_m.muni_name)}}}",
            rf"\newcommand{{\nRichGdp}}{{{hi_m.gdp_pc / 1000:.0f}k}}",
            rf"\newcommand{{\nPoorName}}{{{tex_escape(lo_m.muni_name)}}}",
            rf"\newcommand{{\nPoorGdp}}{{{lo_m.gdp_pc / 1000:.0f}k}}",
            rf"\newcommand{{\nWithinHalf}}{{{within:.0f}\%}}"]
    for key, reason_name in (("Filler", "filler"), ("Building", "claim_building"), ("Road", "claim_road"),
                             ("Water", "claim_water"), ("Industry", "claim_industry"), ("Farm", "claim_farm"),
                             ("Urban", "claim_urban"), ("KeptN", "kept")):
        nums.append(rf"\newcommand{{\nDrop{key}}}{{{int(vc.get(reason_name, 0)):,}}}")
    gain = 100 * (res.at["V3", "rmse"] - before.at["V3", "rmse"]) / before.at["V3", "rmse"]
    nums.append(rf"\newcommand{{\nCalibGain}}{{$-${abs(gain):.0f}\%}}")
    tx = pd.read_parquet(data_path(cfg, "processed", "texts.parquet"))
    nums.append(rf"\newcommand{{\nClipDropped}}{{{int((~tx.keep_caption.astype(bool)).sum()):,}}}")
    for key, q in (("RQone", "RQ1"), ("RQtwo", "RQ2"), ("RQthree", "RQ3"), ("RQfour", "RQ4")):
        row = rq[rq.index.str.startswith(q)]
        if len(row):
            rr = row.iloc[0]
            nums.append(rf"\newcommand{{\{key}}}{{$\Delta$RMSE {rr.rmse_diff_a_minus_b:+.3f}, "
                        rf"95\% CI [{rr.ci_low:+.3f}, {rr.ci_high:+.3f}]}}")
    (OUT / "numbers.tex").write_text("\n".join(nums) + "\n", encoding="utf-8")
    print("figures written to", OUT)
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
