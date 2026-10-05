"""Figures for the mid-term presentation, generated from the project's real data and results.

Run on the server from the repo root:  python presentation/midterm/make_figures.py
Writes PNG maps / tiles and PDF charts to presentation/midterm/assets/, plus small .tex snippets
(caption example, backup captions, numbers) that slides.tex reads, so slide numbers match the results.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.data.cells import GSED
from gisecon.data.grid import Grid

OUT = Path(__file__).resolve().parent / "assets"  # draft_v2_template keeps its own assets
ACCENT, ACCENT_DARK, GREY = "#2B6CB0", "#1A3E6E", "#A0AEC0"
CAPTION_CELL = 236778          # centre-pivot field the model called "a mine pit" (removed by the industry check)
# backup slide: one clear example per land type, picked by rule in main()
BACKUP_TYPES = {"Forest": ("lc_tree", 0.85, r"forest|trees|woodland|vegetation"),
                "Farmland": ("lc_crop", 0.5, r"farm|field|crop|plot|agricultur"),
                "Village edge": ("lc_built", 0.05, r"house|building|residential|roof"),
                "Town centre": ("lc_built", 0.6, r"dense|urban|grid|roof")}
LABELS = {"V1": "Night lights only", "V2": "Land-cover shares", "V11_gsed": "Ridge, mean GSED",
          "V12": "LightGBM, mean GSED", "V3": "GSED + Stage C", "V4": "GSED + steering (B) + C",
          "V5": "Text-aligned (A) + C", "V6": "A + B + C", "V7": "A + B + C + POI/roads",
          "V8": "Full, raw captions", "V9": "Full, fact sentences only", "V10": "Hybrid: best + night lights"}
OURS = {"V3", "V4", "V5", "V6", "V7", "V8", "V9"}

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


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


def save_map(img, path, cmap=None, norm=None, vmin=None, vmax=None, size=(5, 4.2)):
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def write_results_chart(r: pd.DataFrame, path: Path) -> None:
    """Horizontal R2 bars, best first: hybrid gold, our pipeline navy, baselines grey."""
    X = 15.0  # cm per unit of R2
    lines = [r"\begin{tikzpicture}[x=1cm,y=1cm,",
             r"  mname/.style={font=\sffamily\fontsize{8}{9.6}\selectfont,text=kgpink,anchor=east,inner sep=0pt},",
             r"  val/.style={font=\sffamily\fontsize{7.5}{9}\selectfont,text=kgpink,anchor=west,inner xsep=1.5pt},",
             r"  tick/.style={font=\sffamily\fontsize{7}{8.4}\selectfont,text=mdlgrey,anchor=north},",
             r"  tag/.style={font=\sffamily\bfseries\fontsize{6.5}{7.8}\selectfont,text=white,rounded corners=1.5pt,"
             r"inner xsep=2.5pt,inner ysep=1.2pt,anchor=east}]"]
    n = len(r)
    bottom = -0.62 * (n - 1) - 0.4
    for t in np.arange(0, 0.61, 0.1):
        lines.append(rf"\draw[mdlgrey!35,line width=0.4pt,dashed] ({t * X:.2f},0.35) -- ({t * X:.2f},{bottom:.2f});")
        lines.append(rf"\node[tick] at ({t * X:.2f},{bottom - 0.08:.2f}) {{{t:.1f}}};")
    lines.append(rf"\draw[mdlgrey,line width=0.5pt] (0,{bottom:.2f}) -- ({0.6 * X:.2f},{bottom:.2f});")
    lines.append(rf"\node[tick] at ({0.3 * X:.2f},{bottom - 0.45:.2f}) {{R\textsuperscript{{2}} on held-out municipalities"
                 rf" \quad (5 spatial folds, n = 853)}};")
    for i, (v, row) in enumerate(r.iterrows()):
        y = -0.62 * i
        style = ("draw=kgpgold,fill=kgpgold!75" if v == "V10" else
                 "draw=kgpnavy,fill=kgpnavy!80" if v in OURS else "draw=mdlgrey,fill=mdlgreybg")
        lines.append(rf"\node[mname] at (-0.15,{y:.2f}) {{{tex_escape(LABELS[v])}}};")
        lines.append(rf"\path[{style},line width=0.6pt] (0,{y - 0.2:.2f}) rectangle ({row.r2 * X:.2f},{y + 0.2:.2f});")
        lines.append(rf"\node[val] at ({row.r2 * X + 0.08:.2f},{y:.2f}) {{\textbf{{{row.r2:.2f}}}\enspace"
                     rf"\textcolor{{mdlgrey}}{{RMSE {row.rmse:.2f}}}}};")
        if v == "V10":
            lines.append(rf"\node[tag,fill=kgpnavydark] at ({row.r2 * X - 0.07:.2f},{y:.2f}) {{BEST}};")
    ly = bottom - 0.95
    for k, (style, name) in enumerate([("draw=kgpnavy,fill=kgpnavy!80", "our pipeline"),
                                        ("draw=mdlgrey,fill=mdlgreybg", "baselines"),
                                        ("draw=kgpgold,fill=kgpgold!75", "hybrid: blended with night lights")]):
        x0 = [0, 2.6, 4.9][k]
        lines.append(rf"\path[{style},line width=0.6pt] ({x0},{ly - 0.1:.2f}) rectangle ({x0 + 0.3},{ly + 0.1:.2f});")
        lines.append(rf"\node[val] at ({x0 + 0.35},{ly:.2f}) {{{name}}};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_chart(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> None:
    """Dumbbell: RMSE before -> after the validation calibration, for the two Stage C variants."""
    X0, X = 0.40, 30.0  # RMSE 0.40 at x=0, 30 cm per unit

    def ax(v):
        return (v - X0) * X

    lines = [r"\begin{tikzpicture}[x=1cm,y=1cm,",
             r"  mname/.style={font=\sffamily\fontsize{8}{9.6}\selectfont,text=kgpink,anchor=east,inner sep=0pt},",
             r"  num/.style={font=\sffamily\bfseries\fontsize{8}{9.6}\selectfont,inner sep=1pt},",
             r"  tick/.style={font=\sffamily\fontsize{7}{8.4}\selectfont,text=mdlgrey,anchor=north}]"]
    for t in np.arange(0.40, 0.601, 0.05):
        lines.append(rf"\draw[mdlgrey!35,line width=0.4pt,dashed] ({ax(t):.2f},0.55) -- ({ax(t):.2f},-1.85);")
        lines.append(rf"\node[tick] at ({ax(t):.2f},-1.93) {{{t:.2f}}};")
    lines.append(rf"\draw[mdlgrey,line width=0.5pt] (0,-1.85) -- ({ax(0.60):.2f},-1.85);")
    lines.append(rf"\node[tick] at ({ax(0.50):.2f},-2.3) {{RMSE on held-out municipalities (lower is better)}};")
    for i, (v, name) in enumerate([("V3", "GSED + Stage C"), ("V4", "GSED + steering + C")]):
        y = -1.15 * i
        b, a = before.at[v, "rmse"], after.at[v, "rmse"]
        lines.append(rf"\node[mname] at (-0.2,{y:.2f}) {{{name}}};")
        lines.append(rf"\draw[-{{Stealth[length=2.4mm]}},kgpnavy,line width=1.6pt] ({ax(b) - 0.15:.2f},{y:.2f}) -- "
                     rf"({ax(a) + 0.18:.2f},{y:.2f});")
        lines.append(rf"\node[circle,draw=mdlgrey,fill=white,line width=1pt,minimum size=2.6mm,inner sep=0pt] "
                     rf"at ({ax(b):.2f},{y:.2f}) {{}};")
        lines.append(rf"\node[circle,fill=kgpnavy,minimum size=2.8mm,inner sep=0pt] at ({ax(a):.2f},{y:.2f}) {{}};")
        lines.append(rf"\node[num,text=mdlgrey,anchor=south] at ({ax(b):.2f},{y + 0.14:.2f}) {{{b:.3f}}};")
        lines.append(rf"\node[num,text=kgpnavy,anchor=south] at ({ax(a):.2f},{y + 0.14:.2f}) {{{a:.3f}}};")
        lines.append(rf"\node[num,text=kgpgreen,anchor=north] at ({(ax(a) + ax(b)) / 2:.2f},{y - 0.12:.2f})"
                     rf" {{{100 * (a - b) / b:+.0f}\%}};")
    lines.append(r"\node[circle,draw=mdlgrey,fill=white,line width=1pt,minimum size=2.4mm,inner sep=0pt] at (0.2,1.0) {};")
    lines.append(r"\node[font=\sffamily\fontsize{7.5}{9}\selectfont,anchor=west] at (0.35,1.0) {before};")
    lines.append(r"\node[circle,fill=kgpnavy,minimum size=2.4mm,inner sep=0pt] at (1.6,1.0) {};")
    lines.append(r"\node[font=\sffamily\fontsize{7.5}{9}\selectfont,anchor=west] at (1.75,1.0) "
                 r"{after validation calibration};")
    lines.append(r"\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recall_chart(rec: np.ndarray, path: Path) -> None:
    """Grouped bars: Stage A recall@k on validation captions vs chance (batches of 256)."""
    Y = 0.055  # cm per percentage point
    lines = [r"\begin{tikzpicture}[x=1cm,y=1cm,",
             r"  lbl/.style={font=\sffamily\fontsize{7.5}{9}\selectfont,text=kgpink,anchor=north,inner sep=0pt},",
             r"  num/.style={font=\sffamily\bfseries\fontsize{8}{9.6}\selectfont,anchor=south,inner sep=1pt},",
             r"  tick/.style={font=\sffamily\fontsize{6.5}{7.8}\selectfont,text=mdlgrey,anchor=east,inner sep=0pt}]"]
    for t in range(0, 61, 20):
        lines.append(rf"\draw[mdlgrey!25,line width=0.4pt] (0,{t * Y:.2f}) -- (6.2,{t * Y:.2f});")
        lines.append(rf"\node[tick] at (-0.1,{t * Y:.2f}) {{{t}\%}};")
    for i, (k, v) in enumerate(zip([1, 5, 10], rec * 100)):
        x = 1.0 + 2.0 * i
        ch = k / 256 * 100
        lines.append(rf"\fill[kgpnavy] ({x - 0.62:.2f},0) rectangle ({x - 0.02:.2f},{v * Y:.2f});")
        lines.append(rf"\node[num,text=kgpnavy] at ({x - 0.32:.2f},{v * Y:.2f}) {{{v:.0f}\%}};")
        lines.append(rf"\fill[mdlgrey!45] ({x + 0.02:.2f},0) rectangle ({x + 0.62:.2f},{max(ch * Y, 0.03):.2f});")
        lines.append(rf"\node[num,text=mdlgrey] at ({x + 0.32:.2f},{max(ch * Y, 0.03):.2f}) {{{ch:.0f}\%}};")
        lines.append(rf"\node[lbl] at ({x:.2f},-0.08) {{recall@{k}}};")
    lines.append(r"\fill[kgpnavy] (0.3,3.75) rectangle (0.6,3.95); "
                 r"\node[font=\sffamily\fontsize{7.5}{9}\selectfont,anchor=west] at (0.65,3.85) {Stage A};")
    lines.append(r"\fill[mdlgrey!45] (2.1,3.75) rectangle (2.4,3.95); "
                 r"\node[font=\sffamily\fontsize{7.5}{9}\selectfont,anchor=west] at (2.45,3.85) {chance};")
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

    # GSED principal components as RGB (title slide, data slide)
    g = kept[GSED].to_numpy(np.float64)
    g = g - g.mean(0)
    pcs = g @ np.linalg.svd(g[:: max(1, len(g) // 50_000)], full_matrices=False)[2][:3].T
    lo, hi = np.percentile(pcs, [2, 98], axis=0)
    rgb = raster(np.clip((pcs - lo) / (hi - lo), 0, 1), rows, cols, grid)
    alpha = (~np.isnan(rgb[..., 0])).astype(np.float32)[..., None]
    rgba = np.concatenate([np.nan_to_num(rgb, nan=1.0), alpha], axis=-1)
    save_map(crop(rgba, rows, cols), OUT / "hero_gsed.png", size=(6, 5))

    # thumbnails for the data slide (same extent everywhere)
    save_map(crop(raster(kept.log_ntl.to_numpy(), rows, cols, grid), rows, cols), OUT / "thumb_ntl.png",
             cmap="magma", size=(3, 2.6))
    save_map(crop(raster(np.sqrt(kept.lc_built.to_numpy()), rows, cols, grid), rows, cols),
             OUT / "thumb_built.png", cmap="viridis", vmax=0.5, size=(3, 2.6))
    road = kept.filter(regex=r"^road_.*_km$").sum(axis=1).to_numpy()
    save_map(crop(raster(road, rows, cols, grid), rows, cols), OUT / "thumb_roads.png", cmap="Greys",
             vmax=np.percentile(road, 99), size=(3, 2.6))
    b = pd.read_parquet(data_path(cfg, "interim", "buildings.parquet")).set_index("cell_id")
    nb = kept.cell_id.map(b.ms_buildings).fillna(0).to_numpy() + 1
    save_map(crop(raster(nb, rows, cols, grid), rows, cols), OUT / "thumb_buildings.png", cmap="YlOrBr",
             norm=LogNorm(1, np.percentile(nb, 99.5)), size=(3, 2.6))
    save_map(crop(rgba, rows, cols), OUT / "thumb_gsed.png", size=(3, 2.6))

    # municipal maps: official GDP per capita, folds
    munis = gpd.read_file(data_path(cfg, "interim", "municipios.gpkg"))
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet"))
    m = munis.merge(lab[["muni_code", "gdp_pc"]], on="muni_code")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    m.plot(column="gdp_pc", cmap="viridis", norm=LogNorm(8000, 120000), ax=ax, linewidth=0.05, edgecolor="white")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=LogNorm(8000, 120000))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.02, aspect=30)
    cb.set_ticks([10000, 20000, 50000, 100000])
    cb.set_ticklabels(["R$ 10k", "20k", "50k", "100k"])
    cb.ax.set_xlabel("GDP per capita, 2022 (log scale)", fontsize=10)
    ax.set_axis_off()
    fig.savefig(OUT / "map_gdp.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3, 2.6))
    m.plot(column="gdp_pc", cmap="viridis", norm=LogNorm(8000, 120000), ax=ax, linewidth=0)
    ax.set_axis_off()
    fig.savefig(OUT / "thumb_gdp.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))
    mf = munis.merge(folds[["muni_code", "fold", "block_id"]], on="muni_code")
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for f, c in enumerate(colors):
        mf[mf.fold == f].plot(ax=ax, color=c, linewidth=0)
    mf.dissolve("block_id").boundary.plot(ax=ax, color="white", linewidth=0.6)
    for f, c in enumerate(colors):
        ax.scatter([], [], color=c, s=60, label=f"Fold {f + 1}")
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.set_axis_off()
    fig.savefig(OUT / "map_folds.png", dpi=220, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    # ---- charts as TikZ code (drawn in the deck's own fonts and colours, numbers from the results) ----
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

    # caption example and backup captions
    tiles = data_path(cfg, "raw", "tiles")
    shutil.copy(tiles / f"{CAPTION_CELL}.png", OUT / "caption_tile.png")
    sents = pd.read_parquet(data_path(cfg, "interim", "caption_sentences.parquet"))
    reason = {"filler": "filler", "truncated": "cut off"}
    lines = []
    for x in sents[sents.cell_id == CAPTION_CELL].itertuples():
        dr = x.drop_reason if isinstance(x.drop_reason, str) else ""
        t = tex_escape(x.sentence)
        if not dr:
            lines.append(r"\keep{" + t + "}")
        else:
            why = reason.get(dr, "unsupported " + dr.replace("claim_", "") + " claim")
            lines.append(r"\dropped{" + t + "}{" + why + "}")
    (OUT / "caption_example.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    caps = {json.loads(l)["cell_id"]: json.loads(l)["caption"]
            for l in open(data_path(cfg, "interim", "captions_raw_short_v1.jsonl"))}
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
        first = kept_s[0] if kept_s else re.split(r"(?<=[.!?])\s+", caps[cid].strip())[0]  # first sentence kept by cleaning
        first = re.sub(r"^(The|This) satellite image (shows|captures|depicts) ", "", first)
        blines.append(rf"\newcommand{{\bcapname{chr(64 + i)}}}{{{name}}}"
                      rf"\newcommand{{\bcap{chr(64 + i)}}}{{{tex_escape(first[0].upper() + first[1:])}}}")
    (OUT / "backup_captions.tex").write_text("\n".join(blines) + "\n", encoding="utf-8")

    # numbers used in the slide text
    n = {"munis": len(lab), "cells": len(kept), "captions": len(caps),
         "kept": 100 * sents.drop_reason.isna().mean(), "bestr": res.r2.max(),
         "hybr": res.at["V10", "r2"] if "V10" in res.index else np.nan,
         "ntlr": res.at["V1", "r2"], "rqone": "no clear effect", "msb": int(b.ms_buildings.sum())}
    rq = pd.read_csv(tables / "rq_tests.csv").set_index("question")
    nums = [rf"\newcommand{{\nMunis}}{{{n['munis']}}}", rf"\newcommand{{\nCells}}{{{n['cells'] / 1000:.0f}k}}",
            rf"\newcommand{{\nCaptions}}{{{n['captions']:,}}}", rf"\newcommand{{\nKept}}{{{n['kept']:.0f}\%}}",
            rf"\newcommand{{\nHybridR}}{{{n['hybr']:.2f}}}", rf"\newcommand{{\nNtlR}}{{{n['ntlr']:.2f}}}",
            rf"\newcommand{{\nBuildings}}{{{n['msb'] / 1e6:.0f}M}}",
            rf"\newcommand{{\nHybridRMSE}}{{{res.at['V10', 'rmse']:.2f}}}",
            rf"\newcommand{{\nNtlRMSE}}{{{res.at['V1', 'rmse']:.2f}}}",
            rf"\newcommand{{\nSentences}}{{{len(sents):,}}}"]
    vc = sents.drop_reason.fillna("kept").value_counts()
    for key, reason_name in (("Filler", "filler"), ("Building", "claim_building"), ("Road", "claim_road"),
                             ("Water", "claim_water"), ("Industry", "claim_industry"), ("Farm", "claim_farm"),
                             ("Urban", "claim_urban"), ("KeptN", "kept")):
        nums.append(rf"\newcommand{{\nDrop{key}}}{{{int(vc.get(reason_name, 0)):,}}}")
    gain = 100 * (res.at["V3", "rmse"] - before.at["V3", "rmse"]) / before.at["V3", "rmse"]
    nums.append(rf"\newcommand{{\nCalibGain}}{{$-${abs(gain):.0f}\%}}")
    tx = pd.read_parquet(data_path(cfg, "processed", "texts.parquet"))
    nums.append(rf"\newcommand{{\nClipDropped}}{{{int((~tx.keep_caption.astype(bool)).sum()):,}}}")
    for key, q in (("RQone", "RQ1"), ("RQfour", "RQ4")):
        row = rq[rq.index.str.startswith(q)]
        if len(row):
            rr = row.iloc[0]
            nums.append(rf"\newcommand{{\{key}}}{{{rr.verdict} ($\Delta$RMSE {rr.rmse_diff_a_minus_b:+.3f}, "
                        rf"95\% CI [{rr.ci_low:+.3f}, {rr.ci_high:+.3f}])}}")
    (OUT / "numbers.tex").write_text("\n".join(nums) + "\n", encoding="utf-8")
    print("figures written to", OUT)
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
