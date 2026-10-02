"""Phase 9.9-9.10: text explanations for a few municipalities and the rich-vs-poor word contrast (LIGHT: a minute).

Run after 14_error_analysis.py; needs processed/texts.parquet and the Stage A "full" runs (V5-V7).
  reports/tables/explanations.md     for 5 municipalities picked by rule (largest city, top agribusiness town,
                                     top industry/mining town, poorest town, largest miss of the best model):
                                     the 5 training captions nearest to the municipality's mean cell embedding
                                     in the Stage A space of the round where it is tested
  reports/figures/word_contrast.png  15 words most typical of captions in the top vs bottom 20% predicted municipalities
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.eval.folds import get_split
from gisecon.models.stage_a import StageA
from gisecon.text.words import log_odds_z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="default: complete variant with the lowest validation RMSE")
    args = ap.parse_args()
    cfg = load_config()
    rep = REPO_ROOT / cfg["paths"]["reports"]
    res = pd.read_csv(rep / "tables" / "main_results.csv").set_index("variant")
    best = args.model or res[res.n_rounds == cfg["cv"]["n_folds"]].drop(index="V1", errors="ignore")["val_rmse"].idxmin()
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet"))
    preds = pd.concat([pd.read_parquet(f) for f in data_path(cfg, "processed", "predictions").glob("*.parquet")])
    pred = preds[(preds.variant == best) & (preds.split == "test")].set_index("muni_code")["y_pred"]
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet"), columns=["cell_id", "muni_code", "keep"])
    kept = cells[cells.keep].reset_index(drop=True)
    texts = pd.read_parquet(data_path(cfg, "processed", "texts.parquet"))
    temb_all = np.load(data_path(cfg, "interim", "text_emb_full.npy")).astype(np.float32)
    texts["muni_code"] = texts.cell_id.map(cells.set_index("cell_id").muni_code)

    # 9.9 explanations
    big = lab[lab["pop"] > 20000]
    picks = {"largest city": lab["pop"].idxmax(), "agribusiness town": big.share_agro.idxmax(),
             "industry/mining town": big.share_industry.idxmax(), "poorest town": lab.log_gdp_pc.idxmin(),
             f"largest miss of {best}": (pred - lab.log_gdp_pc.reindex(pred.index)).abs().idxmax()}
    lines = [f"# Nearest training captions (Stage A space, text_full), best model {best}\n"]
    for role, code in picks.items():
        r = int(folds.set_index("muni_code").at[code, "fold"])  # the round where this municipality is tested
        split = get_split(folds, r)
        model = StageA(64, temb_all.shape[1], **{k: cfg["stage_a"][k] for k in
                                                 ("proj_hidden", "emb_dim", "dropout", "temperature_init")})
        model.load_state_dict(torch.load(data_path(cfg, "models", f"round{r}", "stage_a_full.pt")))
        model.eval()
        emb = np.load(data_path(cfg, "interim", f"round{r}", "emb_A_full.npy")).astype(np.float32)
        q = emb[(kept.muni_code == code).to_numpy()].mean(0)
        train = texts.muni_code.isin(split.train).to_numpy()
        with torch.no_grad():
            t = model.embed_text(torch.from_numpy(temb_all[train])).numpy()
        sim = t @ (q / np.linalg.norm(q))
        top = np.argsort(-sim)[:5]
        lines.append(f"## {role}: {lab.at[code, 'muni_name']} ({code}), round {r}\n\nofficial "
                     f"{lab.at[code, 'log_gdp_pc']:.2f}, predicted {pred.get(code, np.nan):.2f}\n")
        lines += [f"{i + 1}. ({sim[j]:.2f}) {texts.text_full.to_numpy()[train][j]}" for i, j in enumerate(top)]
        lines.append("")
    (rep / "tables" / "explanations.md").write_text("\n".join(lines))
    print("wrote reports/tables/explanations.md")

    # 9.10 word contrast: captions only (fact sentences are templated, so their words would dominate)
    q_hi, q_lo = pred.quantile(0.8), pred.quantile(0.2)
    sents = pd.read_parquet(data_path(cfg, "interim", "caption_sentences.parquet"))
    caps = sents[sents.drop_reason.isna()].groupby("cell_id")["sentence"].apply(" ".join).rename("caption")
    caps = caps.to_frame().join(cells.set_index("cell_id").muni_code)
    group = caps.muni_code.map(pred)
    vec = CountVectorizer(stop_words="english", min_df=5, token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X = vec.fit_transform(caps.caption)
    a = np.asarray(X[(group >= q_hi).to_numpy()].sum(0)).ravel()
    b = np.asarray(X[(group <= q_lo).to_numpy()].sum(0)).ravel()
    z = log_odds_z(a, b, np.asarray(X.sum(0)).ravel())
    words = vec.get_feature_names_out()
    order = np.argsort(z)
    sel = np.r_[order[:15], order[-15:]]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(words[sel], z[sel], color=np.where(z[sel] > 0, "seagreen", "indianred"))
    ax.set_xlabel("log-odds z (right: top 20% predicted, left: bottom 20%)")
    ax.set_title(f"Caption words, rich vs poor municipalities ({best})")
    fig.tight_layout()
    fig.savefig(rep / "figures" / "word_contrast.png", dpi=150)
    print("wrote reports/figures/word_contrast.png")


if __name__ == "__main__":
    main()
