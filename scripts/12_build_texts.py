"""Phase 4.6-4.11: fact sentences, caption cleaning, optional CLIP filter, audit sheet, texts.parquet.

  python scripts/12_build_texts.py --captions interim/captions_raw_base_v1.jsonl [--clip]
      LIGHT (a minute) without --clip; MEDIUM with --clip (RemoteCLIP over ~10k tiles, ~15-30 min CPU)
      -> processed/texts.parquet (cell_id, text_raw, text_full, text_facts, clip_score, keep_caption)
         interim/caption_sentences.parquet (every sentence with its drop_reason)
  python scripts/12_build_texts.py --audit-export      300 random captions, blind A/B order (PLAN 4.10)
      -> interim/audit/audit.html (tile + both texts), audit_labels.csv (fill `label`: correct/false/vague;
         optional `label2` by a second person), audit_key.csv (which of A/B is raw; do not open while labelling)
  python scripts/12_build_texts.py --audit-score       -> reports/tables/caption_audit.csv
"""
from __future__ import annotations

import argparse
import html
import json

import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.eval.metrics import cohen_kappa, wilson
from gisecon.text.clean import clean_caption, split_sentences
from gisecon.text.facts import fact_sentences


def clip_scores(cfg, cell_ids, texts) -> np.ndarray:
    """Cosine similarity of each tile and its text under RemoteCLIP ViT-B/32 (PLAN 4.9)."""
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32")
    model.load_state_dict(torch.load(hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
                                     map_location="cpu"))
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    tiles, out = data_path(cfg, "raw", "tiles"), []
    with torch.no_grad():
        for i in range(0, len(cell_ids), 64):
            img = torch.stack([prep(Image.open(tiles / f"{c}.png").convert("RGB")) for c in cell_ids[i:i + 64]])
            a = model.encode_image(img)
            b = model.encode_text(tok(list(texts[i:i + 64])))
            out.append(torch.nn.functional.cosine_similarity(a, b).numpy())
            print(f"  CLIP {min(i + 64, len(cell_ids))}/{len(cell_ids)}", flush=True)
    return np.concatenate(out)


def build(cfg, captions_path, use_clip: bool) -> None:
    caps = pd.DataFrame([json.loads(line) for line in open(captions_path)]).drop_duplicates("cell_id", keep="last")
    sample = pd.read_parquet(data_path(cfg, "interim", "caption_sample.parquet"))
    cells = pd.read_parquet(data_path(cfg, "processed", "cells.parquet")).set_index("cell_id")
    d = sample[["cell_id"]].merge(caps[["cell_id", "caption"]], on="cell_id", how="left")
    print(f"{d.caption.notna().sum():,} of {len(d):,} sampled cells have a caption")
    d["caption"] = d["caption"].fillna("")

    facts, cleaned, sents = [], [], []
    for cid, cap in zip(d["cell_id"], d["caption"]):
        row = cells.loc[cid].to_dict()
        facts.append(fact_sentences(row))
        txt, rows = clean_caption(cap, row)
        cleaned.append(txt)
        sents += [{"cell_id": cid, **r} for r in rows]
    sents = pd.DataFrame(sents)
    sents.to_parquet(data_path(cfg, "interim", "caption_sentences.parquet"), index=False)
    print("sentences per drop reason:\n" + sents["drop_reason"].fillna("kept").value_counts().to_string())

    d["text_raw"], d["text_facts"], d["cleaned"] = d["caption"], facts, cleaned
    d["clip_score"], d["keep_caption"] = np.nan, d["cleaned"].str.len() > 0
    if use_clip:
        has = d["keep_caption"].to_numpy()
        d.loc[has, "clip_score"] = clip_scores(cfg, d.loc[has, "cell_id"].tolist(), d.loc[has, "cleaned"].to_numpy())
        # cut-off fixed on a seeded 200-caption pilot subset (PLAN 4.9); uses no GDP information
        pilot = d.loc[has, "clip_score"].sample(min(200, has.sum()), random_state=cfg["project"]["seed"])
        cut = pilot.quantile(cfg["text"]["clip_drop_share"])
        d.loc[has & (d["clip_score"] < cut).to_numpy(), "keep_caption"] = False
        print(f"CLIP cut-off {cut:.3f}: {int((has & ~d.keep_caption).sum())} captions dropped")
    d["text_full"] = np.where(d["keep_caption"], d["cleaned"] + " " + d["text_facts"], d["text_facts"])
    out = data_path(cfg, "processed", "texts.parquet")
    d[["cell_id", "text_raw", "text_full", "text_facts", "clip_score", "keep_caption"]].to_parquet(out, index=False)
    print(f"wrote {out}")


def audit_export(cfg) -> None:
    texts = pd.read_parquet(data_path(cfg, "processed", "texts.parquet"))
    texts = texts[texts["text_raw"].str.len() > 0].sample(cfg["text"]["audit_n"], random_state=cfg["project"]["seed"])
    rng = np.random.default_rng(cfg["project"]["seed"])
    sents = pd.read_parquet(data_path(cfg, "interim", "caption_sentences.parquet"))
    kept = sents[sents["drop_reason"].isna()].groupby("cell_id")["sentence"].apply(" ".join)
    adir = data_path(cfg, "interim", "audit")
    adir.mkdir(exist_ok=True)
    tiles = data_path(cfg, "raw", "tiles")
    key, labels, page = [], [], ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:1100px'>"]
    for cid in texts["cell_id"]:
        versions = {"raw": texts.set_index("cell_id").at[cid, "text_raw"], "clean": kept.get(cid, "")}
        order = rng.permutation(["raw", "clean"])
        key.append({"cell_id": cid, "A": order[0], "B": order[1]})
        page.append(f"<h3>cell {cid}</h3><img src='file://{tiles / f'{cid}.png'}' width=384>")
        for letter, ver in zip("AB", order):
            ss = split_sentences(versions[ver])
            page.append(f"<p><b>{letter}</b><br>" + "<br>".join(f"{i}. {html.escape(s)}" for i, s in enumerate(ss)))
            labels += [{"cell_id": cid, "version": letter, "sent_idx": i, "sentence": s, "label": "", "label2": ""}
                       for i, s in enumerate(ss)]
    (adir / "audit.html").write_text("\n".join(page))
    pd.DataFrame(labels).to_csv(adir / "audit_labels.csv", index=False)
    pd.DataFrame(key).to_csv(adir / "audit_key.csv", index=False)
    print(f"wrote {adir}/audit.html and audit_labels.csv ({len(labels)} sentences to label)")


def audit_score(cfg) -> None:
    adir = data_path(cfg, "interim", "audit")
    lab = pd.read_csv(adir / "audit_labels.csv", dtype={"label": str, "label2": str})
    key = pd.read_csv(adir / "audit_key.csv").melt("cell_id", var_name="version", value_name="text")
    lab = lab.merge(key, on=["cell_id", "version"])
    assert lab["label"].notna().all(), "some sentences have no label yet"
    per_caption = lab.groupby(["text", "cell_id"])["label"].apply(lambda s: (s == "false").any()).reset_index()
    rows = []
    for ver, g in per_caption.groupby("text"):
        p, lo, hi = wilson(int(g["label"].sum()), len(g))
        rows.append({"version": ver, "captions": len(g), "share_with_false_claim": p, "ci_low": lo, "ci_high": hi})
    two = lab.dropna(subset=["label2"])
    if len(two):
        rows.append({"version": f"kappa over {len(two)} double-labelled sentences",
                     "share_with_false_claim": cohen_kappa(two["label"], two["label2"])})
    out = pd.DataFrame(rows)
    out.round(4).to_csv(REPO_ROOT / cfg["paths"]["reports"] / "tables" / "caption_audit.csv", index=False)
    print(out.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", help="JSONL from kaggle_captions.py, relative to DATA_DIR")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--audit-export", action="store_true")
    ap.add_argument("--audit-score", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.audit_export:
        audit_export(cfg)
    elif args.audit_score:
        audit_score(cfg)
    else:
        assert args.captions, "--captions is required to build texts.parquet"
        build(cfg, data_path(cfg, "raw").parent / args.captions, args.clip)


if __name__ == "__main__":
    main()
