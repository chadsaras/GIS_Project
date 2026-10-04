"""Phase 4.6-4.11: fact sentences, caption cleaning, optional CLIP filter, audit sheet, texts.parquet.

  python scripts/12_build_texts.py --captions interim/captions_raw_base_v1.jsonl [--clip]
      LIGHT (a minute) without --clip; MEDIUM with --clip (RemoteCLIP over ~10k tiles, ~15-30 min CPU)
      -> processed/texts.parquet (cell_id, text_raw, text_full, text_facts, clip_score, keep_caption)
         interim/caption_sentences.parquet (every sentence with its drop_reason)
  python scripts/12_build_texts.py --audit-export      300 random raw captions, every sentence once (PLAN 4.10)
      -> interim/audit/audit.html (tile + numbered sentences), audit_labels.csv (fill `label`: correct/false/vague;
         optional `label2` by a second person). Drop decisions are not shown, so labelling is blind to cleaning.
  python scripts/12_build_texts.py --audit-score       -> reports/tables/caption_audit.csv
      raw vs cleaned false-claim share (Wilson 95% CI), cleaning precision and recall, kappa
"""
from __future__ import annotations

import argparse
import html
import json
import shutil

import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.eval.metrics import cohen_kappa, wilson
from gisecon.text.clean import clean_caption
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
    bfile = data_path(cfg, "interim", "buildings.parquet")  # building evidence for the claim checks (03b)
    if bfile.exists():
        b = pd.read_parquet(bfile).set_index("cell_id")
        cells = cells.join(b).fillna({"ms_buildings": 0, "osm_buildings": 0})
    else:
        print("warning: interim/buildings.parquet missing, building claims use WorldCover and POIs only")
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
        has = d["keep_caption"].to_numpy().copy()  # copy: keep_caption is changed below
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
    """Every sentence of 300 random raw captions, labelled once. The labeller never sees which sentences the
    cleaning dropped, so raw and cleaned results are both derived from the same blind labels."""
    texts = pd.read_parquet(data_path(cfg, "processed", "texts.parquet"))
    cids = texts.loc[texts["text_raw"].str.len() > 0, "cell_id"].sample(
        cfg["text"]["audit_n"], random_state=cfg["project"]["seed"]).tolist()
    sents = _sentences(cfg)
    sents = sents[sents["cell_id"].isin(cids)]
    adir = data_path(cfg, "interim", "audit")
    adir.mkdir(exist_ok=True)
    tiles = data_path(cfg, "raw", "tiles")
    (adir / "tiles").mkdir(exist_ok=True)  # self-contained folder: copy the tiles so it works on any computer
    for cid in cids:
        shutil.copy(tiles / f"{cid}.png", adir / "tiles" / f"{cid}.png")
    page = ["<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:1100px'>",
            "<p>Label each sentence in audit_labels.csv: <b>correct</b>, <b>false</b> (claims something not "
            "in the image) or <b>vague</b> (true but says nothing specific).</p>"]
    for cid in cids:
        s = sents[sents["cell_id"] == cid]
        page.append(f"<h3>cell {cid}</h3><img src='tiles/{cid}.png' width=384><p>"
                    + "<br>".join(f"{i}. {html.escape(t)}" for i, t in zip(s["sent_idx"], s["sentence"])))
    (adir / "audit.html").write_text("\n".join(page))
    out = sents[["cell_id", "sent_idx", "sentence"]].assign(label="", label2="")
    out.to_csv(adir / "audit_labels.csv", index=False)
    print(f"wrote {adir}/audit.html and audit_labels.csv ({len(out)} sentences from {len(cids)} captions)")


def _sentences(cfg) -> pd.DataFrame:
    """caption_sentences.parquet with the sentence position inside its caption."""
    s = pd.read_parquet(data_path(cfg, "interim", "caption_sentences.parquet"))
    return s.assign(sent_idx=s.groupby("cell_id").cumcount())


def audit_score(cfg) -> None:
    """PLAN 4.10: share of captions with at least one false claim, raw vs cleaned, plus how well cleaning did."""
    adir = data_path(cfg, "interim", "audit")
    lab = pd.read_csv(adir / "audit_labels.csv", dtype={"label": str, "label2": str})
    assert lab["label"].notna().all(), "some sentences have no label yet"
    assert set(lab["label"]) <= {"correct", "false", "vague"}, f"unexpected labels: {set(lab['label'])}"
    lab = lab.merge(_sentences(cfg)[["cell_id", "sent_idx", "drop_reason"]], on=["cell_id", "sent_idx"])
    texts = pd.read_parquet(data_path(cfg, "processed", "texts.parquet")).set_index("cell_id")
    lab["caption_kept"] = lab["cell_id"].map(texts["keep_caption"]).astype(bool)  # False: dropped by CLIP / empty
    lab["kept"] = lab["drop_reason"].isna() & lab["caption_kept"]
    is_false = lab["label"] == "false"

    rows = []
    raw = is_false.groupby(lab["cell_id"]).any()
    clean = (is_false & lab["kept"]).groupby(lab["cell_id"]).any()[lab.groupby("cell_id")["kept"].any()]
    for name, per_cap in (("raw captions", raw), ("cleaned captions (those still used)", clean)):
        p, lo, hi = wilson(int(per_cap.sum()), len(per_cap))
        rows.append({"measure": f"share with >= 1 false claim: {name}", "n": len(per_cap),
                     "value": p, "ci_low": lo, "ci_high": hi})
    dropped = lab[lab["drop_reason"].notna()]  # removed by the filler / claim rules (passes 1-2)
    for name, k, n in (("rule-dropped sentences that were false or vague (cleaning precision)",
                        int(dropped["label"].isin(["false", "vague"]).sum()), len(dropped)),
                       ("false sentences removed by any cleaning pass (cleaning recall)",
                        int((is_false & ~lab["kept"]).sum()), int(is_false.sum()))):
        p, lo, hi = wilson(k, n)
        rows.append({"measure": name, "n": n, "value": p, "ci_low": lo, "ci_high": hi})
    two = lab[lab["label2"].notna() & (lab["label2"] != "")]
    if len(two):
        rows.append({"measure": "Cohen's kappa, double-labelled sentences", "n": len(two),
                     "value": cohen_kappa(two["label"], two["label2"])})
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
