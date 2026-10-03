"""Package the caption tiles as a Kaggle dataset (PLAN 4.5) and, if a Kaggle API token is present, upload it.

Usage:
  python scripts/16_kaggle_dataset.py            # zip -> DATA_DIR/kaggle/gisecon-tiles/ (+ dataset-metadata.json)
  python scripts/16_kaggle_dataset.py --upload   # also upload with the kaggle CLI (needs ~/.kaggle/kaggle.json)
Without a token, upload DATA_DIR/kaggle/gisecon-tiles/gisecon-tiles.zip by hand (docs/kaggle_steps.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from gisecon.config import data_path, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--slug", default="gisecon-tiles")
    args = ap.parse_args()
    cfg = load_config()

    tiles = data_path(cfg, "raw", "tiles")
    sample = pd.read_parquet(data_path(cfg, "interim", "caption_sample.parquet"))
    pngs = [tiles / f"{c}.png" for c in sample["cell_id"]]
    missing = [p.name for p in pngs if not p.exists()]
    assert not missing, f"{len(missing)} sampled tiles not downloaded yet (e.g. {missing[:3]}): run 11_download_tiles.py"

    out = data_path(cfg, "raw").parent / "kaggle" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{args.slug}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for p in pngs:
            z.write(p, arcname=f"tiles/{p.name}")
    print(f"wrote {zpath} ({len(pngs):,} tiles, {zpath.stat().st_size / 1e9:.2f} GB)")

    meta = {"title": "gisecon caption tiles (Minas Gerais, 1 km cells)", "id": f"<USERNAME>/{args.slug}",
            "licenses": [{"name": "other"}],
            "description": f"{len(pngs):,} 768x768 PNG satellite tiles, one per sampled 1 km cell, for captioning. "
                           f"Imagery: {cfg['text'].get('tile_attribution', 'see project report')}. Private research use."}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    if not args.upload:
        print("next: upload the zip by hand (docs/kaggle_steps.md), or rerun with --upload after adding a token")
        return
    token = Path.home() / ".kaggle" / "kaggle.json"
    assert token.exists(), f"no Kaggle API token at {token}"
    user = json.loads(token.read_text())["username"]
    meta["id"] = f"{user}/{args.slug}"
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle = shutil.which("kaggle") or str(Path(sys.prefix) / "bin" / "kaggle")
    subprocess.run([kaggle, "datasets", "create", "-p", str(out)], check=True)  # private by default
    print(f"uploaded: https://www.kaggle.com/datasets/{user}/{args.slug}")


if __name__ == "__main__":
    main()
