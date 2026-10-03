"""Run notebooks/kaggle_captions.ipynb on Kaggle (2x T4) from the server and fetch the captions (PLAN 4.4-4.5).

Usage:
  python scripts/17_kaggle_captions_run.py --limit 200 --wait          # pilot: 200 random tiles
  python scripts/17_kaggle_captions_run.py --wait                      # full run (all tiles)
  python scripts/17_kaggle_captions_run.py --resume --wait             # continue after a 12 h cut-off
  python scripts/17_kaggle_captions_run.py --fetch                     # only download the latest output
Needs Kaggle credentials (~/.kaggle/kaggle.json or access_token) and the dataset from 16_kaggle_dataset.py.
The captions land in DATA_DIR/interim/captions_raw_<prompt>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gisecon.config import REPO_ROOT, data_path, load_config

SLUG = "gisecon-captions"


def kaggle_bin() -> str:
    return shutil.which("kaggle") or str(Path(sys.prefix) / "bin" / "kaggle")


def username() -> str:
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    if legacy.exists():
        return json.loads(legacy.read_text())["username"]
    view = subprocess.run([kaggle_bin(), "config", "view"], capture_output=True, text=True, check=True).stdout
    return next(l.split(":", 1)[1].strip() for l in view.splitlines() if l.strip().startswith("- username:"))


def build(folder: Path, user: str, limit: int | None, prompt: str, resume: bool, dataset: str) -> None:
    """Copy the notebook with LIMIT / PROMPT_ID set, and write kernel-metadata.json."""
    nb = json.loads((REPO_ROOT / "notebooks" / "kaggle_captions.ipynb").read_text(encoding="utf-8"))
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        new = re.sub(r"(?m)^LIMIT = .*$", f"LIMIT = {limit}", src)
        new = re.sub(r'(?m)^PROMPT_ID = .*$', f'PROMPT_ID = "{prompt}"', new)
        hits += new != src
        c["source"] = new.splitlines(True)
    assert hits, "could not find the LIMIT / PROMPT_ID settings in the notebook"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kaggle_captions.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": "kaggle_captions.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",  # 2x T4 (PLAN 4.5)
            "dataset_sources": [f"{user}/{dataset}"], "competition_sources": [],
            "kernel_sources": [f"{user}/{SLUG}"] if resume else [], "model_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def status(ref: str) -> str:
    out = subprocess.run([kaggle_bin(), "kernels", "status", ref], capture_output=True, text=True).stdout
    m = re.search(r'status "?([\w.]+)"?', out)
    return m.group(1) if m else out.strip()


def fetch(cfg, ref: str, prompt: str) -> None:
    dest = data_path(cfg, "raw").parent / "kaggle" / SLUG / "output"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    subprocess.run([kaggle_bin(), "kernels", "output", ref, "-p", str(dest)], check=True)
    files = sorted(dest.rglob(f"captions_raw_{prompt}.jsonl"))
    assert files, f"no captions_raw_{prompt}.jsonl in the kernel output ({[p.name for p in dest.iterdir()]})"
    target = data_path(cfg, "interim", files[0].name)
    shutil.copy(files[0], target)
    n = sum(1 for _ in target.open())
    print(f"fetched {n:,} captions -> {target}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="pilot size (default: all tiles)")
    ap.add_argument("--prompt", default="base_v1")
    ap.add_argument("--resume", action="store_true", help="attach the previous version's output and continue")
    ap.add_argument("--dataset", default="gisecon-tiles")
    ap.add_argument("--wait", action="store_true", help="poll until the run ends, then fetch the output")
    ap.add_argument("--fetch", action="store_true", help="only download the latest output")
    args = ap.parse_args()
    cfg = load_config()
    user = username()
    ref = f"{user}/{SLUG}"

    if not args.fetch:
        folder = data_path(cfg, "raw").parent / "kaggle" / SLUG / "push"
        build(folder, user, args.limit, args.prompt, args.resume, args.dataset)
        subprocess.run([kaggle_bin(), "kernels", "push", "-p", str(folder)], check=True)
        print(f"pushed {ref} (LIMIT={args.limit}, prompt={args.prompt}, resume={args.resume}); "
              f"https://www.kaggle.com/code/{ref}")
    if args.wait:
        t0 = time.time()
        time.sleep(60)
        while (s := status(ref)).split(".")[-1].lower() in ("queued", "running"):
            print(f"  {time.time() - t0:5.0f} s: {s}", flush=True)
            time.sleep(120)
        print(f"run ended: {s}")
        if s.split(".")[-1].lower() != "complete":
            print("not complete: check the log on Kaggle; a 12 h cut-off can continue with --resume")
    if args.fetch or args.wait:
        fetch(cfg, ref, args.prompt)


if __name__ == "__main__":
    main()
