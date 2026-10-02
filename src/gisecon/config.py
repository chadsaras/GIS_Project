"""Load configs/config.yaml (or the file in $GISECON_CONFIG) and resolve data paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or os.environ.get("GISECON_CONFIG") or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_dir = os.environ.get("GISECON_DATA_DIR") or cfg["paths"]["data_dir"]
    cfg["paths"]["data_dir"] = str(_expand(data_dir))
    return cfg


def data_path(cfg: dict[str, Any], stage: str, *parts: str) -> Path:
    """Path under DATA_DIR/<stage>/..., e.g. data_path(cfg, "raw", "ibge", "pib_2022.csv")."""
    root = Path(cfg["paths"]["data_dir"]) / cfg["paths"][stage]
    return root.joinpath(*parts)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for stage in ("raw", "interim", "processed", "models"):
        data_path(cfg, stage).mkdir(parents=True, exist_ok=True)
    _expand(cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / cfg["paths"]["reports"] / "figures").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / cfg["paths"]["reports"] / "tables").mkdir(parents=True, exist_ok=True)
