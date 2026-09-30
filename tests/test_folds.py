import numpy as np
import pandas as pd
import pytest

from gisecon.config import data_path, load_config
from gisecon.eval.folds import assign_folds, get_split


def _check_rotation(folds: pd.DataFrame, k: int) -> None:
    all_codes = set(folds.muni_code)
    tested = []
    for r in range(k):
        s = get_split(folds, r)
        tr, va, te = set(s.train), set(s.val), set(s.test)
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert tr | va | te == all_codes
        tested += list(te)
    assert sorted(tested) == sorted(all_codes)  # each municipality tested exactly once


def test_synthetic_assignment_and_rotation():
    rng = np.random.default_rng(0)
    blocks = pd.DataFrame({"n": rng.integers(3, 20, 30), "m": rng.normal(10, 1, 30)})
    fold, _ = assign_folds(blocks, 5, 200, 1, {"n": "sum", "m": "mean"})
    assert fold.value_counts().max() - fold.value_counts().min() <= 1
    munis = pd.DataFrame({"muni_code": np.arange(blocks.n.sum()),
                          "fold": np.repeat(fold.to_numpy(), blocks.n.to_numpy())})
    _check_rotation(munis, 5)


def test_real_folds_file():
    cfg = load_config()
    p = data_path(cfg, "processed", "folds.parquet")
    if not p.exists():
        pytest.skip("folds.parquet not built yet")
    folds = pd.read_parquet(p)
    assert len(folds) == cfg["study"]["n_municipalities"]
    # a block never spans two folds
    assert (folds.groupby("block_id").fold.nunique() == 1).all()
    _check_rotation(folds, cfg["cv"]["n_folds"])
