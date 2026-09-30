"""Spatial cross-validation: blocks of neighbouring municipalities assigned to balanced folds.

Round r: test = fold r, validation = fold (r + 1) % k, train = the other folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    round: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def assign_folds(blocks: pd.DataFrame, n_folds: int, n_tries: int, seed: int,
                 balance_cols: dict[str, str]) -> tuple[pd.Series, float]:
    """Assign blocks to folds, keeping the fold profiles as alike as possible.

    blocks: one row per block with a municipality count column 'n' and, for each entry of
    balance_cols (name -> how), a column to balance: how='sum' balances fold totals (counts),
    how='mean' balances municipality-weighted means.
    Each try deals blocks round-robin after a random shuffle, so fold block counts differ by at most 1.
    Returns (fold per block, imbalance score of the best try).
    """
    rng = np.random.default_rng(seed)
    n = blocks["n"].to_numpy(float)
    order_base = np.arange(len(blocks))
    best, best_score = None, np.inf
    cols = {c: blocks[c].to_numpy(float) for c in balance_cols}
    for _ in range(n_tries):
        order = rng.permutation(order_base)
        fold = np.empty(len(blocks), dtype=int)
        fold[order] = np.arange(len(blocks)) % n_folds
        score = 0.0
        for c, how in balance_cols.items():
            v = cols[c]
            if how == "sum":
                per = np.bincount(fold, weights=v, minlength=n_folds)
            else:  # municipality-weighted mean
                per = np.bincount(fold, weights=v * n, minlength=n_folds) / np.bincount(fold, weights=n, minlength=n_folds)
            score += per.std() / (abs(per.mean()) + 1e-12)
        if score < best_score:
            best, best_score = fold.copy(), score
    return pd.Series(best, index=blocks.index, name="fold"), best_score


def get_split(folds: pd.DataFrame, rnd: int, code_col: str = "muni_code") -> Split:
    k = int(folds["fold"].max()) + 1
    test_f, val_f = rnd % k, (rnd + 1) % k
    f = folds["fold"].to_numpy()
    codes = folds[code_col].to_numpy()
    return Split(rnd, train=codes[(f != test_f) & (f != val_f)], val=codes[f == val_f], test=codes[f == test_f])
