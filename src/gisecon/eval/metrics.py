"""Scores and block-bootstrap comparisons on log GDP per capita (PLAN 8.7-8.8)."""
from __future__ import annotations

import numpy as np


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """R2, RMSE, MAE, and the calibration line official = a + b * predicted (b near 1, a near 0 = no bias)."""
    e = pred - y
    b, a = np.polyfit(pred, y, 1)
    return {"r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mae": float(np.abs(e).mean()), "calib_slope": float(b), "calib_intercept": float(a)}


def block_bootstrap_rmse_diff(y, pred_a, pred_b, block, reps: int, seed: int) -> tuple[float, float, float]:
    """RMSE(a) - RMSE(b) and its 95% interval, resampling whole blocks (immediate regions) with replacement.

    Negative = a is better. An interval that contains 0 means no clear difference.
    """
    _, bidx = np.unique(block, return_inverse=True)
    nb = bidx.max() + 1
    sa = np.bincount(bidx, (pred_a - y) ** 2, nb)
    sb = np.bincount(bidx, (pred_b - y) ** 2, nb)
    n = np.bincount(bidx, minlength=nb)
    w = np.random.default_rng(seed).multinomial(nb, np.full(nb, 1 / nb), size=reps)  # times each block is drawn
    diff = np.sqrt(w @ sa / (w @ n)) - np.sqrt(w @ sb / (w @ n))
    point = np.sqrt(sa.sum() / n.sum()) - np.sqrt(sb.sum() / n.sum())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Share k/n with its 95% Wilson interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    return p, mid - c, mid + c


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
