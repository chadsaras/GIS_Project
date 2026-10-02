"""Words that separate two groups of texts: log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008)."""
from __future__ import annotations

import numpy as np


def log_odds_z(a: np.ndarray, b: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """z-score per word; positive = typical of group a. a, b: word counts per group; prior: word counts in
    a background corpus (here all captions), which shrinks rare words toward zero."""
    na, nb, p0 = a.sum(), b.sum(), prior.sum()
    la = np.log((a + prior) / (na + p0 - a - prior))
    lb = np.log((b + prior) / (nb + p0 - b - prior))
    return (la - lb) / np.sqrt(1 / (a + prior) + 1 / (b + prior))
