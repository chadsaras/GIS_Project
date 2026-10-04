"""Which cells get a caption (PLAN 4.2): a fixed number per fold, spread evenly over built-up x night-light strata."""
from __future__ import annotations

import numpy as np
import pandas as pd


BUILT_EDGES = [0.001, 0.01, 0.05, 0.20]  # built-up share classes: ~0, <1%, 1-5%, 5-20%, >20%


def strata(kept: pd.DataFrame) -> np.ndarray:
    """Built-up class (5) x night-light class (dark, then quartiles of the lit cells: 5) = 25 strata.

    Fixed meaningful bins, not rank quantiles: most cells have no built-up land and ~80% have no
    light, so rank quantiles would split the empty cells into many look-alike strata and the equal
    quotas would mostly sample empty land.
    """
    built = np.digitize(kept["lc_built"].to_numpy(), BUILT_EDGES)
    ntl = kept["ntl"].to_numpy()
    lit = ntl > 0
    light = np.zeros(len(kept), dtype=int)
    if lit.any():
        light[lit] = 1 + np.digitize(ntl[lit], np.quantile(ntl[lit], [0.25, 0.5, 0.75]))
    return built * 5 + light


def caption_sample(kept: pd.DataFrame, fold_of_muni: pd.Series, per_fold: int, seed: int) -> pd.DataFrame:
    """kept: kept cells with cell_id, muni_code, lc_built, ntl.

    Strata = built-up class x night-light class (see strata()). Each stratum of a fold gets the same
    quota, which oversamples towns relative to the many rural pasture cells; small strata are taken
    whole and the shortfall is filled at random from the rest of the fold.
    weight = cells in the stratum / cells sampled from it (inverse sampling probability).
    """
    rng = np.random.default_rng(seed)
    d = kept[["cell_id", "muni_code"]].copy()
    d["fold"] = d["muni_code"].map(fold_of_muni).to_numpy()
    d["stratum"] = strata(kept)

    picked = []
    for _, f in d.groupby("fold"):
        groups = list(f.groupby("stratum").groups.values())
        quota = per_fold // len(groups)
        take = [rng.choice(g, min(quota, len(g)), replace=False) for g in groups]
        chosen = np.concatenate(take)
        rest = f.index.difference(chosen)
        short = min(per_fold - len(chosen), len(rest))
        picked.append(np.concatenate([chosen, rng.choice(rest, short, replace=False)]))
    out = d.loc[np.concatenate(picked)].copy()
    out["weight"] = d.groupby(["fold", "stratum"]).size().reindex(
        pd.MultiIndex.from_frame(out[["fold", "stratum"]])).to_numpy() / out.groupby(["fold", "stratum"])[
        "cell_id"].transform("size").to_numpy()
    return out.reset_index(drop=True)
