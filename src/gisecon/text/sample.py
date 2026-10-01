"""Which cells get a caption (PLAN 4.2): a fixed number per fold, spread evenly over built-up x night-light strata."""
from __future__ import annotations

import numpy as np
import pandas as pd


def caption_sample(kept: pd.DataFrame, fold_of_muni: pd.Series, per_fold: int, seed: int) -> pd.DataFrame:
    """kept: kept cells with cell_id, muni_code, lc_built, ntl.

    Strata = built-up tercile x night-light decile (computed over all kept cells). Each stratum of a
    fold gets the same quota, which oversamples towns relative to the many rural pasture cells; small
    strata are taken whole and the shortfall is filled at random from the rest of the fold.
    weight = cells in the stratum / cells sampled from it (inverse sampling probability).
    """
    rng = np.random.default_rng(seed)
    d = kept[["cell_id", "muni_code"]].copy()
    d["fold"] = d["muni_code"].map(fold_of_muni).to_numpy()
    built = pd.qcut(kept["lc_built"].rank(method="first"), 3, labels=False)
    light = pd.qcut(kept["ntl"].rank(method="first"), 10, labels=False)  # rank: ties at 0 still split evenly
    d["stratum"] = (built * 10 + light).to_numpy()

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
