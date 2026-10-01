"""Phase 8.7-8.8: main results table and the block-bootstrap test for each research question.

Reads every processed/predictions/*.parquet (from 07_run_experiment.py and 08_baselines.py) and
scores test-fold rows only. A variant is "complete" when it has all 5 rounds, i.e. every municipality
tested once; incomplete variants are still listed, with n_rounds showing what is missing.

Outputs: reports/tables/main_results.csv, reports/tables/rq_tests.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gisecon.config import REPO_ROOT, data_path, load_config
from gisecon.eval.metrics import block_bootstrap_rmse_diff, scores

RQ_PAIRS = [("RQ1 night-light steering", "V4", "V3"), ("RQ2 text + steering", "V6", "V4"),
            ("RQ3 POI/roads on top", "V7", "V6")]  # RQ4 (V10 vs best single) is added below


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tables = REPO_ROOT / cfg["paths"]["reports"] / "tables"

    files = sorted(data_path(cfg, "processed", "predictions").glob("*.parquet"))
    assert files, "no predictions yet: run 07_run_experiment.py / 08_baselines.py first"
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lab = pd.read_parquet(data_path(cfg, "processed", "municipal_labels.parquet")).set_index("muni_code")
    folds = pd.read_parquet(data_path(cfg, "processed", "folds.parquet")).set_index("muni_code")
    n_rounds = cfg["cv"]["n_folds"]

    test = preds[preds["split"] == "test"]
    rows = []
    for v, d in test.groupby("variant"):
        flagged = lab.loc[d["muni_code"], "flag_extreme"].to_numpy()
        per_round = np.sqrt(((d.y_pred - d.y_true) ** 2).groupby(d["round"]).mean())
        val = preds[(preds["variant"] == v) & (preds["split"] == "val")]
        rows.append({"variant": v, "n_rounds": d["round"].nunique(), "n_munis": len(d),
                     **scores(d["y_true"].to_numpy(), d["y_pred"].to_numpy()),
                     "rmse_round_mean": per_round.mean(), "rmse_round_sd": per_round.std(),
                     "rmse_unflagged": np.sqrt(((d.y_pred - d.y_true)[~flagged] ** 2).mean()),
                     "val_rmse": np.sqrt(((val.y_pred - val.y_true) ** 2).mean())})
    res = pd.DataFrame(rows).sort_values("variant", key=lambda s: s.str.extract(r"V(\d+)")[0].astype(int))
    res.round(4).to_csv(tables / "main_results.csv", index=False)
    print(res.round(3).to_string(index=False))

    complete = set(res.loc[res["n_rounds"] == n_rounds, "variant"])
    singles = [v for v in (f"V{i}" for i in range(3, 10)) if v in complete]
    pairs = list(RQ_PAIRS)
    if "V10" in complete and singles:  # best single chosen on validation, never on test
        best = res.set_index("variant").loc[singles, "val_rmse"].idxmin()
        pairs.append(("RQ4 hybrid vs best single", "V10", best))

    tests = []
    for rq, a, b in pairs:
        if not {a, b} <= complete:
            print(f"{rq}: skipped, needs all {n_rounds} rounds of {a} and {b}")
            continue
        pa = test[test.variant == a].set_index("muni_code")
        pb = test[test.variant == b].set_index("muni_code").loc[pa.index]
        diff, lo, hi = block_bootstrap_rmse_diff(pa["y_true"].to_numpy(), pa["y_pred"].to_numpy(),
                                                 pb["y_pred"].to_numpy(), folds.loc[pa.index, "block_id"].to_numpy(),
                                                 cfg["evaluation"]["bootstrap_reps"], cfg["project"]["seed"])
        verdict = "no clear effect" if lo <= 0 <= hi else (f"{a} better" if hi < 0 else f"{b} better")
        tests.append({"question": rq, "a": a, "b": b, "rmse_diff_a_minus_b": diff, "ci_low": lo, "ci_high": hi,
                      "verdict": verdict})
    if tests:
        t = pd.DataFrame(tests)
        t.round(4).to_csv(tables / "rq_tests.csv", index=False)
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
