"""End-to-end checks of scripts 07 -> 08 -> 09 on a tiny synthetic data directory."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
import yaml

from gisecon.config import DEFAULT_CONFIG, REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner, baselines, evaluate = _load("07_run_experiment"), _load("08_baselines"), _load("09_evaluate")


def _fake_data(root, n_munis=40, cells_per=15, n_captions=300):
    rng = np.random.default_rng(0)
    proc = root / "processed"
    proc.mkdir(parents=True)
    (root / "interim").mkdir()
    codes = np.arange(3100000, 3100000 + n_munis)
    muni = np.repeat(codes, cells_per)
    n = len(muni)
    gsed = rng.normal(size=(n, 64)).astype(np.float32)
    cells = pd.DataFrame(gsed, columns=runner.GSED)
    cells.insert(0, "cell_id", np.arange(n))
    cells.insert(1, "muni_code", muni)
    cells["keep"] = rng.random(n) > 0.05
    cells["log_ntl"] = np.maximum(gsed[:, 0], 0)
    cells["ntl"] = np.expm1(cells["log_ntl"])
    cells["lc_built"] = rng.random(n) * 0.3
    cells["lc_tree"] = 1 - cells["lc_built"]
    cells["poi_retail"] = rng.poisson(1, n)
    cells["road_major_km"] = rng.random(n)
    cells.to_parquet(proc / "cells.parquet")
    texts = pd.DataFrame({"cell_id": rng.choice(n, n_captions, replace=False), "text_full": "x"})
    texts.to_parquet(proc / "texts.parquet")
    np.save(root / "interim" / "text_emb_full.npy", rng.normal(size=(n_captions, 768)).astype(np.float32))
    pd.DataFrame({"muni_code": codes, "log_pop": rng.normal(9, 1, n_munis), "log_gdp_pc": rng.normal(10, 0.5, n_munis),
                  "flag_extreme": np.arange(n_munis) % 13 == 0}).to_parquet(proc / "municipal_labels.parquet")
    pd.DataFrame({"muni_code": codes, "block_id": np.arange(n_munis) // 5 * 5 + np.arange(n_munis) % 5,
                  "fold": np.arange(n_munis) % 5}).to_parquet(proc / "folds.parquet")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    _fake_data(tmp_path / "data")
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["stage_a"].update(max_epochs=2, batch_size=64)
    cfg["stage_b"].update(max_epochs=2)
    cfg["stage_c"].update(max_epochs=10, rolling_window=3, n_seeds=2)
    cfg["evaluation"]["bootstrap_reps"] = 200
    cfg["paths"]["logs"] = str(tmp_path / "logs")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    monkeypatch.setenv("GISECON_DATA_DIR", str(tmp_path / "data"))
    for mod in (runner, evaluate):
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)  # keep test tables out of reports/
    return tmp_path, str(cfg_file)


def test_runner_end_to_end(setup):
    tmp_path, cfg_file = setup
    runner.main(["--variant", "V7", "--round", "0", "--config", cfg_file])
    runner.main(["--variant", "V7", "--round", "0", "--config", cfg_file])  # second run hits the caches

    d = tmp_path / "data"
    for f in ["interim/round0/emb_A_full.npy", "interim/round0/z_A_full.npy", "interim/round0/ntl_pred_A_full.npy",
              "models/round0/gsed_scaler.json", "models/round0/stage_c_V7_seed1.pt"]:
        assert (d / f).exists(), f
    preds = pd.read_parquet(d / "processed/predictions/V7_r0.parquet")
    assert preds.groupby("split").size().to_dict() == {"test": 8, "val": 8}  # folds 0 and 1
    assert set(preds[preds.split == "test"].muni_code % 5) == {0}
    assert (preds.ens_min <= preds.y_pred).all() and (preds.y_pred <= preds.ens_max).all()
    grid = runner.load_config(cfg_file)["stage_c"]["grid"]
    assert len(pd.read_csv(tmp_path / "reports/tables/stage_c_tuning/V7_r0.csv")) == int(np.prod([len(v) for v in grid.values()]))
    assert {"y_pred_raw", "calib_a", "calib_b"} <= set(preds.columns)


def test_baselines_and_evaluation_over_all_rounds(setup):
    tmp_path, cfg_file = setup
    for r in range(5):
        for v in ("V3", "V4"):
            runner.main(["--variant", v, "--round", str(r), "--config", cfg_file])
        baselines.main(["--round", str(r), "--config", cfg_file])
    evaluate.main(["--config", cfg_file])

    res = pd.read_csv(tmp_path / "reports/tables/main_results.csv").set_index("variant")
    assert {"V1", "V2", "V3", "V4", "V10", "V11_gsed", "V11_z_gsed", "V12"} <= set(res.index)
    assert (res["n_rounds"] == 5).all() and (res["n_munis"] == 40).all()  # every municipality tested once
    v10 = pd.read_parquet(tmp_path / "data/processed/predictions/V10_r0.parquet")
    assert v10.note.iloc[0].startswith("base=V")
    rq = pd.read_csv(tmp_path / "reports/tables/rq_tests.csv")
    assert rq.question.str.startswith("RQ1").any() and rq.question.str.startswith("RQ4").any()
    assert (rq.ci_low <= rq.rmse_diff_a_minus_b).all() and (rq.rmse_diff_a_minus_b <= rq.ci_high).all()
