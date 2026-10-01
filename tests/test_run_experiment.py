"""End-to-end check of scripts/06_run_experiment.py on a tiny synthetic data directory."""
import importlib.util

import numpy as np
import pandas as pd
import yaml

from gisecon.config import DEFAULT_CONFIG, REPO_ROOT

spec = importlib.util.spec_from_file_location("run_experiment", REPO_ROOT / "scripts" / "06_run_experiment.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


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
    cells["poi_retail"] = rng.poisson(1, n)
    cells["road_major_km"] = rng.random(n)
    cells.to_parquet(proc / "cells.parquet")
    texts = pd.DataFrame({"cell_id": rng.choice(n, n_captions, replace=False), "text_full": "x"})
    texts.to_parquet(proc / "texts.parquet")
    np.save(root / "interim" / "text_emb_full.npy", rng.normal(size=(n_captions, 768)).astype(np.float32))
    pd.DataFrame({"muni_code": codes, "log_pop": rng.normal(9, 1, n_munis),
                  "log_gdp_pc": rng.normal(10, 0.5, n_munis)}).to_parquet(proc / "municipal_labels.parquet")
    pd.DataFrame({"muni_code": codes, "fold": np.arange(n_munis) % 5}).to_parquet(proc / "folds.parquet")


def test_runner_end_to_end(tmp_path, monkeypatch):
    _fake_data(tmp_path / "data")
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["stage_a"].update(max_epochs=2, batch_size=64)
    cfg["stage_b"].update(max_epochs=2)
    cfg["stage_c"].update(max_epochs=10, rolling_window=3, n_seeds=2)
    cfg["paths"]["logs"] = str(tmp_path / "logs")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("GISECON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)  # keep test tables out of reports/

    runner.main(["--variant", "V7", "--round", "0", "--config", str(cfg_file)])
    runner.main(["--variant", "V7", "--round", "0", "--config", str(cfg_file)])  # second run hits the caches

    d = tmp_path / "data"
    for f in ["interim/round0/emb_A_full.npy", "interim/round0/z_A_full.npy", "interim/round0/ntl_pred_A_full.npy",
              "models/round0/gsed_scaler.json", "models/round0/stage_c_V7_seed1.pt"]:
        assert (d / f).exists(), f
    preds = pd.read_parquet(d / "processed/predictions/V7_r0.parquet")
    assert preds.groupby("split").size().to_dict() == {"test": 8, "val": 8}  # folds 0 and 1
    assert set(preds[preds.split == "test"].muni_code % 5) == {0}
    assert (preds.ens_min <= preds.y_pred).all() and (preds.y_pred <= preds.ens_max).all()
    assert len(pd.read_csv(tmp_path / "reports/tables/stage_c_tuning/V7_r0.csv")) == 16
