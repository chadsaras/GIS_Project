from gisecon.config import data_path, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["study"]["state_code"] == 31
    assert cfg["grid"]["cell_size_m"] == 1000
    assert cfg["cv"]["n_folds"] == 5


def test_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GISECON_DATA_DIR", str(tmp_path))
    cfg = load_config()
    assert data_path(cfg, "raw", "x.csv") == tmp_path / "raw" / "x.csv"
