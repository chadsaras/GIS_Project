import numpy as np

from gisecon.eval.metrics import block_bootstrap_rmse_diff, scores
from gisecon.models import baselines as bl


def test_scores_perfect_prediction():
    y = np.array([9.0, 10.0, 11.0, 12.0])
    s = scores(y, y)
    assert s["r2"] == 1 and s["rmse"] == 0 and abs(s["calib_slope"] - 1) < 1e-9


def test_block_bootstrap_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 1, 400)
    block = np.repeat(np.arange(40), 10)
    good, bad = y + rng.normal(0, 0.1, 400), y + rng.normal(0, 0.5, 400)
    diff, lo, hi = block_bootstrap_rmse_diff(y, good, bad, block, 500, 0)
    assert diff < 0 and hi < 0
    assert block_bootstrap_rmse_diff(y, good, good, block, 100, 0) == (0.0, 0.0, 0.0)


def test_ntl_only_picks_quadratic_for_curved_data():
    rng = np.random.default_rng(0)
    x = {k: rng.uniform(-3, 3, 200) for k in ("train", "val", "test")}
    y = {k: 10 + 0.5 * v - 0.3 * v ** 2 for k, v in x.items()}
    pred, form = bl.ntl_only(x, y)
    assert form == "quadratic" and np.allclose(pred["test"], y["test"])


def test_blend_chooses_weight_on_validation():
    y_val = np.array([1.0, 2.0, 3.0])
    model, ntl = {"val": y_val, "test": np.zeros(2)}, {"val": y_val + 1, "test": np.ones(2)}
    pred, a = bl.blend(model, ntl, y_val, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert a == 1.0 and np.allclose(pred["test"], 0)
