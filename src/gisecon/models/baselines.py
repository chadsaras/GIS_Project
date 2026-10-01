"""Municipal-level baselines (PLAN 7.7-7.8, 8.1-8.3). Every choice (form, alpha, blend weight) uses validation only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-3, 3, 13)


def _mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def ntl_only(x: dict, y: dict) -> tuple[dict, str]:
    """V1: OLS of log GDP pc on x = log(total NTL / pop), linear or quadratic, whichever has lower val MSE.

    x, y: {"train"|"val"|"test": array}. Returns ({"val", "test"} predictions, chosen form).
    """
    best = None
    for form, deg in (("linear", 1), ("quadratic", 2)):
        coef = np.polyfit(x["train"], y["train"], deg)
        mse = _mse(np.polyval(coef, x["val"]), y["val"])
        if best is None or mse < best[0]:
            best = (mse, form, coef)
    return {k: np.polyval(best[2], x[k]) for k in ("val", "test")}, best[1]


def ridge(X: dict, y: dict) -> tuple[dict, float]:
    """Standardize (train stats) + Ridge, alpha chosen on validation. Returns ({"val", "test"} predictions, alpha)."""
    best = None
    for a in RIDGE_ALPHAS:
        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X["train"], y["train"])
        mse = _mse(m.predict(X["val"]), y["val"])
        if best is None or mse < best[0]:
            best = (mse, a, m)
    return {k: best[2].predict(X[k]) for k in ("val", "test")}, float(best[1])


def lightgbm(X: dict, y: dict, seed: int) -> dict:
    """V12: default LightGBM with early stopping on validation."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=2000, random_state=seed, verbose=-1)
    m.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return {k: m.predict(X[k]) for k in ("val", "test")}


def blend(model: dict, ntl: dict, y_val: np.ndarray, alphas) -> tuple[dict, float]:
    """V10: alpha * model + (1 - alpha) * NTL-only, alpha chosen on validation."""
    a = min(alphas, key=lambda a: _mse(a * model["val"] + (1 - a) * ntl["val"], y_val))
    return {k: a * model[k] + (1 - a) * ntl[k] for k in ("val", "test")}, float(a)
