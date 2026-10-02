import numpy as np
import pandas as pd

from gisecon.text.clean import clean_caption
from gisecon.text.facts import fact_sentences
from gisecon.text.sample import caption_sample


def test_fact_sentences_town_and_empty_cell():
    town = {"lc_built": 0.65, "lc_grass": 0.20, "lc_tree": 0.10, "poi_education": 12, "poi_health": 4,
            "poi_retail": 8, "road_major_km": 8.2, "road_minor_km": 15.0, "water_km": 0.0, "ntl": 40.0}
    assert fact_sentences(town) == ("Mostly built-up: 65% built-up, 20% grassland, 10% trees. "
                                    "12 schools, 4 health facilities and 8 shops are mapped. "
                                    "8.2 km of major roads and 15.0 km of minor roads. No mapped rivers.")
    assert "light" not in fact_sentences(town).lower()  # NTL never leaks into text
    rural = {"lc_grass": 0.7, "lc_tree": 0.3, "poi_fuel": 1, "water_km": 2.04}
    assert fact_sentences(rural) == ("Mostly grassland: 70% grassland, 30% trees. 1 fuel station is mapped. "
                                     "No mapped roads. 2.0 km of mapped rivers or streams.")


def test_clean_caption_drops_filler_and_false_claims():
    cell = {"lc_built": 0.3, "lc_tree": 0.02, "lc_grass": 0.4, "lc_water": 0.0, "water_km": 0.0, "poi_retail": 3}
    text = ("The image shows a satellite view of a town. Dense houses line the streets. "
            "A river runs through the centre. A dense forest covers the hills. No lakes are visible. "
            "Pastures surround the town.")
    cleaned, rows = clean_caption(text, cell)
    assert [r["drop_reason"] for r in rows] == ["filler", None, "claim_water", "claim_forest", None, None]
    assert cleaned == "Dense houses line the streets. No lakes are visible. Pastures surround the town."


def test_caption_sample_quota_and_weights():
    rng = np.random.default_rng(0)
    n = 6000
    kept = pd.DataFrame({"cell_id": np.arange(n), "muni_code": rng.integers(0, 50, n),
                         "lc_built": rng.beta(0.5, 5, n), "ntl": np.where(rng.random(n) < 0.6, 0, rng.exponential(5, n))})
    folds = pd.Series(np.arange(50) % 5, index=np.arange(50))
    s = caption_sample(kept, folds, per_fold=300, seed=1)
    assert s.groupby("fold").size().eq(300).all()
    assert s.cell_id.is_unique
    # weights add back up to the fold's cell count, so weighted statistics are unbiased
    fold_sizes = kept.muni_code.map(folds).value_counts().sort_index()
    np.testing.assert_allclose(s.groupby("fold").weight.sum().to_numpy(), fold_sizes.to_numpy(), rtol=0.15)


def test_caption_sample_oversamples_towns():
    """Realistic shares: ~80% dark, most cells with no built-up land. Towns must be over-represented."""
    rng = np.random.default_rng(2)
    n = 20000
    town = rng.random(n) < 0.03
    kept = pd.DataFrame({"cell_id": np.arange(n), "muni_code": rng.integers(0, 50, n),
                         "lc_built": np.where(town, rng.uniform(0.05, 0.9, n), np.where(rng.random(n) < 0.3, 0.002, 0)),
                         "ntl": np.where(town, rng.exponential(10, n), np.where(rng.random(n) < 0.15, 0.3, 0))})
    folds = pd.Series(np.arange(50) % 5, index=np.arange(50))
    s = caption_sample(kept, folds, per_fold=400, seed=1).merge(kept, on="cell_id")
    assert s.lc_built.mean() > 3 * kept.lc_built.mean()
    assert (s.ntl > 0).mean() > 2 * (kept.ntl > 0).mean()


def test_log_odds_marks_group_words():
    from gisecon.text.words import log_odds_z
    a, b = np.array([50, 5, 20]), np.array([5, 50, 20])  # word 0 typical of a, word 1 of b, word 2 shared
    z = log_odds_z(a, b, a + b)
    assert z[0] > 2 and z[1] < -2 and abs(z[2]) < 0.5


def test_tile_range_covers_box():
    import importlib.util
    from gisecon.config import REPO_ROOT
    spec = importlib.util.spec_from_file_location("tiles", REPO_ROOT / "scripts" / "11_download_tiles.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    (c0, c1, r0, r1), tm = t.tile_range(0.0, -1.0, 1.0, 0.0, 1)  # box just right of / below the origin
    assert (c0, c1, r0, r1) == (1, 1, 1, 1) and abs(tm - t.HALF) < 1e-6  # zoom 1: 2 x 2 tiles, origin at the centre
