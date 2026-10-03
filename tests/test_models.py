"""Stages A-C on small synthetic data: each must learn a planted signal well above chance."""
import numpy as np
import torch

from gisecon.config import load_config
from gisecon.models.stage_a import embed_cells as embed_a, recall_at_k, train_stage_a
from gisecon.models.stage_b import embed_cells as embed_b, train_stage_b
from gisecon.models.stage_c import MuniSet, StageC, fit_calibration, predict, train_stage_c

CFG = load_config()


def _r2(pred, y):
    return 1 - ((pred - y) ** 2).mean() / y.var()


def test_stage_a_aligns_planted_pairs():
    torch.manual_seed(0)
    mix = torch.randn(64, 768)
    img = torch.randn(4000, 64)
    txt = img @ mix + 2.0 * torch.randn(4000, 768)  # text is a noisy function of the image
    cfg = {**CFG["stage_a"], "max_epochs": 30, "patience": 5}
    model, log = train_stage_a(img[:3000], txt[:3000], img[3000:], txt[3000:], cfg)
    r = recall_at_k(model, img[3000:], txt[3000:])
    assert r["recall@10"] > 0.3, r  # chance is about 0.04
    assert log[-1]["epoch"] < cfg["max_epochs"]
    assert embed_a(model, img).shape == (4000, cfg["emb_dim"])


def test_stage_b_learns_ntl_and_exposes_z():
    torch.manual_seed(0)
    x = torch.randn(20000, 64)
    y = torch.tanh(x[:, :8].sum(1) / 3) + 0.1 * torch.randn(20000)
    y = (y - y[:15000].mean()) / y[:15000].std()
    cfg = {**CFG["stage_b"], "max_epochs": 15}
    model, _ = train_stage_b(x[:15000], y[:15000], x[15000:], y[15000:], cfg)
    z, pred = embed_b(model, x[15000:])
    assert z.shape == (5000, 32)
    assert (z >= 0).all()  # taken after a ReLU
    assert _r2(pred, y[15000:]) > 0.8


def _munis(n_munis, gen):
    """Cells with a planted per-cell output; municipal GDP is the sum over its cells."""
    sizes = torch.randint(1, 60, (n_munis,), generator=gen)
    idx = torch.repeat_interleave(torch.arange(n_munis), sizes)
    x = torch.randn(len(idx), 8, generator=gen)
    cell_gdp = torch.exp(x[:, 0] + 0.5 * x[:, 1] + 16.0)  # real scale: log total GDP ~19
    total = torch.zeros(n_munis).index_add_(0, idx, cell_gdp)
    log_pop = torch.log(sizes.float()) + 0.3 * torch.randn(n_munis, generator=gen) + 6.0  # log pop ~9, log GDP pc ~10
    return MuniSet(x, idx, log_pop, torch.log(total) - log_pop)


def test_stage_c_pooling_is_order_invariant():
    torch.manual_seed(0)
    s = _munis(20, torch.Generator().manual_seed(0))
    perm = torch.randperm(len(s.idx))
    shuffled = MuniSet(s.x[perm], s.idx[perm], s.log_pop)
    for pooling in ("sum", "logsum", "mean"):
        m = StageC(8, pooling=pooling)
        assert torch.allclose(predict(m, s), predict(m, shuffled), atol=1e-5)


def test_stage_c_learns_from_municipal_totals():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    train, val = _munis(300, gen), _munis(100, gen)
    cfg = {**CFG["stage_c"], "max_epochs": 800, "rolling_window": 30}
    model, log = train_stage_c(train, val, cfg, hidden=32, weight_decay=1e-4, dropout=0.1, pooling="logsum")
    assert _r2(predict(model, val), val.y) > 0.5, log[-1]


def test_mean_pooling_sees_municipality_size():
    """Two municipalities with identical cells but different counts must get different predictions."""
    torch.manual_seed(0)
    x = torch.ones(5, 8)
    s = MuniSet(x, torch.tensor([0, 1, 1, 1, 1]), torch.zeros(2))
    out = predict(StageC(8, pooling="mean"), s)
    assert not torch.isclose(out[0], out[1])


def test_fit_calibration_recovers_line():
    pred = np.linspace(9, 11, 50)
    y = 2.0 + 0.8 * pred
    a, b = fit_calibration(pred, y)
    assert abs(a - 2.0) < 1e-9 and abs(b - 0.8) < 1e-9
