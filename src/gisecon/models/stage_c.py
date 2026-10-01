"""Stage C: municipal log GDP per capita from cell embeddings by sum-pooling (PLAN Phase 7).

Each cell gets a non-negative 16-number contribution p(x); contributions are summed per
municipality (optionally log(1 + sum), since municipalities range from 1 to ~10,000 cells), and
h maps the pooled vector to log total GDP. Subtracting log population gives log GDP per capita.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MuniSet:
    """Cells of a set of municipalities. idx maps each cell to its municipality, 0..len(log_pop)-1."""
    x: torch.Tensor        # (cells, d), standardized with training-cell stats
    idx: torch.Tensor      # (cells,) long
    log_pop: torch.Tensor  # (munis,)
    y: torch.Tensor | None = None  # (munis,) official log GDP per capita


class StageC(nn.Module):
    def __init__(self, d_in: int, cell_hidden: int = 32, cell_out: int = 16, muni_hidden: int = 16,
                 dropout: float = 0.1, pooling: str = "sum"):
        super().__init__()
        assert pooling in ("sum", "logsum")
        self.pooling = pooling
        self.cell = nn.Sequential(nn.Linear(d_in, cell_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(cell_hidden, cell_out), nn.Softplus())
        self.muni = nn.Sequential(nn.Linear(cell_out, muni_hidden), nn.ReLU(), nn.Linear(muni_hidden, 1))

    def forward(self, s: MuniSet) -> torch.Tensor:
        c = self.cell(s.x)
        pooled = torch.zeros(len(s.log_pop), c.shape[1], dtype=c.dtype).index_add_(0, s.idx, c)
        if self.pooling == "logsum":
            pooled = torch.log1p(pooled)
        return self.muni(pooled).squeeze(-1) - s.log_pop


def train_stage_c(train: MuniSet, val: MuniSet, cfg: dict, hidden: int, weight_decay: float, dropout: float,
                  pooling: str) -> tuple[StageC, list[dict]]:
    """Full-batch training, MSE on log GDP per capita. cfg is configs/config.yaml's stage_c section;
    the other arguments are one point of its tuning grid.

    Early stopping uses the rolling mean of validation MSE over `rolling_window` epochs: the best
    weights are those at the lowest rolling mean, and training stops after `rolling_window` epochs
    without a new low (or at max_epochs).
    """
    model = StageC(train.x.shape[1], hidden, cfg["cell_out"], cfg["muni_hidden"], dropout, pooling)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg["rolling_window"] // 2,
                                                       min_lr=cfg["lr_min"])
    w = cfg["rolling_window"]
    vals, log, best, best_state, since = [], [], float("inf"), None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        loss = F.mse_loss(model(train), train.y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = F.mse_loss(model(val), val.y).item()
        sched.step(v)
        vals.append(v)
        roll = float(np.mean(vals[-w:]))
        log.append({"epoch": epoch, "train": loss.item(), "val": v, "val_roll": roll})
        if roll < best:
            best, best_state, since = roll, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= w:
                break
    model.load_state_dict(best_state)
    return model, log


@torch.no_grad()
def predict(model: StageC, s: MuniSet) -> torch.Tensor:
    model.eval()
    return model(s)

