"""Stage B: predict each cell's log night lights from its embedding (PLAN Phase 6).

The 32-number activation after the last hidden ReLU is z, the economy-aware embedding.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gisecon.models.train import fit


class StageB(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64, 32)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(z, predicted standardized log NTL)."""
        z = self.body(x)
        return z, self.head(z).squeeze(-1)


def train_stage_b(x_tr: torch.Tensor, y_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor,
                  cfg: dict) -> tuple[StageB, list[dict]]:
    """cfg is configs/config.yaml's stage_b section; y is log NTL standardized with train-cell stats."""
    model = StageB(x_tr.shape[1], tuple(cfg["hidden"]))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    log = fit(model, lambda m, x, y: F.mse_loss(m(x)[1], y), (x_tr, y_tr), (x_va, y_va), opt,
              cfg["batch_size"], cfg["max_epochs"], cfg["patience"])
    return model, log


@torch.no_grad()
def embed_cells(model: StageB, x: torch.Tensor, batch: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    """z and predicted (standardized) log NTL for every cell."""
    model.eval()
    zs, ps = zip(*(model(x[i:i + batch]) for i in range(0, len(x), batch)))
    return torch.cat(zs), torch.cat(ps)
