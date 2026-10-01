"""Stage A: align each cell's GSED vector with its caption (CLIP-style, PLAN Phase 5).

Both encoders are frozen upstream (GSED, all-mpnet-base-v2); only two small projections and a
temperature are trained, with a symmetric InfoNCE loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gisecon.models.train import fit


def _proj(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))


class StageA(nn.Module):
    def __init__(self, img_dim: int = 64, txt_dim: int = 768, proj_hidden: int = 256, emb_dim: int = 128,
                 dropout: float = 0.1, temperature_init: float = 0.07):
        super().__init__()
        self.img = _proj(img_dim, proj_hidden, emb_dim, dropout)
        self.txt = _proj(txt_dim, proj_hidden, emb_dim, dropout)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))

    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.img(x), dim=-1)

    def embed_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.txt(t), dim=-1)

    def loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE: row i of x matches row i of t, every other row is a negative."""
        logits = self.embed_image(x) @ self.embed_text(t).T * self.log_scale.clamp(max=math.log(100)).exp()
        target = torch.arange(len(x))
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def train_stage_a(img_tr: torch.Tensor, txt_tr: torch.Tensor, img_va: torch.Tensor, txt_va: torch.Tensor,
                  cfg: dict) -> tuple[StageA, list[dict]]:
    """cfg is configs/config.yaml's stage_a section. Inputs are already standardized (train-cell stats)."""
    model = StageA(img_tr.shape[1], txt_tr.shape[1], cfg["proj_hidden"], cfg["emb_dim"], cfg["dropout"],
                   cfg["temperature_init"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    log = fit(model, lambda m, x, t: m.loss(x, t), (img_tr, txt_tr), (img_va, txt_va), opt,
              cfg["batch_size"], cfg["max_epochs"], cfg["patience"], drop_last=True)
    return model, log


@torch.no_grad()
def recall_at_k(model: StageA, x: torch.Tensor, t: torch.Tensor, ks=(1, 5, 10), chunk: int = 256) -> dict[str, float]:
    """Image->text recall@k within chunks of `chunk` pairs (chance at k=10, chunk=256 is about 4%)."""
    model.eval()
    hits = {k: 0 for k in ks}
    for i in range(0, len(x), chunk):
        sim = model.embed_image(x[i:i + chunk]) @ model.embed_text(t[i:i + chunk]).T
        rank = (sim > sim.diag()[:, None]).sum(1)  # how many captions beat the true one
        for k in ks:
            hits[k] += int((rank < k).sum())
    return {f"recall@{k}": hits[k] / len(x) for k in ks}


@torch.no_grad()
def embed_cells(model: StageA, x: torch.Tensor, batch: int = 65536) -> torch.Tensor:
    """128-number Stage A embedding for every cell; no text needed."""
    model.eval()
    return torch.cat([model.embed_image(x[i:i + batch]) for i in range(0, len(x), batch)])
