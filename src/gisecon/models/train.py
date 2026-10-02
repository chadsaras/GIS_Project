"""Mini-batch training loop with early stopping, shared by Stages A and B."""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn


def fit(model: nn.Module, loss_fn: Callable[..., torch.Tensor], train: tuple[torch.Tensor, ...],
        val: tuple[torch.Tensor, ...], opt: torch.optim.Optimizer, batch_size: int, max_epochs: int,
        patience: int, drop_last: bool = False) -> list[dict]:
    """Train until validation loss stops improving for `patience` epochs; restore the best weights.

    loss_fn(model, *batch) -> scalar loss. Validation loss is computed on the whole val set at once.
    drop_last=True skips a short final batch (contrastive losses need full batches of negatives).
    Returns one {epoch, train, val} row per epoch.
    """
    n = len(train[0])
    best, best_state, wait, log = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm, total, seen = torch.randperm(n), 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if drop_last and len(idx) < batch_size and seen:
                break
            loss = loss_fn(model, *(t[idx] for t in train))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, seen = total + loss.item() * len(idx), seen + len(idx)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model, *val).item()
        log.append({"epoch": epoch, "train": total / seen, "val": v})
        if v < best:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    assert best_state is not None, "validation loss was never finite: check the inputs for NaN/inf"
    model.load_state_dict(best_state)
    return log
