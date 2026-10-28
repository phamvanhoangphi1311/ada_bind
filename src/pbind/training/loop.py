"""Generic early-stopping training loop (best validation AUPRC), shared by all custom trainers."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..utils.torch_utils import clone_state


@dataclass(frozen=True)
class TrainConfig:
    batch: int
    epochs: int
    patience: int
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 2026
    seed_offset: int = 0  # notebook convention: local +0, V1 group +100, nuclear V2 +700, small V2 +900
    grad_clip: float = 5.0
    log_every: int = 5

    def seed_for(self, fold: int) -> int:
        return self.seed + self.seed_offset + fold

    def with_(self, **kw) -> "TrainConfig":
        return replace(self, **kw)


@dataclass
class CVResult:
    """Output of a resume-safe official 5-fold CV run."""
    name: str
    summary: dict
    folds: pd.DataFrame
    oof: np.ndarray
    states: dict = field(repr=False, default_factory=dict)

    @property
    def score(self) -> float:
        return float(self.summary["CV_AUPRC_mean"])


def fit_early_stopping(model: torch.nn.Module, train_loader, step: Callable, validate: Callable,
                       cfg: TrainConfig, tag: str, extra_log: Callable | None = None):
    """``step(model, batch) -> (loss, aux_float)``; ``validate(model) -> (val_ap, info_dict)``.

    Keeps the best-val-AP state (improvement > 1e-5), stops after ``patience`` bad epochs.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses, auxs = [], []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss, aux = step(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            auxs.append(float(aux))
        val_ap, info = validate(model)
        if val_ap > best_ap + 1e-5:
            best_ap, best_state, bad = val_ap, clone_state(model), 0
        else:
            bad += 1
        if epoch == 1 or epoch % cfg.log_every == 0:
            msg = f"{tag} epoch {epoch} loss {np.mean(losses):.4f} aux {np.mean(auxs):.4f} val_AP {val_ap:.5f} best {best_ap:.5f}"
            extra = {**(info or {}), **(extra_log(model) if extra_log else {})}
            if extra:
                msg += " " + " ".join(f"{k} {v:.4f}" for k, v in extra.items())
            print(msg)
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return model, best_state
