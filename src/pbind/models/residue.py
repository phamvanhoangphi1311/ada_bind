"""Per-residue classifiers of the GEAR standardized trainer (input = concatenated feature blocks, (N, d)).

State-dict keys are those of the original ``gear_core`` classes, so old and new checkpoints load both ways.
"""
from __future__ import annotations

import torch.nn as nn

try:  # optional, only for the KAN model family
    from kan import KAN  # type: ignore

    HAS_KAN = True
except Exception:  # pragma: no cover
    KAN, HAS_KAN = None, False


class ResidueMLP(nn.Module):
    """d -> 256 (GELU, LayerNorm, dropout) -> 64 (GELU, dropout) -> 1 logit."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidueKAN(nn.Module):
    """Linear(d, 32) -> KAN [32, 8, 2] (grid 2, k 2) -> 2 logits; needs ``pykan``."""

    def __init__(self, d: int, seed: int):
        super().__init__()
        if not HAS_KAN:
            raise RuntimeError("KAN unavailable: pip install pykan")
        self.pre = nn.Linear(d, 32)
        err = None
        for make in (lambda: KAN(width=[32, 8, 2], grid=2, k=2, seed=seed, auto_save=False),
                     lambda: KAN(width=[32, 8, 2], grid=2, k=2, seed=seed),
                     lambda: KAN([32, 8, 2], grid=2, k=2)):
            try:
                self.kan = make()
                break
            except TypeError as exc:
                err = exc
        else:
            raise err

    def forward(self, x):
        return self.kan(self.pre(x))
