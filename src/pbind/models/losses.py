"""Losses used by the custom trainers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sqrt_pos_weight(y: np.ndarray, device=None) -> torch.Tensor:
    """sqrt(neg/pos) computed on the TRAINING fold only."""
    npos = max(float(np.sum(y)), 1.0)
    nneg = max(float(len(y) - np.sum(y)), 1.0)
    return torch.tensor(np.sqrt(nneg / npos), dtype=torch.float32, device=device)


def protein_balanced_bce(logits, y, protein_index, pos_weight):
    """Weighted BCE averaged within each protein, then across proteins (long proteins don't dominate)."""
    raw = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
    nprot = int(protein_index.max().item()) + 1
    sums = raw.new_zeros((nprot,)).index_add(0, protein_index, raw)
    counts = raw.new_zeros((nprot,)).index_add(0, protein_index, raw.new_ones(len(raw)))
    return (sums / counts.clamp_min(1.0)).mean()


def hard_negative_rank(logits, y, protein_index, margin=0.5, k_min=4, k_max=32):
    """Within-protein ranking: top-k negatives should score below the mean positive logit."""
    nprot = int(protein_index.max().item()) + 1
    terms = []
    for p in range(nprot):
        mask = protein_index == p
        lp, yp = logits[mask], y[mask] > 0.5
        if not yp.any() or (~yp).sum() == 0:
            continue
        pos, neg = lp[yp], lp[~yp]
        k = min(max(k_min, int(2 * len(pos))), k_max, len(neg))
        hard = torch.topk(neg, k=k).values
        terms.append(F.softplus(hard - pos.mean() + margin).mean())
    return torch.stack(terms).mean() if terms else logits.new_tensor(0.0)


def v2_loss(logits, y, protein_index, pos_weight, hardneg: bool, lam: float, margin: float = 0.5,
            k_min: int = 4, k_max: int = 32):
    """Returns (total, rank_term). nuclear: lam=.20,k=4..32 ; small: lam=.15,k=8..48."""
    base = protein_balanced_bce(logits, y, protein_index, pos_weight)
    if not hardneg:
        return base, base.new_tensor(0.0)
    rank = hard_negative_rank(logits, y, protein_index, margin, k_min, k_max)
    return base + lam * rank, rank
