"""Shared helpers for residue-group (patch / pocket) builders."""
from __future__ import annotations

import numpy as np


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def to_edges(groups, feat_dim: int):
    """[(members, feat), ...] -> (edge_res int64, edge_group int64, group_feat float32 (G, feat_dim))."""
    edge_res, edge_group, feats = [], [], []
    for gidx, (members, feat) in enumerate(groups):
        feats.append(feat)
        for r in members:
            edge_res.append(int(r))
            edge_group.append(int(gidx))
    return (np.asarray(edge_res, dtype=np.int64),
            np.asarray(edge_group, dtype=np.int64),
            np.asarray(feats, dtype=np.float32).reshape(-1, feat_dim))
