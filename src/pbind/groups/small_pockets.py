"""Candidate small-molecule POCKETS (label-free) for the pocket hypergraph.

V1  (22-d):  one 8 Å pocket per W/F/Y/H/C/M seed, top-160 by a chemistry + packing score.
MS  (23-d):  multi-scale 6/8/10/12 Å pockets (22 V1 features + radius), dedupe + Jaccard NMS 0.65, top-64.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..constants import HYDROPATHY
from ..features.geometry import composition_entropy, frac, pairwise_stats, shape_descriptors
from ..features.small import GROUPS
from .common import to_edges
from .nuclear_patches import _dedupe_nms

SEEDS = set("WFYHCM")
V1_FEAT_DIM = 22
MS_FEAT_DIM = 23
MS_RADII = (6.0, 8.0, 10.0, 12.0)


def pocket_features_v1(seq: str, coords: np.ndarray, members) -> np.ndarray:
    G = GROUPS
    members = np.asarray(members, dtype=int)
    aa = [seq[i] for i in members]
    cc = coords[members]
    n = len(members)
    radial = np.linalg.norm(cc - cc.mean(axis=0)[None, :], axis=1)
    pair, seqsep = pairwise_stats(members, cc)
    hydro = np.asarray([HYDROPATHY.get(a, 0.0) / 4.5 for a in aa], dtype=np.float32)
    f = np.zeros(V1_FEAT_DIM, dtype=np.float32)
    # composition
    f[0] = min(n, 40) / 40.0
    for k, g in enumerate(("AROM", "SULFUR", "HYDRO", "POLAR", "ACIDIC", "HBD", "HBA"), start=1):
        f[k] = frac(G[g], aa)
    f[8] = float(np.mean(hydro))
    f[9] = frac(G["GP"], aa)
    # geometry
    f[10] = min(float(radial.mean()) / 12.0, 1.0)
    f[11] = min(float(radial.std()) / 8.0, 1.0)
    f[12] = min(float(radial.max()) / 15.0, 1.0)
    f[13] = min(float(pair.mean()) / 15.0, 1.0)
    f[14] = min(float(pair.std()) / 10.0, 1.0)
    f[15] = min((members.max() - members.min()) / 100.0, 1.0)
    f[16] = composition_entropy(aa)
    if n >= 2:  # tertiary-contact fraction
        f[17] = float(np.mean(seqsep > 10))
        f[18] = min(float(seqsep.mean()) / 50.0, 1.0)
    if n >= 3:
        f[19], f[20], f[21] = shape_descriptors(cc)
    return f


def _v1_score(f: np.ndarray) -> float:
    """Enriched chemistry + compact packing, mild penalty for acidic environments."""
    return (1.4 * f[1] + 0.8 * f[2] + 0.8 * f[3] + 0.5 * f[6] + 0.5 * f[7] - 0.5 * f[5]
            + 0.25 * f[0] + 0.15 * f[17])


def build_pockets_v1(coords: np.ndarray, seq: str, radius: float = 8.0, max_pockets: int = 160):
    tree = cKDTree(coords)
    candidates = {}
    for seed, a in enumerate(seq):
        if a not in SEEDS:
            continue
        members = tuple(sorted(tree.query_ball_point(coords[seed], radius)))
        if len(members) < 4:
            continue
        f = pocket_features_v1(seq, coords, members)
        score = _v1_score(f)
        old = candidates.get(members)
        if old is None or score > old[0]:
            candidates[members] = (score, f)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:max_pockets]
    return to_edges([(m, f) for m, (_, f) in ranked], V1_FEAT_DIM)


def build_pockets_ms(coords: np.ndarray, seq: str, radii=MS_RADII, max_pockets: int = 64, nms: float = 0.65):
    tree = cKDTree(coords)
    candidates = []
    for seed, a in enumerate(seq):
        if a not in SEEDS:
            continue
        for radius in radii:
            members = tuple(sorted(tree.query_ball_point(coords[seed], radius)))
            if len(members) < 4:
                continue
            f22 = pocket_features_v1(seq, coords, members)
            f23 = np.concatenate([f22.astype(np.float32), np.array([radius / 12.0], dtype=np.float32)])
            # V1 score + very mild preference against unnecessarily large radius
            candidates.append((float(_v1_score(f22) - 0.05 * (radius / 12.0)), members, f23))
    return _dedupe_nms(candidates, max_pockets, nms, MS_FEAT_DIM)
