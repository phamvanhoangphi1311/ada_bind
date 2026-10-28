"""Candidate nucleic-acid-binding SURFACE PATCHES (label-free) for the patch hypergraph.

V1  (18-d):  one 10 Å patch per K/R/H/N/Q/S/T/Y seed containing ≥1 K/R/H, top-128 by a
             cationic/polar plausibility score.
MS  (24-d):  selective multi-scale (6/8/10/12 Å) surface patches: exposure-gated seeds/members,
             duplicate removal + Jaccard NMS (0.65), top-48.
Output of every builder: (edge_res, edge_patch, patch_feat).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..constants import CHARGE
from ..features.geometry import composition_entropy, exposure_proxy, frac, pairwise_stats, shape_descriptors
from ..features.nuclear import GROUPS
from .common import jaccard, to_edges

V1_SEEDS = set("KRHNQSTY")
V1_FEAT_DIM = 18
MS_FEAT_DIM = 24
MS_RADII = (6.0, 8.0, 10.0, 12.0)


def patch_features_v1(seq: str, coords: np.ndarray, members) -> np.ndarray:
    G = GROUPS
    members = np.asarray(members, dtype=int)
    aa = [seq[i] for i in members]
    cc = coords[members]
    n = len(members)
    q = np.asarray([CHARGE.get(x, 0.0) for x in aa], dtype=np.float32)
    radial = np.linalg.norm(cc - cc.mean(axis=0)[None, :], axis=1)
    pair, _ = pairwise_stats(members, cc)
    f = np.zeros(V1_FEAT_DIM, dtype=np.float32)
    f[0] = min(n, 40) / 40.0
    for k, g in enumerate(("KR", "KRH", "DE", "POLAR", "AROM", "HYDRO"), start=1):
        f[k] = frac(G[g], aa)
    f[7] = float(np.mean(q))
    f[8] = frac(G["HBD"], aa)
    f[9] = frac(G["HBA"], aa)
    f[10] = min(float(radial.mean()) / 12.0, 1.0)
    f[11] = min(float(radial.std()) / 8.0, 1.0)
    f[12] = min(float(radial.max()) / 15.0, 1.0)
    f[13] = min((members.max() - members.min()) / 100.0, 1.0)
    f[14] = min(float(pair.mean()) / 15.0, 1.0)
    f[15] = min(float(pair.std()) / 10.0, 1.0)
    f[16] = composition_entropy(aa, dtype=np.float64)
    f[17] = float(np.tanh((sum(x in G["KRH"] for x in aa) - sum(x in G["DE"] for x in aa)) / 5.0))
    return f


def build_patches_v1(coords: np.ndarray, seq: str, radius: float = 10.0, max_patches: int = 128):
    tree = cKDTree(coords)
    candidates = {}
    for seed, a in enumerate(seq):
        if a not in V1_SEEDS:
            continue
        members = tuple(sorted(tree.query_ball_point(coords[seed], radius)))
        if len(members) < 4 or not any(seq[j] in GROUPS["KRH"] for j in members):
            continue
        f = patch_features_v1(seq, coords, members)
        # label-independent cationic/polar recognition plausibility
        score = 2.0 * f[2] + 0.7 * f[4] - 0.7 * f[3] + 0.2 * f[0]
        old = candidates.get(members)
        if old is None or score > old[0]:
            candidates[members] = (score, f)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[:max_patches]
    return to_edges([(m, f) for m, (_, f) in ranked], V1_FEAT_DIM)


def patch_features_ms(seq: str, coords: np.ndarray, exposure: np.ndarray, members, radius: float) -> np.ndarray:
    G = GROUPS
    members = np.asarray(members, dtype=int)
    aa = [seq[j] for j in members]
    cc = coords[members]
    qq = np.asarray([CHARGE.get(a, 0.0) for a in aa], dtype=np.float32)
    n = len(members)
    radial = np.linalg.norm(cc - cc.mean(axis=0)[None, :], axis=1)
    pair, seqsep = pairwise_stats(members, cc)
    f = np.zeros(MS_FEAT_DIM, dtype=np.float32)
    f[0] = radius / 12.0
    f[1] = min(n, 40) / 40.0
    f[2] = float(np.mean(exposure[members]))
    f[3] = float(np.min(exposure[members]))
    for k, g in enumerate(("KR", "KRH", "POLAR", "DE", "HBD"), start=4):
        f[k] = frac(G[g], aa)
    f[9] = float(np.mean(qq))
    f[10] = float(np.std(qq))
    f[11] = float(np.tanh((sum(a in G["KRH"] for a in aa) - sum(a in G["DE"] for a in aa)) / 5.0))
    f[12] = min(float(radial.mean()) / 12.0, 1.0)
    f[13] = min(float(radial.std()) / 8.0, 1.0)
    f[14] = min(float(radial.max()) / 15.0, 1.0)
    f[15] = min(float(pair.mean()) / 15.0, 1.0)
    f[16] = min(float(pair.std()) / 10.0, 1.0)
    f[17] = float(np.mean(seqsep > 10))
    f[18] = min(float(seqsep.mean()) / 50.0, 1.0)
    f[19] = composition_entropy(aa)
    if n >= 3:
        f[20], f[21], f[22] = shape_descriptors(cc)
    f[23] = float(np.mean(np.abs(members - int(np.median(members))) > 10))
    return f


def build_patches_ms(coords: np.ndarray, seq: str, radii=MS_RADII, max_patches: int = 48,
                     nms: float = 0.65):
    tree = cKDTree(coords)
    exposure, _ = exposure_proxy(coords)
    candidates = []
    for seed, a in enumerate(seq):
        if a in GROUPS["KRH"]:
            if exposure[seed] < 0.35:  # basic seeds may be only moderately exposed
                continue
        elif a in GROUPS["POLAR"]:
            if exposure[seed] < 0.65:  # polar-only seeds must be strongly exposed
                continue
        else:
            continue
        for radius in radii:
            members = tuple(sorted(j for j in tree.query_ball_point(coords[seed], radius) if exposure[j] >= 0.25))
            if len(members) < 4 or not any(seq[j] in GROUPS["KRH"] for j in members):
                continue
            f = patch_features_ms(seq, coords, exposure, members, radius)
            score = 2.0 * f[5] + 0.70 * f[6] - 0.70 * f[7] + 1.20 * f[2] + 0.25 * f[17] - 0.10 * f[0]
            candidates.append((float(score), members, f))
    return _dedupe_nms(candidates, max_patches, nms, MS_FEAT_DIM)


def _dedupe_nms(candidates, max_groups: int, nms: float, feat_dim: int):
    unique = {}
    for score, members, feat in candidates:
        if members not in unique or score > unique[members][0]:
            unique[members] = (score, feat)
    ranked = sorted([(s, m, f) for m, (s, f) in unique.items()], key=lambda x: x[0], reverse=True)
    selected = []
    for score, members, feat in ranked:
        if any(jaccard(members, m2) >= nms for _, m2, _ in selected):
            continue
        selected.append((score, members, feat))
        if len(selected) >= max_groups:
            break
    return to_edges([(m, f) for _, m, f in selected], feat_dim)
