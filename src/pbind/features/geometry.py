"""Small geometric / compositional primitives shared by the nuclear and small feature builders."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree


def frac(chars: set, aa) -> float:
    """Fraction of residues in ``aa`` belonging to ``chars`` (caller guarantees len(aa) > 0)."""
    return sum(x in chars for x in aa) / len(aa)


def window_frac(seq: str, i: int, chars: set, radius: int) -> float:
    lo, hi = max(0, i - radius), min(len(seq), i + radius + 1)
    w = seq[lo:hi]
    return sum(x in chars for x in w) / max(len(w), 1)


def group_index(seq: str, chars: set) -> np.ndarray:
    return np.asarray([j for j, x in enumerate(seq) if x in chars], dtype=int)


def nearest_distance(coords: np.ndarray, idx: np.ndarray, i: int, scale: float = 20.0) -> float:
    """min distance from residue i to residues ``idx`` (excluding i), /scale, clipped at 1."""
    idx = idx[idx != i]
    if len(idx) == 0:
        return 1.0
    d = np.linalg.norm(coords[idx] - coords[i][None, :], axis=1)
    return min(float(d.min()) / scale, 1.0)


def component_sizes(indices, coords, threshold: float = 8.0) -> dict[int, int]:
    """Union-find connected components of the given residues at a distance threshold.

    Returns residue_index -> size of its component.
    """
    if len(indices) == 0:
        return {}
    idx = np.asarray(indices, dtype=int)
    cc = np.asarray(coords, dtype=np.float32)
    parent = list(range(len(idx)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if len(idx) > 1:
        for a, b in cKDTree(cc).query_pairs(threshold):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    comp = defaultdict(list)
    for j, rid in enumerate(idx):
        comp[find(j)].append(int(rid))
    out = {}
    for members in comp.values():
        for r in members:
            out[r] = len(members)
    return out


def exposure_proxy(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label-free surface exposure: fewer residues within 10 Å = more exposed. Ranked to [0, 1]."""
    tree = cKDTree(coords)
    n10 = np.array([max(len(tree.query_ball_point(coords[i], 10.0)) - 1, 0)
                    for i in range(len(coords))], dtype=np.float32)
    if len(coords) <= 1:
        return np.ones(len(coords), dtype=np.float32), n10
    order = np.argsort(np.argsort(n10))
    exposure = 1.0 - order.astype(np.float32) / (len(coords) - 1)
    return exposure.astype(np.float32), n10


def shape_descriptors(points: np.ndarray) -> tuple[float, float, float]:
    """PCA linearity / planarity / isotropy of a point cloud (n >= 3)."""
    z = points - points.mean(axis=0, keepdims=True)
    cov = z.T @ z / max(len(points) - 1, 1)
    eig = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    total = float(eig.sum() + 1e-8)
    return (
        float(np.clip((eig[0] - eig[1]) / (eig[0] + 1e-8), 0, 1)),
        float(np.clip((eig[1] - eig[2]) / (eig[0] + 1e-8), 0, 1)),
        float(np.clip(eig[2] / total, 0, 1)),
    )


def composition_entropy(aa, dtype=np.float32) -> float:
    """Shannon entropy of residue composition, normalized by log(20).

    (nuclear V1 patches computed it in float64, the later builders in float32)
    """
    n = len(aa)
    probs = np.asarray([c / n for c in Counter(aa).values()], dtype=dtype)
    return float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(20))


def pairwise_stats(members: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All pairwise distances and sequence separations (upper triangle, row-major order)."""
    pair, seqsep = [], []
    n = len(members)
    for x in range(n):
        for y in range(x + 1, n):
            pair.append(float(np.linalg.norm(cc[x] - cc[y])))
            seqsep.append(abs(int(members[x]) - int(members[y])))
    return (np.asarray(pair if pair else [0.0], dtype=np.float32),
            np.asarray(seqsep if seqsep else [0.0], dtype=np.float32))
