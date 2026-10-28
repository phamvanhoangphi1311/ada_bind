"""Small-molecule-binding residue feature blocks.

Cache: cache/small_ultimate_v1/small_explicit_v1/<split>/<pid>.npz
  SM_SEQCHEM32    hydropathy / aromatic / sulfur / acidic / G-P sequence context (radii 2,4,7,10)
  SM_POCKET3D44   pocket composition at 5/7/9/11 Å, nearest chemistry, shell occupancy, PCA shape
  SM_LIGCHEM20    own interaction chemistry + local donor/acceptor/aromatic environment
  SM_TOPO16       packing, aromatic/hydrophobic components, clustering, neighbour entropy
  SM_AROMSTACK12  aromatic identity, density, ring-normal coherence (stacking proxy)
  SM_PROTCTX8     whole-protein composition context
Registered feature group: SM_V1 (the six blocks); the per-protein cache above is imported by the feature store.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from ..constants import HYDROPATHY
from ..data.structures import load_chain_residues, representative_coord
from .api import Protein, feature_group
from .geometry import component_sizes, frac, group_index, nearest_distance

GROUPS = {
    "AROM": set("FWYH"),
    "STRONG_AROM": set("FWY"),
    "SULFUR": set("CM"),
    "HYDRO": set("AVILMFWY"),
    "ALIPHATIC": set("AVILM"),
    "POLAR": set("NQSTY"),
    "ACIDIC": set("DE"),
    "BASIC": set("KRH"),
    "GP": set("GP"),
    "HBD": set("KRHNQSTYW"),
    "HBA": set("DEHNQSTYC"),
    "AMIDE": set("NQ"),
    "HYDROXYL": set("STY"),
}
RING_ATOMS = {
    "PHE": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "TYR": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
    "HIS": ["CG", "ND1", "CE1", "NE2", "CD2"],
    "TRP": ["CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2"],
}
V1_DIMS = {
    "SM_SEQCHEM32": 32,
    "SM_POCKET3D44": 44,
    "SM_LIGCHEM20": 20,
    "SM_TOPO16": 16,
    "SM_AROMSTACK12": 12,
    "SM_PROTCTX8": 8,
}


def hydropathy(seq: str) -> np.ndarray:
    return np.asarray([HYDROPATHY.get(a, 0.0) / 4.5 for a in seq], dtype=np.float32)


def ring_geometry(res):
    """(centroid, unit normal) of an aromatic ring, or (None, None)."""
    atoms = RING_ATOMS.get(res.resname.strip().upper())
    if atoms is None:
        return None, None
    cc = [np.asarray(res[a].coord, dtype=np.float32) for a in atoms if a in res]
    if len(cc) < 3:
        return None, None
    x = np.stack(cc)
    centroid = x.mean(axis=0)
    centered = x - centroid[None, :]
    eigval, eigvec = np.linalg.eigh(centered.T @ centered / max(len(x) - 1, 1))
    normal = eigvec[:, np.argmin(eigval)]
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    return centroid.astype(np.float32), normal.astype(np.float32)


# ------------------------------------------------------------------ sequence-only
def sequence_chem_block(seq: str) -> np.ndarray:
    """SM_SEQCHEM32."""
    G = GROUPS
    L = len(seq)
    out = np.zeros((L, 32), dtype=np.float32)
    hyd = hydropathy(seq)
    for i, a in enumerate(seq):
        out[i, :8] = [hyd[i], a in G["AROM"], a in G["SULFUR"], a in G["HYDRO"], a in G["ACIDIC"],
                      a in G["POLAR"], a in G["GP"], a in G["BASIC"]]
        col = 8
        for rad in (2, 4, 7, 10):
            lo, hi = max(0, i - rad), min(L, i + rad + 1)
            w = seq[lo:hi]
            out[i, col:col + 6] = [frac(G["AROM"], w), frac(G["HYDRO"], w), frac(G["SULFUR"], w),
                                   frac(G["ACIDIC"], w), frac(G["GP"], w), float(np.mean(hyd[lo:hi]))]
            col += 6
    return out


def protein_context_block(seq: str) -> np.ndarray:
    """SM_PROTCTX8."""
    G = GROUPS
    L = len(seq)
    p = np.array([math.log1p(L) / 8.0, frac(G["AROM"], seq), frac(G["HYDRO"], seq), frac(G["SULFUR"], seq),
                  frac(G["POLAR"], seq), frac(G["ACIDIC"], seq), frac(G["GP"], seq),
                  float(np.mean([HYDROPATHY.get(a, 0.0) / 4.5 for a in seq]))], dtype=np.float32)
    return np.repeat(p[None, :], L, axis=0)


# ------------------------------------------------------------------ structure-based
def structure_blocks(residues, seq: str):
    """SM_POCKET3D44, SM_LIGCHEM20, SM_TOPO16, SM_AROMSTACK12."""
    G = GROUPS
    coords = np.stack([representative_coord(r) for r in residues], axis=0).astype(np.float32)
    L = len(seq)
    tree = cKDTree(coords)
    hyd = hydropathy(seq)

    arom_idx = [i for i, a in enumerate(seq) if a in G["AROM"]]
    hydro_idx = [i for i, a in enumerate(seq) if a in G["HYDRO"]]
    empty = np.zeros((0, 3), dtype=np.float32)
    arom_comp = component_sizes(arom_idx, coords[arom_idx] if arom_idx else empty, 8.0)
    hydro_comp = component_sizes(hydro_idx, coords[hydro_idx] if hydro_idx else empty, 8.0)
    gidx = {k: group_index(seq, G[k]) for k in ("AROM", "SULFUR", "POLAR", "HYDRO")}

    ring_centroid, ring_normal = {}, {}
    for i, res in enumerate(residues):
        c, n = ring_geometry(res)
        if c is not None:
            ring_centroid[i], ring_normal[i] = c, n

    pocket = np.zeros((L, 44), dtype=np.float32)
    ligchem = np.zeros((L, 20), dtype=np.float32)
    topo = np.zeros((L, 16), dtype=np.float32)
    arom = np.zeros((L, 12), dtype=np.float32)
    neigh6 = [tree.query_ball_point(coords[i], 6.0) for i in range(L)]
    neigh8 = [tree.query_ball_point(coords[i], 8.0) for i in range(L)]
    neigh10 = [tree.query_ball_point(coords[i], 10.0) for i in range(L)]

    for i, a in enumerate(seq):
        # ---- SM_LIGCHEM20
        ligchem[i, :10] = [a in G[k] for k in ("HBD", "HBA", "AROM", "SULFUR", "HYDRO", "POLAR",
                                                "ACIDIC", "BASIC", "AMIDE", "HYDROXYL")]
        c = 10
        for radius in (6.0, 10.0):
            ids = np.asarray(tree.query_ball_point(coords[i], radius), dtype=int)
            aa = [seq[j] for j in ids[ids != i]]
            den = max(len(aa), 1)
            ligchem[i, c:c + 5] = [sum(x in G[k] for x in aa) / den
                                   for k in ("HBD", "HBA", "AROM", "SULFUR", "POLAR")]
            c += 5

        # ---- SM_POCKET3D44: four radii x eight descriptors
        c = 0
        for radius in (5.0, 7.0, 9.0, 11.0):
            ids = np.asarray(tree.query_ball_point(coords[i], radius), dtype=int)
            ids = ids[ids != i]
            if len(ids):
                aa = [seq[j] for j in ids]
                dd = np.linalg.norm(coords[ids] - coords[i][None, :], axis=1)
                vals = [min(len(ids), 40) / 40.0, frac(G["AROM"], aa), frac(G["HYDRO"], aa),
                        frac(G["SULFUR"], aa), frac(G["POLAR"], aa), frac(G["ACIDIC"], aa),
                        float(np.mean(hyd[ids])), min(float(np.mean(dd)) / 12.0, 1.0)]
            else:
                vals = [0.0] * 8
            pocket[i, c:c + 8] = vals
            c += 8
        pocket[i, 32] = nearest_distance(coords, gidx["AROM"], i)
        pocket[i, 33] = nearest_distance(coords, gidx["SULFUR"], i)
        pocket[i, 34] = nearest_distance(coords, gidx["POLAR"], i)
        pocket[i, 35] = nearest_distance(coords, gidx["HYDRO"], i)
        # shell occupancy: packing / cavity proxy
        d_all = np.linalg.norm(coords - coords[i][None, :], axis=1)
        shells = [np.sum((d_all > 0.1) & (d_all <= 4.0)), np.sum((d_all > 4.0) & (d_all <= 6.0)),
                  np.sum((d_all > 6.0) & (d_all <= 8.0)), np.sum((d_all > 8.0) & (d_all <= 10.0))]
        pocket[i, 36:40] = np.clip(np.asarray(shells, dtype=np.float32) / 20.0, 0, 1)
        ids = np.asarray(neigh8[i], dtype=int)
        ids = ids[ids != i]
        if len(ids) >= 3:
            cc = coords[ids] - coords[ids].mean(axis=0, keepdims=True)
            eig = np.sort(np.clip(np.linalg.eigvalsh(cc.T @ cc / max(len(ids) - 1, 1)), 0, None))[::-1]
            total = float(eig.sum() + 1e-8)
            pocket[i, 40] = np.clip((eig[0] - eig[1]) / (eig[0] + 1e-8), 0, 1)
            pocket[i, 41] = np.clip((eig[1] - eig[2]) / (eig[0] + 1e-8), 0, 1)
            pocket[i, 42] = np.clip(eig[2] / total, 0, 1)
            pocket[i, 43] = float(np.mean(np.abs(ids - i) > 10))

        # ---- SM_TOPO16
        n6 = [j for j in neigh6[i] if j != i]
        n8 = [j for j in neigh8[i] if j != i]
        n10 = [j for j in neigh10[i] if j != i]
        a8 = [j for j in n8 if seq[j] in G["AROM"]]
        h8 = [j for j in n8 if seq[j] in G["HYDRO"]]
        s8 = [j for j in n8 if seq[j] in G["SULFUR"]]
        p8 = [j for j in n8 if seq[j] in G["POLAR"]]
        d8 = [j for j in n8 if seq[j] in G["ACIDIC"]]
        topo[i, 0] = min(len(n6), 25) / 25.0
        topo[i, 1] = min(len(n8), 35) / 35.0
        topo[i, 2] = min(len(n10), 45) / 45.0
        topo[i, 3] = min(len(a8), 12) / 12.0
        topo[i, 4] = min(len(h8), 20) / 20.0
        topo[i, 5] = min(len(s8), 8) / 8.0
        topo[i, 6] = min(len(p8), 20) / 20.0
        topo[i, 7] = min(len(d8), 15) / 15.0
        topo[i, 8] = min(arom_comp.get(i, 0), 15) / 15.0
        topo[i, 9] = min(hydro_comp.get(i, 0), 25) / 25.0
        if len(n8) >= 2:
            edges = len(cKDTree(coords[n8]).query_pairs(8.0))
            topo[i, 10] = edges / max(len(n8) * (len(n8) - 1) / 2, 1)
            seqsep = np.abs(np.asarray(n8) - i)
            topo[i, 11] = float(np.mean(seqsep > 10))
            topo[i, 12] = min(float(np.mean(seqsep)) / 50.0, 1.0)
            probs = np.asarray([x / len(n8) for x in Counter(seq[j] for j in n8).values()], dtype=float)
            topo[i, 13] = float(-np.sum(probs * np.log(probs + 1e-12)) / math.log(20))
        topo[i, 14] = np.tanh((len(n6) - (len(n10) - len(n6))) / 10.0)  # inner/outer packing balance
        topo[i, 15] = np.tanh((len(h8) + len(a8) - len(d8)) / 10.0)

        # ---- SM_AROMSTACK12
        arom[i, :5] = [a in G["AROM"], a == "W", a == "F", a == "Y", a == "H"]
        for col2, radius in zip((5, 6, 7), (6.0, 8.0, 10.0)):
            ids = [j for j in tree.query_ball_point(coords[i], radius) if j != i]
            arom[i, col2] = sum(seq[j] in G["AROM"] for j in ids) / max(len(ids), 1)
        arom[i, 8] = nearest_distance(coords, gidx["AROM"], i)
        arom[i, 9] = min(arom_comp.get(i, 0), 15) / 15.0
        ar_nei = [j for j in neigh10[i] if j in ring_normal and j != i]
        dots = [abs(float(np.dot(ring_normal[ar_nei[x]], ring_normal[ar_nei[y]])))
                for x in range(len(ar_nei)) for y in range(x + 1, len(ar_nei))]
        arom[i, 10] = float(np.mean(dots)) if dots else 0.0
        if ar_nei:
            cent = np.mean(np.stack([ring_centroid[j] for j in ar_nei]), axis=0)
            arom[i, 11] = min(float(np.linalg.norm(coords[i] - cent)) / 20.0, 1.0)

    return pocket, ligchem, topo, arom


def build_v1_blocks(pdb, seq: str) -> dict[str, np.ndarray]:
    pocket, ligchem, topo, arom = structure_blocks(load_chain_residues(pdb, seq), seq)
    return {
        "SM_SEQCHEM32": sequence_chem_block(seq),
        "SM_POCKET3D44": pocket,
        "SM_LIGCHEM20": ligchem,
        "SM_TOPO16": topo,
        "SM_AROMSTACK12": arom,
        "SM_PROTCTX8": protein_context_block(seq),
    }


# ------------------------------------------------------------------ registered feature group
@feature_group("SM_V1", columns={b: [f"{b}_{k}" for k in range(d)] for b, d in V1_DIMS.items()},
               structure="exact_chain_a", legacy_subdir="small_ultimate_v1/small_explicit_v1",
               legacy_cache={b: "{split}/{pid}.npz" for b in V1_DIMS})
def sm_v1(p: Protein) -> dict:
    """Small-molecule V1 blocks: sequence chemistry, pocket 3-D, ligand chemistry, topology, stacking, context."""
    return build_v1_blocks(p.pdb, p.sequence)
