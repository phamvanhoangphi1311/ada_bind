"""Nuclear-binding (nucleic-acid) residue feature blocks.

V1 blocks (cache: cache/nuclear_ultimate_v1/nuclear_explicit_v1/<split>/<pid>.npz)
  NA_SEQELECTRO32  sequence charge / basic / acidic / polar context at radii 2,4,7,10
  NA_PATCH3D40     3D electrostatic patch summaries at 6/8/10/12 Å + distances + basic components
  NA_HBOND16       own H-bond chemistry + local donor/acceptor environment
  NA_TOPO12        spatial packing / basic-acidic topology / clustering coefficient
  NA_PROTCTX8      whole-protein composition context (repeated per residue)
  NA_AROMCTRL8     aromatic control block (must earn its place)
V2 block (cache: cache/nuclear_v2_selective_multiscale/surftopo16/<split>/<pid>.npy)
  NA_SURFTOPO16    surface-restricted electrostatic topology using a geometric exposure proxy
Registered feature groups: NA_V1 (six V1 blocks), NA_SURFTOPO (V2 block); the per-protein caches above are
imported by the feature store when present.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ..constants import CHARGE
from ..data.structures import structure_coords
from .api import Protein, feature_group
from .geometry import component_sizes, exposure_proxy, frac, group_index, nearest_distance, window_frac

GROUPS = {
    "KR": set("KR"),
    "KRH": set("KRH"),
    "DE": set("DE"),
    "POLAR": set("NQSTY"),
    "AROM": set("FWYH"),
    "HYDRO": set("AVILMFWY"),
    "HBD": set("KRHNQSTYW"),
    "HBA": set("DEHNQSTYC"),
}

V1_DIMS = {
    "NA_SEQELECTRO32": 32,
    "NA_PATCH3D40": 40,
    "NA_HBOND16": 16,
    "NA_TOPO12": 12,
    "NA_PROTCTX8": 8,
    "NA_AROMCTRL8": 8,
}
SURFTOPO_DIMS = {"NA_SURFTOPO16": 16}


def charges(seq: str) -> np.ndarray:
    return np.array([CHARGE.get(a, 0.0) for a in seq], dtype=np.float32)


# ------------------------------------------------------------------ sequence-only
def sequence_electro_block(seq: str) -> np.ndarray:
    """NA_SEQELECTRO32."""
    G = GROUPS
    L = len(seq)
    out = np.zeros((L, 32), dtype=np.float32)
    q = charges(seq)
    for i, a in enumerate(seq):
        out[i, :8] = [q[i], a == "K", a == "R", a == "H", a in G["DE"], a in G["POLAR"],
                      a in G["AROM"], a in G["HYDRO"]]
        col = 8
        for radius in (2, 4, 7, 10):
            lo, hi = max(0, i - radius), min(L, i + radius + 1)
            w, qq = seq[lo:hi], q[lo:hi]
            out[i, col:col + 6] = [frac(G["KR"], w), frac(G["KRH"], w), frac(G["DE"], w),
                                   frac(G["POLAR"], w), float(np.mean(qq)), frac(G["HYDRO"], w)]
            col += 6
    return out


def protein_context_block(seq: str) -> np.ndarray:
    """NA_PROTCTX8 (same vector for every residue)."""
    G = GROUPS
    L = len(seq)
    p = np.array([math.log1p(L) / 8.0, frac(G["KR"], seq), frac(G["KRH"], seq), frac(G["DE"], seq),
                  frac(G["POLAR"], seq), frac(G["HYDRO"], seq), frac(G["AROM"], seq),
                  float(np.mean(charges(seq)))], dtype=np.float32)
    return np.repeat(p[None, :], L, axis=0)


# ------------------------------------------------------------------ structure-based
def structure_blocks(coords: np.ndarray, seq: str):
    """NA_PATCH3D40, NA_HBOND16, NA_TOPO12, NA_AROMCTRL8 from representative coordinates."""
    G = GROUPS
    L = len(seq)
    tree = cKDTree(coords)
    q = charges(seq)

    basic_idx = [i for i, a in enumerate(seq) if a in G["KRH"]]
    basic_comp = component_sizes(basic_idx, coords[basic_idx] if basic_idx else np.zeros((0, 3), np.float32), 8.0)
    gidx = {k: group_index(seq, G[k]) for k in ("KR", "DE", "POLAR", "AROM", "KRH")}

    patch = np.zeros((L, 40), dtype=np.float32)
    hbond = np.zeros((L, 16), dtype=np.float32)
    topo = np.zeros((L, 12), dtype=np.float32)
    arom = np.zeros((L, 8), dtype=np.float32)
    neigh8 = [tree.query_ball_point(coords[i], 8.0) for i in range(L)]
    neigh10 = [tree.query_ball_point(coords[i], 10.0) for i in range(L)]

    def centroid_distance(idx, i):
        if len(idx) == 0:
            return 1.0
        d = np.linalg.norm(coords[idx] - coords[i][None, :], axis=1)
        sel = idx[d <= 10.0]
        if len(sel) == 0:
            return 1.0
        return min(float(np.linalg.norm(coords[i] - coords[sel].mean(axis=0))) / 20.0, 1.0)

    for i, a in enumerate(seq):
        # own H-bond chemistry
        hbond[i, :8] = [a in G["HBD"], a in G["HBA"], a == "R", a == "K", a == "H",
                        a in set("NQ"), a in set("STY"), a in set("DE")]

        # 3D radius summaries
        col = 0
        for radius in (6.0, 8.0, 10.0, 12.0):
            ids = np.asarray(tree.query_ball_point(coords[i], radius), dtype=int)
            ids = ids[ids != i]
            if len(ids):
                aa = [seq[j] for j in ids]
                d = np.linalg.norm(coords[ids] - coords[i][None, :], axis=1)
                qn = q[ids]
                vals = [min(len(ids), 40) / 40.0, frac(G["KR"], aa), frac(G["KRH"], aa), frac(G["DE"], aa),
                        frac(G["POLAR"], aa), frac(G["HYDRO"], aa), float(np.mean(qn)),
                        float(np.tanh(np.sum(qn / (d + 1.0))))]
            else:
                vals = [0.0] * 8
            patch[i, col:col + 8] = vals
            col += 8

        # local H-bond-capable environment
        col = 8
        for radius in (6.0, 10.0):
            ids = np.asarray(tree.query_ball_point(coords[i], radius), dtype=int)
            aa = [seq[j] for j in ids[ids != i]]
            den = max(len(aa), 1)
            hbond[i, col:col + 4] = [sum(x in G[k] for x in aa) / den for k in ("HBD", "HBA", "KRH", "POLAR")]
            col += 4

        # nearest chemistry distances
        patch[i, 32] = nearest_distance(coords, gidx["KR"], i)
        patch[i, 33] = nearest_distance(coords, gidx["DE"], i)
        patch[i, 34] = nearest_distance(coords, gidx["POLAR"], i)
        patch[i, 35] = nearest_distance(coords, gidx["AROM"], i)

        # higher-order basic component proximity
        max_comp = max([basic_comp[j] for j in neigh10[i] if j in basic_comp], default=0)
        patch[i, 36] = min(max_comp, 20) / 20.0
        patch[i, 37] = centroid_distance(gidx["KRH"], i)
        patch[i, 38] = centroid_distance(gidx["POLAR"], i)
        ids = np.asarray(neigh10[i], dtype=int)
        ids = ids[ids != i]
        if len(ids):
            patch[i, 39] = min(float(np.mean(np.linalg.norm(coords[ids] - coords[i][None, :], axis=1))) / 20.0, 1.0)

        # topology
        n8 = [j for j in neigh8[i] if j != i]
        n10 = [j for j in neigh10[i] if j != i]
        b8 = [j for j in n8 if seq[j] in G["KRH"]]
        p8 = [j for j in n8 if seq[j] in G["POLAR"]]
        d8 = [j for j in n8 if seq[j] in G["DE"]]
        topo[i, 0] = min(len(n8), 30) / 30.0
        topo[i, 1] = min(len(n10), 40) / 40.0
        topo[i, 2] = min(len(b8), 15) / 15.0
        topo[i, 3] = min(len(p8), 20) / 20.0
        topo[i, 4] = min(len(d8), 15) / 15.0
        topo[i, 5] = np.tanh((len(b8) - len(d8)) / 5.0)
        topo[i, 6] = min(max_comp, 20) / 20.0
        topo[i, 7] = min(len(b8) + len(d8), 20) / 20.0
        if n8:
            seqsep = np.abs(np.asarray(n8) - i)
            topo[i, 8] = float(np.mean(seqsep > 10))
            topo[i, 9] = min(float(np.mean(seqsep)) / 50.0, 1.0)
            if len(n8) >= 2:  # local clustering coefficient
                n_edges = len(cKDTree(coords[n8]).query_pairs(8.0))
                topo[i, 10] = n_edges / max(len(n8) * (len(n8) - 1) / 2, 1)
        if b8:
            bd = np.linalg.norm(coords[b8] - coords[i][None, :], axis=1)
            topo[i, 11] = min((len(b8) / (1.0 + float(np.mean(bd)))) / 3.0, 1.0)

        # aromatic control
        arom[i, 0] = float(a in G["AROM"])
        arom[i, 1] = float(a in set("YH"))
        for col2, radius in zip((2, 3, 4), (6.0, 8.0, 10.0)):
            ids = [j for j in tree.query_ball_point(coords[i], radius) if j != i]
            arom[i, col2] = sum(seq[j] in G["AROM"] for j in ids) / max(len(ids), 1)
        arom[i, 5] = window_frac(seq, i, G["AROM"], 4)
        arom[i, 6] = window_frac(seq, i, G["AROM"], 10)
        arom[i, 7] = nearest_distance(coords, gidx["AROM"], i)

    return patch, hbond, topo, arom


def build_v1_blocks(pdb, seq: str) -> dict[str, np.ndarray]:
    """All six V1 blocks for one protein (the per-protein cache unit)."""
    patch, hbond, topo, arom = structure_blocks(structure_coords(pdb, seq), seq)
    return {
        "NA_SEQELECTRO32": sequence_electro_block(seq),
        "NA_PATCH3D40": patch,
        "NA_HBOND16": hbond,
        "NA_TOPO12": topo,
        "NA_PROTCTX8": protein_context_block(seq),
        "NA_AROMCTRL8": arom,
    }


# ------------------------------------------------------------------ V2 surface topology
def surface_topology_block(coords: np.ndarray, seq: str) -> np.ndarray:
    """NA_SURFTOPO16: chemistry/electrostatics restricted to the geometrically exposed half."""
    G = GROUPS
    tree = cKDTree(coords)
    exposure, n10 = exposure_proxy(coords)
    L = len(seq)
    q = charges(seq)
    surface = exposure >= 0.50
    surf_basic = [i for i, a in enumerate(seq) if surface[i] and a in G["KRH"]]
    comp = component_sizes(surf_basic, coords[surf_basic] if surf_basic else np.zeros((0, 3), np.float32), 8.0)
    surf_basic_arr = np.asarray(surf_basic, dtype=int)

    out = np.zeros((L, 16), dtype=np.float32)
    for i in range(L):
        ids8 = np.asarray(tree.query_ball_point(coords[i], 8.0), dtype=int)
        ids8 = ids8[ids8 != i]
        surf8 = ids8[surface[ids8]] if len(ids8) else np.array([], dtype=int)

        out[i, 0] = exposure[i]
        out[i, 1] = min(float(n10[i]) / 40.0, 1.0)
        out[i, 2] = len(surf8) / max(len(ids8), 1)
        if len(surf8):
            aa = [seq[j] for j in surf8]
            qq = q[surf8]
            dd = np.linalg.norm(coords[surf8] - coords[i][None, :], axis=1)
            out[i, 3] = frac(G["KRH"], aa)
            out[i, 4] = frac(G["KR"], aa)
            out[i, 5] = frac(G["POLAR"], aa)
            out[i, 6] = frac(G["DE"], aa)
            out[i, 7] = float(np.mean(qq))
            out[i, 8] = float(np.tanh(np.sum(qq / (dd + 1.0))))
            out[i, 12] = float(np.std(qq))
            out[i, 13] = float(np.tanh((sum(a in G["KRH"] for a in aa) - sum(a in G["DE"] for a in aa)) / 5.0))
            out[i, 14] = frac(G["HBD"], aa)
            out[i, 15] = float(np.mean(np.abs(surf8 - i) > 10))
        out[i, 9] = nearest_distance(coords, surf_basic_arr, i)
        out[i, 10] = min(comp.get(i, 0), 20) / 20.0
        ids12 = np.asarray(tree.query_ball_point(coords[i], 12.0), dtype=int)
        pos = [j for j in ids12 if surface[j] and seq[j] in G["KRH"]]
        out[i, 11] = (min(float(np.linalg.norm(coords[i] - coords[pos].mean(axis=0))) / 20.0, 1.0)
                      if pos else 1.0)
    return out


def build_surftopo_block(pdb, seq: str) -> dict[str, np.ndarray]:
    return {"NA_SURFTOPO16": surface_topology_block(structure_coords(pdb, seq), seq)}


# ------------------------------------------------------------------ registered feature groups
@feature_group("NA_V1", columns={b: [f"{b}_{k}" for k in range(d)] for b, d in V1_DIMS.items()},
               structure="exact_chain_a", legacy_subdir="nuclear_ultimate_v1/nuclear_explicit_v1",
               legacy_cache={b: "{split}/{pid}.npz" for b in V1_DIMS})
def na_v1(p: Protein) -> dict:
    """Nucleic-acid V1 blocks: sequence electrostatics, 3-D electro patches, H-bond, topology, context, aromatic."""
    return build_v1_blocks(p.pdb, p.sequence)


@feature_group("NA_SURFTOPO", columns={"NA_SURFTOPO16": [f"NA_SURFTOPO16_{k}" for k in range(16)]},
               structure="exact_chain_a", legacy_subdir="nuclear_v2_selective_multiscale/surftopo16",
               legacy_cache={"NA_SURFTOPO16": "{split}/{pid}.npy"})
def na_surftopo(p: Protein) -> dict:
    """Nucleic-acid V2 block: surface-restricted electrostatic topology."""
    return build_surftopo_block(p.pdb, p.sequence)
