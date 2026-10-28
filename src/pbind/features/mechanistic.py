"""Mechanism-first blocks (port of GEAR ``gear_mechanistic``). All label-independent.

MECH1  = ACCESS(10) + SHELL2(12) + ENV(8) + CONF(4)
MULTI2 = METALSPEC(18) + ELEC3D(8) + TOPO(12) + SOLV(10) + SEQMOTIF(10)
SEQCTX = 182-d local sequence context
Registered feature groups: MECH1, MULTI2, SEQCTX. Original backend caches (imported by the feature store):
    <BASE>/cache/spatial_revamp/<split>_mechanistic_v1.pkl, <split>_multisignal_v2.pkl   {prot_id: {block: (n, d)}}
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from ..constants import AA20, HYDROPATHY
from ..data.structures import safe_seq1
from .api import Protein, feature_group
from .coordination import ATOM_DONOR_ATOMS

MECH1_BLOCKS = ("ACCESS", "SHELL2", "ENV", "CONF")
MULTI2_BLOCKS = ("METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF")
MECH1_DIMS = {"ACCESS": 10, "SHELL2": 12, "ENV": 8, "CONF": 4}
MULTI2_DIMS = {"METALSPEC": 18, "ELEC3D": 8, "TOPO": 12, "SOLV": 10, "SEQMOTIF": 10}

PRIMARY_DONORS = ATOM_DONOR_ATOMS
HYDROPHOBIC, POLAR, AROMATIC, ACIDIC, BASIC = set("AVILMFWC"), set("STNQYC"), set("FWYH"), set("DE"), set("KRH")
POLAR_SUPPORT_AA = {"SER", "THR", "TYR", "ASN", "GLN"}
ACIDIC_RES, BASIC_RES = {"ASP", "GLU"}, {"LYS", "ARG", "HIS"}

METALS = ["Zn", "Mg", "Ca", "Mn", "Fe", "Cu"]
CANONICAL_DONORS = {"CYS": {"SG": "S"}, "HIS": {"ND1": "N", "NE2": "N"}, "ASP": {"OD1": "O", "OD2": "O"},
                    "GLU": {"OE1": "O", "OE2": "O"}}
BROAD_DONORS = {**CANONICAL_DONORS, "SER": {"OG": "O"}, "THR": {"OG1": "O"}, "TYR": {"OH": "O"},
                "ASN": {"OD1": "O"}, "GLN": {"OE1": "O"}, "MET": {"SD": "S"}}
DONOR_AA = set("HCDE")
METAL_PROTO = {
    "Zn": {"r0": 2.10, "coord": 4.0, "weights": {"O": 0.85, "N": 1.00, "S": 1.00}},
    "Mg": {"r0": 2.10, "coord": 6.0, "weights": {"O": 1.00, "N": 0.35, "S": 0.10}},
    "Ca": {"r0": 2.40, "coord": 7.0, "weights": {"O": 1.00, "N": 0.25, "S": 0.10}},
    "Mn": {"r0": 2.20, "coord": 6.0, "weights": {"O": 1.00, "N": 0.60, "S": 0.25}},
    "Fe": {"r0": 2.15, "coord": 6.0, "weights": {"O": 0.90, "N": 0.85, "S": 0.60}},
    "Cu": {"r0": 2.05, "coord": 4.0, "weights": {"O": 0.65, "N": 1.00, "S": 0.95}},
}

# Column names (same as gear_mechanistic).
FEATURE_NAMES = {
    "ACCESS": ["other_heavy_count_3A", "other_heavy_count_4A", "other_heavy_count_6A", "other_heavy_count_8A",
               "nearest_other_heavy_A", "other_residue_count_6A", "other_residue_count_8A",
               "open_octant_fraction_6A", "directional_anisotropy_6A", "relative_surface_radius"],
    "SHELL2": ["focal_primary_donor_count", "support_hetero_atoms_3p5A", "support_residues_3p5A",
               "support_hetero_atoms_4A", "support_residues_4A", "primary_donor_support_fraction",
               "min_support_distance_A", "mean_support_distance_A", "acidic_support_residues_4A",
               "basic_support_residues_4A", "polar_support_residues_4A", "backbone_support_atoms_3p5A"],
    "ENV": ["neighbor_residues_6A", "hydrophobic_fraction_6A", "polar_fraction_6A", "aromatic_fraction_6A",
            "acidic_fraction_6A", "basic_fraction_6A", "mean_hydropathy_6A", "hydropathy_std_6A"],
    "CONF": ["focal_plddt", "local6A_plddt_mean", "local6A_plddt_std", "local_donor_plddt_mean"],
    "METALSPEC": [f"{m}_{s}" for m in METALS for s in ("distance_fit", "coordination_fit", "donor_type_fit")],
    "ELEC3D": ["net_charge_8A", "field_magnitude_8A", "field_radial_component_8A", "dipole_magnitude_8A",
               "positive_negative_centroid_distance", "charge_anisotropy_8A", "net_charge_12A",
               "field_magnitude_12A"],
    "TOPO": ["donor_residue_count_8A", "donor_type_diversity", "donor_graph_edges", "donor_graph_density",
             "largest_component_fraction", "donor_triangles", "longrange_donor_count", "mean_sequence_separation",
             "max_sequence_separation", "donor_radius_of_gyration", "min_donor_pair_distance",
             "mean_donor_pair_distance"],
    "SOLV": ["polar_atoms_3A", "polar_atoms_4A", "polar_atoms_6A", "polar_fraction_4A", "polar_fraction_6A",
             "nonpolar_heavy_atoms_4A", "polar_shell_4_6_ratio", "local_void_proxy", "buried_polar_proxy",
             "polar_anisotropy_6A"],
    "SEQMOTIF": ["donor_fraction_w2", "donor_fraction_w4", "donor_fraction_w8", "donor_fraction_w16",
                 "acidic_fraction_w4", "acidic_fraction_w8", "acidic_fraction_w16", "donor_type_diversity_w8",
                 "donor_type_diversity_w16", "longrange_sequence_donor_density"],
    "SEQCTX": [f"aa_{a}_at_{o:+d}" for o in (-4, -3, -2, -1, 1, 2, 3, 4) for a in AA20]
              + [f"comp_{a}_w8" for a in AA20] + ["relative_position", "terminus_proximity"],
}


def _anchor(res, donor_names) -> np.ndarray:
    """Mean of canonical donor atoms, else Cβ, else Cα."""
    xyz = [np.asarray(res[n].coord, np.float32) for n in donor_names if n in res]
    if xyz:
        return np.mean(np.stack(xyz), axis=0).astype(np.float32)
    if "CB" in res:
        return np.asarray(res["CB"].coord, np.float32)
    return np.asarray(res["CA"].coord, np.float32)


def _principal_fraction(u: np.ndarray) -> float:
    eig = np.maximum(np.linalg.eigvalsh(np.cov(u.T, bias=True)), 0.0)
    return float(eig.max() / max(eig.sum(), 1e-8))


# ------------------------------------------------------------------ MECH1
def extract_mechanistic_features(residues) -> dict:
    n = len(residues)
    ACCESS = np.zeros((n, 10), np.float32)
    SHELL2 = np.zeros((n, 12), np.float32)
    ENV = np.zeros((n, 8), np.float32)
    CONF = np.zeros((n, 4), np.float32)
    resnames = [str(r.resname).upper() for r in residues]
    anchors = np.stack([_anchor(r, PRIMARY_DONORS.get(rn, [])) for r, rn in zip(residues, resnames)]).astype(np.float32)
    ca = np.stack([np.asarray(r["CA"].coord, np.float32) for r in residues])
    aa = [safe_seq1(r.resname) for r in residues]
    plddt = np.asarray([float(np.clip(float(r["CA"].bfactor), 0.0, 100.0)) if "CA" in r else 0.0 for r in residues],
                       np.float32)
    ca_tree = cKDTree(ca)
    radial = np.linalg.norm(ca - ca.mean(axis=0), axis=1)
    radial_scale = max(float(np.percentile(radial, 95)), 1e-6)

    heavy_xyz, heavy_res, support_xyz, support_res, support_bb, primary_res, primary_xyz = [], [], [], [], [], [], []
    for ri, res in enumerate(residues):
        canonical = set(PRIMARY_DONORS.get(resnames[ri], []))
        for atom in res:
            elem = str(atom.element).upper()
            if elem == "H":
                continue
            xyz = np.asarray(atom.coord, np.float32)
            name = str(atom.name)
            heavy_xyz.append(xyz)
            heavy_res.append(ri)
            if name in canonical:
                primary_xyz.append(xyz)
                primary_res.append(ri)
            if elem in {"N", "O", "S"} and name not in canonical:  # second-shell polar support
                support_xyz.append(xyz)
                support_res.append(ri)
                support_bb.append(name in {"N", "O", "OXT"})
    heavy_xyz, heavy_res = np.asarray(heavy_xyz, np.float32), np.asarray(heavy_res, np.int32)
    support_xyz, support_res = np.asarray(support_xyz, np.float32), np.asarray(support_res, np.int32)
    support_bb = np.asarray(support_bb, bool)
    primary_res = np.asarray(primary_res, np.int32)
    heavy_tree = cKDTree(heavy_xyz) if len(heavy_xyz) else None
    support_tree = cKDTree(support_xyz) if len(support_xyz) else None
    primary_tree = cKDTree(np.asarray(primary_xyz, np.float32)) if len(primary_xyz) else None

    for i in range(n):
        anchor = anchors[i]
        # ---- ACCESSIBILITY / PACKING
        idx = [j for j in heavy_tree.query_ball_point(anchor, r=8.0) if heavy_res[j] != i] if heavy_tree is not None else []
        coords = heavy_xyz[idx] if len(idx) else np.empty((0, 3), np.float32)
        rids = heavy_res[idx] if len(idx) else np.empty(0, np.int32)
        d = np.linalg.norm(coords - anchor, axis=1) if len(coords) else np.empty(0, np.float32)
        for col, radius in enumerate((3.0, 4.0, 6.0, 8.0)):
            ACCESS[i, col] = float(np.sum(d <= radius))
        ACCESS[i, 4] = float(d.min()) if len(d) else 8.0
        ACCESS[i, 5] = float(len(set(rids[d <= 6.0].tolist())))
        ACCESS[i, 6] = float(len(set(rids[d <= 8.0].tolist())))
        near = coords[d <= 6.0] if len(d) else np.empty((0, 3), np.float32)
        if len(near):
            v = near - anchor
            octants = (v[:, 0] >= 0).astype(int) * 4 + (v[:, 1] >= 0).astype(int) * 2 + (v[:, 2] >= 0).astype(int)
            ACCESS[i, 7] = 1.0 - len(np.unique(octants)) / 8.0
            unit = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
            if len(unit) >= 3:
                eig = np.sort(np.maximum(np.linalg.eigvalsh(np.cov(unit.T, bias=True)), 0.0))
                ACCESS[i, 8] = float(eig[-1] / max(eig.sum(), 1e-8))
        else:
            ACCESS[i, 7] = 1.0
            ACCESS[i, 8] = 1.0
        ACCESS[i, 9] = float(np.clip(radial[i] / radial_scale, 0.0, 2.0))

        # ---- ENVIRONMENT
        nbr = [j for j in ca_tree.query_ball_point(ca[i], r=6.0) if j != i]
        ENV[i, 0] = len(nbr)
        if nbr:
            aas = [aa[j] for j in nbr]
            den = float(len(aas))
            ENV[i, 1:6] = [sum(a in g for a in aas) / den for g in (HYDROPHOBIC, POLAR, AROMATIC, ACIDIC, BASIC)]
            hyd = np.asarray([HYDROPATHY.get(a, 0.0) for a in aas], np.float32)
            ENV[i, 6] = float(hyd.mean())
            ENV[i, 7] = float(hyd.std())

        # ---- CONFIDENCE (pLDDT from the CA B-factor, 0-100 scale)
        local = ca_tree.query_ball_point(ca[i], r=6.0)
        CONF[i, 0] = plddt[i] / 100.0
        if local:
            vals = plddt[local] / 100.0
            CONF[i, 1] = float(vals.mean())
            CONF[i, 2] = float(vals.std())
        CONF[i, 3] = CONF[i, 1]
        if primary_tree is not None:
            pidx = primary_tree.query_ball_point(anchor, r=6.0)
            if pidx:
                CONF[i, 3] = float((plddt[np.unique(primary_res[pidx])] / 100.0).mean())

        # ---- SECOND COORDINATION SHELL (must stay last: uses `continue`)
        focal_res = residues[i]
        focal_donors = [np.asarray(focal_res[a].coord, np.float32)
                        for a in PRIMARY_DONORS.get(resnames[i], []) if a in focal_res]
        SHELL2[i, 0] = len(focal_donors)
        if not focal_donors or support_tree is None:
            SHELL2[i, 6] = 6.0
            SHELL2[i, 7] = 6.0
            continue
        s35, s40, r35, r40, sdist, bb35 = set(), set(), set(), set(), [], set()
        supported = 0
        for fd in focal_donors:
            idx35 = [j for j in support_tree.query_ball_point(fd, r=3.5) if support_res[j] != i]
            idx40 = [j for j in support_tree.query_ball_point(fd, r=4.0) if support_res[j] != i]
            if idx35:
                supported += 1
            for j in idx35:
                s35.add(int(j))
                r35.add(int(support_res[j]))
                sdist.append(float(np.linalg.norm(support_xyz[j] - fd)))
                if support_bb[j]:
                    bb35.add(int(j))
            for j in idx40:
                s40.add(int(j))
                r40.add(int(support_res[j]))
        SHELL2[i, 1:6] = [len(s35), len(r35), len(s40), len(r40), supported / max(len(focal_donors), 1)]
        SHELL2[i, 6] = min(sdist) if sdist else 6.0
        SHELL2[i, 7] = float(np.mean(sdist)) if sdist else 6.0
        SHELL2[i, 8] = sum(resnames[j] in ACIDIC_RES for j in r40)
        SHELL2[i, 9] = sum(resnames[j] in BASIC_RES for j in r40)
        SHELL2[i, 10] = sum(resnames[j] in POLAR_SUPPORT_AA for j in r40)
        SHELL2[i, 11] = len(bb35)
    return {"ACCESS": ACCESS, "SHELL2": SHELL2, "ENV": ENV, "CONF": CONF}


# ------------------------------------------------------------------ MULTI2
def _sphere_pair_centers(a, b, r=2.10, samples=12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = b - a
    d = np.linalg.norm(v)
    if d < 1e-6 or d > 2 * r:
        return np.empty((0, 3), np.float32)
    u = v / d
    helper = np.array([1.0, 0.0, 0.0])
    if abs(u @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, helper)
    e1 /= max(np.linalg.norm(e1), 1e-8)
    e2 = np.cross(u, e1)
    mid = (a + b) / 2
    h = math.sqrt(max(r * r - (d / 2) ** 2, 0.0))
    if h < 1e-8:
        return mid.reshape(1, 3).astype(np.float32)
    return np.asarray([mid + h * (np.cos(t) * e1 + np.sin(t) * e2)
                       for t in np.linspace(0, 2 * np.pi, samples, endpoint=False)], np.float32)


def _charge_center(res, rn):
    names = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "LYS": ("NZ",), "ARG": ("NH1", "NH2", "CZ")}.get(rn)
    if names is None:
        return None
    pts = [np.asarray(res[x].coord, np.float32) for x in names if x in res]
    if not pts:
        return None
    return np.mean(pts, axis=0), (+1.0 if rn in ("LYS", "ARG") else -1.0)


def extract_new2_features(residues, sequence: str) -> dict:
    n = len(residues)
    METALSPEC = np.zeros((n, 18), np.float32)
    ELEC3D = np.zeros((n, 8), np.float32)
    TOPO = np.zeros((n, 12), np.float32)
    SOLV = np.zeros((n, 10), np.float32)
    SEQMOTIF = np.zeros((n, 10), np.float32)
    resnames = [str(r.resname).upper() for r in residues]
    anchors = np.stack([_anchor(r, CANONICAL_DONORS.get(rn, {})) for r, rn in zip(residues, resnames)]).astype(np.float32)
    ca = np.stack([np.asarray(r["CA"].coord, np.float32) for r in residues])

    donor_xyz, donor_type, donor_res = [], [], []
    can_xyz, can_res = [], []
    heavy_xyz, heavy_res, heavy_polar = [], [], []
    charge_xyz, charge_q = [], []
    for ri, res in enumerate(residues):
        rn = resnames[ri]
        for atom_name, dtype in BROAD_DONORS.get(rn, {}).items():
            if atom_name in res:
                donor_xyz.append(np.asarray(res[atom_name].coord, np.float32))
                donor_type.append(dtype)
                donor_res.append(ri)
        for atom_name in CANONICAL_DONORS.get(rn, {}):
            if atom_name in res:
                can_xyz.append(np.asarray(res[atom_name].coord, np.float32))
                can_res.append(ri)
        for atom in res:
            elem = str(atom.element).upper()
            if elem == "H":
                continue
            heavy_xyz.append(np.asarray(atom.coord, np.float32))
            heavy_res.append(ri)
            heavy_polar.append(elem in {"N", "O", "S"})
        cc = _charge_center(res, rn)
        if cc is not None:
            charge_xyz.append(cc[0])
            charge_q.append(cc[1])
    donor_xyz = np.asarray(donor_xyz, np.float32)
    donor_type = np.asarray(donor_type, object)
    donor_res = np.asarray(donor_res, np.int32)
    can_res = np.asarray(can_res, np.int32)
    heavy_xyz, heavy_res = np.asarray(heavy_xyz, np.float32), np.asarray(heavy_res, np.int32)
    heavy_polar = np.asarray(heavy_polar, bool)
    charge_xyz, charge_q = np.asarray(charge_xyz, np.float32), np.asarray(charge_q, np.float32)
    donor_tree = cKDTree(donor_xyz) if len(donor_xyz) else None
    can_tree = cKDTree(np.asarray(can_xyz, np.float32)) if len(can_xyz) else None
    heavy_tree = cKDTree(heavy_xyz) if len(heavy_xyz) else None
    charge_tree = cKDTree(charge_xyz) if len(charge_xyz) else None
    ca_center = ca.mean(axis=0)

    for i in range(n):
        anchor = anchors[i]
        # ---- A. METAL-SPECIFIC COMPATIBILITY
        local_donors = [j for j in donor_tree.query_ball_point(anchor, r=6.0) if donor_res[j] != i] \
            if donor_tree is not None else []
        focal = [np.asarray(residues[i][a].coord, np.float32)
                 for a in CANONICAL_DONORS.get(resnames[i], {}) if a in residues[i]]
        best_center = None
        if focal and local_donors:
            centers = [cc for fc in focal for j in local_donors
                       for cc in [_sphere_pair_centers(fc, donor_xyz[j], r=2.10, samples=8)] if len(cc)]
            if centers:
                best_score = None
                for center in np.concatenate(centers, axis=0):
                    distances = np.linalg.norm(donor_xyz - center, axis=1)
                    near = np.where(np.abs(distances - 2.10) <= 0.50)[0]
                    if not np.any(donor_res[near] == i):
                        continue
                    rmse = float(np.sqrt(np.mean((distances[near] - 2.10) ** 2))) if len(near) else 9.0
                    score = (len(near), -rmse)
                    if best_score is None or score > best_score:
                        best_score, best_center = score, center
        if best_center is not None:
            distance = np.linalg.norm(donor_xyz - best_center, axis=1)
            for k, metal in enumerate(METALS):
                proto = METAL_PROTO[metal]
                near = np.where(np.abs(distance - proto["r0"]) <= 0.55)[0]
                if len(near):
                    rmse = float(np.sqrt(np.mean((distance[near] - proto["r0"]) ** 2)))
                    METALSPEC[i, 3 * k] = math.exp(-(rmse / 0.35) ** 2)
                    METALSPEC[i, 3 * k + 1] = math.exp(-abs(len(near) - proto["coord"]) / 2.0)
                    METALSPEC[i, 3 * k + 2] = float(np.mean([proto["weights"].get(str(donor_type[j]), 0.0) for j in near]))

        # ---- B. 3D ELECTROSTATIC FIELD
        for radius in (8.0, 12.0):
            if charge_tree is None:
                continue
            idx = charge_tree.query_ball_point(anchor, r=radius)
            if not idx:
                continue
            xyz, q = charge_xyz[idx], charge_q[idx]
            rvec = xyz - anchor
            dist = np.maximum(np.linalg.norm(rvec, axis=1), 1.5)
            field = np.sum(q[:, None] * (-rvec) / (dist[:, None] ** 3), axis=0)
            net = float(q.sum())
            if radius == 8.0:
                ELEC3D[i, 0] = net
                ELEC3D[i, 1] = float(np.linalg.norm(field))
                radial_vec = anchor - ca_center
                radial_norm = np.linalg.norm(radial_vec)
                if radial_norm > 1e-8:
                    radial_vec = radial_vec / radial_norm
                    ELEC3D[i, 2] = float(field @ radial_vec)
                centroid = np.average(xyz, axis=0, weights=np.abs(q))
                ELEC3D[i, 3] = float(np.linalg.norm(np.sum(q[:, None] * (xyz - centroid), axis=0)))
                pos, neg = xyz[q > 0], xyz[q < 0]
                if len(pos) and len(neg):
                    ELEC3D[i, 4] = float(np.linalg.norm(pos.mean(axis=0) - neg.mean(axis=0)))
                if len(xyz) >= 3:
                    u = xyz - anchor
                    ELEC3D[i, 5] = _principal_fraction(u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8))
            else:
                ELEC3D[i, 6] = net
                ELEC3D[i, 7] = float(np.linalg.norm(field))

        # ---- C. DONOR NETWORK TOPOLOGY
        idx = list(can_tree.query_ball_point(anchor, r=8.0)) if can_tree is not None else []
        if idx:
            unique_res = np.unique(can_res[idx])
            coords = np.asarray([anchors[r] for r in unique_res], np.float32)
            m = len(unique_res)
            TOPO[i, 0] = m
            TOPO[i, 1] = len({resnames[r] for r in unique_res})
            edges, pair_dist = [], []
            for a, b in combinations(range(m), 2):
                dd = float(np.linalg.norm(coords[a] - coords[b]))
                pair_dist.append(dd)
                if dd <= 5.0:
                    edges.append((a, b))
            TOPO[i, 2] = len(edges)
            possible = m * (m - 1) / 2
            TOPO[i, 3] = len(edges) / possible if possible else 0.0
            parent = list(range(m))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in edges:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            comps = {}
            for j in range(m):
                root = find(j)
                comps[root] = comps.get(root, 0) + 1
            TOPO[i, 4] = (max(comps.values()) if comps else 0) / max(m, 1)
            edge_set = {tuple(sorted(x)) for x in edges}
            TOPO[i, 5] = sum(1 for a, b, c in combinations(range(m), 3)
                             if (a, b) in edge_set and (a, c) in edge_set and (b, c) in edge_set)
            seq_sep = np.abs(unique_res - i)
            TOPO[i, 6] = int(np.sum(seq_sep >= 12))
            TOPO[i, 7] = float(seq_sep.mean())
            TOPO[i, 8] = float(seq_sep.max())
            center = coords.mean(axis=0)
            TOPO[i, 9] = float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))
            if pair_dist:
                TOPO[i, 10] = float(np.min(pair_dist))
                TOPO[i, 11] = float(np.mean(pair_dist))

        # ---- D. SOLVATION / POLAR ORGANIZATION
        idx = [j for j in heavy_tree.query_ball_point(anchor, r=6.0) if heavy_res[j] != i] if heavy_tree is not None else []
        if idx:
            xyz, polar = heavy_xyz[idx], heavy_polar[idx]
            d = np.linalg.norm(xyz - anchor, axis=1)
            for col, radius in enumerate((3.0, 4.0, 6.0)):
                SOLV[i, col] = int(np.sum(polar & (d <= radius)))
            mask4, mask6 = d <= 4.0, d <= 6.0
            SOLV[i, 3] = polar[mask4].mean() if mask4.any() else 0.0
            SOLV[i, 4] = polar[mask6].mean() if mask6.any() else 0.0
            SOLV[i, 5] = int(np.sum((~polar) & mask4))
            core_polar = np.sum(polar & (d <= 4.0))
            shell_polar = np.sum(polar & (d > 4.0) & (d <= 6.0))
            SOLV[i, 6] = shell_polar / (core_polar + 1.0)
            SOLV[i, 7] = 1.0 / (1.0 + np.sum(d <= 3.0))
            SOLV[i, 8] = core_polar / (np.sum(mask4) + 1.0)
            pxyz = xyz[polar & mask6]
            if len(pxyz) >= 3:
                u = pxyz - anchor
                SOLV[i, 9] = _principal_fraction(u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8))

        # ---- E. EXPLICIT SEQUENCE MOTIF
        def window(w, keep):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            return [sequence[j] for j in range(lo, hi) if j != i and keep(sequence[j])]

        for col, w in enumerate((2, 4, 8, 16)):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            chars = [sequence[j] for j in range(lo, hi) if j != i]
            SEQMOTIF[i, col] = sum(c in DONOR_AA for c in chars) / max(len(chars), 1)
        for col, w in zip((4, 5, 6), (4, 8, 16)):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            chars = [sequence[j] for j in range(lo, hi) if j != i]
            SEQMOTIF[i, col] = sum(c in {"D", "E"} for c in chars) / max(len(chars), 1)
        for col, w in ((7, 8), (8, 16)):
            SEQMOTIF[i, col] = len(set(window(w, lambda c: c in DONOR_AA)))
        SEQMOTIF[i, 9] = len([j for j, c in enumerate(sequence[:n]) if c in DONOR_AA and abs(j - i) >= 12]) / max(n, 1)
    return {"METALSPEC": METALSPEC, "ELEC3D": ELEC3D, "TOPO": TOPO, "SOLV": SOLV, "SEQMOTIF": SEQMOTIF}


# ------------------------------------------------------------------ SEQCTX
def make_seqctx(sequence: str) -> np.ndarray:
    """160: one-hot at offsets ±1..4; 20: ±8 composition (focal excluded); relative position; terminus proximity."""
    aa_i = {a: k for k, a in enumerate(AA20)}
    seq, n = str(sequence), len(sequence)
    out = np.zeros((n, 182), dtype=np.float32)
    for i in range(n):
        for oi, off in enumerate((-4, -3, -2, -1, 1, 2, 3, 4)):
            j = i + off
            if 0 <= j < n and seq[j] in aa_i:
                out[i, oi * 20 + aa_i[seq[j]]] = 1.0
        denom = 0
        for j in range(max(0, i - 8), min(n, i + 9)):
            if j != i and seq[j] in aa_i:
                out[i, 160 + aa_i[seq[j]]] += 1.0
                denom += 1
        if denom:
            out[i, 160:180] /= float(denom)
        out[i, 180] = 0.0 if n <= 1 else i / float(n - 1)
        out[i, 181] = 0.0 if n <= 1 else min(i, n - 1 - i) / max((n - 1) / 2.0, 1.0)
    return out


# ------------------------------------------------------------------ registered feature groups
@feature_group("MECH1", columns={b: FEATURE_NAMES[b] for b in MECH1_BLOCKS},
               legacy_cache={b: "{split}_mechanistic_v1.pkl" for b in MECH1_BLOCKS})
def mech1(p: Protein) -> dict:
    """Accessibility, second-shell support, chemical environment, pLDDT confidence (ACCESS/SHELL2/ENV/CONF)."""
    return extract_mechanistic_features(p.residues)


@feature_group("MULTI2", columns={b: FEATURE_NAMES[b] for b in MULTI2_BLOCKS},
               legacy_cache={b: "{split}_multisignal_v2.pkl" for b in MULTI2_BLOCKS})
def multi2(p: Protein) -> dict:
    """Metal-specific site fit, 3-D electrostatics, donor-network topology, solvation, sequence motifs."""
    return extract_new2_features(p.residues, p.sequence[:len(p.residues)])


@feature_group("SEQCTX", columns={"SEQCTX": FEATURE_NAMES["SEQCTX"]}, needs="sequence")
def seqctx(p: Protein) -> dict:
    """Local sequence context: ±4 one-hot, ±8 composition, relative position, terminus proximity (182)."""
    return {"SEQCTX": make_seqctx(p.sequence)}
