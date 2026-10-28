"""ATOM (15) and CHEM (24) metal-coordination blocks (port of GEAR ``gear_core``).

ATOM: geometry around canonical donor atoms (Cys SG, His ND1/NE2, Asp OD1/OD2, Glu OE1/OE2):
      donor counts/distances, best candidate metal centre (2.10 Å two-sphere circles), clearance, angles.
CHEM: donor-type counts at 4/6 Å (carboxylate O, imidazole N, thiol S, hydroxyl O, amide O, thioether S,
      backbone O) + charged-side-chain environment and Coulomb-like potentials + O/N/S donor fractions.
Registered feature group: ATOM_CHEM. Original backend caches (imported by the feature store when present):
    <BASE>/cache/spatial_revamp/<split>_atomic_coordination_v1.pkl, <split>_chemistry_v1_24d.pkl
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .api import Protein, feature_group

DONOR_SEARCH_RADIUS = 5.0
IDEAL_METAL_DONOR_DISTANCE = 2.10
CENTER_TOL, CENTER_TOL_LOOSE = 0.35, 0.50
CIRCLE_SAMPLES = 12
CLEARANCE_THRESHOLD = 1.20
CHEM_INNER, CHEM_OUTER, CHEM_CHARGE_OUTER = 4.0, 6.0, 8.0

ATOM_FEATURE_NAMES = [
    "focal_has_donor", "n_focal_donor_atoms", "n_neighbor_donor_atoms_5A", "n_neighbor_donor_residues_5A",
    "min_focal_neighbor_donor_dist_A", "mean_focal_neighbor_donor_dist_A", "n_focal_neighbor_pairs_le4p2A",
    "best_coord_n_tol035", "best_coord_n_tol050", "best_coord_rmse_A", "best_center_clearance_A",
    "has_empty_center_ge2", "has_empty_center_ge3", "best_min_donor_angle_deg", "best_mean_donor_angle_deg",
]
ATOM_DONOR_ATOMS = {"CYS": ["SG"], "HIS": ["ND1", "NE2"], "ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]}

CHEM_DONOR_CATEGORIES = {
    "carboxylate_O": {"ASP": ["OD1", "OD2"], "GLU": ["OE1", "OE2"]},
    "imidazole_N": {"HIS": ["ND1", "NE2"]},
    "thiol_S": {"CYS": ["SG"]},
    "hydroxyl_O": {"SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"]},
    "amide_O": {"ASN": ["OD1"], "GLN": ["OE1"]},
    "thioether_S": {"MET": ["SD"]},
    "backbone_O": "BACKBONE_O",
}
CHEM_FEATURE_NAMES = [f"n_{cat}_{r}A" for r in (4, 6) for cat in CHEM_DONOR_CATEGORIES] + [
    "n_negative_residues_6A", "n_positive_residues_6A", "n_histidine_residues_6A",
    "signed_charge_potential_6A", "signed_charge_potential_8A",
    "negative_charge_potential_6A", "positive_charge_potential_6A",
    "O_donor_fraction_6A", "N_donor_fraction_6A", "S_donor_fraction_6A",
]
assert len(ATOM_FEATURE_NAMES) == 15 and len(CHEM_FEATURE_NAMES) == 24


# ------------------------------------------------------------------ ATOM
def sphere_pair_centers(a, b, r: float = IDEAL_METAL_DONOR_DISTANCE, samples: int = CIRCLE_SAMPLES):
    """Points at distance r from both a and b (circle), sampled; empty if impossible."""
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
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    mid = (a + b) / 2
    h = np.sqrt(max(r * r - (d / 2) ** 2, 0))
    if h < 1e-8:
        return mid.reshape(1, 3).astype(np.float32)
    ts = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    return np.asarray([mid + h * (np.cos(t) * e1 + np.sin(t) * e2) for t in ts], np.float32)


def angle_stats(center, coords):
    if len(coords) < 2:
        return 0.0, 0.0
    V = coords - center
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    ang = [np.degrees(np.arccos(np.clip(V[i] @ V[j], -1, 1))) for i, j in combinations(range(len(V)), 2)]
    return (float(np.min(ang)), float(np.mean(ang))) if ang else (0.0, 0.0)


def atomic_geometry(residues) -> np.ndarray:
    n = len(residues)
    out = np.zeros((n, len(ATOM_FEATURE_NAMES)), np.float32)
    donors, heavy, heavy_is_donor = [], [], []
    for ri, res in enumerate(residues):
        dn = set(ATOM_DONOR_ATOMS.get(str(res.resname).upper(), []))
        for atom in res:
            if str(atom.element).upper() == "H":
                continue
            xyz = np.asarray(atom.coord, np.float32)
            isdon = atom.name in dn
            heavy.append(xyz)
            heavy_is_donor.append(isdon)
            if isdon:
                donors.append((ri, atom.name, xyz))
    if not donors:
        out[:, 4:6] = 8.0
        out[:, 9] = 2.0
        return out
    dcoord = np.stack([x[2] for x in donors])
    dres = np.asarray([x[0] for x in donors], np.int32)
    dtree = cKDTree(dcoord)
    heavy = np.asarray(heavy, np.float32)
    ndheavy = heavy[~np.asarray(heavy_is_donor, bool)]
    htree = cKDTree(ndheavy) if len(ndheavy) else None
    r0 = IDEAL_METAL_DONOR_DISTANCE

    for i in range(n):
        fi = np.where(dres == i)[0]
        if len(fi) == 0:
            out[i, 4:6] = 8.0
            out[i, 9] = 2.0
            continue
        out[i, 0] = 1.0
        out[i, 1] = len(fi)
        nbr, fd = set(), []
        for fc in dcoord[fi]:
            for j in dtree.query_ball_point(fc, r=DONOR_SEARCH_RADIUS):
                if dres[j] == i:
                    continue
                nbr.add(int(j))
                fd.append(float(np.linalg.norm(dcoord[j] - fc)))
        nbr = sorted(nbr)
        out[i, 2] = len(nbr)
        out[i, 3] = len(set(int(dres[j]) for j in nbr))
        if fd:
            out[i, 4] = np.min(fd)
            out[i, 5] = np.mean(fd)
            out[i, 6] = np.sum(np.asarray(fd) <= 4.2)
        else:
            out[i, 4:6] = 8.0

        centers = []
        for aidx in fi:
            for bidx in nbr:
                c = sphere_pair_centers(dcoord[aidx], dcoord[bidx])
                if len(c):
                    centers.append(c)
        if not centers:
            out[i, 9] = 2.0
            continue
        best = None
        for c in np.concatenate(centers):
            dist = np.linalg.norm(dcoord - c, axis=1)
            q35 = np.where(np.abs(dist - r0) <= CENTER_TOL)[0]
            q50 = np.where(np.abs(dist - r0) <= CENTER_TOL_LOOSE)[0]
            if not np.any(dres[q50] == i):
                continue
            rmse = float(np.sqrt(np.mean((dist[q50] - r0) ** 2))) if len(q50) else 2.0
            clearance = float(htree.query(c, k=1)[0]) if htree is not None else 9.0
            mina, meana = angle_stats(c, dcoord[q50])
            rec = (len(q35), len(q50), clearance, -rmse, rmse, mina, meana)
            if best is None or rec[:4] > best[:4]:
                best = rec
        if best is None:
            out[i, 9] = 2.0
            continue
        n35, n50, clearance, _, rmse, mina, meana = best
        out[i, 7:11] = [n35, n50, rmse, clearance]
        empty = clearance >= CLEARANCE_THRESHOLD
        out[i, 11] = float(empty and n35 >= 2)
        out[i, 12] = float(empty and n35 >= 3)
        out[i, 13] = mina
        out[i, 14] = meana
    return out


# ------------------------------------------------------------------ CHEM
def chem_reference_point(res) -> np.ndarray:
    """Side-chain chemistry centre if available, otherwise Cβ / Cα."""
    resname = str(res.resname).upper()
    atoms = []
    for spec in CHEM_DONOR_CATEGORIES.values():
        if spec == "BACKBONE_O":
            continue
        for name in spec.get(resname, []):
            if name in res:
                atoms.append(np.asarray(res[name].coord, np.float32))
    if atoms:
        return np.mean(np.stack(atoms), axis=0).astype(np.float32)
    if "CB" in res:
        return np.asarray(res["CB"].coord, np.float32)
    return np.asarray(res["CA"].coord, np.float32)


def chemistry_features(residues) -> np.ndarray:
    n = len(residues)
    out = np.zeros((n, len(CHEM_FEATURE_NAMES)), np.float32)

    donor_records = []  # (residue index, category, element, xyz)
    for ri, res in enumerate(residues):
        rn = str(res.resname).upper()
        for cat, spec in CHEM_DONOR_CATEGORIES.items():
            if spec == "BACKBONE_O":
                if "O" in res:
                    donor_records.append((ri, cat, "O", np.asarray(res["O"].coord, np.float32)))
                continue
            for atom_name in spec.get(rn, []):
                if atom_name in res:
                    elem = str(res[atom_name].element).upper() or atom_name[0]
                    donor_records.append((ri, cat, elem, np.asarray(res[atom_name].coord, np.float32)))
    if donor_records:
        donor_xyz = np.stack([z[3] for z in donor_records]).astype(np.float32)
        donor_res = np.asarray([z[0] for z in donor_records], np.int32)
        donor_cat = np.asarray([z[1] for z in donor_records], dtype=object)
        donor_elem = np.asarray([z[2] for z in donor_records], dtype=object)
        donor_tree = cKDTree(donor_xyz)
    else:
        donor_res = np.empty((0,), np.int32)
        donor_cat = donor_elem = np.empty((0,), dtype=object)
        donor_tree = None

    # formal-charge centres at dominant near-neutral-pH states
    charge_records, histidine_centers = [], []
    for ri, res in enumerate(residues):
        rn = str(res.resname).upper()
        if rn == "ASP" and "OD1" in res and "OD2" in res:
            xyz = (np.asarray(res["OD1"].coord) + np.asarray(res["OD2"].coord)) / 2
            charge_records.append((ri, -1.0, "negative", xyz.astype(np.float32)))
        elif rn == "GLU" and "OE1" in res and "OE2" in res:
            xyz = (np.asarray(res["OE1"].coord) + np.asarray(res["OE2"].coord)) / 2
            charge_records.append((ri, -1.0, "negative", xyz.astype(np.float32)))
        elif rn == "LYS" and "NZ" in res:
            charge_records.append((ri, +1.0, "positive", np.asarray(res["NZ"].coord, np.float32)))
        elif rn == "ARG":
            names = [a for a in ("NE", "NH1", "NH2") if a in res]
            if names:
                xyz = np.mean(np.stack([np.asarray(res[a].coord, np.float32) for a in names]), axis=0)
                charge_records.append((ri, +1.0, "positive", xyz.astype(np.float32)))
        if rn == "HIS":
            names = [a for a in ("ND1", "NE2") if a in res]
            if names:
                histidine_centers.append(
                    (ri, np.mean(np.stack([np.asarray(res[a].coord, np.float32) for a in names]), axis=0)))
    if charge_records:
        charge_xyz = np.stack([z[3] for z in charge_records]).astype(np.float32)
        charge_res = np.asarray([z[0] for z in charge_records], np.int32)
        charge_q = np.asarray([z[1] for z in charge_records], np.float32)
        charge_cls = np.asarray([z[2] for z in charge_records], dtype=object)
        charge_tree = cKDTree(charge_xyz)
    else:
        charge_tree = None
    if histidine_centers:
        his_res = np.asarray([z[0] for z in histidine_centers], np.int32)
        his_tree = cKDTree(np.stack([z[1] for z in histidine_centers]).astype(np.float32))
    else:
        his_tree = None
    cat_names = list(CHEM_DONOR_CATEGORIES)
    empty_idx = np.empty((0,), dtype=np.int64)

    def potentials(indices, ref):
        if len(indices) == 0:
            return 0.0, 0.0, 0.0
        d = np.maximum(np.linalg.norm(charge_xyz[indices] - ref, axis=1), 1.0)
        q = charge_q[indices]
        return float(np.sum(q / d)), float(np.sum((q < 0) / d)), float(np.sum((q > 0) / d))

    for i, res in enumerate(residues):
        ref = chem_reference_point(res)
        col = 0
        selected_outer = empty_idx
        for radius in (CHEM_INNER, CHEM_OUTER):  # donor-type counts, focal residue excluded
            idx = empty_idx if donor_tree is None else np.asarray(donor_tree.query_ball_point(ref, r=radius), np.int64)
            if len(idx):
                idx = idx[donor_res[idx] != i]
            if math.isclose(radius, CHEM_OUTER):
                selected_outer = idx
            for cat in cat_names:
                out[i, col] = float(np.sum(donor_cat[idx] == cat)) if len(idx) else 0.0
                col += 1

        if charge_tree is not None:  # charged side-chain environment
            idx6 = np.asarray(charge_tree.query_ball_point(ref, r=CHEM_OUTER), np.int64)
            idx8 = np.asarray(charge_tree.query_ball_point(ref, r=CHEM_CHARGE_OUTER), np.int64)
            if len(idx6):
                idx6 = idx6[charge_res[idx6] != i]
            if len(idx8):
                idx8 = idx8[charge_res[idx8] != i]
            nneg = float(np.sum(charge_cls[idx6] == "negative")) if len(idx6) else 0.0
            npos = float(np.sum(charge_cls[idx6] == "positive")) if len(idx6) else 0.0
            signed6, negpot6, pospot6 = potentials(idx6, ref)
            signed8, _, _ = potentials(idx8, ref)
        else:
            nneg = npos = signed6 = negpot6 = pospot6 = signed8 = 0.0
        if his_tree is not None:
            ih = np.asarray(his_tree.query_ball_point(ref, r=CHEM_OUTER), np.int64)
            if len(ih):
                ih = ih[his_res[ih] != i]
            nhis = float(len(ih))
        else:
            nhis = 0.0
        out[i, col:col + 7] = [nneg, npos, nhis, signed6, signed8, negpot6, pospot6]
        col += 7

        if len(selected_outer):  # element fractions among donor atoms within 6 Å
            elems = donor_elem[selected_outer]
            denom = max(len(elems), 1)
            out[i, col:col + 3] = [float(np.sum(elems == e) / denom) for e in ("O", "N", "S")]
        col += 3
    return out


# ------------------------------------------------------------------ registered feature group
@feature_group("ATOM_CHEM", columns={"ATOM": ATOM_FEATURE_NAMES, "CHEM": CHEM_FEATURE_NAMES},
               legacy_cache={"ATOM": "{split}_atomic_coordination_v1.pkl", "CHEM": "{split}_chemistry_v1_24d.pkl"})
def atom_chem(p: Protein) -> dict:
    """Metal-coordination geometry around donor atoms (ATOM 15) + donor chemistry / charge field (CHEM 24)."""
    return {"ATOM": atomic_geometry(p.residues), "CHEM": chemistry_features(p.residues)}
