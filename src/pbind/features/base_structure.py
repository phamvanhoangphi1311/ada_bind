"""AA (20) / SPATIAL (112) / LOCAL (8) blocks (port of GEAR ``gear_structure``).

SPATIAL: per Cβ shell (0-4, 4-6, 6-8, 8-10 Å) AA counts + total + long-range counts, nearest distance
per AA type (cap 12 Å), neighbour distance / sequence-separation summaries.
LOCAL: DSSP H/E/C one-hot, sin/cos φ/ψ, relative SASA proxy (MDTraj).
Registered feature groups: AA (sequence only), STRUCTURE (SPATIAL + LOCAL).
Original backend cache (imported by the feature store when present):
    <BASE>/cache/spatial_revamp/<split>_spatial_local_v2.pkl   {prot_id: {"AA","SPATIAL","LOCAL", ...}}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..constants import AA20
from ..data.structures import _parser, safe_seq1
from .api import Protein, feature_group

AA_TO_IDX = {a: i for i, a in enumerate(AA20)}
SHELL_EDGES = [0.0, 4.0, 6.0, 8.0, 10.0]
SHELL_PAIRS = list(zip(SHELL_EDGES[:-1], SHELL_EDGES[1:]))
SPATIAL_MAX_RADIUS = SHELL_EDGES[-1]
LONG_RANGE_SEQ_SEP = 12
NEAREST_DISTANCE_CAP = 12.0
SPATIAL_DIM, LOCAL_DIM = 112, 8
# Column names (same as gear_structure).
FEATURE_NAMES = {
    "AA": [f"focal_AA_{a}" for a in AA20],
    "SPATIAL": [n for lo, hi in SHELL_PAIRS for n in
                [f"n_{a}_{lo:g}_{hi:g}A" for a in AA20] + [f"n_total_{lo:g}_{hi:g}A", f"n_longrange_{lo:g}_{hi:g}A"]]
               + [f"nearest_{a}_A" for a in AA20]
               + ["neighbor_mean_distance", "neighbor_std_distance", "neighbor_mean_seqsep", "neighbor_max_seqsep"],
    "LOCAL": ["SS_H", "SS_E", "SS_C", "sin_phi", "cos_phi", "sin_psi", "cos_psi", "rel_sasa_proxy"],
}
assert len(FEATURE_NAMES["SPATIAL"]) == SPATIAL_DIM
MAX_ASA = {"A": 1.29, "R": 2.74, "N": 1.95, "D": 1.93, "C": 1.67, "E": 2.23, "Q": 2.25, "G": 1.04, "H": 2.24,
           "I": 1.97, "L": 2.01, "K": 2.36, "M": 2.24, "F": 2.40, "P": 1.59, "S": 1.55, "T": 1.72, "W": 2.85,
           "Y": 2.63, "V": 1.74}


def parse_residues_and_coords(pdb, n_keep: int):
    model = next(_parser().get_structure(Path(pdb).stem, str(pdb)).get_models())
    aa, ca, cb = [], [], []
    for chain in model:
        for res in chain:
            if res.id[0] != " " or "CA" not in res:
                continue
            aa.append(safe_seq1(res.resname))
            ca_coord = np.asarray(res["CA"].coord, dtype=np.float32)
            ca.append(ca_coord)
            cb.append(np.asarray(res["CB"].coord, dtype=np.float32) if "CB" in res else ca_coord.copy())
            if len(aa) >= n_keep:
                return "".join(aa), np.asarray(ca, np.float32), np.asarray(cb, np.float32)
    return "".join(aa), np.asarray(ca, np.float32), np.asarray(cb, np.float32)


def focal_aa_onehot(seq: str) -> np.ndarray:
    x = np.zeros((len(seq), 20), dtype=np.float32)
    for i, a in enumerate(seq):
        if a in AA_TO_IDX:
            x[i, AA_TO_IDX[a]] = 1.0
    return x


def spatial_microenvironment(seq_struct: str, cb: np.ndarray) -> np.ndarray:
    n = len(seq_struct)
    out = np.zeros((n, SPATIAL_DIM), dtype=np.float32)
    if n == 0:
        return out
    tree = cKDTree(cb)
    shell_block = 22
    nearest_offset = len(SHELL_PAIRS) * shell_block
    summary_offset = nearest_offset + 20
    for i in range(n):
        nbrs = [j for j in tree.query_ball_point(cb[i], r=SPATIAL_MAX_RADIUS) if j != i]
        if not nbrs:
            out[i, nearest_offset:nearest_offset + 20] = NEAREST_DISTANCE_CAP
            continue
        idx = np.asarray(nbrs, dtype=np.int32)
        dists = np.linalg.norm(cb[idx] - cb[i], axis=1)
        seqseps = np.abs(idx - i)
        for sidx, (lo, hi) in enumerate(SHELL_PAIRS):
            last = sidx == len(SHELL_PAIRS) - 1
            mask = ((dists >= lo) & (dists <= hi)) if last else ((dists >= lo) & (dists < hi))
            base = sidx * shell_block
            for j in idx[mask]:
                a = seq_struct[j]
                if a in AA_TO_IDX:
                    out[i, base + AA_TO_IDX[a]] += 1.0
            out[i, base + 20] = float(np.sum(mask))
            out[i, base + 21] = float(np.sum(seqseps[mask] >= LONG_RANGE_SEQ_SEP))
        nearest = np.full(20, NEAREST_DISTANCE_CAP, dtype=np.float32)
        for j, d in zip(idx, dists):
            a = seq_struct[j]
            if a in AA_TO_IDX:
                k = AA_TO_IDX[a]
                if d < nearest[k]:
                    nearest[k] = min(float(d), NEAREST_DISTANCE_CAP)
        out[i, nearest_offset:nearest_offset + 20] = nearest
        out[i, summary_offset + 0] = float(np.mean(dists))
        out[i, summary_offset + 1] = float(np.std(dists))
        out[i, summary_offset + 2] = float(np.mean(seqseps))
        out[i, summary_offset + 3] = float(np.max(seqseps))
    return out


def local_structure_features(pdb, n_keep: int) -> np.ndarray:
    import mdtraj as md

    traj = md.load(str(pdb))
    protein_res = []
    for r in traj.topology.residues:
        has_ca = any(a.name == "CA" for a in r.atoms)
        if getattr(r, "is_protein", has_ca) and has_ca:
            protein_res.append(r)
    n = min(n_keep, len(protein_res))
    if n == 0:
        raise ValueError(f"No protein residues found by MDTraj in {pdb}")
    protein_res = protein_res[:n]
    row_of = {r.index: i for i, r in enumerate(protein_res)}
    out = np.zeros((n, LOCAL_DIM), dtype=np.float32)
    dssp = md.compute_dssp(traj, simplified=True)[0]
    ss_map = {"H": 0, "E": 1, "C": 2, "NA": 2}
    for row, r in enumerate(protein_res):
        s = str(dssp[r.index]) if r.index < len(dssp) else "C"
        out[row, ss_map.get(s, 2)] = 1.0
    phi_idx, phi_val = md.compute_phi(traj)
    psi_idx, psi_val = md.compute_psi(traj)
    if phi_val.size:
        for k, q in enumerate(phi_idx):
            ri = traj.topology.atom(int(q[1])).residue.index
            if ri in row_of:
                ang = float(phi_val[0, k])
                out[row_of[ri], 3], out[row_of[ri], 4] = np.sin(ang), np.cos(ang)
    if psi_val.size:
        for k, q in enumerate(psi_idx):
            ri = traj.topology.atom(int(q[0])).residue.index
            if ri in row_of:
                ang = float(psi_val[0, k])
                out[row_of[ri], 5], out[row_of[ri], 6] = np.sin(ang), np.cos(ang)
    sasa_all = md.shrake_rupley(traj, mode="residue")[0]
    md_seq = "".join(safe_seq1(r.name) for r in protein_res)
    sasa = np.asarray([sasa_all[r.index] for r in protein_res], dtype=np.float32)
    denom = np.asarray([MAX_ASA.get(a, 2.0) for a in md_seq], dtype=np.float32)
    out[:, 7] = np.clip(sasa / np.maximum(denom, 1e-6), 0.0, 2.0)
    return out


def structure_record(pdb, csv_seq: str) -> dict:
    """AA/SPATIAL/LOCAL of one protein (``csv_seq`` already truncated to max_residues)."""
    seq_struct, _, cb = parse_residues_and_coords(pdb, len(csv_seq))
    if not seq_struct:
        raise ValueError(f"{pdb}: no residues parsed")
    nc = min(len(csv_seq), len(seq_struct))
    ident = float(np.mean([a == b for a, b in zip(csv_seq[:nc], seq_struct[:nc])])) if nc else 0.0
    n = min(len(csv_seq), len(seq_struct))
    aa = focal_aa_onehot(csv_seq[:n])
    spatial = spatial_microenvironment(seq_struct[:n], cb[:n])
    local = local_structure_features(pdb, n)
    n2 = min(n, len(local))
    return {"AA": aa[:n2].astype(np.float32), "SPATIAL": spatial[:n2].astype(np.float32),
            "LOCAL": local[:n2].astype(np.float32), "n": int(n2), "sequence_identity_to_csv": ident,
            "csv_length": len(csv_seq), "pdb_length": len(seq_struct), "pdb_path": str(pdb)}


# ------------------------------------------------------------------ registered feature groups
@feature_group("AA", columns={"AA": FEATURE_NAMES["AA"]}, needs="sequence")
def aa(p: Protein) -> dict:
    """Focal amino-acid one-hot (20)."""
    return {"AA": focal_aa_onehot(p.sequence)}


@feature_group("STRUCTURE", columns={b: FEATURE_NAMES[b] for b in ("SPATIAL", "LOCAL")},
               legacy_cache={b: "{split}_spatial_local_v2.pkl" for b in ("SPATIAL", "LOCAL")})
def structure(p: Protein) -> dict:
    """Cβ-shell micro-environment (SPATIAL 112) + DSSP / torsion / SASA (LOCAL 8, MDTraj)."""
    rec = structure_record(p.pdb, p.sequence)
    if rec["sequence_identity_to_csv"] < 0.95:  # same refusal threshold as the original backend
        raise ValueError(f"{p.pid}: PDB/CSV identity {rec['sequence_identity_to_csv']:.3f} < 0.95")
    return {"SPATIAL": rec["SPATIAL"], "LOCAL": rec["LOCAL"]}
