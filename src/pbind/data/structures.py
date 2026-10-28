"""Exact-sequence, chain-A-normalized PDB cache shared by nuclear + small (+ coordinate loaders).

Cache layers (fastest first):
  1. local scratch  ``<local>/shared_exact_pdb_v1/<split>/<pid>.pdb``
  2. Drive archive  ``<BASE>/cache/shared_exact_pdb_v1.tar.gz`` (restored to 1 when 1 is missing)
  3. older normalized copies (nuclear/small/metal caches) — copied as-is if exact
  4. raw sources (ESMFold, experimental, bindNode24 AlphaFold) — normalized on demand
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import AA3, BACKBONE_ATOMS, SPLIT_ALIASES
from ..paths import Paths

ARCHIVE_NAME = "shared_exact_pdb_v1"


def _parser():
    from Bio.PDB import PDBParser

    return PDBParser(QUIET=True)


# ------------------------------------------------------------------ residue helpers
def standard_residues(chain) -> list:
    return [r for r in chain if r.id[0] == " " and r.resname.strip().upper() in AA3]


def residue_sequence(rr) -> str:
    from Bio.SeqUtils import seq1

    return "".join(seq1(r.resname.strip().upper()) for r in rr)


def representative_coord(res) -> np.ndarray:
    """Side-chain heavy-atom centroid; CA for glycine-like residues; else all-atom mean."""
    side = [np.asarray(a.coord, dtype=np.float32) for a in res
            if a.element != "H" and a.name not in BACKBONE_ATOMS]
    if side:
        return np.mean(np.stack(side), axis=0)
    if "CA" in res:
        return np.asarray(res["CA"].coord, dtype=np.float32)
    atoms = [np.asarray(a.coord, dtype=np.float32) for a in res]
    if not atoms:
        raise RuntimeError("Residue has no atoms")
    return np.mean(np.stack(atoms), axis=0)


def pdb_exact_ok(path: str | Path, target_seq: str) -> bool:
    """True if ``path`` is a chain-A PDB whose standard residues spell ``target_seq``."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        model = next(_parser().get_structure("check", str(path)).get_models())
        if "A" not in model:
            return False
        return residue_sequence(standard_residues(model["A"])) == target_seq
    except Exception:
        return False


def normalize_exact_chain_A(src: Path, target_seq: str, dst: Path) -> None:
    """Copy the chain that exactly matches ``target_seq`` into a fresh chain A numbered 1..L."""
    from Bio.PDB import PDBIO
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    model = next(_parser().get_structure("src", str(src)).get_models())
    exact, diag = [], []
    for ch in model:
        rr = standard_residues(ch)
        seq = residue_sequence(rr)
        diag.append((str(ch.id), len(seq)))
        if seq == target_seq:
            exact.append((ch, rr))
    if not exact:
        raise RuntimeError(f"{src}: no exact chain match; target={len(target_seq)}, chains={diag}")
    exact.sort(key=lambda x: (0 if str(x[0].id) == "A" else 1, str(x[0].id)))
    _, rr = exact[0]

    out_st, out_model, out_chain = Structure("normalized"), Model(0), Chain("A")
    out_st.add(out_model)
    out_model.add(out_chain)
    for i, res in enumerate(rr, start=1):
        # Residue.copy() writes byte-identical PDBs to the notebooks' deepcopy(res), without deep-copying the
        # whole parent structure for every residue (that made normalization O(L^2), ~30 s per protein)
        r2 = res.copy()
        r2.detach_parent()
        r2.id = (" ", i, " ")
        out_chain.add(r2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(out_st)
    tmp = dst.with_name(f".{dst.name}.tmp")
    io.save(str(tmp))
    os.replace(tmp, dst)


# ------------------------------------------------------------------ source lookup
def _old_normalized_roots(paths: Paths) -> list[Path]:
    return [
        paths.cache / "nuclear_ultimate_v1" / "normalized_pdb",
        paths.cache / "small_ultimate_v1" / "normalized_pdb",
        paths.baseline_work / "cache" / "metal_external_normalized_structures",
    ]


def _normalized_candidates(paths: Paths, split: str, pid: str) -> list[Path]:
    out = []
    for root in _old_normalized_roots(paths):
        out.append(root / f"{pid}.pdb")
        for alias in SPLIT_ALIASES.get(split, ()):
            out.append(root / alias / f"{pid}.pdb")
    return out


def _raw_roots(paths: Paths, bn24: Path | None) -> list[Path]:
    roots = [paths.esmfold_pdb, paths.experimental_pdb]
    if bn24 is not None:
        roots.append(bn24 / "dataset" / "AlphaFold_predictions")
    return roots


class _RawIndex:
    """Built at most once, only if a direct raw-path lookup misses (avoids rglob over Drive)."""

    def __init__(self, roots):
        self.roots, self._idx = roots, None

    def get(self, pid: str) -> Path | None:
        if self._idx is None:
            idx = {}
            for root in self.roots:
                if not root.exists():
                    continue
                print(" scanning:", root)
                for dirpath, _, files in os.walk(root):
                    for fn in files:
                        if fn.endswith(".pdb"):
                            idx.setdefault(Path(fn).stem, Path(dirpath) / fn)
            print("fallback PDBs indexed:", len(idx))
            self._idx = idx
        return self._idx.get(pid)


# ------------------------------------------------------------------ cache build
def local_pdb_root(paths: Paths) -> Path:
    return paths.local / ARCHIVE_NAME


def drive_archive(paths: Paths) -> Path:
    return paths.cache / f"{ARCHIVE_NAME}.tar.gz"


def _extract(tar_fp: Path, dest: Path) -> None:
    with tarfile.open(tar_fp, "r:gz") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(dest, filter="data")
        else:
            tf.extractall(dest)


def restore_archive(paths: Paths) -> bool:
    """Drive tar.gz -> local scratch (copy locally first: much faster than streaming from Drive)."""
    root, archive = local_pdb_root(paths), drive_archive(paths)
    if root.exists() or not archive.exists():
        return False
    print("Restoring shared PDB archive from Drive ...")
    local_tar = paths.local / archive.name
    shutil.copy2(archive, local_tar)
    _extract(local_tar, paths.local)
    local_tar.unlink(missing_ok=True)
    return True


def pack_archive(paths: Paths) -> Path:
    root, archive = local_pdb_root(paths), drive_archive(paths)
    local_tar = paths.local / archive.name
    with tarfile.open(local_tar, "w:gz") as tf:
        tf.add(root, arcname=ARCHIVE_NAME)
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(f".{archive.name}.tmp")
    shutil.copy2(local_tar, tmp)
    os.replace(tmp, archive)
    local_tar.unlink(missing_ok=True)
    return archive


def ensure_normalized_pdbs(paths: Paths, splits: dict[str, pd.DataFrame], bn24: Path | None = None,
                           repack: str = "if_changed") -> dict:
    """Make every protein of every split available as an exact chain-A PDB in local scratch.

    ``repack``: "if_changed" (default), "always" or "never" — upload the archive to Drive.
    """
    restore_archive(paths)
    root = local_pdb_root(paths)
    raw_index = _RawIndex(_raw_roots(paths, bn24))
    stats = {"already_local": 0, "reused_old_normalized": 0, "normalized_raw": 0}
    missing, done = [], 0
    total = sum(len(df) for df in splits.values())

    for split, df in splits.items():
        outdir = root / split
        outdir.mkdir(parents=True, exist_ok=True)
        for r in df.itertuples():
            pid, dst = str(r.prot_id), outdir / f"{r.prot_id}.pdb"
            done += 1
            if pdb_exact_ok(dst, r.sequence):
                stats["already_local"] += 1
                continue
            reused = False
            for cand in _normalized_candidates(paths, split, pid):
                if cand.exists() and pdb_exact_ok(cand, r.sequence):
                    shutil.copy2(cand, dst)
                    stats["reused_old_normalized"] += 1
                    reused = True
                    break
            if reused:
                continue
            src = next((p for p in (rt / f"{pid}.pdb" for rt in _raw_roots(paths, bn24))
                        if p.exists() and p.stat().st_size > 0), None)
            if src is None:
                src = raw_index.get(pid)
            if src is None:
                missing.append((split, pid))
                continue
            normalize_exact_chain_A(src, r.sequence, dst)
            if not pdb_exact_ok(dst, r.sequence):
                raise RuntimeError(f"{split}/{pid}: normalized PDB failed validation")
            stats["normalized_raw"] += 1
            if done % 100 == 0:
                print(f"  {done}/{total}", stats)

    if missing:
        raise FileNotFoundError(f"Missing source PDB for {len(missing)} proteins: {missing[:20]}")
    changed = stats["reused_old_normalized"] + stats["normalized_raw"] > 0
    if repack == "always" or (repack == "if_changed" and (changed or not drive_archive(paths).exists())):
        print("Packing shared PDB cache -> Drive ...")
        pack_archive(paths)
    print("Exact-sequence shared structures: PASS", stats)
    return stats


# ------------------------------------------------------------------ loaders
def pdb_path(paths: Paths, split: str, pid: str) -> Path:
    return local_pdb_root(paths) / split / f"{pid}.pdb"


def load_chain_residues(pdb: str | Path, seq: str) -> list:
    model = next(_parser().get_structure("p", str(pdb)).get_models())
    rr = standard_residues(model["A"])
    if residue_sequence(rr) != seq:
        raise RuntimeError(f"{pdb}: normalized PDB / sequence mismatch")
    return rr


# ------------------------------------------------------------------ GEAR-style raw-PDB access
# (used by the ESM/ATOM/CHEM/MECH1/MULTI2 ports: all chains, residues with CA, no renumbering)
def safe_seq1(resname) -> str:
    from Bio.SeqUtils import seq1

    from ..constants import AA20

    try:
        x = seq1(str(resname))
        return x if x in AA20 else "X"
    except Exception:
        return "X"


def parse_protein_residues(pdb: str | Path, n_keep: int) -> list:
    model = next(_parser().get_structure(Path(pdb).stem, str(pdb)).get_models())
    residues = []
    for chain in model:
        for res in chain:
            if res.id[0] != " " or "CA" not in res:
                continue
            residues.append(res)
            if len(residues) >= n_keep:
                return residues
    return residues


def parsed_sequence(residues) -> str:
    return "".join(safe_seq1(r.resname) for r in residues)


def validate_feature_alignment(pid: str, csv_seq: str, residues, min_identity: float = 0.95):
    """(n aligned residues, identity); refuses silent misalignment below ``min_identity``."""
    pseq = parsed_sequence(residues)
    n = min(len(csv_seq), len(pseq))
    if n == 0:
        raise ValueError(f"{pid}: no aligned PDB residues")
    identity = sum(a == b for a, b in zip(csv_seq[:n], pseq[:n])) / n
    if identity < min_identity:
        raise ValueError(f"{pid}: PDB/CSV sequence identity {identity:.3f} < {min_identity:.2f}")
    return n, identity


def resolve_predicted_pdb(paths: Paths, pid: str, split: str | None = None) -> Path:
    """ESMFold PDB of a protein (same search order as the GEAR backend).

    Falls back to the exact chain-A copy from ``cache/shared_exact_pdb_v1.tar.gz`` when the raw
    file is missing or empty (some Drive uploads are 0-byte files).
    """
    pid = str(pid)
    candidates = [paths.esmfold_pdb / f"{pid}.pdb", paths.base / "data" / "pdb_files" / f"{pid}.pdb",
                  paths.base / "pdb" / "all" / f"{pid}.pdb"]
    if split:
        candidates += [paths.base / "pdb" / split / f"{pid}.pdb", paths.base / "data" / "pdb_files" / split / f"{pid}.pdb"]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    restore_archive(paths)
    for s in ([split] if split else []) + [s for s in SPLIT_ALIASES if s != split]:
        p = pdb_path(paths, s, pid)
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"No (non-empty) predicted PDB for {pid}; checked {candidates} and the shared archive")


@lru_cache(maxsize=4096)
def _coords_cached(pdb: str, seq: str) -> np.ndarray:
    rr = load_chain_residues(pdb, seq)
    out = np.stack([representative_coord(r) for r in rr], axis=0).astype(np.float32)
    out.setflags(write=False)
    return out


def structure_coords(pdb: str | Path, seq: str) -> np.ndarray:
    """(L, 3) representative coordinates; memoized in RAM per (pdb, sequence)."""
    return _coords_cached(str(pdb), seq)
