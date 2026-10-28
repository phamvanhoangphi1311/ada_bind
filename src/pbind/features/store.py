"""Feature store: every residue-level block is computed ONCE and kept as one column file per block.

    <BASE>/feature_store/
        manifest.json            blocks -> {dim, dtype, group, columns}; splits -> {n_proteins, n_residues, built}
        <split>/index.csv        prot_id, start, end, sequence    (rows [start, end) of each protein)
        <split>/labels.npz       metal_binding / nuclear_binding / small_binding / any_ligand_binding_sites (int8)
        <split>/<BLOCK>.npy      (n_residues, dim); dtype as registered (ESM float16, everything else float32)

Using it = picking columns:
    fs = FeatureStore(paths.feature_store)
    X = fs.matrix("TestSet300", ["ESM", "ATOM", "CHEM"])               # (n_residues, 1319) float32
    df = fs.frame("TestSet300", ["ATOM"])                             # pandas, named columns + prot_id
    store = fs.protein_store("DevSet1010", "metal_binding", blocks)     # {pid: {"sequence", "y", block: view}}

Filling it: ``build_feature_store(paths, frames, groups)`` calls the functions registered in
``pbind.features.api.FEATURE_GROUPS`` (``@feature_group``) for every split whose block files are missing.
Proteins already present in an original GEAR backend cache (``legacy_cache``) are imported instead of recomputed.
Every block must have exactly one row per residue of the (max 1000-residue) sequence; a shorter block aborts
the export instead of silently misaligning rows.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import ALL_LABELS
from ..data.structures import resolve_predicted_pdb
from ..utils.cache import load_json, load_pickle_dict, save_json, save_npy, save_npz, save_table
from . import base_structure, coordination, esm, mechanistic  # noqa: F401  (register the feature groups)
from .api import FEATURE_GROUPS, BuildOptions, FeatureGroup, Protein, all_blocks, group_of

STORE_VERSION = 1


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root)

    # ---------------------------------------------------------------- metadata
    @property
    def manifest(self) -> dict:
        return load_json(self.root / "manifest.json", None) or {"version": STORE_VERSION, "blocks": {}, "splits": {}}

    def split_dir(self, split: str) -> Path:
        return self.root / split

    def index(self, split: str) -> pd.DataFrame:
        return pd.read_csv(self.split_dir(split) / "index.csv", dtype={"prot_id": str})

    def n_residues(self, split: str) -> int:
        return int(self.manifest["splits"][split]["n_residues"])

    def has(self, split: str, block: str) -> bool:
        fp = self.split_dir(split) / f"{block}.npy"
        meta = self.manifest["splits"].get(split)
        if meta is None or not fp.exists():
            return False
        try:
            arr = np.load(fp, mmap_mode="r")
        except (OSError, ValueError):
            return False
        g = group_of(block)
        return arr.shape == (meta["n_residues"], g.dim(block)) and arr.dtype == g.dtype(block)

    def blocks(self, split: str) -> list[str]:
        return [b for b in all_blocks() if self.has(split, b)]

    def columns(self, blocks) -> list[str]:
        return [f"{b}:{c}" for b in blocks for c in group_of(b).columns[b]]

    # ---------------------------------------------------------------- reading
    def block(self, split: str, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.split_dir(split) / f"{name}.npy", mmap_mode="r" if mmap else None)

    def labels(self, split: str, label: str) -> np.ndarray:
        with np.load(self.split_dir(split) / "labels.npz") as z:
            return z[label]

    def matrix(self, split: str, blocks, rows=None) -> np.ndarray:
        """Residue matrix of the chosen blocks (all rows, or ``rows`` = index array / slice), float32."""
        rows = slice(None) if rows is None else rows
        return np.concatenate([np.asarray(self.block(split, b)[rows], np.float32) for b in blocks], axis=1)

    def frame(self, split: str, blocks, labels=()) -> pd.DataFrame:
        """Residue table: prot_id, residue_index, the chosen blocks' named columns and optional label columns."""
        idx = self.index(split)
        lens = (idx.end - idx.start).to_numpy()
        out = pd.DataFrame({"prot_id": np.repeat(idx.prot_id.to_numpy(), lens),
                            "residue_index": np.concatenate([np.arange(n) for n in lens]) if len(lens) else []})
        out = pd.concat([out, pd.DataFrame(self.matrix(split, blocks), columns=self.columns(blocks))], axis=1)
        for lab in labels:
            out[lab] = self.labels(split, lab)
        return out

    def protein_store(self, split: str, label: str, blocks=None, order=None) -> dict:
        """{prot_id: {"sequence", "y", block: (L, d) memory-mapped view}} in ``order`` (default: index order)."""
        idx = self.index(split).set_index("prot_id")
        blocks = list(blocks or self.blocks(split))
        missing = [b for b in blocks if not self.has(split, b)]
        if missing:
            raise KeyError(f"{split}: feature store lacks {missing}; run build_feature_store first")
        arrs = {b: self.block(split, b) for b in blocks}
        y = self.labels(split, label)
        out = {}
        for pid in (order if order is not None else idx.index):
            r = idx.loc[str(pid)]
            a, b = int(r.start), int(r.end)
            out[str(pid)] = {"sequence": str(r.sequence), "y": y[a:b], **{k: v[a:b] for k, v in arrs.items()}}
        return out


# -------------------------------------------------------------------- computing one group
def _load_legacy_file(fp: Path):
    """A legacy cache file: per-split pickle -> dict; per-protein .npz -> {block: arr}; .npy -> arr; else None."""
    if not fp.exists():
        return None
    try:
        if fp.suffix == ".npz":
            with np.load(fp) as z:
                return {k: z[k] for k in z.files}
        if fp.suffix == ".npy":
            return np.load(fp)
        return load_pickle_dict(fp)
    except Exception:  # unreadable / truncated legacy file -> recompute
        return None


def _legacy_blocks(group: FeatureGroup, legacy_dir: Path, split: str, lengths: dict) -> dict:
    """{pid: {block: arr}} for proteins whose every block is in the original cache with shape (L, d)."""
    found, split_files = {}, {}
    for pid, L in lengths.items():
        blocks, protein_files = {}, {}
        for b, pattern in group.legacy_cache.items():
            fp = Path(legacy_dir) / pattern.format(split=split, pid=pid)
            if "{pid}" in pattern:  # one file per protein: {block: arr} (.npz) or arr (.npy)
                if fp not in protein_files:
                    protein_files[fp] = _load_legacy_file(fp)
                v = protein_files[fp]
            else:  # one pickle per split: {pid: arr or {block: arr}}
                if fp not in split_files:
                    split_files[fp] = _load_legacy_file(fp) or {}
                v = split_files[fp].get(pid)
            arr = v.get(b) if isinstance(v, dict) else v
            if arr is not None and np.shape(arr) == (L, group.dim(b)):
                blocks[b] = np.asarray(arr)
        if set(blocks) == set(group.blocks):
            found[pid] = blocks
    return found


def compute_group(group: FeatureGroup, df: pd.DataFrame, split: str, paths, opts: BuildOptions,
                  legacy_dir: Path | None = None, require_legacy: bool = False) -> dict:
    """Run one registered feature function over a split -> {prot_id: {block: (L, d)}}.
    ``require_legacy``: every protein must come from the original cache (exact reproduction); raise otherwise."""
    from tqdm.auto import tqdm

    seqs = {str(r.prot_id): str(r.sequence)[:opts.max_residues] for r in df.itertuples()}
    data = _legacy_blocks(group, legacy_dir, split, {p: len(s) for p, s in seqs.items()}) \
        if legacy_dir is not None and group.legacy_cache else {}
    todo = [p for p in seqs if p not in data]
    print(f"[{split}] {group.name}: {len(data)} imported from backend cache, {len(todo)} to compute")
    if todo and require_legacy and group.legacy_cache:
        raise FileNotFoundError(f"{group.name}/{split}: {len(todo)} proteins missing from the original cache under "
                                f"{legacy_dir} (e.g. {todo[:5]}); copy the original cache files there first")
    if not todo:
        return data
    proteins = [Protein(p, seqs[p], split, pdb) for p, pdb in zip(todo, _structures(group, df, split, paths, todo))]
    if group.needs == "batch":
        data.update(group.fn(proteins, opts))
        return data
    for prot in tqdm(proteins, desc=f"{group.name} {split}"):
        try:
            data[prot.pid] = group.fn(prot)
        except Exception as e:
            raise RuntimeError(f"feature group {group.name} failed on {split}/{prot.pid}: {e}") from e
    return data


def _structures(group: FeatureGroup, df: pd.DataFrame, split: str, paths, pids: list) -> list:
    """The PDB each protein is computed from: ESMFold file, or its exact-sequence chain-A copy (nuclear/small)."""
    if group.needs != "structure":
        return [None] * len(pids)
    if group.structure == "esmfold":
        return [resolve_predicted_pdb(paths, p, split) for p in pids]
    from ..data.structures import ensure_normalized_pdbs, pdb_path

    ensure_normalized_pdbs(paths, {split: df[df.prot_id.astype(str).isin(set(pids))]}, None, repack="never")
    return [pdb_path(paths, split, p) for p in pids]


# -------------------------------------------------------------------- writing
def _frame_index(df: pd.DataFrame, max_residues: int) -> pd.DataFrame:
    seqs = [str(s)[:max_residues] for s in df.sequence]
    lens = np.asarray([len(s) for s in seqs], dtype=np.int64)
    ends = np.cumsum(lens)
    return pd.DataFrame({"prot_id": df.prot_id.astype(str).to_numpy(), "start": ends - lens, "end": ends,
                         "sequence": seqs})


def _ensure_index(fs: FeatureStore, split: str, df: pd.DataFrame, max_residues: int, manifest: dict) -> pd.DataFrame:
    """Write index + labels for a split; a split whose proteins/sequences changed is wiped and rebuilt."""
    idx = _frame_index(df, max_residues)
    d = fs.split_dir(split)
    if (d / "index.csv").exists():
        old = fs.index(split)
        if old[["prot_id", "sequence"]].equals(idx[["prot_id", "sequence"]]) and (d / "labels.npz").exists():
            return idx
        print(f"[{split}] feature store index changed -> rebuilding this split")
        shutil.rmtree(d)
        manifest["splits"].pop(split, None)
    labels = {c: np.concatenate([np.asarray(v[:max_residues], np.int8) for v in df[c]])
              for c in ALL_LABELS if c in df.columns}
    save_table(d / "index.csv", idx)
    save_npz(d / "labels.npz", **labels)
    manifest["splits"][split] = {"n_proteins": len(idx), "n_residues": int(idx.end.iloc[-1]) if len(idx) else 0,
                                 "max_residues": max_residues, "labels": sorted(labels), "built": {}}
    save_json(fs.root / "manifest.json", manifest)
    return idx


def _write_group(fs: FeatureStore, split: str, group: FeatureGroup, data: dict, idx: pd.DataFrame, manifest: dict,
                 sec: float):
    n = int(idx.end.iloc[-1]) if len(idx) else 0
    for b in group.blocks:
        dim = group.dim(b)
        arr = np.empty((n, dim), group.dtype(b))
        for r in idx.itertuples():
            x = np.asarray(data[r.prot_id][b])
            if x.shape != (r.end - r.start, dim):
                raise ValueError(f"{split}/{r.prot_id}/{b}: shape {x.shape} != ({r.end - r.start}, {dim}) "
                                 "(block must cover every residue of the sequence)")
            arr[r.start:r.end] = x
        save_npy(fs.split_dir(split) / f"{b}.npy", arr)
        manifest["blocks"][b] = {"dim": dim, "dtype": group.dtype(b).name, "group": group.name, "needs": group.needs,
                                 "columns": list(group.columns[b])}
        manifest["splits"][split]["built"][b] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(sec, 1)}
    save_json(fs.root / "manifest.json", manifest)


def build_feature_store(paths, frames: dict, groups=None, max_residues: int = 1000, device=None,
                        esm_batch_size: int = 8, force: bool = False, legacy: bool = True,
                        require_legacy: bool = False) -> FeatureStore:
    """frames = {split: DataFrame(prot_id, sequence, label columns)} in the row order to store.
    groups = names in ``FEATURE_GROUPS`` (default: every registered group).
    force  = rewrite the groups' block files even if present.
    legacy = import proteins found in the original caches (``cache/spatial_revamp/*.pkl``, per-protein NA_/SM_ caches,
             e.g. copied from Drive) instead of recomputing them; ``force=True, legacy=True`` rebuilds the store from them.
    require_legacy = groups that have an original cache must be fully imported from it (exact-reproduction runs)."""
    groups = list(groups or FEATURE_GROUPS)
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise KeyError(f"unknown feature groups {unknown}; registered: {list(FEATURE_GROUPS)}")
    fs = FeatureStore(paths.feature_store)
    opts = BuildOptions(device=device, max_residues=max_residues, esm_batch_size=esm_batch_size)
    manifest = fs.manifest
    for split, df in frames.items():
        idx = _ensure_index(fs, split, df, max_residues, manifest)
        for name in groups:
            g = FEATURE_GROUPS[name]
            if not force and all(fs.has(split, b) for b in g.blocks):
                print(f"[{split}] feature store: reuse {name} {list(g.blocks)}")
                continue
            t0 = time.time()
            data = compute_group(g, df, split, paths, opts, legacy_dir=paths.cache / g.legacy_subdir if legacy else None,
                                 require_legacy=require_legacy)
            _write_group(fs, split, g, data, idx, manifest, time.time() - t0)
            print(f"[{split}] feature store: wrote {name} {list(g.blocks)} ({time.time() - t0:.0f}s)")
    return fs
