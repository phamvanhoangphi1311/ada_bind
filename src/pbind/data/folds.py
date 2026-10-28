"""Official bindNode24 protein-level 5-fold split of DevSet1010."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import Paths
from .loading import residue_offsets

BN24_URL = "https://github.com/erckert/bindNode24.git"
FOLD_SUBDIR = Path("dataset") / "cv_splits" / "predicted_structures"
# The Drive copy of ids_split1..5.txt has 1014 IDs: the official bindNode24 split (1010; 202/203/203/202/200,
# same order) plus these 4, whose ESMFold structure does not reproduce the CSV sequence. They are not in DevSet1010.
EXCLUDED_DEV_IDS = frozenset({"C8BD48", "P84801", "Q00277", "Q9NZV6"})


def find_bindnode24(paths: Paths, clone: bool = True) -> Path | None:
    """bindNode24 repo (Drive copy, sibling roots, or a local clone). None if absent and clone=False."""
    roots = [paths.base, *paths.siblings()]
    candidates = []
    for root in roots:
        candidates += [
            root / "baseline_reproduction_master_v6" / "repos" / "bindnode24",
            root / "repos" / "bindnode24",
        ]
    candidates.append(paths.local / "bindnode24")
    for c in candidates:
        if (c / FOLD_SUBDIR / "ids_split1.txt").exists():
            return c
    if not clone:
        return None
    local = paths.local / "bindnode24"
    subprocess.run(["git", "clone", "-q", BN24_URL, str(local)], check=True)
    return local


def find_fold_dir(paths: Paths, clone: bool = True) -> Path:
    """Folder holding ids_split1..5.txt: ``data/development_set/`` first, else the bindNode24 repo."""
    local = paths.base / "data" / "development_set"
    if all((local / f"ids_split{k}.txt").exists() for k in range(1, 6)):
        return local
    repo = find_bindnode24(paths, clone=clone)
    if repo is None:
        raise FileNotFoundError("ids_split1..5.txt not found (data/development_set/ or bindNode24 repo)")
    return repo / FOLD_SUBDIR


def load_official_folds(fold_dir: Path) -> dict[int, list[str]]:
    """``fold_dir`` = folder with ids_split1..5.txt (or a bindNode24 repo root)."""
    fold_dir = Path(fold_dir)
    if not (fold_dir / "ids_split1.txt").exists():
        fold_dir = fold_dir / FOLD_SUBDIR
    folds, seen = {}, set()
    for k in range(1, 6):
        fp = fold_dir / f"ids_split{k}.txt"
        ids = [x.strip() for x in fp.read_text().splitlines() if x.strip() and x.strip() not in EXCLUDED_DEV_IDS]
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError(f"Fold overlap at fold {k}: {sorted(overlap)[:10]}")
        folds[k] = ids
        seen.update(ids)
    return folds


def official_order(folds: dict[int, list[str]]) -> list[str]:
    return [pid for k in sorted(folds) for pid in folds[k]]


def fold_frames(dev: pd.DataFrame, folds: dict[int, list[str]], fold: int):
    """(train_df, val_df) for one fold, preserving DevSet1010 order (as the notebooks did)."""
    val = set(folds[fold])
    return dev[~dev.prot_id.isin(val)].copy(), dev[dev.prot_id.isin(val)].copy()


def fold_residue_indices(dev: pd.DataFrame, folds: dict[int, list[str]]) -> dict[int, np.ndarray]:
    """Residue indices (into the concatenated Dev vector) of each official fold."""
    offsets, _ = residue_offsets(dev)
    out = {}
    for fold, ids in folds.items():
        idx = []
        for pid in ids:
            a, b = offsets[str(pid)]
            idx.extend(range(a, b))
        out[fold] = np.asarray(idx, dtype=np.int64)
    return out


def train_val_residue_idx(pids: np.ndarray, folds: dict[int, list[str]], fold: int):
    """Residue-level train/val indices from a per-residue prot_id vector."""
    val = set(folds[fold])
    mask = np.array([p in val for p in pids])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def scatter_fold_predictions(dev: pd.DataFrame, val_df: pd.DataFrame, pred: np.ndarray,
                             oof: np.ndarray) -> None:
    """Write a fold's concatenated validation prediction into the Dev OOF vector."""
    offsets, _ = residue_offsets(dev)
    pos = 0
    for r in val_df.itertuples():
        L = len(r.sequence)
        a, b = offsets[str(r.prot_id)]
        oof[a:b] = pred[pos:pos + L]
        pos += L
    if pos != len(pred):
        raise RuntimeError("fold prediction length mismatch")
