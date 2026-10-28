"""Blocks derived from other blocks or from other tasks' predictions (no structure parsing)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backend.gear import get_block, set_block


def evo_access_residual(evo: np.ndarray, access: np.ndarray) -> np.ndarray:
    """EVO48 with the (quadratic) ACCESS-explainable part regressed out, per protein, z-scored."""
    E = np.asarray(evo, dtype=np.float64)
    A = np.asarray(access, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    X = np.concatenate([np.ones((len(A), 1)), A, A * A], axis=1)
    if len(E) > X.shape[1] + 2:
        beta = np.linalg.lstsq(X, E, rcond=None)[0]
        R = E - X @ beta
    else:
        R = E - E.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True)
    return (R / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def attach_evo_access_residual(store, df: pd.DataFrame, block: str = "EVO_ACCESS_RES_CLEAN") -> None:
    for r in df.itertuples():
        set_block(store, r.prot_id, block,
                  evo_access_residual(get_block(store, r.prot_id, "EVO"), get_block(store, r.prot_id, "ACCESS")))


def attach_crosstask_block(store, df: pd.DataFrame, xdf: pd.DataFrame, columns, block: str) -> None:
    """Attach another task's per-residue OOF probabilities as a feature block (Dev only).

    ``xdf`` must have prot_id, residue_index and ``columns``; every protein must be complete.
    """
    xdf = xdf.copy()
    xdf["prot_id"] = xdf["prot_id"].astype(str)
    xdf["residue_index"] = xdf["residue_index"].astype(int)
    groups = {pid: g.sort_values("residue_index") for pid, g in xdf.groupby("prot_id", sort=False)}
    for r in df.itertuples():
        pid, L = str(r.prot_id), len(r.sequence)
        sub = groups[pid]
        if len(sub) != L or not np.array_equal(sub.residue_index.to_numpy(), np.arange(L)):
            raise RuntimeError(f"{pid}: cross-task rows are not exactly residues 0..{L - 1}")
        arr = sub[list(columns)].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{pid}: non-finite cross-task probability")
        set_block(store, pid, block, arr)
