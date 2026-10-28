"""Decision thresholds — always chosen on Dev OOF predictions only, then frozen."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef


def best_mcc_threshold(y, p, grid=None) -> tuple[float, float]:
    """Grid search (0.01..0.99, 199 points) as in the V1 notebooks. Returns (threshold, mcc)."""
    grid = np.linspace(0.01, 0.99, 199) if grid is None else grid
    best = (-999.0, None)
    for t in grid:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best[0]:
            best = (m, float(t))
    return best[1], best[0]


def optimize_thresholds_exact(y, p) -> dict:
    """Exact search over unique OOF cutoffs for pred = (p >= t): MCC-, F1- and Youden-J-optimal."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp_c, fp_c = np.cumsum(ys == 1), np.cumsum(ys == 0)
    ends = np.r_[np.where(ps[:-1] != ps[1:])[0], len(ps) - 1]  # include ties at each cutoff
    tp, fp = tp_c[ends].astype(np.float64), fp_c[ends].astype(np.float64)
    P, N = float(np.sum(y == 1)), float(np.sum(y == 0))
    fn, tn, thr = P - tp, N - fp, ps[ends]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(tp), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    tpr = np.divide(tp, P, out=np.zeros_like(tp), where=P > 0)
    tnr = np.divide(tn, N, out=np.zeros_like(tn), where=N > 0)
    return {
        "fixed_0.5": 0.5,
        "dev_oof_mcc_opt": float(thr[int(np.nanargmax(mcc))]),
        "dev_oof_positive_f1_opt": float(thr[int(np.nanargmax(f1))]),
        "dev_oof_youden_j_opt": float(thr[int(np.nanargmax(tpr + tnr - 1.0))]),
    }
