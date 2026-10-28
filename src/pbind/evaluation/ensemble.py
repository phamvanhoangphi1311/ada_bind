"""Dev-only two-model probability ensembles."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ap, fold_auprc

WEIGHT_GRID = np.linspace(0.0, 1.0, 21)


def weight_search(y, p1, p2, fold_idx: dict, names=("V1", "MS"), grid=WEIGHT_GRID) -> pd.DataFrame:
    """w*p1 + (1-w)*p2 scored by mean official-fold AUPRC; ties prefer more weight on p1."""
    a, b = names
    rows = []
    for w in grid:
        p = w * p1 + (1.0 - w) * p2
        aps = fold_auprc(y, p, fold_idx)
        rows.append({f"w_{a}": float(w), f"w_{b}": float(1.0 - w), "CV_AUPRC_mean": float(np.mean(aps)),
                     "CV_AUPRC_std": float(np.std(aps, ddof=1)), "OOF_AUPRC": ap(y, p),
                     **{f"fold{i + 1}_AUPRC": float(v) for i, v in enumerate(aps)}})
    return (pd.DataFrame(rows).sort_values(["CV_AUPRC_mean", f"w_{a}"], ascending=[False, False])
            .reset_index(drop=True))


def crossfit_audit(y, p1, p2, fold_idx: dict, names=("V1", "MS"), grid=WEIGHT_GRID) -> pd.DataFrame:
    """Selection-bias check: weight for each held fold is chosen on the other four only."""
    a, b = names
    rows = []
    folds = sorted(fold_idx)
    for held in folds:
        others = [f for f in folds if f != held]
        cand = []
        for w in grid:
            p = w * p1 + (1.0 - w) * p2
            cand.append((float(np.mean([ap(y[fold_idx[f]], p[fold_idx[f]]) for f in others])), float(w)))
        cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
        w = cand[0][1]
        idx = fold_idx[held]
        rows.append({
            "held_fold": held,
            f"selected_w_{a}_from_other_4_folds": w,
            f"selected_w_{b}_from_other_4_folds": 1.0 - w,
            "held_fold_AUPRC_crossfit_ensemble": ap(y[idx], w * p1[idx] + (1 - w) * p2[idx]),
            f"held_fold_AUPRC_{a}": ap(y[idx], p1[idx]),
            f"held_fold_AUPRC_{b}": ap(y[idx], p2[idx]),
        })
    return pd.DataFrame(rows)
