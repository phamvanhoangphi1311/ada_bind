"""Dev-only selection rules (representation lock, complexity-aware locks, verdicts)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..constants import MIN_DEV_GAIN


def rank_vs_reference(cv: pd.DataFrame, reference: str, score: str = "CV_AUPRC",
                      key: str = "representation", dims: dict | None = None) -> pd.DataFrame:
    """Sort by score and add ``delta_vs_<reference>`` (+ feature_dim if block dims are given)."""
    out = cv.copy()
    ref = float(out.loc[out[key].eq(reference), score].iloc[0])
    out[f"delta_vs_{reference}"] = out[score] - ref
    if dims is not None:
        out["feature_dim"] = out[key].map(dims)
    return out.sort_values(score, ascending=False).reset_index(drop=True)


def best_mean(cv: pd.DataFrame, score: str = "CV_AUPRC", key: str = "representation") -> str:
    return str(cv.sort_values(score, ascending=False).iloc[0][key])


def one_se_rule(folds: pd.DataFrame, feature_dims: dict, score: str = "AUPRC",
                key: str = "representation") -> tuple[str, pd.DataFrame]:
    """Lowest-dimensional representation within one SE of the best fold-mean."""
    q = folds.groupby(key)[score].agg(["mean", "std", "count"]).reset_index()
    q["se"] = q["std"] / np.sqrt(q["count"].clip(lower=1))
    q["feature_dim"] = q[key].map(feature_dims)
    best = q.sort_values("mean", ascending=False).iloc[0]
    eligible = q[q["mean"] >= float(best["mean"] - best["se"])].sort_values(["feature_dim", "mean"],
                                                                              ascending=[True, False])
    return str(eligible.iloc[0][key]), q.sort_values("mean", ascending=False)


def lock_if_gain(best_name: str, best_score: float, ref_name: str, ref_score: float,
                 margin: float = MIN_DEV_GAIN) -> str:
    """Keep the simpler reference unless the best beats it by >= margin."""
    return best_name if best_score - ref_score >= margin else ref_name


def complexity_lock(candidates: Sequence[tuple[str, float]], margin: float = MIN_DEV_GAIN) -> tuple[str, float]:
    """``candidates`` ordered simplest -> most complex; a more complex one wins only if it beats the
    current winner by >= margin."""
    winner = candidates[0]
    for name, score in candidates[1:]:
        if score - winner[1] >= margin:
            winner = (name, score)
    return winner


def verdict(gain: float, min_gain: float = MIN_DEV_GAIN) -> str:
    if gain >= 0.01:
        return "STRONG"
    if gain >= 0.005:
        return "CLEAR"
    if gain >= min_gain:
        return "MODEST_BUT_REAL"
    if gain > 0:
        return "WEAK"
    return "NO_GAIN"


def block_dim(blocks, dims: dict) -> int:
    return int(sum(dims[b] for b in blocks))
