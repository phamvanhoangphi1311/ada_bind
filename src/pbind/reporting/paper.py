"""Paper-ready final report: Dev-locked V1/MS ensemble, threshold locks, full external metrics.

Port of the "FINAL PAPER-READY REPORT — ONE CELL" used by both nuclear and small.
External probabilities are cached in ``<out>/prediction_cache`` (same file names as before).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..constants import EXTERNAL_SPLITS, MIN_DEV_GAIN
from ..data.loading import concat_labels, n_residues
from ..evaluation.ensemble import WEIGHT_GRID, crossfit_audit, weight_search
from ..evaluation.metrics import full_global_metrics, protein_mean_metrics
from ..evaluation.thresholds import optimize_thresholds_exact
from ..utils.cache import PredictionCache, save_json, save_table
from .baselines import baseline_full_rows
from .predictions import residue_prediction_frame

HEADLINE_COLS = [
    "dataset", "method", "threshold", "AUPRC_global_residue", "AUROC_global_residue", "Accuracy_global_residue",
    "Precision_positive_global_residue", "Recall_positive_global_residue", "F1_positive_global_residue",
    "MCC_global_residue", "F1_micro_all_classes", "F1_macro_two_classes",
    "protein_mean_F1_baseline_convention", "protein_mean_MCC_baseline_convention",
]
FINAL_COLS = [
    "dataset", "method", "threshold_rule", "threshold", "AUPRC_global_residue", "AUROC_global_residue",
    "Accuracy_global_residue", "Balanced_accuracy_global_residue", "Precision_positive_global_residue",
    "Recall_positive_global_residue", "Specificity_global_residue", "F1_positive_global_residue",
    "MCC_global_residue", "F1_micro_all_classes", "F1_macro_two_classes",
    "protein_mean_F1_baseline_convention", "protein_mean_MCC_baseline_convention",
    "protein_n_included_baseline_convention",
]


def paper_ready_report(*, task_name: str, label: str, splits: dict, dev_split: str, fold_idx: dict,
                       v1_oof: np.ndarray, ms_oof: np.ndarray, v1_dev_score: float,
                       predict_v1: Callable[[str], np.ndarray], predict_ms: Callable[[str], np.ndarray],
                       baselines: pd.DataFrame, out_dir: Path, v1_name: str, ms_name: str,
                       final_name: str = "FINAL_DEV_LOCKED_ENSEMBLE", min_gain: float = MIN_DEV_GAIN,
                       force_predictions: bool = False, tags: dict | None = None) -> dict:
    """``tags`` = {"V1_Attention": fingerprint, "MS_Attention": fingerprint} invalidates cached
    external predictions when fold models change."""
    tags = tags or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_df = splits[dev_split]
    y_dev = concat_labels(dev_df, label)
    v1_oof = np.asarray(v1_oof, dtype=np.float64)
    ms_oof = np.asarray(ms_oof, dtype=np.float64)
    assert v1_oof.shape == ms_oof.shape == y_dev.shape and np.all(np.isfinite(v1_oof)) and np.all(np.isfinite(ms_oof))

    # D. Dev-only ensemble weight lock
    ws = weight_search(y_dev, v1_oof, ms_oof, fold_idx, ("V1", "MS"), WEIGHT_GRID)
    best = ws.iloc[0]
    cand_w, cand_dev = float(best.w_V1), float(best.CV_AUPRC_mean)
    promote = cand_dev - v1_dev_score >= min_gain
    w_v1, w_ms = (cand_w, 1.0 - cand_w) if promote else (1.0, 0.0)
    final_dev = cand_dev if promote else v1_dev_score
    final_oof = w_v1 * v1_oof + w_ms * ms_oof
    save_table(out_dir / "DEV_ENSEMBLE_WEIGHT_SEARCH.csv", ws)

    # E. cross-fitted audit
    crossfit = crossfit_audit(y_dev, v1_oof, ms_oof, fold_idx, ("V1", "MS"), WEIGHT_GRID)
    save_table(out_dir / "DEV_CROSSFIT_ENSEMBLE_AUDIT.csv", crossfit)

    # F. thresholds per method, Dev OOF only
    dev_probs = {"V1_Attention": v1_oof, "MS_Attention": ms_oof, final_name: final_oof}
    thresholds, thr_rows, dev_rows = {}, [], []
    for method, p in dev_probs.items():
        thresholds[method] = optimize_thresholds_exact(y_dev, p)
        for rule, thr in thresholds[method].items():
            sel = "fixed" if rule == "fixed_0.5" else f"{dev_split} OOF only"
            thr_rows.append({"method": method, "threshold_rule": rule, "threshold": float(thr),
                             "threshold_selected_on": sel})
            dev_rows.append({"dataset": f"{dev_split}_OOF", "method": method, "threshold_rule": rule,
                             "threshold": float(thr), **full_global_metrics(y_dev, p, thr),
                             **protein_mean_metrics(dev_df, p, thr, label)})
    threshold_df, dev_metrics = pd.DataFrame(thr_rows), pd.DataFrame(dev_rows)
    save_table(out_dir / "DEV_THRESHOLD_LOCK.csv", threshold_df)
    save_table(out_dir / "DEV_OOF_FULL_METRICS.csv", dev_metrics)

    # G/H. external predictions (cached) + full metrics at every locked threshold
    cache = PredictionCache(out_dir / "prediction_cache")
    ext_rows, pred_files = [], {}
    for split in EXTERNAL_SPLITS:
        df = splits[split]
        y = concat_labels(df, label)
        n = n_residues(df)
        p_v1 = cache.get_or_compute("V1_Attention", split, n, lambda s=split: predict_v1(s), force_predictions,
                                    tags.get("V1_Attention"))
        p_ms = cache.get_or_compute("MS_Attention", split, n, lambda s=split: predict_ms(s), force_predictions,
                                    tags.get("MS_Attention"))
        p_final = w_v1 * p_v1 + w_ms * p_ms
        for method, p in {"V1_Attention": p_v1, "MS_Attention": p_ms, final_name: p_final}.items():
            for rule, thr in thresholds[method].items():
                ext_rows.append({
                    "dataset": split, "method": method, "threshold_rule": rule, "threshold": float(thr),
                    "threshold_selected_on": "fixed" if rule == "fixed_0.5" else f"{dev_split} OOF only",
                    "directly_comparable_to_baseline_fixed0.5": rule == "fixed_0.5",
                    **full_global_metrics(y, p, thr), **protein_mean_metrics(df, p, thr, label),
                })
        fp = out_dir / f"FINAL_RESIDUE_PREDICTIONS_{split}.csv.gz"
        save_table(fp, residue_prediction_frame(df, label, prob_V1=p_v1, prob_MS=p_ms, prob_FINAL=p_final))
        pred_files[split] = str(fp)
    external = pd.DataFrame(ext_rows)

    # I/J/K. baselines + tables
    all_metrics = pd.concat([external, baseline_full_rows(baselines)], ignore_index=True, sort=False)
    save_table(out_dir / "EXTERNAL_FULL_METRICS_LONG.csv", all_metrics)
    headline = (all_metrics[all_metrics.threshold_rule.eq("fixed_0.5")]
                .reindex(columns=HEADLINE_COLS)
                .sort_values(["dataset", "AUPRC_global_residue"], ascending=[True, False]))
    save_table(out_dir / "HEADLINE_FIXED05_COMPARISON.csv", headline)
    final_report = external[external.method.eq(final_name)].reindex(columns=FINAL_COLS)
    save_table(out_dir / "FINAL_MODEL_ALL_THRESHOLDS.csv", final_report)

    decision = {
        "task": task_name, "final_method": final_name, "V1_name": v1_name, "MS_name": ms_name,
        "candidate_w_V1": cand_w, "candidate_w_MS": 1.0 - cand_w, "final_w_V1": w_v1, "final_w_MS": w_ms,
        "V1_Dev_mean_AUPRC": v1_dev_score, "candidate_ensemble_Dev_mean_AUPRC": cand_dev,
        "final_Dev_mean_AUPRC": final_dev, "candidate_Dev_gain_vs_V1": cand_dev - v1_dev_score,
        f"promoted_by_{min_gain}_rule": bool(promote),
        "crossfit_mean_AUPRC_ensemble": float(crossfit.held_fold_AUPRC_crossfit_ensemble.mean()),
        "crossfit_mean_AUPRC_V1": float(crossfit.held_fold_AUPRC_V1.mean()),
        "metric_naming": {
            "F1_positive_global_residue": "binary positive-class F1 after pooling all residues",
            "MCC_global_residue": "MCC from one pooled residue-level confusion matrix",
            "F1_micro_all_classes": "formal micro-F1 across both classes; equals accuracy here",
            "protein_mean_metrics": "per-protein metrics averaged after excluding TN-only proteins",
        },
        "threshold_rules": {m: {k: float(v) for k, v in r.items()} for m, r in thresholds.items()},
        "prediction_files": pred_files,
    }
    save_json(out_dir / "FINAL_DECISION_AND_DEFINITIONS.json", decision)
    return {"decision": decision, "weight_search": ws, "crossfit": crossfit, "thresholds": threshold_df,
            "dev_metrics": dev_metrics, "external": external, "headline": headline, "final": final_report,
            "final_oof": final_oof, "weights": (w_v1, w_ms)}
