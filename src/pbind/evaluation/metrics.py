"""Residue-level metrics (global pooled, and protein-mean with the baseline convention)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ap(y, p) -> float:
    return float(average_precision_score(y, p))


def binary_metrics(y, p, threshold: float = 0.5) -> dict:
    """Compact set used during model development."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "AUPRC": ap(y, p),
        "AUROC": float(roc_auc_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }


def confusion_counts(y, pred):
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    return (int(np.sum((y == 1) & (pred == 1))), int(np.sum((y == 0) & (pred == 0))),
            int(np.sum((y == 0) & (pred == 1))), int(np.sum((y == 1) & (pred == 0))))


def mcc_from_counts(tp, tn, fp, fn) -> float:
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


def _div(a, b) -> float:
    return 0.0 if b == 0 else float(a / b)


def full_global_metrics(y, p, threshold: float) -> dict:
    """Paper-ready pooled residue-level metrics (one confusion matrix over all residues)."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int8)
    tp, tn, fp, fn = confusion_counts(y, pred)
    recall = _div(tp, tp + fn)
    return {
        "n_residues": int(len(y)), "n_positive": int(np.sum(y == 1)), "prevalence": float(np.mean(y)),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "AUPRC_global_residue": ap(y, p),
        "AUROC_global_residue": float(roc_auc_score(y, p)),
        "Accuracy_global_residue": float(accuracy_score(y, pred)),
        "Balanced_accuracy_global_residue": float(balanced_accuracy_score(y, pred)),
        "Precision_positive_global_residue": _div(tp, tp + fp),
        "Recall_positive_global_residue": recall,
        "Sensitivity_global_residue": recall,
        "Specificity_global_residue": _div(tn, tn + fp),
        "NPV_global_residue": _div(tn, tn + fn),
        "F1_positive_global_residue": _div(2 * tp, 2 * tp + fp + fn),
        "MCC_global_residue": mcc_from_counts(tp, tn, fp, fn),
        # formal micro-F1 over both classes == accuracy for binary single-label
        "F1_micro_all_classes": float(f1_score(y, pred, average="micro", zero_division=0)),
        "F1_macro_two_classes": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def protein_mean_metrics(df: pd.DataFrame, p, threshold: float, label: str) -> dict:
    """Per-protein F1/MCC averaged over proteins with TP+FP+FN > 0 (TN-only proteins excluded),
    the convention of the reproduced bindEmbed21DL / bindNode24 baselines."""
    p = np.asarray(p, dtype=np.float64)
    pos, f1s, mccs, n_total, n_excl = 0, [], [], 0, 0
    for r in df.itertuples():
        L = len(r.sequence)
        y = np.asarray(getattr(r, label), dtype=np.int8)
        tp, tn, fp, fn = confusion_counts(y, (p[pos:pos + L] >= float(threshold)).astype(np.int8))
        pos += L
        n_total += 1
        if tp + fp + fn == 0:
            n_excl += 1
            continue
        f1s.append(_div(2 * tp, 2 * tp + fp + fn))
        mccs.append(mcc_from_counts(tp, tn, fp, fn))
    assert pos == len(p)
    return {
        "protein_mean_F1_baseline_convention": float(np.mean(f1s)) if f1s else np.nan,
        "protein_mean_MCC_baseline_convention": float(np.mean(mccs)) if mccs else np.nan,
        "protein_n_total": n_total,
        "protein_n_included_baseline_convention": n_total - n_excl,
        "protein_n_excluded_TN_only": n_excl,
    }


def fold_auprc(y, p, fold_idx: dict) -> list[float]:
    return [ap(y[idx], p[idx]) for _, idx in sorted(fold_idx.items())]
