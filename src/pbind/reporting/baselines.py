"""Read-only access to the frozen baseline reproductions (no baseline inference here)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import Paths

DIRECT_FILE = "FULL_DIRECT_BASELINE_METRICS_300_46.csv"
TRANSFER_FILES = {"ESMBind": "ESMBind_TRANSFER_FULL_METRICS_300_46.csv",
                  "MetalBind": "MetalBind_TRANSFER_FULL_METRICS_300_46.csv"}
DIRECT_METHODS = ("bindNode24", "bindEmbed21DL")
SHOW_COLS = ["method", "dataset", "label", "AP_AUPRC", "AUPRC", "AUROC", "global_positive_F1", "global_MCC",
             "protein_mean_F1", "protein_mean_MCC"]


def _read(fp, label, operating_point="fixed_0.5") -> pd.DataFrame:
    if not fp.exists():
        print("Missing baseline file:", fp)
        return pd.DataFrame()
    z = pd.read_csv(fp)
    if "label" in z.columns:
        z = z[z.label.eq(label)].copy()
    if operating_point and "operating_point" in z.columns:
        z = z[z.operating_point.eq(operating_point)].copy()
    return z


def load_baselines(paths: Paths, label: str, include_transfer: bool = False) -> pd.DataFrame:
    """bindEmbed21DL / bindNode24 (+ ESMBind / MetalBind transfer rows for metal)."""
    frames = []
    z = _read(paths.baseline_results / DIRECT_FILE, label)
    if len(z):
        z["baseline_group"] = "direct"
        frames.append(z)
    if include_transfer:
        for group, fn in TRANSFER_FILES.items():
            z = _read(paths.baseline_results / fn, label)
            if len(z):
                z["baseline_group"] = group
                frames.append(z)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def baseline_view(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in SHOW_COLS if c in df.columns]] if len(df) else df


def _first(row, cols, default=np.nan):
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
    return default


def baseline_compact_rows(df: pd.DataFrame, methods=DIRECT_METHODS) -> pd.DataFrame:
    """dataset/method/AUPRC/AUROC/F1/MCC rows for quick comparisons."""
    rows = []
    for _, r in df.iterrows():
        if methods and str(r.method) not in methods:
            continue
        rows.append({"dataset": str(r.dataset), "method": str(r.method), "AUPRC": _first(r, ["AP_AUPRC", "AUPRC"]),
                     "AUROC": _first(r, ["AUROC"]), "F1": _first(r, ["global_positive_F1", "F1"]),
                     "MCC": _first(r, ["global_MCC", "MCC"])})
    return pd.DataFrame(rows)


def baseline_full_rows(df: pd.DataFrame, methods=DIRECT_METHODS) -> pd.DataFrame:
    """Baseline rows in the paper-report schema (fixed 0.5 only)."""
    rows = []
    for _, r in df.iterrows():
        if methods and str(r.method) not in methods:
            continue
        acc = _first(r, ["accuracy", "Accuracy", "global_accuracy"])
        rec = _first(r, ["global_positive_recall", "recall", "Recall"])
        rows.append({
            "dataset": str(r.dataset), "method": str(r.method), "threshold_rule": "fixed_0.5", "threshold": 0.5,
            "threshold_selected_on": "reproduced baseline operating point",
            "directly_comparable_to_baseline_fixed0.5": True,
            "AUPRC_global_residue": _first(r, ["AP_AUPRC", "AUPRC"]),
            "AUROC_global_residue": _first(r, ["AUROC"]),
            "Accuracy_global_residue": acc,
            "Precision_positive_global_residue": _first(r, ["global_positive_precision", "precision", "Precision"]),
            "Recall_positive_global_residue": rec,
            "Sensitivity_global_residue": rec,
            "F1_positive_global_residue": _first(r, ["global_positive_F1", "F1", "f1"]),
            "MCC_global_residue": _first(r, ["global_MCC", "MCC"]),
            "F1_micro_all_classes": acc,
            "protein_mean_F1_baseline_convention": _first(r, ["protein_mean_F1"]),
            "protein_mean_MCC_baseline_convention": _first(r, ["protein_mean_MCC"]),
        })
    return pd.DataFrame(rows)


def prior_gear_controls(paths: Paths, label: str, runs) -> pd.DataFrame:
    """cv_summary.csv of earlier GEAR runs for this label, if present."""
    frames = []
    for run in runs:
        fp = paths.results / run / label / "cv_summary.csv"
        if fp.exists():
            z = pd.read_csv(fp)
            z.insert(0, "run_version", run)
            frames.append(z)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
