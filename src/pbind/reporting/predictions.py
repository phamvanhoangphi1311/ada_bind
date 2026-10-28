"""Per-residue prediction tables and run manifests."""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loading import concat_labels, residue_ids
from ..utils.cache import save_json, save_table


def residue_prediction_frame(df: pd.DataFrame, label: str, **probs) -> pd.DataFrame:
    pids, ridx = residue_ids(df)
    out = pd.DataFrame({"prot_id": pids, "residue_index": ridx, "y_true": concat_labels(df, label)})
    for k, v in probs.items():
        out[k] = np.asarray(v)
    return out


def save_residue_predictions(fp: Path, df: pd.DataFrame, label: str, **probs) -> Path:
    return save_table(fp, residue_prediction_frame(df, label, **probs))


def write_manifest(fp: Path, **fields) -> dict:
    import torch

    manifest = {"python": platform.python_version(), "torch": torch.__version__, **fields}
    save_json(fp, manifest)
    return manifest
