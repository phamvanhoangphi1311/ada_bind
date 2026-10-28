"""Residue-label CSVs → DataFrames, and the exact DevSet1010 / TestSet300 / TestSetNew46 splits."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import ALL_LABELS, DEV, SPLIT_SIZES, TEST46, TEST300
from ..paths import Paths


def parse_list(v):
    if isinstance(v, str):
        return list(ast.literal_eval(v))
    return list(v)


def load_split_csv(path: str | Path, labels=ALL_LABELS, require_unique: bool = True) -> pd.DataFrame:
    """Read one grouped CSV; label columns become python lists; lengths are checked."""
    df = pd.read_csv(path)
    df["prot_id"] = df["prot_id"].astype(str)
    df["sequence"] = df["sequence"].astype(str)
    present = [c for c in labels if c in df.columns]
    for c in present:
        df[c] = df[c].apply(parse_list)
    if require_unique and not df.prot_id.is_unique:
        raise RuntimeError(f"{path}: duplicate prot_id")
    for r in df.itertuples():
        for c in present:
            if len(getattr(r, c)) != len(r.sequence):
                raise RuntimeError(f"{path}/{r.prot_id}/{c}: label length != sequence length")
    return df


def load_raw_frames(paths: Paths) -> dict[str, pd.DataFrame]:
    """Raw CSVs. The raw dev CSV has 1014 proteins; the official DevSet1010 is a subset."""
    return {
        "raw_train": load_split_csv(paths.train_csv),
        TEST300: load_split_csv(paths.test300_csv),
        TEST46: load_split_csv(paths.test46_csv),
    }


def backend_frames(paths: Paths, official_ids, labels=ALL_LABELS) -> dict[str, pd.DataFrame]:
    """Frames of the GEAR store ({"train", "test300", "test46"}): Dev = official IDs in CSV order (the original
    backend's store order, which fixes the training-batch order), external sets as in their CSVs."""
    frames = {"train": load_split_csv(paths.train_csv, labels), "test300": load_split_csv(paths.test300_csv, labels),
              "test46": load_split_csv(paths.test46_csv, labels)}
    official = {str(p) for p in official_ids}
    frames["train"] = frames["train"][frames["train"].prot_id.isin(official)].reset_index(drop=True)
    return frames


def build_splits(raw: dict[str, pd.DataFrame], official_order: list[str]) -> dict[str, pd.DataFrame]:
    """DevSet1010 in official fold order + the two external sets."""
    raw_train = raw["raw_train"]
    missing = set(official_order) - set(raw_train.prot_id)
    if missing:
        raise RuntimeError(f"Official Dev IDs missing from project CSV: {sorted(missing)[:20]}")
    dev = raw_train.set_index("prot_id", drop=False).loc[official_order].reset_index(drop=True)
    splits = {DEV: dev, TEST300: raw[TEST300], TEST46: raw[TEST46]}
    for name, df in splits.items():
        if len(df) != SPLIT_SIZES[name]:
            raise RuntimeError(f"{name}: {len(df)} proteins, expected {SPLIT_SIZES[name]}")
    return splits


# ------------------------------------------------------------------ residue bookkeeping
def residue_offsets(df: pd.DataFrame) -> tuple[dict[str, tuple[int, int]], int]:
    """prot_id -> (start, end) in the concatenated residue vector, and total length."""
    out, cur = {}, 0
    for r in df.itertuples():
        L = len(r.sequence)
        out[str(r.prot_id)] = (cur, cur + L)
        cur += L
    return out, cur


def concat_labels(df: pd.DataFrame, label: str, dtype=np.int8) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=dtype) for x in df[label]])


def residue_ids(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(prot_id per residue, residue index per residue)."""
    pids, ridx = [], []
    for r in df.itertuples():
        L = len(r.sequence)
        pids.extend([str(r.prot_id)] * L)
        ridx.extend(range(L))
    return np.asarray(pids, dtype=object), np.asarray(ridx, dtype=np.int32)


def n_residues(df: pd.DataFrame) -> int:
    return int(df.sequence.str.len().sum())
