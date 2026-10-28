"""Semantic modality grouping of a locked block list + residue-level packs."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..backend.gear import concat_blocks, store_of

FLOAT16_MODALITIES = frozenset({"ESM", "BIO"})


def modality_spec(blocks: Sequence[str], groups: Sequence[tuple[str, Sequence[str]]]) -> dict[str, list[str]]:
    """Assign blocks to named modalities; order inside a modality follows ``groups``.

    ``groups`` = [(modality, candidate_blocks), ...] (TaskSpec.modality_groups).
    Unassigned blocks go to OTHER.
    """
    blocks = list(blocks)
    spec = {}
    for name, candidates in groups:
        chosen = [b for b in candidates if b in blocks]
        if chosen:
            spec[name] = chosen
    assigned = {x for v in spec.values() for x in v}
    other = [b for b in blocks if b not in assigned]
    if other:
        spec["OTHER"] = other
    return spec


def modality_dims(ctx, split: str, spec: dict, pid: str | None = None) -> dict[str, int]:
    store = store_of(ctx, split)
    pid = pid or next(iter(store))
    return {m: concat_blocks(store, pid, b).shape[1] for m, b in spec.items()}


def build_pack(ctx, split: str, df: pd.DataFrame, spec: dict, label: str,
               float16=FLOAT16_MODALITIES) -> dict:
    """Concatenate all residues of a split: {'mods': {m: (N, d)}, 'y', 'pids', 'ridx'}."""
    store = store_of(ctx, split)
    mods = {k: [] for k in spec}
    yy, pids, ridx = [], [], []
    for r in df.itertuples():
        L = len(r.sequence)
        for m, blocks in spec.items():
            mods[m].append(concat_blocks(store, r.prot_id, blocks))
        yy.append(np.asarray(getattr(r, label), dtype=np.float32))
        pids.extend([str(r.prot_id)] * L)
        ridx.extend(range(L))
    mods = {k: np.concatenate(v, axis=0).astype(np.float16 if k in float16 else np.float32, copy=False)
            for k, v in mods.items()}
    return {"mods": mods, "y": np.concatenate(yy).astype(np.float32),
            "pids": np.asarray(pids, dtype=object), "ridx": np.asarray(ridx, dtype=np.int32)}


def build_packs(ctx, splits: dict[str, pd.DataFrame], spec: dict, label: str) -> dict:
    packs = {s: build_pack(ctx, s, df, spec, label) for s, df in splits.items()}
    for s, p in packs.items():
        print(s, {k: v.shape for k, v in p["mods"].items()}, "positive=", int(p["y"].sum()))
    return packs
