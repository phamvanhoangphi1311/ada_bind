"""Datasets / collate functions for residue-level (flat) and protein-level (group) training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..backend.gear import concat_blocks, store_of


class MultiArrayDataset(Dataset):
    """One item = one residue (dict of modality vectors, label)."""

    def __init__(self, mods: dict, y: np.ndarray):
        self.mods, self.y, self.keys = mods, y, sorted(mods)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = {k: torch.from_numpy(np.asarray(self.mods[k][i], dtype=np.float32)) for k in self.keys}
        return x, torch.tensor(self.y[i], dtype=torch.float32)


def collate_multi(batch):
    keys = sorted(batch[0][0])
    return ({k: torch.stack([b[0][k] for b in batch], dim=0) for k in keys},
            torch.stack([b[1] for b in batch]))


class GroupProteinDataset(Dataset):
    """One item = one protein with its modality matrix and its patch/pocket bank entry."""

    def __init__(self, ctx, split: str, df: pd.DataFrame, spec: dict, bank, label: str):
        self.split, self.df, self.spec, self.bank, self.label = split, df.reset_index(drop=True), spec, bank, label
        self.store = store_of(ctx, split)

    def __len__(self):
        return len(self.df)

    def labels(self) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=np.float32) for x in self.df[self.label]])

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = str(r.prot_id)
        er, eg, gf = self.bank[(self.split, pid)]
        return {
            "pid": pid,
            "mods": {m: concat_blocks(self.store, pid, b).astype(np.float32, copy=False) for m, b in self.spec.items()},
            "y": np.asarray(r[self.label], dtype=np.float32),
            "edge_res": er, "edge_group": eg, "group_feat": gf,
        }


def make_group_collate(feat_dim: int):
    """Concatenate proteins into one disjoint graph (with per-residue protein_index)."""

    def collate(batch):
        keys = sorted(batch[0]["mods"])
        mods = {k: torch.from_numpy(np.concatenate([b["mods"][k] for b in batch], axis=0)).float() for k in keys}
        y = torch.from_numpy(np.concatenate([b["y"] for b in batch])).float()
        er_all, eg_all, gf_all, pidx = [], [], [], []
        roff = goff = 0
        for p, b in enumerate(batch):
            L, G = len(b["y"]), len(b["group_feat"])
            pidx.append(torch.full((L,), p, dtype=torch.long))
            if G:
                gf_all.append(torch.from_numpy(b["group_feat"]).float())
            if len(b["edge_res"]):
                er_all.append(torch.from_numpy(b["edge_res"]).long() + roff)
                eg_all.append(torch.from_numpy(b["edge_group"]).long() + goff)
            roff += L
            goff += G
        return {
            "mods": mods,
            "y": y,
            "protein_index": torch.cat(pidx, 0),
            "edge_res": torch.cat(er_all, 0) if er_all else torch.zeros((0,), dtype=torch.long),
            "edge_group": torch.cat(eg_all, 0) if eg_all else torch.zeros((0,), dtype=torch.long),
            "group_feat": torch.cat(gf_all, 0) if gf_all else torch.zeros((0, feat_dim), dtype=torch.float32),
        }

    return collate
