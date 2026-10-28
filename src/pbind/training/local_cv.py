"""Residue-level (flat) multimodal fusion models: official-fold CV, resume-safe."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset

from ..data.folds import train_val_residue_idx
from ..models.losses import sqrt_pos_weight
from ..utils.cache import FoldArtifacts
from ..utils.torch_utils import free_gpu, seed_everything
from .datasets import MultiArrayDataset, collate_multi
from .loop import CVResult, TrainConfig, fit_early_stopping


def _loader(ds, batch, shuffle, device):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=(device.type == "cuda"), collate_fn=collate_multi)


@torch.no_grad()
def predict_flat(model, dataset, device, indices=None, batch_size=8192) -> np.ndarray:
    ds = dataset if indices is None else Subset(dataset, np.asarray(indices).tolist())
    model.eval()
    out = []
    for x, _ in _loader(ds, batch_size, False, device):
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out)


def train_local_fold(factory: Callable, dims: dict, pack: dict, train_idx, val_idx, fold: int,
                     cfg: TrainConfig, device, tag: str):
    seed_everything(cfg.seed_for(fold))
    ds = MultiArrayDataset(pack["mods"], pack["y"])
    loader = _loader(Subset(ds, train_idx.tolist()), cfg.batch, True, device)
    model = factory(dims).to(device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=sqrt_pos_weight(pack["y"][train_idx], device))

    def step(m, batch):
        x, y = batch
        x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
        return crit(m(x), y.to(device, non_blocking=True)), 0.0

    def validate(m):
        return float(average_precision_score(pack["y"][val_idx], predict_flat(m, ds, device, val_idx))), {}

    return fit_early_stopping(model, loader, step, validate, cfg, f"{tag} fold {fold}")


def run_local_cv(arch: str, factory: Callable, dims: dict, pack: dict, folds: dict, state_root: Path,
                 rep_name: str, rep_blocks, cfg: TrainConfig, device, force: bool = False) -> CVResult:
    """5-fold CV of one fusion architecture. Fold checkpoints are reused when meta matches."""
    y, pids = pack["y"], pack["pids"]
    ds = MultiArrayDataset(pack["mods"], y)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    rows, states = [], {}
    for fold in sorted(folds):
        train_idx, val_idx = train_val_residue_idx(pids, folds, fold)
        art = FoldArtifacts(Path(state_root) / arch, fold)
        cached = None if force else art.load({"representation": rep_name, "n_val": int(len(val_idx))}, len(val_idx))
        if cached is not None:
            print(arch, "fold", fold, "REUSE")
            state, pred = cached
        else:
            print(arch, "fold", fold, "TRAIN")
            model, state = train_local_fold(factory, dims, pack, train_idx, val_idx, fold, cfg, device, arch)
            pred = predict_flat(model, ds, device, val_idx).astype(np.float32)
            art.save(state, pred, {"representation": rep_name, "blocks": list(rep_blocks), "n_val": int(len(val_idx))})
            del model
            free_gpu(device)
        states[fold] = state
        oof[val_idx] = pred
        rows.append({"architecture": arch, "fold": fold, "AUPRC": float(average_precision_score(y[val_idx], pred)),
                     "n_val": int(len(val_idx)), "n_positive": int(y[val_idx].sum())})
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f"{arch}: incomplete OOF")
    f = pd.DataFrame(rows)
    summary = {"architecture": arch, "CV_AUPRC_mean": float(f.AUPRC.mean()),
               "CV_AUPRC_std": float(f.AUPRC.std(ddof=1)), "OOF_AUPRC": float(average_precision_score(y, oof))}
    return CVResult(arch, summary, f, oof, states)


def predict_local_ensemble(factory: Callable, dims: dict, states: dict, pack: dict, device) -> np.ndarray:
    """Mean of the five fold models on a whole split."""
    ds = MultiArrayDataset(pack["mods"], pack["y"])
    preds = []
    for _, state in sorted(states.items()):
        model = factory(dims).to(device)
        model.load_state_dict(state)
        preds.append(predict_flat(model, ds, device))
        del model
        free_gpu(device)
    return np.mean(np.stack(preds), axis=0)
