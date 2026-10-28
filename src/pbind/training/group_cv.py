"""Protein-level patch/pocket hypergraph training: official-fold CV, resume-safe.

Covers V1 (GroupHypergraph, plain weighted BCE) and V2 (GroupMessageV2,
protein-balanced BCE ± hard-negative ranking) with the same loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from ..data.folds import fold_frames, scatter_fold_predictions
from ..data.loading import concat_labels, n_residues
from ..models.hypergraph import GroupHypergraph, GroupMessageV2
from ..models.losses import sqrt_pos_weight, v2_loss
from ..utils.cache import FoldArtifacts
from ..utils.torch_utils import free_gpu, seed_everything, to_device
from .loop import CVResult, TrainConfig, fit_early_stopping


@dataclass(frozen=True)
class LossConfig:
    kind: str = "bce"  # "bce" (V1) | "v2" (protein-balanced, optional hard-negative)
    hardneg: bool = False
    lam: float = 0.20
    margin: float = 0.50
    k_min: int = 4
    k_max: int = 32


def _forward(model, b, aux=False):
    args = (b["mods"], b["edge_res"], b["edge_group"], b["group_feat"])
    if isinstance(model, GroupMessageV2):
        return model(*args, return_aux=True) if aux else model(*args)
    out = model(*args)
    return (out, {}) if aux else out


def _loader(ds, collate, batch, shuffle, device):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=(device.type == "cuda"), collate_fn=collate)


@torch.no_grad()
def predict_groups(model, ds, collate, batch, device):
    """(probabilities, labels, mean gate) over a protein dataset, in dataset order."""
    model.eval()
    pp, yy, gates = [], [], []
    for b in _loader(ds, collate, batch, False, device):
        b = to_device(b, device)
        logits, aux = _forward(model, b, aux=True)
        pp.append(torch.sigmoid(logits).cpu().numpy())
        yy.append(b["y"].cpu().numpy())
        if "gate_mean" in aux:
            gates.append(float(aux["gate_mean"].detach().cpu()))
    return np.concatenate(pp), np.concatenate(yy), (float(np.mean(gates)) if gates else 0.0)


def train_group_fold(make_model: Callable, tr_ds, va_ds, collate, fold: int, cfg: TrainConfig,
                     loss: LossConfig, device, tag: str):
    seed_everything(cfg.seed_for(fold))
    model = make_model().to(device)
    pos_weight = sqrt_pos_weight(tr_ds.labels(), device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def step(m, b):
        b = to_device(b, device)
        logits = _forward(m, b)
        if loss.kind == "bce":
            return bce(logits, b["y"]), 0.0
        total, rank = v2_loss(logits, b["y"], b["protein_index"], pos_weight, loss.hardneg, loss.lam,
                              loss.margin, loss.k_min, loss.k_max)
        return total, float(rank.detach().cpu())

    def validate(m):
        p, y, gate = predict_groups(m, va_ds, collate, cfg.batch, device)
        return float(average_precision_score(y, p)), ({"gate": gate} if isinstance(m, GroupMessageV2) else {})

    def extra(m):
        return {"alpha": m.alpha} if isinstance(m, GroupHypergraph) else {}

    loader = _loader(tr_ds, collate, cfg.batch, True, device)
    return fit_early_stopping(model, loader, step, validate, cfg, f"{tag} fold {fold}", extra)


def run_group_cv(name: str, make_model: Callable, make_dataset: Callable, collate, dev_df: pd.DataFrame,
                 folds: dict, label: str, state_dir: Path, meta: dict, cfg: TrainConfig,
                 loss: LossConfig = LossConfig(), device=None, force: bool = False,
                 name_key: str = "architecture", report_gate: bool = False) -> CVResult:
    """5-fold CV of one hypergraph model.

    ``meta`` must contain the keys that make a cached fold valid (e.g. representation,
    and code_version for V2); ``n_val`` is checked automatically.
    ``make_dataset(df) -> GroupProteinDataset``.
    """
    oof = np.full(n_residues(dev_df), np.nan, dtype=np.float32)
    rows, states = [], {}
    check_keys = {k: v for k, v in meta.items() if k != "blocks"}
    for fold in sorted(folds):
        tr_df, va_df = fold_frames(dev_df, folds, fold)
        tr_ds, va_ds = make_dataset(tr_df), make_dataset(va_df)
        expected = n_residues(va_df)
        art = FoldArtifacts(state_dir, fold)
        cached = None if force else art.load({**check_keys, "n_val": expected}, expected)
        gate = 0.0
        if cached is not None:
            print(name, "fold", fold, "REUSE")
            state, pred = cached
            if report_gate:
                model = make_model().to(device)
                model.load_state_dict(state)
                _, _, gate = predict_groups(model, va_ds, collate, cfg.batch, device)
                del model
        else:
            print(name, "fold", fold, "TRAIN")
            model, state = train_group_fold(make_model, tr_ds, va_ds, collate, fold, cfg, loss, device, name)
            pred, _, gate = predict_groups(model, va_ds, collate, cfg.batch, device)
            art.save(state, pred, {**meta, "n_val": expected})
            del model
            free_gpu(device)
        states[fold] = state
        scatter_fold_predictions(dev_df, va_df, pred, oof)
        yv = concat_labels(va_df, label, np.float32)
        row = {name_key: name, "fold": fold, "AUPRC": float(average_precision_score(yv, pred)),
               "n_val": len(pred), "n_positive": int(yv.sum())}
        if report_gate:
            row["gate_mean"] = gate
        rows.append(row)
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f"{name}: incomplete OOF")
    f = pd.DataFrame(rows)
    yall = concat_labels(dev_df, label, np.float32)
    summary = {name_key: name, "CV_AUPRC_mean": float(f.AUPRC.mean()), "CV_AUPRC_std": float(f.AUPRC.std(ddof=1)),
               "OOF_AUPRC": float(average_precision_score(yall, oof))}
    if report_gate:
        summary["gate_mean"] = float(f.gate_mean.mean())
    return CVResult(name, summary, f, oof, states)


def predict_group_ensemble(make_model: Callable, states: dict, ds, collate, batch, device) -> np.ndarray:
    """Mean of the five fold models on a whole split (external evaluation)."""
    preds = []
    for _, state in sorted(states.items()):
        model = make_model().to(device)
        model.load_state_dict(state)
        p, _, _ = predict_groups(model, ds, collate, batch, device)
        preds.append(p)
        del model
        free_gpu(device)
    return np.mean(np.stack(preds), axis=0)


def oof_from_saved_folds(state_dir: Path, dev_df: pd.DataFrame, folds: dict, label: str):
    """Rebuild Dev OOF from saved ``fold{k}_val.npy`` only (no model loading). Returns (oof, y)."""
    oof = np.full(n_residues(dev_df), np.nan, dtype=np.float32)
    for fold in sorted(folds):
        _, va_df = fold_frames(dev_df, folds, fold)
        expected = n_residues(va_df)
        pred = FoldArtifacts(state_dir, fold).load_val_pred(expected)
        if pred is None:
            raise FileNotFoundError(f"{state_dir}/fold{fold}_val.npy missing or wrong shape (expected {expected})")
        scatter_fold_predictions(dev_df, va_df, pred, oof)
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("Incomplete reconstructed OOF")
    return oof, concat_labels(dev_df, label)
