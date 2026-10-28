"""Standardized residue trainer of the GEAR backend (port of ``gear_core``) for every ``kind="residue"`` model of
``pbind.models.registry`` (MLP, KAN, user models) and for XGBoost (``kind="tree"``).

Used for every same-MLP feature screen and for the final metal model.
Protein-level batches (8 proteins), per-fold train-only standardizer, BCE with clipped pos_weight,
early stopping on validation AUPRC (patience 6), seed = cfg.seed + fold + 1000 * model.seed_offset.
"""
from __future__ import annotations

import gc
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from ..models.registry import get_model
from ..models.residue import HAS_KAN, ResidueKAN, ResidueMLP  # noqa: F401  (backend interface)

try:  # optional, only for the XGBoost model family
    import xgboost  # noqa: F401

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False



@dataclass
class GearConfig:
    """Defaults identical to the original backend Config (only the fields used here)."""
    base: str = "."
    label_column: str = "metal_binding"
    run_version: str = "standardized_chem_v1"
    models: tuple = ("MLP",)
    max_residues: int = 1000
    seed: int = 42
    neural_epochs: int = 30
    neural_patience: int = 6
    proteins_per_batch_mlp: int = 8
    proteins_per_batch_kan: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.20
    grad_clip: float = 2.0
    xgb_estimators: int = 700
    xgb_depth: int = 7
    xgb_lr: float = 0.04
    xgb_subsample: float = 0.85
    xgb_colsample: float = 0.65
    xgb_min_child_weight: float = 3.0
    xgb_reg_lambda: float = 1.0
    xgb_patience: int = 40
    xgb_max_bin: int = 256
    xgb_train_max_residues: Optional[int] = None
    fixed_threshold: float = 0.50
    save_models: bool = True  # original default was False; weights are kept here
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_pos_weight(store, ids) -> float:
    pos = sum(int(store[p]["y"].sum()) for p in ids)
    total = sum(len(store[p]["y"]) for p in ids)
    return float(np.clip((total - pos) / max(pos, 1), 1.0, 30.0))


def choose_f1_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(np.asarray(y), np.asarray(p))
    if len(th) == 0:
        return 0.5
    f = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[np.nanargmax(f)])


def binary_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(np.int64)
    two = len(np.unique(y)) > 1
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "auprc": float(average_precision_score(y, p)) if two else np.nan,
        "auroc": float(roc_auc_score(y, p)) if two else np.nan,
    }


# ------------------------------------------------------------------ data
def concat_features(item, blocks: Sequence[str]) -> np.ndarray:
    return np.concatenate([np.asarray(item[b], dtype=np.float32) for b in blocks], axis=1).astype(np.float32)


def matrix_from_store(store, ids, blocks):
    return (np.concatenate([concat_features(store[p], blocks) for p in ids], axis=0),
            np.concatenate([np.asarray(store[p]["y"], dtype=np.int64) for p in ids], axis=0))


def fit_streaming_standardizer(store, ids, blocks) -> dict:
    n, s, ss = 0, None, None
    for pid in ids:
        X = concat_features(store[pid], blocks).astype(np.float64)
        if s is None:
            s, ss = np.zeros(X.shape[1]), np.zeros(X.shape[1])
        s += X.sum(0)
        ss += (X * X).sum(0)
        n += len(X)
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(ss / max(n, 1) - mean * mean, 1e-8))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class ProteinDataset(Dataset):
    def __init__(self, store, ids, blocks, scaler):
        self.store, self.ids, self.blocks, self.scaler = store, list(ids), list(blocks), scaler

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]
        X = ((concat_features(self.store[pid], self.blocks) - self.scaler["mean"]) / self.scaler["std"]).astype(np.float32)
        return pid, torch.from_numpy(X), torch.from_numpy(np.asarray(self.store[pid]["y"], dtype=np.int64))


def protein_collate(batch):
    pids, xs, ys = zip(*batch)
    return list(pids), torch.cat(xs, 0).float(), torch.cat(ys, 0).long(), [len(y) for y in ys]


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ------------------------------------------------------------------ train / predict
@torch.no_grad()
def predict_neural(model, kind: str, store, ids, blocks, scaler, device, batch_size: int):
    loader = DataLoader(ProteinDataset(store, ids, blocks, scaler), batch_size=batch_size, shuffle=False,
                        collate_fn=protein_collate, num_workers=0)
    model.eval()
    ys, ps = [], []
    for _, X, y, _ in loader:
        logits = model(X.to(device=device, dtype=torch.float32)).float()
        p = torch.sigmoid(logits) if get_model(kind).output == "logit" else torch.softmax(logits, dim=-1)[:, 1]
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps).astype(np.float32)


def train_neural(kind: str, store, tr_ids, va_ids, blocks, cfg: GearConfig, seed: int, device):
    """Train registered residue model ``kind`` on one fold (MLP / KAN reproduce the original backend exactly)."""
    spec = get_model(kind)
    if spec.kind != "residue":
        raise ValueError(f"{kind} is a {spec.kind} model, not a residue model")
    set_seed(seed)
    scaler = fit_streaming_standardizer(store, tr_ids, blocks)
    d = sum(np.asarray(store[tr_ids[0]][b]).shape[1] for b in blocks)
    model = spec.build(d, cfg, seed).to(device=device, dtype=torch.float32)
    batch_size = getattr(cfg, spec.batch_attr)
    if spec.output == "logit":
        posw = torch.tensor([class_pos_weight(store, tr_ids)], dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=posw)
    else:
        w = torch.tensor([1.0, class_pos_weight(store, tr_ids)], dtype=torch.float32, device=device)

        def loss_fn(logits, y):
            return nn.functional.cross_entropy(logits, y.long(), weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(ProteinDataset(store, tr_ids, blocks, scaler), batch_size=batch_size, shuffle=True,
                        collate_fn=protein_collate, num_workers=0)
    best_ap, best_state, best_epoch, bad = -np.inf, None, 0, 0
    for epoch in range(1, cfg.neural_epochs + 1):
        model.train()
        for _, X, y, _ in loader:
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(X).float()
            loss = loss_fn(logits, y.float() if spec.output == "logit" else y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        yv, pv = predict_neural(model, kind, store, va_ids, blocks, scaler, device, batch_size)
        ap = average_precision_score(yv, pv)
        if ap > best_ap + 1e-4:
            best_ap, best_epoch, bad = float(ap), epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.neural_patience:
            break
    if best_state is None:
        raise RuntimeError(f"{kind}: no valid checkpoint")
    model.load_state_dict(best_state)
    return model, scaler, {"best_epoch": best_epoch, "best_val_auprc": best_ap, "n_params": count_params(model)}


def stratified_cap(X, y, max_rows, seed):
    if max_rows is None or len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n1 = min(len(idx1), max(1, int(max_rows * len(idx1) / max(len(y), 1))))
    take = np.concatenate([rng.choice(idx0, min(len(idx0), max_rows - n1), replace=False),
                           rng.choice(idx1, n1, replace=False)])
    rng.shuffle(take)
    return X[take], y[take]


def train_xgboost(store, tr_ids, va_ids, blocks, cfg: GearConfig, seed: int, device):
    import xgboost as xgb

    Xtr, ytr = matrix_from_store(store, tr_ids, blocks)
    Xv, yv = matrix_from_store(store, va_ids, blocks)
    Xtr, ytr = stratified_cap(Xtr, ytr, cfg.xgb_train_max_residues, seed)
    kwargs = dict(n_estimators=cfg.xgb_estimators, max_depth=cfg.xgb_depth, learning_rate=cfg.xgb_lr,
                  subsample=cfg.xgb_subsample, colsample_bytree=cfg.xgb_colsample,
                  min_child_weight=cfg.xgb_min_child_weight, reg_lambda=cfg.xgb_reg_lambda,
                  scale_pos_weight=float(np.clip((len(ytr) - int(ytr.sum())) / max(int(ytr.sum()), 1), 1.0, 30.0)),
                  eval_metric="aucpr", tree_method="hist", max_bin=cfg.xgb_max_bin, random_state=seed, n_jobs=-1,
                  early_stopping_rounds=cfg.xgb_patience)
    if getattr(device, "type", str(device)) == "cuda":
        kwargs["device"] = "cuda"
    model = xgb.XGBClassifier(**kwargs)
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    meta = {"best_epoch": int(getattr(model, "best_iteration", cfg.xgb_estimators - 1)) + 1,
            "best_val_auprc": float(average_precision_score(yv, model.predict_proba(Xv)[:, 1])),
            "n_params": np.nan, "xgb_train_rows": int(len(ytr))}
    del Xtr, ytr, Xv, yv
    gc.collect()
    return model, None, meta


def predict_xgboost(model, store, ids, blocks):
    X, y = matrix_from_store(store, ids, blocks)
    p = model.predict_proba(X)[:, 1].astype(np.float32)
    del X
    gc.collect()
    return y, p
