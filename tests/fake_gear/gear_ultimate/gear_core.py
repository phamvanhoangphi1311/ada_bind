"""Fake core: same function names/signatures/file layout as the real gear_core, tiny training."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve

HAS_KAN = False


@dataclass
class Config:
    base: str = "."
    label_column: str = "metal_binding"
    run_version: str = "x"
    models: tuple = ("MLP",)
    seed: int = 42
    dropout: float = 0.2
    neural_epochs: int = 2
    lr: float = 1e-3
    proteins_per_batch_mlp: int = 8
    proteins_per_batch_kan: int = 8


def make_paths(cfg):
    base = Path(cfg.base)
    run_root = base / "results" / cfg.run_version / cfg.label_column
    out = {"base": base, "run_root": run_root, "model_root": base / "trained_models" / cfg.run_version / cfg.label_column,
           "pred_root": run_root / "fold_predictions", "ensemble_root": run_root / "ensemble_predictions"}
    for k in ("run_root", "model_root", "pred_root", "ensemble_root"):
        out[k].mkdir(parents=True, exist_ok=True)
    return out


def job_pred_path(paths, model_name, rep_name, fold):
    return paths["pred_root"] / f"{model_name}__{rep_name.replace('+', '__')}__fold{fold + 1}.pkl"


def job_model_path(paths, model_name, rep_name, fold):
    ext = ".json" if model_name == "XGBoost" else ".pt"
    return paths["model_root"] / f"{model_name}__{rep_name.replace('+', '__')}__fold{fold + 1}{ext}"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def config_hash(cfg, representations):
    payload = {"config": asdict(cfg), "representations": {k: list(v) for k, v in representations.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def get_fold_train_val_ids(fold_idx, fold_ids, train_store):
    va = [p for p in fold_ids[fold_idx] if p in train_store]
    return [p for p in train_store if p not in set(va)], va


def choose_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    if len(th) == 0:
        return 0.5
    f = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[np.nanargmax(f)])


def _x(store, ids, blocks, scaler):
    X = np.concatenate([np.concatenate([np.asarray(store[p][b], np.float32) for b in blocks], 1) for p in ids])
    return (X - scaler["mean"]) / scaler["std"]


class ResidueMLP(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout),
                                 nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict_neural(model, kind, store, ids, blocks, scaler, device, batch_size):
    model.eval()
    X = torch.from_numpy(_x(store, ids, blocks, scaler).astype(np.float32)).to(device)
    y = np.concatenate([store[p]["y"] for p in ids]).astype(np.int64)
    return y, torch.sigmoid(model(X)).cpu().numpy().astype(np.float32)


def train_neural(kind, store, tr_ids, va_ids, blocks, cfg, seed, device):
    torch.manual_seed(seed)
    X = np.concatenate([np.concatenate([np.asarray(store[p][b], np.float32) for b in blocks], 1) for p in tr_ids])
    scaler = {"mean": X.mean(0).astype(np.float32), "std": (X.std(0) + 1e-4).astype(np.float32)}
    model = ResidueMLP(X.shape[1], cfg.dropout).to(device)
    Xt = torch.from_numpy(_x(store, tr_ids, blocks, scaler).astype(np.float32)).to(device)
    yt = torch.from_numpy(np.concatenate([store[p]["y"] for p in tr_ids]).astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.neural_epochs):
        model.train()
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(Xt), yt).backward()
        opt.step()
    yv, pv = predict_neural(model, kind, store, va_ids, blocks, scaler, device, 8)
    return model, scaler, {"best_epoch": cfg.neural_epochs, "best_val_auprc": float(average_precision_score(yv, pv))}
