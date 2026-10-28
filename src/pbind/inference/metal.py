"""Metal-binding inference with the Dev-locked GEAR model.

Default model = the locked result of GEAR_METAL_ONLY_ULTIMATE_V2:
MLP on ESM + ATOM + CHEM + MULTI2 (1377-d), 5 official-fold models, ensemble = mean.

Default weights = ``<BASE>/final_model/metal/mlp/`` (Colab T4 rerun of the original notebook, bit-identical
predictions), linked into the backend's layout
    <BASE>/trained_models/metal_v2_06_final_locked_external/metal_binding/MLP__ESM__Atomic__Chem__MULTI2__fold<k>.pt
``run_version`` selects another folder (``trained_models/<run_version>/metal_binding/``); a fold missing there is
trained ONCE with the backend's own ``train_neural`` / ``train_xgboost`` (same config, seed formula and official
folds) and saved; every later call only loads + predicts.

Thresholds (Dev only): fixed 0.5 and the backend's ``OOF_tuned_threshold`` = F1-optimal threshold on the
pooled Dev out-of-fold predictions of the same five models.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..constants import EXTERNAL_SPLITS, STORE_KEY
from ..evaluation.metrics import full_global_metrics, protein_mean_metrics
from ..models.registry import get_model
from ..pipelines.workspace import Workspace
from ..tasks import metal as M
from ..utils.cache import PredictionCache, load_json, load_npz_valid, save_json, save_npz, save_table
from ..utils.torch_utils import free_gpu
from .final import install_final_models

LABEL = M.LABEL
DEFAULT_LOCK = M.FINAL_LOCK


@dataclass
class FoldModel:
    fold: int  # 1-based
    model: object
    scaler: dict | None
    path: Path
    val_ids: list
    val_y: np.ndarray
    val_p: np.ndarray
    trained_now: bool


# ------------------------------------------------------------------ config
def locked_config(ws: Workspace) -> dict:
    """Locked representation/model from the metal v2 decision files (falls back to the known lock)."""
    root = ws.paths.results / M.RUN_ROOT
    lock = load_json(root / "locked_model.json") or load_json(root / "locked_representation.json") or {}
    out = {**DEFAULT_LOCK, **{k: lock[k] for k in DEFAULT_LOCK if k in lock}}
    out["run_version"] = M.RUN_VERSIONS["final"]
    return out


def run_context(ws: Workspace, lock: dict):
    """Backend context of the final locked run (same cfg -> same trained_models/ and fold_predictions/ paths)."""
    ws.require_backend()
    return ws.gear.suite.clone_context_for_run(ws.ctx, run_version=lock["run_version"],
                                               models=(lock["locked_model"],))


# ------------------------------------------------------------------ fold models
def _val_sidecar(path: Path) -> Path:
    return path.with_name(path.stem + ".val.npz")


def _ckpt_sig(path: Path) -> str:
    """Content hash of a checkpoint: a sidecar is only valid for the exact weights it was computed from."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def _predict(core, fm: FoldModel, kind: str, store, ids, blocks, cfg, device):
    if get_model(kind).kind == "tree":
        return core.predict_xgboost(fm.model, store, ids, blocks)
    return core.predict_neural(fm.model, kind, store, ids, blocks, fm.scaler, device,
                               getattr(cfg, get_model(kind).batch_attr))


def load_or_train_fold(ws: Workspace, rctx, lock: dict, fold0: int, force_retrain: bool = False) -> FoldModel:
    """One official fold model (0-based ``fold0``), loaded from trained_models/ or trained once and saved."""
    core, cfg, device = ws.gear.core, rctx.cfg, ws.device
    kind, rep, blocks = lock["locked_model"], lock["locked_representation"], list(lock["locked_blocks"])
    train_store = rctx.stores["train"]
    tr_ids, va_ids = core.get_fold_train_val_ids(fold0, rctx.fold_ids, train_store)
    spec = get_model(kind)
    seed = cfg.seed + fold0 + 1000 * spec.seed_offset
    path = Path(core.job_model_path(rctx.paths, kind, rep, fold0))
    d = sum(np.asarray(train_store[tr_ids[0]][b]).shape[1] for b in blocks)

    if path.exists() and not force_retrain:
        if spec.kind == "tree":
            import xgboost as xgb

            model, scaler = xgb.XGBClassifier(), None
            model.load_model(path)
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)  # numpy scaler inside
            if list(ck["blocks"]) != blocks:
                raise RuntimeError(f"{path}: checkpoint blocks {ck['blocks']} != locked {blocks}")
            model = spec.build(d, cfg, seed).to(device)
            model.load_state_dict(ck["state_dict"])
            scaler = ck["scaler"]
        fm = FoldModel(fold0 + 1, model, scaler, path, va_ids, np.zeros(0), np.zeros(0), trained_now=False)
        side, sig = load_npz_valid(_val_sidecar(path)), _ckpt_sig(path)
        if side is not None and list(side["val_ids"]) == list(va_ids) and "ckpt" in side and str(side["ckpt"]) == sig:
            fm.val_y, fm.val_p = side["val_y"], side["val_p"]
        else:  # no / stale stored OOF (e.g. weights replaced by a Colab run): recompute on the fold's Dev proteins
            fm.val_y, fm.val_p = _predict(core, fm, kind, train_store, va_ids, blocks, cfg, device)
            save_npz(_val_sidecar(path), val_ids=np.asarray(va_ids), val_y=fm.val_y, val_p=fm.val_p, ckpt=sig)
        print(f"LOAD {kind} | {rep} | fold {fold0 + 1}: {path.name}")
        return fm

    print(f"TRAIN {kind} | {rep} | fold {fold0 + 1}/5 | train={len(tr_ids)} val={len(va_ids)} | seed={seed}")
    if spec.kind == "tree":
        model, scaler, meta = core.train_xgboost(train_store, tr_ids, va_ids, blocks, cfg, seed, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".tmp.{path.name}")
        model.save_model(tmp)
        tmp.replace(path)
    else:
        model, scaler, meta = core.train_neural(kind, train_store, tr_ids, va_ids, blocks, cfg, seed, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".tmp.{path.name}")
        torch.save({"state_dict": model.state_dict(), "scaler": scaler, "blocks": blocks,
                    "config_hash": core.config_hash(cfg, {rep: blocks}), "meta": meta}, tmp)
        tmp.replace(path)
    fm = FoldModel(fold0 + 1, model, scaler, path, va_ids, np.zeros(0), np.zeros(0), trained_now=True)
    fm.val_y, fm.val_p = _predict(core, fm, kind, train_store, va_ids, blocks, cfg, device)
    save_npz(_val_sidecar(path), val_ids=np.asarray(va_ids), val_y=fm.val_y, val_p=fm.val_p, ckpt=_ckpt_sig(path))
    return fm


def models_fingerprint(paths) -> str:
    h = hashlib.sha1()
    for p in sorted(Path(x) for x in paths):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ reference: the original run's predictions
def original_run_ensemble(ws: Workspace, rctx, lock: dict, split: str) -> np.ndarray | None:
    """Mean of ``<split>_p`` from the original run's fold_predictions/*.pkl (if all five exist)."""
    core, key = ws.gear.core, f"{STORE_KEY[split]}_p"
    preds = []
    for fold0 in range(5):
        fp = Path(core.job_pred_path(rctx.paths, lock["locked_model"], lock["locked_representation"], fold0))
        if not fp.exists():
            return None
        z = core.load_pickle(fp)
        if key not in z:
            return None
        preds.append(np.asarray(z[key], dtype=np.float64))
    return np.mean(np.stack(preds), axis=0)


# ------------------------------------------------------------------ main entry
def run_metal_inference(ws: Workspace, split: str = "TestSet300", force_retrain: bool = False,
                        force_predict: bool = False, out_dir: str | Path | None = None,
                        run_version: str | None = None) -> dict:
    """Ensemble inference of the locked metal model on an external split; writes residue/protein/metric tables.
    Default weights = ``final_model/metal/mlp`` (the original final model, saved by the Colab T4 rerun); a fold
    that is missing there is trained once. ``run_version`` selects another set of fold weights
    (``trained_models/<run_version>/metal_binding/``).

    Outputs (``results/METAL_INFERENCE/<representation>/<split>/``):
      residue_predictions.csv.gz  prot_id, residue_index, AA, y_true, prob, prob_fold1..5, pred_at_0.5, pred_at_oof_thr
      protein_summary.csv         per protein: predicted binding residues at both thresholds (+ true count)
      metrics.csv / inference_summary.json
    """
    if split not in EXTERNAL_SPLITS:
        raise ValueError(f"split must be one of {EXTERNAL_SPLITS} (Dev predictions are the CV out-of-fold ones)")
    if run_version is None:
        if force_retrain:
            raise ValueError("the default weights are final_model/metal/mlp; retrain into another folder with "
                             "run_version=... (e.g. scripts/train_metal.py)")
        install_final_models(ws.paths, ["metal"])  # the final weights in final_model/metal/mlp (no training)
    lock = locked_config(ws)
    if run_version:
        lock["run_version"] = run_version
    rctx = run_context(ws, lock)
    core, kind, rep, blocks = ws.gear.core, lock["locked_model"], lock["locked_representation"], lock["locked_blocks"]
    store = rctx.stores[STORE_KEY[split]]
    ids = list(store)
    n = sum(len(store[p]["y"]) for p in ids)
    out = Path(out_dir) if out_dir else ws.paths.results_dir("METAL_INFERENCE", rep.replace("+", "__"),
                                                               *([run_version] if run_version else []), split)
    out.mkdir(parents=True, exist_ok=True)
    print("LOCKED:", kind, rep, blocks)

    # 1. five fold models (load, or train once)
    folds = []
    for fold0 in range(5):
        folds.append(load_or_train_fold(ws, rctx, lock, fold0, force_retrain))
    tag = models_fingerprint([fm.path for fm in folds])

    # 2. Dev-OOF threshold of THESE models (backend convention: F1-optimal on pooled OOF)
    oof_y = np.concatenate([fm.val_y for fm in folds]).astype(np.int64)
    oof_p = np.concatenate([fm.val_p for fm in folds]).astype(np.float64)
    oof_thr = float(core.choose_f1_threshold(oof_y, oof_p))

    # 3. per-fold external predictions (cached per checkpoint set) -> mean
    cache = PredictionCache(out / "prediction_cache")
    per_fold = []
    for fm in folds:
        per_fold.append(cache.get_or_compute(
            f"{kind}__{rep.replace('+', '__')}__fold{fm.fold}", split, n,
            lambda fm=fm: _predict(core, fm, kind, store, ids, blocks, rctx.cfg, ws.device)[1],
            force=force_predict, tag=tag))
    for fm in folds:
        fm.model = None
    free_gpu(ws.device)
    prob = np.mean(np.stack(per_fold), axis=0)

    # 4. tables
    y = np.concatenate([np.asarray(store[p]["y"], dtype=np.int64) for p in ids])
    rows_pid, rows_idx, rows_aa = [], [], []
    for p in ids:
        seq = store[p]["sequence"]
        rows_pid += [p] * len(seq)
        rows_idx += list(range(len(seq)))
        rows_aa += list(seq)
    res = pd.DataFrame({"prot_id": rows_pid, "residue_index": rows_idx, "AA": rows_aa, "y_true": y, "prob": prob,
                        **{f"prob_fold{k + 1}": pf for k, pf in enumerate(per_fold)},
                        "pred_at_0.5": (prob >= 0.5).astype(np.int8),
                        "pred_at_oof_thr": (prob >= oof_thr).astype(np.int8)})
    save_table(out / "residue_predictions.csv.gz", res)

    prot_rows, cur = [], 0
    for p in ids:
        seq, yy = store[p]["sequence"], np.asarray(store[p]["y"])
        pp = prob[cur:cur + len(yy)]
        cur += len(yy)
        hit = np.flatnonzero(pp >= oof_thr)
        prot_rows.append({"prot_id": p, "length": len(yy), "n_true_binding": int(yy.sum()),
                          "max_prob": float(pp.max()), "n_pred_at_0.5": int((pp >= 0.5).sum()),
                          "n_pred_at_oof_thr": len(hit),
                          "pred_residues_at_oof_thr": " ".join(f"{seq[i]}{i + 1}" for i in hit)})  # 1-based
    prot = pd.DataFrame(prot_rows)
    save_table(out / "protein_summary.csv", prot)

    # 5. metrics (labels are available for the benchmark sets)
    eval_df = pd.DataFrame({"prot_id": ids, "sequence": [store[p]["sequence"] for p in ids],
                            LABEL: [list(store[p]["y"]) for p in ids]})
    metrics = pd.DataFrame([
        {"dataset": split, "model": kind, "representation": rep, "threshold_rule": rule, "threshold": thr,
         **full_global_metrics(y, prob, thr), **protein_mean_metrics(eval_df, prob, thr, LABEL)}
        for rule, thr in (("fixed_0.5", 0.5), ("dev_oof_f1_opt", oof_thr))
    ])
    save_table(out / "metrics.csv", metrics)

    # 6. agreement with the original locked run (its saved fold predictions)
    ref = original_run_ensemble(ws, rctx, lock, split)
    agreement = None
    if ref is not None and ref.shape == prob.shape:
        from ..evaluation.metrics import ap

        agreement = {"original_run_AUPRC": ap(y, ref), "this_inference_AUPRC": ap(y, prob),
                     "pearson_r": float(np.corrcoef(ref, prob)[0, 1]),
                     "max_abs_diff": float(np.max(np.abs(ref - prob)))}
    summary = {
        "split": split, "model": kind, "representation": rep, "blocks": blocks, "run_version": lock["run_version"],
        "fold_checkpoints": [str(fm.path) for fm in folds],
        "folds_trained_in_this_call": [fm.fold for fm in folds if fm.trained_now],
        "dev_oof_f1_threshold": oof_thr, "n_proteins": len(ids), "n_residues": int(n),
        "AUPRC": float(metrics.AUPRC_global_residue.iloc[0]), "AUROC": float(metrics.AUROC_global_residue.iloc[0]),
        "agreement_with_original_run": agreement, "output_dir": str(out),
    }
    save_json(out / "inference_summary.json", summary)
    print(metrics[["threshold_rule", "threshold", "AUPRC_global_residue", "AUROC_global_residue",
                   "F1_positive_global_residue", "MCC_global_residue"]].to_string(index=False))
    if agreement:
        print("Agreement with original run:", agreement)
    return {"summary": summary, "residues": res, "proteins": prot, "metrics": metrics}
