"""One-call experiment API: pick the task, the features and the model by name.

    from pbind import api

    ws = api.workspace("metal")                    # data, official folds, feature store -> per-protein stores
    api.list_features()                            # registered feature blocks (group, dim, needs)
    api.list_models()                              # registered models (kind, description)

    r = api.cross_validate(ws, model="MLP", features="ESM+Atomic+Chem+MULTI2")         # DevSet1010 5-fold only
    r.dev["CV_AUPRC"], r.folds
    f = api.train_final(ws, model="MLP", features="ESM+Atomic+Chem+MULTI2")            # 5 fold weights + 300/46
    p = api.predict(ws, model="MLP", features="ESM+Atomic+Chem+MULTI2", split="TestSet300")   # residue table

    ws = api.workspace("nuclear")
    api.cross_validate(ws, model="ConcatMLP", features="ESM+ACCESS+NA_PATCH3D40+NA_TOPO12")
    api.cross_validate(ws, model="NA_PatchHG_Attention", features="ESM+ACCESS+NA_PATCH3D40+NA_TOPO12", bank="v1")

    @api.register_model("WideMLP", kind="residue")           # a new model is one function
    def wide_mlp(d, cfg, seed): ...
    api.cross_validate(ws, model="WideMLP", features="ESM+Atomic+Chem", neural_epochs=20)

features  "A+B+C" string or list; tokens = blocks, feature-group names (MULTI2, MECH1, ...) or notebook aliases
          (Atomic, Chem, Structure, SeqCtx). Task blocks (NA_*, SM_*) are built and attached on demand.
model     any name in ``pbind.models.registry`` or a task alias (NA_PatchHG_Attention, PatchResidualFusion, ...).
**train   training overrides: GearConfig fields for residue/tree models (neural_epochs, lr, dropout, ...),
          TrainConfig fields for fusion/group models (epochs, lr, batch, ...). Overrides get their own run folder.

Nothing here re-implements training: every call routes to the existing runner of the model's kind (GEAR residue
trainer / local fusion CV / group hypergraph CV), so caches, seeds, fold checkpoints and file names are those of the
notebook pipelines. Selection stays Dev-only: ``cross_validate`` touches DevSet1010 only; the external sets are
used by ``train_final`` / ``predict``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import DEV, EXTERNAL_SPLITS, STORE_KEY
from .data.loading import concat_labels, n_residues
from .evaluation.metrics import binary_metrics
from .evaluation.thresholds import best_mcc_threshold
from .features.api import feature_group, list_features, resolve_blocks
from .features.registry import FEATURE_FAMILIES
from .features.store import FeatureStore
from .models.registry import ModelSpec, list_models, register_model, resolve_model
from .pipelines.features import build_features, group_bank
from .pipelines.workspace import Workspace, open_workspace
from .reporting.predictions import residue_prediction_frame
from .training.datasets import GroupProteinDataset, make_group_collate
from .training.group_cv import predict_group_ensemble, run_group_cv
from .training.local_cv import predict_local_ensemble, run_local_cv
from .training.loop import TrainConfig
from .training.modalities import build_packs, modality_dims, modality_spec
from .utils.cache import PredictionCache, save_table, states_fingerprint

__all__ = ["Experiment", "workspace", "feature_store", "list_features", "list_models", "register_model",
           "feature_group", "cross_validate", "train_final", "predict"]

# fusion models on metal (no TaskSpec): the nuclear/small local trainer settings
DEFAULT_LOCAL_TRAIN = TrainConfig(batch=4096, epochs=45, patience=7, lr=3e-4, weight_decay=1e-4)


@dataclass
class Experiment:
    """What one api call trained / evaluated."""
    task: str
    model: str
    kind: str
    name: str  # representation name, used in file names
    blocks: list
    dev: dict  # Dev 5-fold summary; always has "CV_AUPRC"
    folds: pd.DataFrame  # per-fold Dev metrics
    external: pd.DataFrame | None = None  # train_final: TestSet300 / TestSetNew46 metrics
    oof: np.ndarray | None = None  # fusion/group: Dev out-of-fold probabilities (ws.dev residue order)
    artifacts: dict = field(default_factory=dict)  # folders of weights / predictions (+ in-memory fold states)


# ------------------------------------------------------------------ entry points
def workspace(task: str = "metal", **kw) -> Workspace:
    """``open_workspace`` with EVO/ESM-IF off by default (enable with attach_evo=True / attach_esmif=True)."""
    kw.setdefault("attach_evo", False)
    kw.setdefault("attach_esmif", False)
    return open_workspace(task, **kw)


def feature_store(ws: Workspace) -> FeatureStore:
    """Column access to every computed block: ``.matrix(split, blocks)``, ``.frame(...)``, ``.labels(...)``."""
    return FeatureStore(ws.paths.feature_store)


def cross_validate(ws: Workspace, model: str, features, *, name: str | None = None, run_version: str | None = None,
                   bank: str = "v1", modalities=None, force: bool = False, **train) -> Experiment:
    """Official DevSet1010 5-fold CV of ``model`` on ``features`` (fold results are cached and reused)."""
    spec, blocks, rep = _resolve(ws, model, features, name)
    if spec.kind in ("residue", "tree"):
        return _residue_cv(ws, spec, blocks, rep, run_version, train)
    return _torch_cv(ws, spec, model, blocks, rep, bank, modalities, force, train)


def train_final(ws: Workspace, model: str, features, *, name: str | None = None, run_version: str | None = None,
                bank: str = "v1", modalities=None, force: bool = False, **train) -> Experiment:
    """Five official-fold models (weights kept), their mean evaluated once on TestSet300 / TestSetNew46 at 0.5 and
    at the Dev-OOF threshold (residue/tree: F1-optimal, as GEAR; fusion/group: MCC-optimal, as nuclear/small)."""
    spec, blocks, rep = _resolve(ws, model, features, name)
    if spec.kind in ("residue", "tree"):
        _require_backend_model(ws, spec)
        rv = _run_version(ws, "final", train, run_version)
        over = {"save_models": True, **train} if ws.gear.kind == "gear" else dict(train)
        run_ctx = ws.gear.suite.clone_context_for_run(ws.ctx, rv, models=(spec.name,), **over)
        cv, ext, _ = ws.gear.suite.run_grid(run_ctx, {rep: blocks}, models=(spec.name,))
        run_root = Path(run_ctx.paths["run_root"])
        return Experiment(_task(ws), spec.name, spec.kind, rep, blocks, cv.iloc[0].to_dict(),
                          pd.read_csv(run_root / "fold_metrics.csv"), external=ext,
                          artifacts={"run_version": rv, "run_root": run_root,
                                     "weights": Path(run_ctx.paths["model_root"])})
    exp = _torch_cv(ws, spec, model, blocks, rep, bank, modalities, force, train)
    thr, _ = best_mcc_threshold(concat_labels(ws.dev, ws.label, int), exp.oof)
    rows = []
    for split in EXTERNAL_SPLITS:
        y = concat_labels(ws.splits[split], ws.label, int)
        p = _ensemble_predict(ws, exp, split)
        row = {"model": model, "representation": rep, "dataset": split, "Dev_CV_AUPRC": exp.dev["CV_AUPRC"],
               "Dev_OOF_MCC_threshold": thr}
        row.update({f"{k}_fixed0.5": v for k, v in binary_metrics(y, p, 0.5).items()})
        row.update({f"{k}_OOFthr": v for k, v in binary_metrics(y, p, thr).items()})
        rows.append(row)
    exp.external = pd.DataFrame(rows)
    save_table(exp.artifacts["state_dir"] / "external_metrics.csv", exp.external)
    return exp


def predict(ws: Workspace, model: str, features, split: str = "TestSet300", *, name: str | None = None,
            run_version: str | None = None, bank: str = "v1", modalities=None, **train) -> pd.DataFrame:
    """Residue table (prot_id, residue_index, y_true, probability) of the 5-fold ensemble on an external split.
    residue/tree models load the weights saved by ``train_final``; fusion/group models reuse their CV fold states."""
    if split not in EXTERNAL_SPLITS:
        raise ValueError(f"split must be one of {EXTERNAL_SPLITS} (Dev predictions are the CV out-of-fold ones)")
    spec, blocks, rep = _resolve(ws, model, features, name)
    if spec.kind in ("residue", "tree"):
        p = _residue_predict(ws, spec, blocks, rep, split, _run_version(ws, "final", train, run_version), train)
    else:
        p = _ensemble_predict(ws, _torch_cv(ws, spec, model, blocks, rep, bank, modalities, False, train), split)
    return residue_prediction_frame(ws.splits[split], ws.label, probability=p)


# ------------------------------------------------------------------ resolution
def _task(ws: Workspace) -> str:
    return ws.task.name if ws.task is not None else "metal"


def _safe(rep: str) -> str:
    return rep.replace("+", "__")


def _resolve(ws: Workspace, model: str, features, name: str | None) -> tuple[ModelSpec, list, str]:
    spec = resolve_model(model, ws.task)
    blocks = resolve_blocks(features)
    _ensure_blocks(ws, blocks)
    return spec, blocks, name or (features if isinstance(features, str) else "+".join(blocks))


def _ensure_blocks(ws: Workspace, blocks: list) -> None:
    """Attach task feature families (NA_*, SM_*) that ``blocks`` need; fail clearly on anything else."""
    ws.require_backend()

    def missing():
        item = next(iter(ws.ctx.stores["train"].values()))
        return [b for b in blocks if b not in item]

    need = missing()
    families = [f.name for f in FEATURE_FAMILIES.values() if set(need) & set(f.dims)]
    if families:
        build_features(ws, families=families)
        need = missing()
    if need:
        raise KeyError(f"blocks {need} are not available; see api.list_features() and the task feature families")


def _run_version(ws: Workspace, stage: str, train: dict, given: str | None) -> str:
    if given:
        return given
    tag = "_" + hashlib.sha1(json.dumps(train, sort_keys=True, default=str).encode()).hexdigest()[:8] if train else ""
    return f"api_{_task(ws)}_{stage}{tag}"


def _require_backend_model(ws: Workspace, spec: ModelSpec) -> None:
    if ws.gear.kind == "gear" and spec.name not in ("MLP", "KAN", "XGBoost"):
        raise RuntimeError(f"{spec.name}: registered models need the native backend (open_workspace(backend='native'))")


# ------------------------------------------------------------------ residue / tree models (GEAR trainer)
def _residue_cv(ws, spec, blocks, rep, run_version, train) -> Experiment:
    _require_backend_model(ws, spec)
    rv = _run_version(ws, "cv", train, run_version)
    ctx = ws.gear.suite.clone_context_for_run(ws.ctx, rv, models=(spec.name,), **train) if train else ws.ctx
    cv, folds = ws.gear.suite.run_dev_cv_only(ctx, {rep: blocks}, models=(spec.name,), run_version=rv)
    return Experiment(_task(ws), spec.name, spec.kind, rep, blocks, cv.iloc[0].to_dict(), folds,
                      artifacts={"run_version": rv, "run_root": ws.paths.results / rv / ws.label})


def _residue_predict(ws, spec, blocks, rep, split, rv, train) -> np.ndarray:
    core = ws.gear.core
    run_ctx = ws.gear.suite.clone_context_for_run(ws.ctx, rv, models=(spec.name,), **train)
    store = ws.ctx.stores[STORE_KEY[split]]
    ids = list(store)
    if ids != list(ws.splits[split].prot_id.astype(str)):
        raise RuntimeError(f"{split}: store order differs from the split frame")
    d = sum(np.asarray(store[ids[0]][b]).shape[1] for b in blocks)
    preds = []
    for fold in range(5):
        path = Path(core.job_model_path(run_ctx.paths, spec.name, rep, fold))
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run api.train_final(ws, {spec.name!r}, {rep!r}) first")
        if spec.kind == "tree":
            import xgboost as xgb

            booster = xgb.XGBClassifier()
            booster.load_model(path)
            preds.append(core.predict_xgboost(booster, store, ids, blocks)[1])
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)  # numpy scaler inside
        if list(ck["blocks"]) != list(blocks):
            raise RuntimeError(f"{path}: checkpoint blocks {ck['blocks']} != {blocks}")
        net = spec.build(d, run_ctx.cfg, run_ctx.cfg.seed + fold + 1000 * spec.seed_offset).to(ws.device)
        net.load_state_dict(ck["state_dict"])
        preds.append(core.predict_neural(net, spec.name, store, ids, blocks, ck["scaler"], ws.device,
                                         getattr(run_ctx.cfg, spec.batch_attr))[1])
    return np.mean(np.stack(preds), axis=0)


# ------------------------------------------------------------------ fusion / group models (torch CV trainers)
def _packs(ws: Workspace, mspec: dict) -> dict:
    key = ("api_packs", tuple((k, tuple(v)) for k, v in mspec.items()))
    if key not in ws.state:
        ws.state[key] = build_packs(ws.ctx, ws.splits, mspec, ws.label)
    return ws.state[key]


def _torch_cv(ws, spec, model, blocks, rep, bank, modalities, force, train) -> Experiment:
    if ws.task is None and (modalities is None or spec.kind == "group"):
        raise ValueError(f"{spec.kind} models need a nuclear/small workspace (or modalities=[(name, blocks), ...] "
                         "for fusion models on metal)")
    mspec = modality_spec(blocks, modalities or ws.task.modality_groups)
    dims = modality_dims(ws.ctx, DEV, mspec, str(ws.dev.iloc[0].prot_id))
    folder = _safe(rep) + ("_" + hashlib.sha1(json.dumps(train, sort_keys=True).encode()).hexdigest()[:8] if train else "")
    if spec.kind == "fusion":
        cfg = (ws.task.local_train if ws.task else DEFAULT_LOCAL_TRAIN).with_(**train)

        def factory(d):
            return spec.build(d, ws.task)

        root = ws.run_dir("api", "fusion", folder)
        res = run_local_cv(model, factory, dims, _packs(ws, mspec)[DEV], ws.folds, root, rep, blocks, cfg, ws.device,
                           force)
        art = {"state_dir": root / model, "factory": factory}
    else:
        gb = group_bank(ws, bank)
        collate = make_group_collate(gb.feat_dim)
        cfg = ws.task.group_train.with_(**train)

        def make_model():
            return spec.build(dims, gb.feat_dim, ws.task)

        state_dir = ws.run_dir("api", "group", folder, f"{model}__{bank}")
        res = run_group_cv(model, make_model, lambda df: GroupProteinDataset(ws.ctx, DEV, df, mspec, gb, ws.label),
                           collate, ws.dev, ws.folds, ws.label, state_dir,
                           {"representation": rep, "blocks": blocks, "model": model, "bank": bank}, cfg,
                           device=ws.device, force=force)
        art = {"state_dir": state_dir, "make_model": make_model, "bank": gb, "collate": collate}
    art.update(result=res, dims=dims, modalities=mspec, cfg=cfg)
    return Experiment(_task(ws), model, spec.kind, rep, blocks, {**res.summary, "CV_AUPRC": res.summary["CV_AUPRC_mean"]},
                      res.folds, oof=res.oof, artifacts=art)


def _ensemble_predict(ws: Workspace, exp: Experiment, split: str) -> np.ndarray:
    a = exp.artifacts
    cache = PredictionCache(a["state_dir"] / "prediction_cache")
    if exp.kind == "fusion":
        pack = _packs(ws, a["modalities"])[split]
        n = len(pack["y"])

        def fn():
            return predict_local_ensemble(a["factory"], a["dims"], a["result"].states, pack, ws.device)
    else:
        ds = GroupProteinDataset(ws.ctx, split, ws.splits[split], a["modalities"], a["bank"], ws.label)
        n = n_residues(ws.splits[split])

        def fn():
            return predict_group_ensemble(a["make_model"], a["result"].states, ds, a["collate"], a["cfg"].batch,
                                          ws.device)
    return cache.get_or_compute(exp.model, split, n, fn, False, states_fingerprint(a["state_dir"]))
