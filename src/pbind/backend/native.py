"""Native backend: the subset of GEAR ``gear_ultimate`` that the final models need, implemented in pbind.

Exposes the same interface the pipelines call on the Drive backend (``ws.gear.core / suite / reporting``),
so nothing else changes. Feature caches use the original file names under ``cache/spatial_revamp/``,
so caches computed by either backend are interchangeable.

Not ported (use ``open_workspace(..., backend="gear")`` with the Drive package): EVO/MSA, graph
experiments, historical KAN protocol, the old GEAR_ULTIMATE grids.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from .. import __version__
from ..constants import MECH1, MULTI2
from ..data.folds import find_fold_dir, load_official_folds
from ..data.loading import backend_frames
from ..features import coordination as CO
from ..features import store as FS
from ..models.registry import get_model
from ..paths import resolve_paths
from ..training import gear_mlp as T
from ..utils.cache import load_pickle_dict, save_pickle

SPLITS = (("train", "DevSet1010"), ("test300", "TestSet300"), ("test46", "TestSetNew46"))


@dataclass
class Context:
    cfg: T.GearConfig
    paths: dict
    frames: dict
    fold_ids: list
    stores: dict
    device: torch.device


# ------------------------------------------------------------------ paths / ids
def make_paths(cfg: T.GearConfig) -> dict:
    base = Path(cfg.base)
    run_root = base / "results" / cfg.run_version / cfg.label_column
    out = {"base": base, "cache": base / "cache" / "spatial_revamp", "run_root": run_root,
           "model_root": base / "trained_models" / cfg.run_version / cfg.label_column,
           "pred_root": run_root / "fold_predictions", "ensemble_root": run_root / "ensemble_predictions"}
    for k in ("cache", "run_root", "model_root", "pred_root", "ensemble_root"):
        out[k].mkdir(parents=True, exist_ok=True)
    return out


def _safe(rep: str) -> str:
    return rep.replace("+", "__")


def job_pred_path(paths, model_name, rep_name, fold) -> Path:
    return paths["pred_root"] / f"{model_name}__{_safe(rep_name)}__fold{fold + 1}.pkl"


def _is_tree(model_name: str) -> bool:
    return get_model(model_name).kind == "tree"


def _seed(cfg, fold: int, model_name: str) -> int:
    return cfg.seed + fold + 1000 * get_model(model_name).seed_offset


def job_model_path(paths, model_name, rep_name, fold) -> Path:
    ext = ".json" if _is_tree(model_name) else ".pt"
    return paths["model_root"] / f"{model_name}__{_safe(rep_name)}__fold{fold + 1}{ext}"


def load_pickle(path):
    return load_pickle_dict(path)


def config_hash(cfg, representations) -> str:
    payload = {"config": asdict(cfg), "representations": {k: list(v) for k, v in representations.items()},
               "atom_features": CO.ATOM_FEATURE_NAMES, "chem_features": CO.CHEM_FEATURE_NAMES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def get_fold_train_val_ids(fold_idx, fold_ids, train_store):
    va = [p for p in fold_ids[fold_idx] if p in train_store]
    vs = set(va)
    return [p for p in train_store if p not in vs], va


# ------------------------------------------------------------------ context
def _structure_available() -> bool:
    try:
        import mdtraj  # noqa: F401

        return True
    except ImportError:
        return False


def prepare_context(base, run_version="native_context", models=("MLP",), label_column="metal_binding",
                    ensure_base=True, add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=8, device=None,
                    with_structure: bool | None = None, feature_groups=None, require_legacy: bool = False):
    """Fill the feature store once (missing groups only) and build the per-protein stores from its columns."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = resolve_paths(base)
    cfg = T.GearConfig(base=str(paths.base), label_column=label_column, run_version=run_version, models=tuple(models))
    gp = make_paths(cfg)
    folds = load_official_folds(find_fold_dir(paths))
    frames = backend_frames(paths, [p for ids in folds.values() for p in ids])
    if with_structure is None:
        with_structure = _structure_available()
        if not with_structure:
            print("[native] mdtraj not installed -> SPATIAL/LOCAL skipped (not used by the final models)")

    skip = {"STRUCTURE": not with_structure, "MECH1": not add_mech, "MULTI2": not add_multi, "SEQCTX": not add_seqctx}
    # every registered group (incl. user-registered ones) unless the caller restricts them
    groups = list(feature_groups) if feature_groups else [g for g in FS.FEATURE_GROUPS if not skip.get(g, False)]
    fs = FS.build_feature_store(paths, {split: frames[key] for key, split in SPLITS}, groups, cfg.max_residues,
                                device, esm_batch_size, require_legacy=require_legacy)
    blocks = [b for g in groups for b in FS.FEATURE_GROUPS[g].blocks]
    stores = {key: fs.protein_store(split, label_column, blocks, order=list(frames[key].prot_id.astype(str)))
              for key, split in SPLITS}
    print("Store sizes:", {k: len(v) for k, v in stores.items()}, "| feature store:", fs.root)
    return Context(cfg, gp, frames, [list(folds[k]) for k in sorted(folds)], stores, device)


def clone_context_for_run(ctx: Context, run_version: str, models=("MLP",), **overrides) -> Context:
    cfg = replace(ctx.cfg, run_version=run_version, models=tuple(models), **overrides)
    return Context(cfg, make_paths(cfg), ctx.frames, ctx.fold_ids, ctx.stores, ctx.device)


def audit_context(ctx: Context) -> pd.DataFrame:
    rows = []
    for k, store in ctx.stores.items():
        nres = sum(len(x["y"]) for x in store.values())
        pos = sum(int(np.asarray(x["y"]).sum()) for x in store.values())
        rows.append({"split": k, "proteins": len(store), "residues": nres, "positives": pos,
                     "prevalence": pos / max(nres, 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ training jobs
def _train_predict(model_name, store, tr, va, blocks, cfg, seed, device):
    if not _is_tree(model_name):
        model, scaler, meta = T.train_neural(model_name, store, tr, va, blocks, cfg, seed, device)
        batch = getattr(cfg, get_model(model_name).batch_attr)

        def predict(st, ids):
            return T.predict_neural(model, model_name, st, ids, blocks, scaler, device, batch)
    else:
        model, scaler, meta = T.train_xgboost(store, tr, va, blocks, cfg, seed, device)

        def predict(st, ids):
            return T.predict_xgboost(model, st, ids, blocks)
    return model, scaler, meta, predict


def run_dev_cv_only(ctx: Context, representations, models=("MLP",), run_version="native_dev_only"):
    """Official 5-fold Dev CV; each fold resumes from ``dev_only_predictions/<model>__<rep>__fold<k>.pkl``."""
    root = Path(ctx.cfg.base) / "results" / run_version / ctx.cfg.label_column
    pred = root / "dev_only_predictions"
    pred.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in models:
        for rep, blocks in representations.items():
            for fold in range(5):
                pp = pred / f"{model_name}__{_safe(rep)}__fold{fold + 1}.pkl"
                z = load_pickle_dict(pp)
                if z is None:
                    tr, va = get_fold_train_val_ids(fold, ctx.fold_ids, ctx.stores["train"])
                    missing = [p for p in tr + va if any(b not in ctx.stores["train"][p] for b in blocks)]
                    if missing:
                        raise RuntimeError(f"{rep}: {len(missing)} Dev proteins lack blocks; e.g. {missing[:5]}")
                    seed = _seed(ctx.cfg, fold, model_name)
                    _, _, meta, predict = _train_predict(model_name, ctx.stores["train"], tr, va, blocks, ctx.cfg,
                                                         seed, ctx.device)
                    y, p = predict(ctx.stores["train"], va)
                    z = {"model": model_name, "representation": rep, "fold": fold + 1, "y": y, "p": p, "meta": meta}
                    save_pickle(pp, z)
                    tag = "DONE"
                else:
                    tag = "REUSE"
                m = T.binary_metrics(z["y"], z["p"], 0.5)
                rows.append({"model": model_name, "representation": rep, "fold": fold + 1, "AUPRC": m["auprc"],
                             "AUROC": m["auroc"], "F1@0.5": m["f1"], "MCC@0.5": m["mcc"]})
                print(tag, model_name, rep, fold + 1, "AUPRC", round(m["auprc"], 5))
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(root / "fold_metrics.csv", index=False)
    cv = fold_df.groupby(["model", "representation"], as_index=False).agg(
        CV_AUPRC=("AUPRC", "mean"), CV_AUPRC_SD=("AUPRC", "std"), CV_AUROC=("AUROC", "mean"))
    cv.to_csv(root / "cv_summary.csv", index=False)
    return cv, fold_df


def _supersede(files, root: Path, old_hash) -> None:
    """Move results of another config out of the way (``<root>/superseded/<old_hash>/``) instead of overwriting
    them, so e.g. the original run's fold_predictions on Drive are never lost."""
    dest = Path(root) / "superseded" / str(old_hash or "unknown")
    for fp in map(Path, files):
        if fp.exists():
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / fp.name
            if not target.exists():
                fp.replace(target)
                print(f"kept previous result (config {old_hash}) -> {target}")


def run_one_job(ctx: Context, model_name, rep, blocks, fold, run_hash):
    """One fold of the final protocol: train, predict val + both external sets, save model + predictions."""
    pp = job_pred_path(ctx.paths, model_name, rep, fold)
    old = load_pickle_dict(pp)
    if old is not None and old.get("config_hash") == run_hash:
        print(f"SKIP completed: {model_name} | {rep} | fold {fold + 1}")
        return old
    if old is not None:
        _supersede([pp, job_model_path(ctx.paths, model_name, rep, fold)], ctx.paths["run_root"], old.get("config_hash"))
    tr, va = get_fold_train_val_ids(fold, ctx.fold_ids, ctx.stores["train"])
    seed = _seed(ctx.cfg, fold, model_name)
    print(f"RUN {model_name} | {rep} | fold {fold + 1}/5 | train={len(tr)} val={len(va)}")
    t0 = time.time()
    model, scaler, meta, predict = _train_predict(model_name, ctx.stores["train"], tr, va, blocks, ctx.cfg, seed,
                                                  ctx.device)
    yv, pv = predict(ctx.stores["train"], va)
    y300, p300 = predict(ctx.stores["test300"], list(ctx.stores["test300"]))
    y46, p46 = predict(ctx.stores["test46"], list(ctx.stores["test46"]))
    if ctx.cfg.save_models:
        # tmp + os.replace: a symlinked checkpoint (final_model/) is replaced as a link, never written through
        mp = Path(job_model_path(ctx.paths, model_name, rep, fold))
        tmp = mp.with_name(f".tmp.{mp.name}")  # keeps the extension (xgboost picks the format from it)
        if _is_tree(model_name):
            model.save_model(tmp)
        else:
            torch.save({"state_dict": model.state_dict(), "scaler": scaler, "blocks": list(blocks),
                        "config_hash": run_hash, "meta": meta}, tmp)
        tmp.replace(mp)
    z = {"config_hash": run_hash, "model": model_name, "representation": rep, "blocks": list(blocks),
         "fold": fold + 1, "val_ids": va, "val_y": np.asarray(yv, np.int8), "val_p": np.asarray(pv, np.float32),
         "test300_y": np.asarray(y300, np.int8), "test300_p": np.asarray(p300, np.float32),
         "test46_y": np.asarray(y46, np.int8), "test46_p": np.asarray(p46, np.float32),
         "fold_tuned_threshold": T.choose_f1_threshold(yv, pv), "meta": meta, "seconds": time.time() - t0}
    save_pickle(pp, z)
    m = T.binary_metrics(yv, pv, ctx.cfg.fixed_threshold)
    print(f"DONE {model_name} | {rep} | fold {fold + 1}: AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f}")
    return z


def _ci95(values) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return np.nan if len(v) < 2 else float(np.std(v) / np.sqrt(len(v)) * student_t.ppf(0.975, len(v)))


def macro_positive_metrics(store, ids, probs, threshold) -> dict:
    cur, f1s, mccs, false_alarm, included = 0, [], [], 0, 0
    for pid in ids:
        y = np.asarray(store[pid]["y"], np.int64)
        pred = (np.asarray(probs[cur:cur + len(y)]) >= threshold).astype(np.int64)
        cur += len(y)
        tp, fp = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum())
        tn, fn = int(((y == 0) & (pred == 0)).sum()), int(((y == 1) & (pred == 0)).sum())
        if tp + fp + fn == 0:
            continue
        included += 1
        false_alarm += int(y.sum() == 0)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mccs.append((tp * tn - fp * fn) / den if den else 0.0)
    return {"macro_positive_f1": float(np.mean(f1s)) if f1s else np.nan, "macro_positive_f1_ci95": _ci95(f1s),
            "macro_positive_mcc": float(np.mean(mccs)) if mccs else np.nan, "macro_positive_mcc_ci95": _ci95(mccs),
            "n_macro_positive": included, "n_false_alarm_only": false_alarm}


def run_grid(ctx: Context, representations, models=None, comparisons=None):
    """5 official folds per (model, representation), then the 5-fold-mean external ensemble report."""
    models = tuple(models or ctx.cfg.models)
    run_hash = config_hash(replace(ctx.cfg, models=models), representations)
    print("RUN HASH:", run_hash, "| RUN ROOT:", ctx.paths["run_root"])
    root = Path(ctx.paths["run_root"])
    prev = pd.read_csv(root / "fold_metrics.csv") if (root / "fold_metrics.csv").exists() else None
    if prev is not None and "config_hash" not in prev:
        prev = None
    old = sorted(set(prev.config_hash.astype(str)) - {str(run_hash)}) if prev is not None else []
    if old:  # summaries / ensemble tables of another config (e.g. the original run on Drive) are kept, not overwritten
        _supersede([root / "cv_summary.csv", root / "external_summary.csv",
                    *Path(ctx.paths["ensemble_root"]).glob("*.csv.gz")], root, "+".join(old))
    fold_rows, ext_rows = [], []
    for model_name in models:
        if model_name == "KAN" and not T.HAS_KAN:
            print("SKIP KAN: pykan unavailable")
            continue
        for rep, blocks in representations.items():
            jobs = [run_one_job(ctx, model_name, rep, blocks, fold, run_hash) for fold in range(5)]
            for z in jobs:
                m = T.binary_metrics(z["val_y"], z["val_p"], ctx.cfg.fixed_threshold)
                mt = T.binary_metrics(z["val_y"], z["val_p"], z["fold_tuned_threshold"])
                fold_rows.append({"config_hash": run_hash, "model": model_name, "representation": rep,
                                  "fold": z["fold"], "n_val_proteins": len(z["val_ids"]),
                                  "n_val_residues": len(z["val_y"]), "val_positive_residues": int(z["val_y"].sum()),
                                  "AUPRC": m["auprc"], "AUROC": m["auroc"], "F1_at_0.5": m["f1"],
                                  "MCC_at_0.5": m["mcc"], "fold_tuned_threshold": z["fold_tuned_threshold"],
                                  "F1_at_fold_tuned": mt["f1"], "MCC_at_fold_tuned": mt["mcc"],
                                  "best_epoch": z["meta"].get("best_epoch", np.nan),
                                  "best_val_auprc": z["meta"].get("best_val_auprc", np.nan),
                                  "n_params": z["meta"].get("n_params", np.nan), "seconds": z.get("seconds", np.nan)})
            oof_thr = T.choose_f1_threshold(np.concatenate([z["val_y"] for z in jobs]),
                                            np.concatenate([z["val_p"] for z in jobs]))
            fold_thr_mean = float(np.mean([z["fold_tuned_threshold"] for z in jobs]))
            for dname, key in (("TestSet300", "test300"), ("TestSetNew46", "test46")):
                store = ctx.stores[key]
                ids = list(store)
                y = np.asarray(jobs[0][f"{key}_y"], np.int64)
                p = np.mean(np.stack([np.asarray(z[f"{key}_p"], float) for z in jobs]), axis=0)
                m05, moof = T.binary_metrics(y, p, ctx.cfg.fixed_threshold), T.binary_metrics(y, p, oof_thr)
                mmean = T.binary_metrics(y, p, fold_thr_mean)
                ext_rows.append({"model": model_name, "representation": rep, "dataset": dname, "n_residues": len(y),
                                 "positive_residues": int(y.sum()), "AUPRC": m05["auprc"], "AUROC": m05["auroc"],
                                 "F1_at_0.5": m05["f1"], "MCC_at_0.5": m05["mcc"],
                                 "precision_at_0.5": m05["precision"], "recall_at_0.5": m05["recall"],
                                 "OOF_tuned_threshold": oof_thr, "F1_at_OOF_threshold": moof["f1"],
                                 "MCC_at_OOF_threshold": moof["mcc"], "precision_at_OOF_threshold": moof["precision"],
                                 "recall_at_OOF_threshold": moof["recall"], "mean_fold_tuned_threshold": fold_thr_mean,
                                 "F1_at_mean_fold_threshold": mmean["f1"], "MCC_at_mean_fold_threshold": mmean["mcc"],
                                 **{f"at_0.5_{k}": v for k, v in macro_positive_metrics(store, ids, p, 0.5).items()},
                                 **{f"at_OOF_{k}": v for k, v in macro_positive_metrics(store, ids, p, oof_thr).items()}})
                rows, cur = [], 0
                for pid in ids:
                    n = len(store[pid]["y"])
                    seq = store[pid]["sequence"]
                    rows += [(pid, j, seq[j], int(y[cur + j]), float(p[cur + j])) for j in range(n)]
                    cur += n
                pd.DataFrame(rows, columns=["prot_id", "residue_index", "AA", "label", "probability"]).to_csv(
                    ctx.paths["ensemble_root"] / f"{model_name}__{_safe(rep)}__{dname}.csv.gz", index=False,
                    compression="gzip")
    fold_df = pd.DataFrame(fold_rows)
    if prev is not None:  # keep other configs' rows (original GEAR append semantics)
        fold_df = pd.concat([prev[prev.config_hash.astype(str) != str(run_hash)], fold_df], ignore_index=True)
    fold_df.to_csv(root / "fold_metrics.csv", index=False)
    fold_df = fold_df[fold_df.config_hash.astype(str) == str(run_hash)]
    cv = fold_df.groupby(["model", "representation"], as_index=False).agg(
        folds=("fold", "nunique"), CV_AUPRC=("AUPRC", "mean"), CV_AUPRC_SD=("AUPRC", "std"),
        CV_AUROC=("AUROC", "mean"), CV_F1_at_0_5=("F1_at_0.5", "mean"), CV_MCC_at_0_5=("MCC_at_0.5", "mean"),
        mean_best_epoch=("best_epoch", "mean"), n_params=("n_params", "mean"), mean_seconds=("seconds", "mean"))
    cv.to_csv(ctx.paths["run_root"] / "cv_summary.csv", index=False)
    ext = pd.DataFrame(ext_rows)
    ext.to_csv(ctx.paths["run_root"] / "external_summary.csv", index=False)
    print("\n=== CV SUMMARY (selection should be based here) ===\n" + cv.to_string(index=False))
    if len(ext):
        print("\n=== EXTERNAL SUMMARY ===\n" + ext[["model", "representation", "dataset", "AUPRC", "AUROC", "F1_at_0.5",
                                                  "MCC_at_0.5", "OOF_tuned_threshold", "F1_at_OOF_threshold",
                                                  "MCC_at_OOF_threshold"]].to_string(index=False))
    return cv, ext, pd.DataFrame()


# ------------------------------------------------------------------ reporting / stubs
def dimension_table(store) -> pd.DataFrame:
    rec = next(iter(store.values()))
    return pd.DataFrame([{"block": k, "dim": v.shape[1]} for k, v in rec.items()
                         if isinstance(v, np.ndarray) and v.ndim == 2]).sort_values("block")


def validate_no_direct_label_features(stores, blocks) -> pd.DataFrame:
    rows = []
    for split, store in stores.items():
        for b in blocks:
            have = [np.asarray(item[b], np.float32) for item in store.values() if b in item]
            rows.append({"split": split, "block": b, "missing_proteins": len(store) - len(have),
                         "mean_finite_fraction": float(np.mean([np.isfinite(x).mean() for x in have])) if have else 0.0})
    return pd.DataFrame(rows)


def _evo_unavailable(*a, **k):
    raise RuntimeError("EVO48 is not ported; use open_workspace(..., backend='gear') with the Drive package")


def load_native() -> SimpleNamespace:
    core = SimpleNamespace(
        HAS_KAN=T.HAS_KAN, Config=T.GearConfig, make_paths=make_paths, job_pred_path=job_pred_path,
        job_model_path=job_model_path, load_pickle=load_pickle, config_hash=config_hash,
        get_fold_train_val_ids=get_fold_train_val_ids, choose_f1_threshold=T.choose_f1_threshold,
        binary_metrics=T.binary_metrics, ResidueMLP=T.ResidueMLP, ResidueKAN=T.ResidueKAN,
        train_neural=T.train_neural, predict_neural=T.predict_neural, train_xgboost=T.train_xgboost,
        predict_xgboost=T.predict_xgboost)
    suite = SimpleNamespace(
        prepare_context=prepare_context, clone_context_for_run=clone_context_for_run, audit_context=audit_context,
        run_dev_cv_only=run_dev_cv_only, run_grid=run_grid, MECH1=list(MECH1), MULTI2=list(MULTI2))
    reporting = SimpleNamespace(dimension_table=dimension_table,
                                validate_no_direct_label_features=validate_no_direct_label_features)
    evo = SimpleNamespace(attach_evo=_evo_unavailable, run_msa_batches=_evo_unavailable)
    return SimpleNamespace(kind="native", dir=None, package=None, version=f"pbind-native-{__version__}",
                           core=core, suite=suite, reporting=reporting, evo=evo, evolution=evo,
                           graph=None, legacy=None, structure=None, mechanistic=None, mech=None, has_kan=T.HAS_KAN,
                           has_xgb=T.HAS_XGB)

