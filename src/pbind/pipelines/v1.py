"""Stages after the representation lock (nuclear / small V1):

modalities -> fusion architecture screen -> patch/pocket hypergraph screen -> global Dev lock
-> Dev-OOF thresholds -> external evaluation -> baselines / insight / manifest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DEV, EXTERNAL_SPLITS, MIN_DEV_GAIN
from ..data.loading import concat_labels, n_residues
from ..evaluation.metrics import binary_metrics
from ..evaluation.selection import complexity_lock, lock_if_gain, verdict
from ..evaluation.thresholds import best_mcc_threshold
from ..reporting.baselines import baseline_view, load_baselines, prior_gear_controls
from ..reporting.predictions import save_residue_predictions, write_manifest
from ..training.datasets import GroupProteinDataset, make_group_collate
from ..training.group_cv import predict_group_ensemble, run_group_cv
from ..training.local_cv import predict_local_ensemble, run_local_cv
from ..training.modalities import build_packs, modality_dims, modality_spec
from ..utils.cache import PredictionCache, save_json, save_table, states_fingerprint
from .features import group_bank
from .screens import representation
from .workspace import Workspace


# ------------------------------------------------------------------ modalities / packs
def modalities(ws: Workspace) -> tuple[dict, dict]:
    """(modality_spec, dims) of the locked representation."""
    if "modalities" not in ws.state:
        rep = representation(ws)
        spec = modality_spec(rep["blocks"], ws.task.modality_groups)
        dims = modality_dims(ws.ctx, DEV, spec, str(ws.dev.iloc[0].prot_id))
        print("MODALITY_SPEC =", spec, "| dims =", dims)
        ws.state["modalities"] = (spec, dims)
    return ws.state["modalities"]


def packs(ws: Workspace) -> dict:
    if "packs" not in ws.state:
        spec, _ = modalities(ws)
        ws.state["packs"] = build_packs(ws.ctx, ws.splits, spec, ws.label)
    return ws.state["packs"]


# ------------------------------------------------------------------ fusion architectures
def architecture_screen(ws: Workspace, archs=None, force: bool = False) -> dict:
    """Matched multimodal fusion screen; ConcatMLP is kept unless another wins by >= 0.002."""
    rep = representation(ws)
    spec, dims = modalities(ws)
    registry = ws.task.fusion_registry(dims)
    archs = list(archs or registry)
    dev_pack = packs(ws)[DEV]
    results = {}
    for arch in archs:
        print("\n" + "=" * 80 + f"\nARCH: {arch}\n" + "=" * 80)
        results[arch] = run_local_cv(arch, registry[arch], dims, dev_pack, ws.folds, ws.run_dir("custom_states"),
                                     rep["representation"], rep["blocks"], ws.task.local_train, ws.device, force)
    summary = (pd.DataFrame([r.summary for r in results.values()])
               .sort_values("CV_AUPRC_mean", ascending=False).reset_index(drop=True))
    save_table(ws.run_dir() / "architecture_cv.csv", summary)
    save_table(ws.run_dir() / "architecture_folds.csv", pd.concat([r.folds for r in results.values()]))
    concat = results["ConcatMLP"].score
    best = str(summary.iloc[0].architecture)
    locked = lock_if_gain(best, results[best].score, "ConcatMLP", concat)
    lock = {"architecture": locked, "dev_cv_auprc": results[locked].score, "concat_cv_auprc": concat,
            "best_raw_architecture": best, "best_raw_cv_auprc": results[best].score}
    save_json(ws.run_dir() / "locked_architecture.json", lock)
    print("LOCKED ARCH:", locked, round(lock["dev_cv_auprc"], 5))
    ws.state["architecture"] = {"summary": summary, "results": results, "lock": lock, "registry": registry}
    return ws.state["architecture"]


# ------------------------------------------------------------------ hypergraph
def _group_dataset(ws, split, df, spec, bank):
    return GroupProteinDataset(ws.ctx, split, df, spec, bank, ws.label)


def hypergraph_screen(ws: Workspace, names=None, force: bool = False) -> dict:
    """V1 patch/pocket hypergraph screen (Gated / Attention / GatedMoE) on the V1 group bank."""
    rep = representation(ws)
    spec, dims = modalities(ws)
    bank = group_bank(ws, "v1")
    collate = make_group_collate(bank.feat_dim)
    names = list(names or ws.task.hypergraphs)
    results = {}
    for name in names:
        print("\n" + "=" * 80 + f"\n{ws.task.group_family}: {name}\n" + "=" * 80)
        results[name] = run_group_cv(
            name, lambda n=name: ws.task.make_hypergraph(n, dims, bank.feat_dim),
            lambda df: _group_dataset(ws, DEV, df, spec, bank), collate, ws.dev, ws.folds, ws.label,
            ws.run_dir(ws.task.group_state_dir, name),
            {"representation": rep["representation"], "blocks": rep["blocks"]},
            ws.task.group_train, device=ws.device, force=force)
    summary = (pd.DataFrame([r.summary for r in results.values()])
               .sort_values("CV_AUPRC_mean", ascending=False).reset_index(drop=True))
    g = ws.task.group
    save_table(ws.run_dir() / f"{g}_hypergraph_cv.csv", summary)
    save_table(ws.run_dir() / f"{g}_hypergraph_folds.csv", pd.concat([r.folds for r in results.values()]))
    best = str(summary.iloc[0].architecture)
    alphas = [float(1 / (1 + np.exp(-float(s["alpha_logit"])))) for _, s in sorted(results[best].states.items())
              if "alpha_logit" in s]
    lock = {"architecture": best, "dev_cv_auprc": results[best].score, "alpha_by_fold": alphas,
            "alpha_mean": float(np.mean(alphas)) if alphas else None}
    save_json(ws.run_dir() / f"locked_{g}_model.json", lock)
    print(f"BEST {g.upper()} MODEL:", best, round(results[best].score, 5), "| mean alpha:", lock["alpha_mean"])
    ws.state["hypergraph"] = {"summary": summary, "results": results, "lock": lock, "bank": bank,
                              "collate": collate}
    return ws.state["hypergraph"]


def _need(ws, key, fn):
    if key not in ws.state:
        fn(ws)
    return ws.state[key]


# ------------------------------------------------------------------ global lock / external
def global_lock(ws: Workspace, margin: float = MIN_DEV_GAIN) -> dict:
    """Simplest family wins unless the more complex one beats it by >= margin on Dev."""
    arch = _need(ws, "architecture", architecture_screen)["lock"]
    hg = _need(ws, "hypergraph", hypergraph_screen)["lock"]
    cands = [("LOCAL_MULTIMODAL", arch["architecture"], arch["dev_cv_auprc"]),
             (ws.task.group_family, hg["architecture"], hg["dev_cv_auprc"])]
    (family, model), score = complexity_lock([((f, m), s) for f, m, s in cands], margin)
    lock = {"family": family, "model": model, "dev_cv_auprc": score, "complexity_margin": margin}
    save_json(ws.run_dir() / "GLOBAL_LOCK.json", lock)
    print("GLOBAL LOCK:", family, model, round(score, 5))
    ws.state["global_lock"] = lock
    return lock


def predict_local(ws: Workspace, split: str, force: bool = False) -> np.ndarray:
    arch = _need(ws, "architecture", architecture_screen)
    name = arch["lock"]["architecture"]
    _, dims = modalities(ws)
    pack = packs(ws)[split]
    cache = PredictionCache(ws.run_dir("prediction_cache"))
    tag = states_fingerprint(ws.run_dir("custom_states", name))
    return cache.get_or_compute(
        f"local_{name}", split, len(pack["y"]),
        lambda: predict_local_ensemble(arch["registry"][name], dims, arch["results"][name].states, pack, ws.device),
        force, tag)


def predict_hypergraph(ws: Workspace, split: str, name: str | None = None, force: bool = False) -> np.ndarray:
    hg = _need(ws, "hypergraph", hypergraph_screen)
    name = name or hg["lock"]["architecture"]
    spec, dims = modalities(ws)
    bank = hg["bank"]
    ds = _group_dataset(ws, split, ws.splits[split], spec, bank)
    cache = PredictionCache(ws.run_dir("prediction_cache"))
    tag = states_fingerprint(ws.run_dir(ws.task.group_state_dir, name))
    return cache.get_or_compute(
        name, split, n_residues(ws.splits[split]),
        lambda: predict_group_ensemble(lambda: ws.task.make_hypergraph(name, dims, bank.feat_dim),
                                       hg["results"][name].states, ds, hg["collate"],
                                       ws.task.group_train.batch, ws.device),
        force, tag)


def external_evaluation(ws: Workspace) -> pd.DataFrame:
    """Both locked family winners on TestSet300 / TestSetNew46 at 0.5 and at the Dev-OOF MCC threshold."""
    arch = _need(ws, "architecture", architecture_screen)
    hg = _need(ws, "hypergraph", hypergraph_screen)
    glock = _need(ws, "global_lock", global_lock)
    y_dev = concat_labels(ws.dev, ws.label, int)
    a_name, h_name = arch["lock"]["architecture"], hg["lock"]["architecture"]
    local_thr, local_mcc = best_mcc_threshold(y_dev, arch["results"][a_name].oof)
    hg_thr, hg_mcc = best_mcc_threshold(y_dev, hg["results"][h_name].oof)
    print("Local threshold:", local_thr, "OOF MCC:", local_mcc, "|", ws.task.group, "threshold:", hg_thr, "OOF MCC:", hg_mcc)
    rows = []
    for split in EXTERNAL_SPLITS:
        y = concat_labels(ws.splits[split], ws.label, int)
        lp, gp = predict_local(ws, split), predict_hypergraph(ws, split)
        for family, model, p, thr, dev in (
                ("LOCAL_MULTIMODAL", a_name, lp, local_thr, arch["lock"]["dev_cv_auprc"]),
                (ws.task.group_family, h_name, gp, hg_thr, hg["lock"]["dev_cv_auprc"])):
            row = {"family": family, "model": model, "dataset": split, "Dev_CV_AUPRC": dev,
                   "Dev_OOF_MCC_threshold": thr}
            row.update({f"{k}_fixed0.5": v for k, v in binary_metrics(y, p, 0.5).items()})
            row.update({f"{k}_OOFthr": v for k, v in binary_metrics(y, p, thr).items()})
            rows.append(row)
        save_residue_predictions(ws.run_dir() / f"predictions_{split}.csv.gz", ws.splits[split], ws.label,
                                 local_probability=lp, **{f"{ws.task.group}_probability": gp})
    ext = pd.DataFrame(rows)
    save_table(ws.run_dir() / "external_family_winners.csv", ext)
    headline = ext[ext.family.eq(glock["family"]) & ext.model.eq(glock["model"])].copy()
    save_table(ws.run_dir() / "FINAL_HEADLINE_EXTERNAL.csv", headline)
    ws.state["external"] = {"table": ext, "headline": headline,
                            "thresholds": {"local": local_thr, ws.task.group: hg_thr}}
    return ext


# ------------------------------------------------------------------ reporting
def baselines(ws: Workspace) -> dict:
    base = load_baselines(ws.paths, ws.label)
    if len(base):
        save_table(ws.run_dir() / f"locked_{ws.task.name}_baselines.csv", base)
    prior = prior_gear_controls(ws.paths, ws.label, ws.task.prior_runs)
    if len(prior):
        save_table(ws.run_dir() / f"prior_{ws.task.name}_cv_controls.csv", prior)
    return {"baselines": baseline_view(base), "prior": prior}


def insight_summary(ws: Workspace) -> dict:
    """Gain decomposition: ESM -> representation -> architecture -> hypergraph, with verdicts."""
    mech = _need(ws, "mechanism", lambda w: representation(w))
    rep = representation(ws)
    arch = _need(ws, "architecture", architecture_screen)["lock"]
    hg = _need(ws, "hypergraph", hypergraph_screen)["lock"]
    glock = _need(ws, "global_lock", global_lock)
    base = float(mech.loc[mech.representation.eq("ESM"), "CV_AUPRC"].iloc[0])
    gains = mech[["representation", "CV_AUPRC"]].rename(columns={"CV_AUPRC": "Dev_CV_AUPRC"})
    gains["delta_vs_ESM"] = gains.Dev_CV_AUPRC - base
    save_table(ws.run_dir() / "INSIGHT_mechanism_gain.csv", gains)
    arch_gain = arch["dev_cv_auprc"] - arch["concat_cv_auprc"]
    hg_gain = hg["dev_cv_auprc"] - arch["dev_cv_auprc"]
    summary = {
        "ESM_baseline": base, "best_representation": rep["representation"],
        "best_representation_dev_auprc": rep["dev_cv_auprc"],
        "best_representation_gain_vs_ESM": rep["dev_cv_auprc"] - base,
        "locked_multimodal_architecture": arch["architecture"], "locked_multimodal_dev_auprc": arch["dev_cv_auprc"],
        "architecture_gain_vs_concat": arch_gain, "architecture_verdict": verdict(arch_gain),
        f"best_{ws.task.group}_hypergraph": hg["architecture"], f"best_{ws.task.group}_dev_auprc": hg["dev_cv_auprc"],
        f"{ws.task.group}_gain_vs_locked_local": hg_gain, f"{ws.task.group}_verdict": verdict(hg_gain),
        f"{ws.task.group}_alpha_mean": hg["alpha_mean"],
        "global_family": glock["family"], "global_model": glock["model"], "global_dev_auprc": glock["dev_cv_auprc"],
        "EVO_available": "EVO" in ws.available, "ESMIF_available": "ESMIF" in ws.available,
    }
    save_json(ws.run_dir() / "INSIGHT_summary.json", summary)
    return summary


def manifest(ws: Workspace) -> dict:
    rep = representation(ws)
    glock = ws.state.get("global_lock", {})
    return write_manifest(
        ws.run_dir() / "FINAL_MANIFEST.json", gear_version=ws.gear.version, device=str(ws.device),
        task=ws.label, development_set="DevSet1010 official bindNode24 5-fold split",
        external_sets=list(EXTERNAL_SPLITS), selection_metric="residue-level Dev CV AUPRC",
        locked_representation=rep, global_lock=glock, optional_blocks=sorted(ws.available))


def run_all(ws: Workspace) -> dict:
    """The whole V1 notebook, in order. Every step resumes from cache."""
    from .screens import family_screen, mechanism_ladder

    family_screen(ws)
    mechanism_ladder(ws)
    architecture_screen(ws)
    hypergraph_screen(ws)
    global_lock(ws)
    external_evaluation(ws)
    out = {"baselines": baselines(ws), "insight": insight_summary(ws)}
    manifest(ws)
    return out
