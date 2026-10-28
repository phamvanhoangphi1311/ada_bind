"""Final stages: V1 attention hypergraph on the multi-scale bank (MS), then the Dev-locked
V1+MS ensemble and the paper-ready report."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import DEV, EXTERNAL_SPLITS
from ..data.loading import concat_labels, n_residues
from ..evaluation.metrics import binary_metrics
from ..evaluation.thresholds import best_mcc_threshold
from ..models.hypergraph import GroupHypergraph
from ..reporting.baselines import baseline_compact_rows, load_baselines
from ..reporting.paper import paper_ready_report
from ..training.datasets import GroupProteinDataset, make_group_collate
from ..training.group_cv import predict_group_ensemble, run_group_cv
from ..utils.cache import PredictionCache, save_table, states_fingerprint
from .features import group_bank
from .screens import representation
from .v1 import _need, external_evaluation, hypergraph_screen, modalities, predict_hypergraph
from .workspace import Workspace


def _ms_root(ws, *parts):
    return ws.paths.results_dir(ws.task.ms_run_root, *parts)


def _make_ms(ws, dims, static_dim):
    return GroupHypergraph(dims, group=ws.task.group, static_dim=static_dim, mode="attention",
                           latent=ws.task.latent, moe=False)


def multiscale_screen(ws: Workspace, force: bool = False):
    """Same V1 attention hypergraph + representation; only the group bank changes (multi-scale)."""
    rep = representation(ws)
    spec, dims = modalities(ws)
    bank = group_bank(ws, "ms")
    collate = make_group_collate(bank.feat_dim)
    name = ws.task.ms_hg
    res = run_group_cv(
        name, lambda: _make_ms(ws, dims, bank.feat_dim),
        lambda df: GroupProteinDataset(ws.ctx, DEV, df, spec, bank, ws.label), collate, ws.dev, ws.folds,
        ws.label, _ms_root(ws, "states", name), {"representation": rep["representation"], "blocks": rep["blocks"]},
        ws.task.group_train, device=ws.device, force=force)
    save_table(_ms_root(ws) / "dev_cv_summary.csv", pd.DataFrame([res.summary]))
    save_table(_ms_root(ws) / "dev_cv_folds.csv", res.folds)
    v1 = _need(ws, "hypergraph", hypergraph_screen)["results"][ws.task.v1_hg]
    print(f"V1 attention Dev AUPRC {v1.score:.6f} | MS {res.score:.6f} | delta {res.score - v1.score:+.6f}")
    ws.state["ms"] = {"result": res, "bank": bank, "collate": collate}
    return res


def _ms_state(ws: Workspace) -> dict:
    if "ms" not in ws.state:
        multiscale_screen(ws)
    return ws.state["ms"]


def predict_ms(ws: Workspace, split: str, force: bool = False):
    st = _ms_state(ws)
    spec, dims = modalities(ws)
    bank = st["bank"]
    ds = GroupProteinDataset(ws.ctx, split, ws.splits[split], spec, bank, ws.label)
    cache = PredictionCache(_ms_root(ws, "prediction_cache"))
    tag = states_fingerprint(_ms_root(ws, "states", ws.task.ms_hg))
    return cache.get_or_compute(
        ws.task.ms_hg, split, n_residues(ws.splits[split]),
        lambda: predict_group_ensemble(lambda: _make_ms(ws, dims, bank.feat_dim), st["result"].states, ds,
                                       st["collate"], ws.task.group_train.batch, ws.device), force, tag)


def multiscale_external(ws: Workspace) -> pd.DataFrame:
    """MS on external sets (0.5 and frozen Dev-OOF MCC threshold) vs V1 headline vs baselines."""
    res = _ms_state(ws)["result"]
    thr, mcc = best_mcc_threshold(concat_labels(ws.dev, ws.label), res.oof)
    print("Frozen Dev threshold:", thr, "Dev OOF MCC:", mcc)
    rows = []
    for split in EXTERNAL_SPLITS:
        y = concat_labels(ws.splits[split], ws.label)
        p = predict_ms(ws, split)
        row = {"dataset": split, "method": ws.task.ms_hg, "Dev_CV_AUPRC": res.score, "Dev_OOF_MCC_threshold": thr}
        row.update({f"{k}_fixed0.5": v for k, v in binary_metrics(y, p, 0.5).items()})
        row.update({f"{k}_OOFthr": v for k, v in binary_metrics(y, p, thr).items()})
        rows.append(row)
    ext = pd.DataFrame(rows)
    save_table(_ms_root(ws) / "external_metrics.csv", ext)
    if "external" not in ws.state:
        external_evaluation(ws)
    head = ws.state["external"]["headline"]
    cmp_rows = [{"dataset": r.dataset, "method": f"MS {r.method}", "AUPRC": r["AUPRC_fixed0.5"],
                 "AUROC": r["AUROC_fixed0.5"], "F1": r["F1_fixed0.5"], "MCC": r["MCC_fixed0.5"]} for _, r in ext.iterrows()]
    cmp_rows += [{"dataset": r.dataset, "method": f"V1 {r.model}", "AUPRC": r["AUPRC_fixed0.5"],
                  "AUROC": r["AUROC_fixed0.5"], "F1": r["F1_fixed0.5"], "MCC": r["MCC_fixed0.5"]} for _, r in head.iterrows()]
    compare = pd.concat([pd.DataFrame(cmp_rows), baseline_compact_rows(load_baselines(ws.paths, ws.label))],
                        ignore_index=True).sort_values(["dataset", "AUPRC"], ascending=[True, False])
    save_table(_ms_root(ws) / "FINAL_COMPARISON.csv", compare)
    return compare


def paper_report(ws: Workspace, force_predictions: bool = False) -> dict:
    """Dev-locked V1 + MS ensemble (>= 0.002 Dev gain rule), cross-fit audit, all Dev-locked
    thresholds, full external metrics incl. protein-mean F1/MCC and direct baselines."""
    hg = _need(ws, "hypergraph", hypergraph_screen)
    v1 = hg["results"][ws.task.v1_hg]
    ms = _ms_state(ws)["result"]
    out = paper_ready_report(
        task_name=ws.task.name.upper(), label=ws.label, splits=ws.splits, dev_split=DEV, fold_idx=ws.fold_idx,
        v1_oof=v1.oof, ms_oof=ms.oof, v1_dev_score=v1.score,
        predict_v1=lambda s: predict_hypergraph(ws, s, ws.task.v1_hg),
        predict_ms=lambda s: predict_ms(ws, s),
        baselines=load_baselines(ws.paths, ws.label), out_dir=ws.paths.results_dir(ws.task.paper_root),
        v1_name=ws.task.v1_hg, ms_name=ws.task.ms_hg, force_predictions=force_predictions,
        tags={"V1_Attention": states_fingerprint(ws.run_dir(ws.task.group_state_dir, ws.task.v1_hg)),
              "MS_Attention": states_fingerprint(_ms_root(ws, "states", ws.task.ms_hg))})
    ws.state["paper"] = out
    return out


def final_models(ws: Workspace, force: bool = False) -> dict:
    """Only the task's final model: V1 attention hypergraph + its multi-scale twin + the Dev-locked ensemble report.
    The original fold checkpoints in ``final_model/<task>/{v1,ms}`` are linked into the pipeline folders and reused
    instead of trained, so the Dev OOF numbers are the original ones."""
    from ..inference.final import install_final_models

    install_final_models(ws.paths, [ws.task.name])
    hypergraph_screen(ws, names=[ws.task.v1_hg], force=force)
    multiscale_screen(ws, force=force)
    out = paper_report(ws)
    if ws.task.original_final:
        reproduction_report(ws)
    return out


def reproduction_report(ws: Workspace) -> pd.DataFrame:
    """Final ensemble of this workspace vs the numbers printed by the original notebook (``TaskSpec.original_final``).
    With the original fold checkpoints in place the Dev rows match exactly (their OOF files are reused); external
    rows match exactly when the features also come from the original caches (Colab), otherwise closely."""
    ref = ws.task.original_final
    out = ws.state.get("paper") or paper_report(ws)
    dec, ext = out["decision"], out["external"]
    rows = [(k, ref[k], float(dec[k])) for k in ("V1_Dev_mean_AUPRC", "final_Dev_mean_AUPRC", "final_w_V1")]
    for (split, method), v in ref["AUPRC"].items():
        got = ext[ext.dataset.eq(split) & ext.method.eq(method)].AUPRC_global_residue
        rows.append((f"{split}_{method}_AUPRC", v, float(got.iloc[0]) if len(got) else np.nan))
    thr = dec["threshold_rules"].get("FINAL_DEV_LOCKED_ENSEMBLE", {})
    rows += [(f"threshold_{k}", v, float(thr.get(k, np.nan))) for k, v in ref["final_thresholds"].items()]
    rep = pd.DataFrame(rows, columns=["metric", "original", "reproduced"])
    rep["delta"] = rep.reproduced - rep.original
    save_table(ws.paths.results_dir(ws.task.paper_root) / "FINAL_reproduction_vs_original.csv", rep)
    print(rep.to_string(index=False))
    return rep
