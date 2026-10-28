"""V2 matched ablations (residue <-> group message model with protein-balanced loss).

nuclear: NA_SURFTOPO16 re-lock -> selective multi-scale patches -> {NoPatch,Patch} x {BCE,HardNeg}
small:   pocket-coverage audit -> V1 pockets -> {No,Mean,Cross}Pocket x {BCE,HardNeg}
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ..constants import DEV, EXTERNAL_SPLITS
from ..data.loading import concat_labels
from ..evaluation.metrics import binary_metrics
from ..evaluation.selection import complexity_lock
from ..evaluation.thresholds import best_mcc_threshold
from ..groups.bank import coverage_enrichment
from ..reporting.baselines import baseline_compact_rows, load_baselines
from ..reporting.predictions import save_residue_predictions
from ..training.datasets import GroupProteinDataset, make_group_collate
from ..training.group_cv import predict_group_ensemble, run_group_cv
from ..training.modalities import modality_dims, modality_spec
from ..utils.cache import PredictionCache, save_json, save_table, states_fingerprint
from .features import build_features, group_bank
from .screens import gear_screen, representation
from .v1 import external_evaluation, modalities
from .workspace import Workspace


def _root(ws: Workspace, *parts):
    return ws.paths.results_dir(ws.task.v2.run_root, *parts)


def v2_representation(ws: Workspace) -> dict:
    """nuclear: re-lock with NA_SURFTOPO16 (Dev only). small: reuse the V1 lock + modalities."""
    if "v2_rep" in ws.state:
        return ws.state["v2_rep"]
    v2 = ws.task.v2
    if v2.relock_grid is None:
        rep = representation(ws)
        spec, dims = modalities(ws)
        out = {"representation": rep["representation"], "blocks": rep["blocks"], "modalities": spec, "dims": dims}
    else:
        build_features(ws, families=[v2.relock_feature_family])
        cv, folds = gear_screen(ws, v2.relock_grid, v2.relock_run_version, reference=None)
        save_table(_root(ws) / "representation_cv.csv", cv)
        save_table(_root(ws) / "representation_folds.csv", folds)
        name = str(cv.iloc[0].representation)
        blocks = list(v2.relock_grid[name])
        spec = modality_spec(blocks, v2.modality_groups)
        dims = modality_dims(ws.ctx, DEV, spec, str(ws.dev.iloc[0].prot_id))
        out = {"representation": name, "blocks": blocks, "modalities": spec, "dims": dims,
               "dev_cv_auprc": float(cv.iloc[0].CV_AUPRC)}
        save_json(_root(ws) / "locked_representation.json", {k: v for k, v in out.items() if k != "dims"})
        print("V2 LOCKED REPRESENTATION:", name, blocks, "| modalities:", spec)
    ws.state["v2_rep"] = out
    return out


def pocket_coverage_audit(ws: Workspace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the label-free generator enrich true binding residues? (Dev only)"""
    bank = group_bank(ws, ws.task.v2.bank)
    audit, per = coverage_enrichment(bank, ws.dev, ws.label, DEV)
    save_table(_root(ws) / f"current_{ws.task.group}_candidate_audit.csv", audit)
    save_table(_root(ws) / f"current_{ws.task.group}_positive_protein_audit.csv", per)
    return audit, per


def _meta(ws, rep, variant):
    v2 = ws.task.v2
    if v2.relock_grid is not None:  # nuclear convention
        return {"code_version": v2.code_version, "representation": rep["representation"], "blocks": rep["blocks"]}
    cfg = v2.variants[variant]  # small convention
    return {"code_version": v2.code_version, "message_mode": cfg["mode"], "hardneg": cfg["hardneg"]}


def v2_ablation(ws: Workspace, variants=None, force: bool = False) -> dict:
    v2 = ws.task.v2
    rep = v2_representation(ws)
    bank = group_bank(ws, v2.bank)
    collate = make_group_collate(bank.feat_dim)
    results = {}
    for variant in list(variants or v2.variants):
        cfg = v2.variants[variant]
        print("\n" + "=" * 88 + f"\nV2: {variant}\n" + "=" * 88)
        loss = replace(v2.loss, hardneg=cfg["hardneg"])
        results[variant] = run_group_cv(
            variant, lambda v=variant: ws.task.make_v2(v, rep["dims"], bank.feat_dim),
            lambda df: GroupProteinDataset(ws.ctx, DEV, df, rep["modalities"], bank, ws.label), collate,
            ws.dev, ws.folds, ws.label, _root(ws, "states", variant), _meta(ws, rep, variant), v2.train, loss,
            ws.device, force, name_key="variant", report_gate=True)
    summary = (pd.DataFrame([r.summary for r in results.values()])
               .sort_values("CV_AUPRC_mean", ascending=False).reset_index(drop=True))
    save_table(_root(ws) / "v2_cv_summary.csv", summary)
    save_table(_root(ws) / "v2_cv_folds.csv", pd.concat([r.folds for r in results.values()]))
    score = {k: r.score for k, r in results.items()}
    ablation = pd.DataFrame([{"comparison": lbl, "delta_AUPRC": score[a] - score[b]}
                             for lbl, a, b in v2.ablation if a in score and b in score])
    save_table(_root(ws) / "matched_ablation.csv", ablation)
    locked, locked_score = complexity_lock([(v, score[v]) for v in v2.complexity if v in score])
    print("V2 LOCKED:", locked, round(locked_score, 5))
    ws.state["v2"] = {"summary": summary, "ablation": ablation, "results": results, "locked": locked,
                      "locked_score": locked_score, "bank": bank, "collate": collate}
    return ws.state["v2"]


def v2_external(ws: Workspace) -> pd.DataFrame:
    """External confirmation of the DEV-locked V2 variant only, compared with V1 + baselines."""
    st = ws.state.get("v2") or v2_ablation(ws)
    rep = v2_representation(ws)
    name, res, bank = st["locked"], st["results"][st["locked"]], st["bank"]
    thr, _ = best_mcc_threshold(concat_labels(ws.dev, ws.label), res.oof)
    cache = PredictionCache(_root(ws, "prediction_cache"))
    tag = states_fingerprint(_root(ws, "states", name))
    rows = []
    for split in EXTERNAL_SPLITS:
        df = ws.splits[split]
        y = concat_labels(df, ws.label)
        ds = GroupProteinDataset(ws.ctx, split, df, rep["modalities"], bank, ws.label)
        p = cache.get_or_compute(name, split, len(y), lambda: predict_group_ensemble(
            lambda: ws.task.make_v2(name, rep["dims"], bank.feat_dim), res.states, ds, st["collate"],
            ws.task.v2.train.batch, ws.device), tag=tag)
        row = {"dataset": split, "variant": name, "Dev_CV_AUPRC": st["locked_score"], "Dev_OOF_MCC_threshold": thr}
        row.update({f"{k}_fixed0.5": v for k, v in binary_metrics(y, p, 0.5).items()})
        row.update({f"{k}_OOFthr": v for k, v in binary_metrics(y, p, thr).items()})
        rows.append(row)
        save_residue_predictions(_root(ws) / f"predictions_{split}.csv.gz", df, ws.label, probability=p)
    ext = pd.DataFrame(rows)
    save_table(_root(ws) / "FINAL_V2_EXTERNAL.csv", ext)
    cmp_rows = [{"dataset": r.dataset, "method": f"V2 {name}", "AUPRC": r["AUPRC_fixed0.5"],
                 "AUROC": r["AUROC_fixed0.5"], "F1": r["F1_fixed0.5"], "MCC": r["MCC_fixed0.5"]}
                for _, r in ext.iterrows()]
    headline = (ws.state.get("external") or {}).get("headline")
    if headline is None:
        external_evaluation(ws)
        headline = ws.state["external"]["headline"]
    cmp_rows += [{"dataset": r.dataset, "method": f"V1 {r.model}", "AUPRC": r["AUPRC_fixed0.5"],
                  "AUROC": r["AUROC_fixed0.5"], "F1": r["F1_fixed0.5"], "MCC": r["MCC_fixed0.5"]}
                 for _, r in headline.iterrows()]
    compare = pd.concat([pd.DataFrame(cmp_rows), baseline_compact_rows(load_baselines(ws.paths, ws.label))],
                        ignore_index=True).sort_values(["dataset", "AUPRC"], ascending=[True, False])
    save_table(_root(ws) / "FINAL_V2_COMPARISON.csv", compare)
    ws.state["v2_external"] = {"table": ext, "compare": compare}
    return compare
