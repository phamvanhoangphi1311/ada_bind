"""Cross-task + EVO add-in (the two "STANDALONE_*_XTASK_ADDIN" notebooks).

Runs in a light workspace (no PDB parsing, no V1/MS retraining):
  1. rebuild this task's V1/MS Dev OOF from saved fold predictions, export it
  2. wait for the counterpart task's export (nuclear <-> small)
  3. attach the counterpart OOF (and, for nuclear, EVO residual beyond ACCESS) as Dev blocks
  4. Dev-only MLP screen of base / +x-task / +EVO / combinations
  5. (nuclear) matched PatchHG vs PatchHG+EVO48, then EVO-PatchHG + MS ensemble vs old V1 + MS
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DEV, EXTERNAL_SPLITS, MIN_DEV_GAIN
from ..data.loading import concat_labels, n_residues, residue_ids
from ..evaluation.ensemble import weight_search
from ..evaluation.metrics import binary_metrics, fold_auprc
from ..evaluation.thresholds import best_mcc_threshold
from ..features.derived import attach_crosstask_block, attach_evo_access_residual
from ..groups.bank import load_bank
from ..training.datasets import GroupProteinDataset, make_group_collate
from ..training.group_cv import oof_from_saved_folds, predict_group_ensemble, run_group_cv
from ..training.modalities import modality_dims
from ..utils.cache import read_table, save_json, save_table
from .features import build_features
from .screens import gear_screen
from .workspace import Workspace

EXPORT_DIR = "CROSSTASK_EXPORTS"


def _addin(ws, *parts) -> Path:
    return ws.paths.results_dir(ws.task.xtask["addin_root"], *parts)


def reconstruct_oof(ws: Workspace) -> dict:
    """V1 / MS Dev OOF from ``fold{k}_val.npy`` + Dev-locked V1/MS weights."""
    if "xtask_oof" in ws.state:
        return ws.state["xtask_oof"]
    t = ws.task
    v1, y = oof_from_saved_folds(ws.paths.results_dir(t.run_root, t.group_state_dir, t.v1_hg), ws.dev, ws.folds, ws.label)
    ms, y2 = oof_from_saved_folds(ws.paths.results_dir(t.ms_run_root, "states", t.ms_hg), ws.dev, ws.folds, ws.label)
    assert np.array_equal(y, y2)
    ws_tab = weight_search(y, v1, ms, ws.fold_idx)
    save_table(_addin(ws) / "DEV_WEIGHT_SEARCH.csv", ws_tab)
    w = float(ws_tab.iloc[0].w_V1)
    out = {"V1": v1, "MS": ms, "y": y, "w_V1": w, "FINAL": w * v1 + (1 - w) * ms,
           "final_dev": float(ws_tab.iloc[0].CV_AUPRC_mean)}
    print("Locked weights:", w, 1 - w, "| Dev mean AUPRC:", out["final_dev"])
    ws.state["xtask_oof"] = out
    return out


def export_dev_oof(ws: Workspace, force: bool = False) -> Path:
    """Write ``results/CROSSTASK_EXPORTS/<TASK>_DEV_OOF.csv.gz`` (real gzip, atomic). Reused if valid."""
    fp = ws.paths.results_dir(EXPORT_DIR) / ws.task.xtask["export_file"]
    expected = n_residues(ws.dev)
    if fp.exists() and not force:
        try:
            old = read_table(fp)
            if len(old) == expected and {"prob_V1", "prob_MS", ws.task.xtask["own_label_col"]} <= set(old.columns):
                print("Reuse export:", fp)
                return fp
        except Exception:
            pass
    o = reconstruct_oof(ws)
    pids, ridx = residue_ids(ws.dev)
    df = pd.DataFrame({"dataset": DEV, "prot_id": pids, "residue_index": ridx,
                       ws.task.xtask["own_label_col"]: o["y"].astype(int),
                       "prob_FINAL": o["FINAL"], "prob_V1": o["V1"], "prob_MS": o["MS"]})
    save_table(fp, df)
    print("Export:", fp, len(df), "rows")
    return fp


def wait_for_counterpart(ws: Workspace, timeout: int = 900, poll: int = 10) -> pd.DataFrame:
    """Poll for the other task's export and validate it strictly."""
    x = ws.task.xtask
    fp = ws.paths.results_dir(EXPORT_DIR) / x["counterpart_file"]
    t0 = time.time()
    while not fp.exists() and time.time() - t0 < timeout:
        print(f"Waiting for {fp.name} ... {int(time.time() - t0)}s", end="\r")
        time.sleep(poll)
    if not fp.exists():
        raise FileNotFoundError(fp)
    xdf = read_table(fp)
    xdf["prot_id"] = xdf["prot_id"].astype(str)
    missing = {"prot_id", "residue_index", x["counterpart_label_col"], "prob_V1", "prob_MS"} - set(xdf.columns)
    if missing:
        raise RuntimeError(f"Wrong/old counterpart export, missing {sorted(missing)}")
    if x["own_label_col"] in xdf.columns:
        raise RuntimeError(f"WRONG FILE: counterpart export contains {x['own_label_col']}")
    if xdf.duplicated(["prot_id", "residue_index"]).any():
        raise RuntimeError("Duplicate (prot_id, residue_index) in counterpart export")
    if len(xdf) != n_residues(ws.dev) or set(xdf.prot_id) != set(ws.dev.prot_id.astype(str)):
        raise RuntimeError("Counterpart export does not match DevSet1010")
    print("Counterpart export integrity: PASS", len(xdf), "rows")
    return xdf


def crosstask_screen(ws: Workspace, xdf: pd.DataFrame | None = None, timeout: int = 900) -> dict:
    """Dev-only MLP screen: base / +x-task / +EVO (/ +EVO residual) / +EVO+x-task."""
    x, names = ws.task.xtask, ws.task.xtask["names"]
    if not set(x["base_blocks"]) <= ws.available:
        build_features(ws, attach_only=True)
    xdf = wait_for_counterpart(ws, timeout) if xdf is None else xdf
    store = ws.store(DEV)
    attach_crosstask_block(store, ws.dev, xdf, ["prob_V1", "prob_MS"], x["xtask_block"])
    evo = "EVO" in ws.available
    base = list(x["base_blocks"])
    grid = {x["base_name"]: base, names["xtask"]: base + [x["xtask_block"]]}
    if evo:
        grid[names["evo"]] = base + ["EVO"]
        if "evo_res" in names:
            attach_evo_access_residual(store, ws.dev)
            grid[names["evo_res"]] = base + ["EVO_ACCESS_RES_CLEAN"]
        grid[names["evo_xtask"]] = base + ["EVO", x["xtask_block"]]
    cv, folds = gear_screen(ws, grid, x["run_version"], reference=None)
    b = float(cv.loc[cv.representation.eq(x["base_name"]), "CV_AUPRC"].iloc[0])
    cv["delta_vs_base"] = cv.CV_AUPRC - b
    save_table(_addin(ws) / "REPRESENTATION_SCREEN_CLEAN_V3.csv", cv)
    save_table(_addin(ws) / "REPRESENTATION_SCREEN_FOLDS_CLEAN_V3.csv", folds)
    score = dict(zip(cv.representation, cv.CV_AUPRC))
    best = cv.iloc[0]
    decision = {
        "run_version": x["run_version"], "base_dev_auprc": b,
        "xtask_dev_auprc": score[names["xtask"]], "xtask_gain_vs_base": score[names["xtask"]] - b,
        "evo_dev_auprc": score.get(names["evo"]), "evo_gain_vs_base": score[names["evo"]] - b if evo else None,
        "evo_plus_xtask_dev_auprc": score.get(names["evo_xtask"]),
        "best_representation": str(best.representation), "best_dev_auprc": float(best.CV_AUPRC),
        "best_gain_vs_base": float(best.delta_vs_base),
        f"promote_best_if_gain_ge_{MIN_DEV_GAIN}": bool(best.delta_vs_base >= MIN_DEV_GAIN),
        "EVO_available": evo, "cross_task_source_columns": ["prob_V1", "prob_MS"],
    }
    save_json(_addin(ws) / "DECISION_CLEAN_V3.json", decision)
    ws.state["xtask"] = {"cv": cv, "folds": folds, "decision": decision}
    return ws.state["xtask"]


# ------------------------------------------------------------------ nuclear: PatchHG + EVO48
def evo_hypergraph_addin(ws: Workspace, force: bool = False) -> dict:
    """Matched base PatchHG vs PatchHG with EVO as an extra BIO modality (V1 patch bank, attention),
    then EVO-PatchHG + MS ensemble vs the old V1 + MS final. External only if Dev promotion passes."""
    x, t = ws.task.xtask, ws.task
    if "evo_hg_root" not in x:
        raise RuntimeError(f"{t.name}: no EVO hypergraph add-in defined")
    if "EVO" not in ws.available:
        raise RuntimeError("EVO48 not attached (open_workspace(..., attach_evo=True) and complete MSA cache)")
    if not set(x["base_blocks"]) <= ws.available:
        build_features(ws, attach_only=True)
    root = ws.paths.results_dir(*x["evo_hg_root"])
    specs = {x["evo_hg_base_name"]: {"ESM": ["ESM"], "ELECTRO": ["ACCESS"], "PATCH": ["NA_PATCH3D40", "NA_TOPO12"]}}
    specs[x["evo_hg_evo_name"]] = {**specs[x["evo_hg_base_name"]], "BIO": ["EVO"]}
    bank, _ = load_bank(ws.paths, {DEV: ws.dev}, t.bank_v1)  # Dev only; reads the existing patch cache
    collate = make_group_collate(bank.feat_dim)
    res = {}
    for name, spec in specs.items():
        dims = modality_dims(ws.ctx, DEV, spec)
        res[name] = run_group_cv(
            name, lambda d=dims: t.make_hypergraph(t.v1_hg, d, bank.feat_dim),
            lambda df, s=spec: GroupProteinDataset(ws.ctx, DEV, df, s, bank, ws.label), collate, ws.dev, ws.folds,
            ws.label, root / "states" / name, {"representation": name}, t.group_train, device=ws.device, force=force)
    base, evo = res[x["evo_hg_base_name"]], res[x["evo_hg_evo_name"]]
    evo_gain = evo.score - base.score
    fold_cmp = base.folds[["fold", "AUPRC"]].merge(evo.folds[["fold", "AUPRC"]], on="fold", suffixes=("_BASE", "_EVO"))
    fold_cmp["delta_EVO_minus_BASE"] = fold_cmp.AUPRC_EVO - fold_cmp.AUPRC_BASE
    save_table(root / "matched_fold_comparison.csv", fold_cmp)

    o = reconstruct_oof(ws)
    ens = weight_search(o["y"], evo.oof, o["MS"], ws.fold_idx, names=("EVO_PATCHHG", "MS"))
    save_table(root / "dev_ensemble_search.csv", ens)
    w_evo = float(ens.iloc[0].w_EVO_PATCHHG)
    new_dev = float(ens.iloc[0].CV_AUPRC_mean)
    old_dev = max(float(np.mean(fold_auprc(o["y"], w * o["V1"] + (1 - w) * o["MS"], ws.fold_idx)))
                  for w in np.linspace(0, 1, 21))
    new_oof = w_evo * evo.oof + (1 - w_evo) * o["MS"]
    thr, mcc = best_mcc_threshold(o["y"], new_oof)
    promote_final = new_dev - old_dev >= MIN_DEV_GAIN
    decision = {
        "matched_base_patchhg_dev_auprc": base.score, "evo_patchhg_dev_auprc": evo.score,
        "evo_patchhg_gain_vs_matched_base": evo_gain,
        "evo_patchhg_fold_wins": int((fold_cmp.delta_EVO_minus_BASE > 0).sum()),
        "promote_evo_architecture": bool(evo_gain >= MIN_DEV_GAIN),
        "old_final_v1_ms_dev_auprc": old_dev, "new_w_EVO_PatchHG": w_evo, "new_w_MS": 1 - w_evo,
        "new_final_dev_auprc": new_dev, "new_final_gain_vs_old_final": new_dev - old_dev,
        "promote_new_final": bool(promote_final), "new_final_dev_mcc_threshold": thr, "new_final_dev_oof_mcc": mcc,
    }
    if promote_final:
        decision["external"] = _evo_external(ws, root, evo, specs[x["evo_hg_evo_name"]], w_evo, thr)
    else:
        print("New ensemble did not beat old final by >= 0.002 Dev AUPRC. External evaluation skipped.")
    save_json(root / "FINAL_DECISION.json", decision)
    ws.state["evo_hg"] = {"results": res, "decision": decision}
    return ws.state["evo_hg"]


def _evo_external(ws, root, evo, spec, w_evo, thr) -> list:
    """Only reached when the Dev rule passes (needs MS predictions -> pipelines.final)."""
    from .final import predict_ms

    t = ws.task
    bank, _ = load_bank(ws.paths, ws.splits, t.bank_v1)
    collate = make_group_collate(bank.feat_dim)
    dims = modality_dims(ws.ctx, DEV, spec)
    rows = []
    for split in EXTERNAL_SPLITS:
        df = ws.splits[split]
        y = concat_labels(df, ws.label)
        ds = GroupProteinDataset(ws.ctx, split, df, spec, bank, ws.label)
        p_evo = predict_group_ensemble(lambda: t.make_hypergraph(t.v1_hg, dims, bank.feat_dim), evo.states, ds,
                                       collate, t.group_train.batch, ws.device)
        p = w_evo * p_evo + (1 - w_evo) * predict_ms(ws, split)
        for rule, th in (("fixed_0.5", 0.5), ("dev_oof_mcc_opt", thr)):
            rows.append({"dataset": split, "threshold_rule": rule, "threshold": th, **binary_metrics(y, p, th)})
    save_table(root / "external_metrics.csv", pd.DataFrame(rows))
    return rows
