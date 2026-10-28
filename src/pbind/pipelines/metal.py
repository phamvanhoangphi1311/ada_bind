"""Metal binding (GEAR_METAL_ONLY_ULTIMATE_V2 + GEAR_ULTIMATE_METAL_BINDING).

Features and the standardized MLP/KAN/XGBoost trainer live in the GEAR backend;
this module only orders the Dev-only decisions and writes the decision records.
Workspace: ``open_workspace("metal")`` (prefers the GEAR_ULTIMATE backend copy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import EXPLICIT_GEAR_BLOCKS
from ..evaluation.selection import best_mean, one_se_rule, rank_vs_reference
from ..reporting.baselines import baseline_view, load_baselines
from ..reporting.predictions import write_manifest
from ..tasks import metal as M
from ..utils.cache import load_json, save_json, save_table
from .workspace import Workspace

POLICIES = ("best_mean", "one_se")


def _gear_only(ws: Workspace, what: str) -> None:
    if getattr(ws.gear, "kind", "gear") != "gear":
        raise RuntimeError(f"{what} needs the Drive backend: open_workspace('metal', backend='gear')")


def _run(ws, grid, key, reference=None, models=("MLP",)):
    cv, folds = ws.gear.suite.run_dev_cv_only(ws.ctx, grid, models=tuple(models), run_version=M.RUN_VERSIONS[key])
    ranked = rank_vs_reference(cv, reference, dims=M.dims_of(grid)) if reference else cv
    return ranked, folds


def audit(ws: Workspace) -> dict:
    return {"context": ws.gear.suite.audit_context(ws.ctx),
            "dims": ws.gear.reporting.dimension_table(ws.store("train")),
            "leakage": ws.gear.reporting.validate_no_direct_label_features(ws.ctx.stores, EXPLICIT_GEAR_BLOCKS)}


def family_screen(ws: Workspace) -> pd.DataFrame:
    """Stage A: same MLP, ESM + one family."""
    ranked, folds = _run(ws, M.FAMILY_GRID, "family", "ESM")
    save_table(ws.run_dir() / "family_screen_cv.csv", ranked)
    save_table(ws.run_dir() / "family_screen_folds.csv", folds)
    ws.state["family"] = ranked
    return ranked


def component_screen(ws: Workspace) -> pd.DataFrame:
    """Stage B: one mechanistic component at a time on top of ESM+ATOM+CHEM."""
    ranked, folds = _run(ws, M.COMPONENT_GRID, "component", "Core_ESM+Atomic+Chem")
    save_table(ws.run_dir() / "component_screen_ranked.csv", ranked)
    save_table(ws.run_dir() / "component_screen_folds.csv", folds)
    ws.state["component"] = ranked
    return ranked


def combination_ladder(ws: Workspace, policy: str = "best_mean") -> dict:
    """Stage C + representation lock (Dev only). policy: best_mean | one_se."""
    assert policy in POLICIES
    cv, folds = _run(ws, M.COMBO_GRID, "combo")
    save_table(ws.run_dir() / "combo_ladder_cv.csv", cv)
    save_table(ws.run_dir() / "combo_ladder_folds.csv", folds)
    dims = M.dims_of(M.COMBO_GRID)
    best = best_mean(cv)
    one_se, fold_stats = one_se_rule(folds, dims)
    locked = best if policy == "best_mean" else one_se
    decision = {"selection_policy": policy, "best_mean_representation": best, "one_se_representation": one_se,
                "locked_representation": locked, "locked_blocks": list(M.COMBO_GRID[locked]),
                "locked_dimension": dims[locked]}
    save_json(ws.run_dir() / "locked_representation.json", decision)
    print("LOCKED_REP =", locked, M.COMBO_GRID[locked])
    ws.state["representation"] = decision
    ws.state["combo"] = {"cv": cv, "fold_stats": fold_stats}
    return decision


def model_family_screen(ws: Workspace) -> dict:
    """Stage D: MLP / KAN / XGBoost on the locked representation (Dev only)."""
    rep = ws.state.get("representation") or combination_ladder(ws)
    models = ["MLP", "XGBoost"]
    if ws.gear.has_kan:
        models.insert(1, "KAN")
    if not getattr(ws.gear, "has_xgb", True):
        print("SKIP XGBoost: xgboost not installed")
        models.remove("XGBoost")
    cv, folds = _run(ws, {rep["locked_representation"]: rep["locked_blocks"]}, "model", models=models)
    cv = cv.sort_values("CV_AUPRC", ascending=False).reset_index(drop=True)
    save_table(ws.run_dir() / "model_family_cv.csv", cv)
    save_table(ws.run_dir() / "model_family_folds.csv", folds)
    decision = {**rep, "candidate_models": models, "locked_model": str(cv.iloc[0].model),
                "locked_model_cv_auprc": float(cv.iloc[0].CV_AUPRC)}
    save_json(ws.run_dir() / "locked_model.json", decision)
    ws.state["model"] = decision
    return decision


def graph_diagnostic(ws: Workspace, include_esm: bool = False):
    """Optional Stage E: residual relation graph on the locked explicit blocks (+ESM optional)."""
    _gear_only(ws, "graph diagnostic")
    rep = ws.state.get("representation") or combination_ladder(ws)
    explicit = ["AA"] + [b for b in rep["locked_blocks"] if b not in ("ESM", "AA")]
    exps = {f"ResidualGraph_ExplicitLocked_{m.title()}": {"blocks": explicit, "architecture": "residual", "edge_mode": m}
            for m in ("rich", "donor")}
    if include_esm:
        exps.update({f"ResidualGraph_ESMLocked_{m.title()}": {"blocks": rep["locked_blocks"],
                                                             "architecture": "residual", "edge_mode": m}
                     for m in ("rich", "donor")})
    g = ws.gear.graph
    cv, ext = g.run_graph_grid(ws.ctx.stores, ws.ctx.fold_ids, ws.paths.base, exps,
                               run_version=M.RUN_VERSIONS["graph"], cfg=g.GraphConfig(), device=ws.device)
    save_table(ws.run_dir() / "graph_cv.csv", cv)
    save_table(ws.run_dir() / "graph_external_exploratory.csv", ext)
    return cv, ext


def final_lock(ws: Workspace) -> dict:
    """Locked representation/model: this session's Dev screens, else their saved decision, else the known lock."""
    return ws.state.get("model") or load_json(ws.run_dir() / "locked_model.json") or dict(M.FINAL_LOCK)


def final_external(ws: Workspace, run_version: str | None = None):
    """Stage F: five official folds with the locked representation/model (fold weights saved), evaluated once
    externally. Folds already trained with the same config are reused. ``run_version`` writes to another run folder
    (e.g. a Colab reproduction next to the original run on Drive); results of another config are never overwritten."""
    dec = final_lock(ws)
    run_ctx = ws.gear.suite.clone_context_for_run(ws.ctx, run_version=run_version or M.RUN_VERSIONS["final"],
                                                  models=(dec["locked_model"],))
    cv, ext, _ = ws.gear.suite.run_grid(run_ctx, {dec["locked_representation"]: dec["locked_blocks"]},
                                        models=(dec["locked_model"],))
    out_dir = Path(run_ctx.paths["run_root"]) if run_version else ws.run_dir()  # a reproduction stays in its own folder
    save_table(out_dir / "FINAL_locked_cv.csv", cv)
    save_table(out_dir / "FINAL_locked_external.csv", ext)
    ws.state["final"] = {"cv": cv, "external": ext, "lock": dec, "run_root": Path(run_ctx.paths["run_root"]),
                         "model_root": Path(run_ctx.paths["model_root"])}
    return cv, ext


def compare_fold_predictions(ws: Workspace, run_version: str, reference: str = M.RUN_VERSIONS["final"]) -> pd.DataFrame:
    """Saved fold predictions of ``run_version`` vs ``reference`` (default: the original final run, i.e. its
    ``fold_predictions/*.pkl`` on Drive), per fold and split (val / test300 / test46).
    ``identical`` everywhere = bit-exact reproduction of the original models."""
    from sklearn.metrics import average_precision_score

    from ..utils.cache import load_pickle_dict

    lock = final_lock(ws)
    model, rep = lock["locked_model"], lock["locked_representation"]

    def preds(rv, fold):
        ctx = ws.gear.suite.clone_context_for_run(ws.ctx, rv, models=(model,))
        fp = Path(ws.gear.core.job_pred_path(ctx.paths, model, rep, fold))
        z = load_pickle_dict(fp)
        if z is None:
            raise FileNotFoundError(f"{fp} missing")
        return z

    rows = []
    for fold in range(5):
        ref, new = preds(reference, fold), preds(run_version, fold)
        if list(ref["val_ids"]) != list(new["val_ids"]):
            raise RuntimeError(f"fold {fold + 1}: validation proteins differ between the two runs")
        for key in ("val", "test300", "test46"):
            y, a, b = (np.asarray(ref[f"{key}_y"]), np.asarray(ref[f"{key}_p"], np.float64),
                       np.asarray(new[f"{key}_p"], np.float64))
            rows.append({"fold": fold + 1, "split": key, "residues": len(a), "identical": bool(np.array_equal(a, b)),
                         "max_abs_diff": float(np.abs(a - b).max()), "AUPRC_reference": average_precision_score(y, a),
                         "AUPRC_run": average_precision_score(y, b)})
    out = pd.DataFrame(rows)
    run_root = Path(ws.gear.suite.clone_context_for_run(ws.ctx, run_version, models=(model,)).paths["run_root"])
    save_table(run_root / f"PREDICTION_PARITY_vs_{reference}.csv", out)
    print(out.to_string(index=False))
    print("BIT-EXACT REPRODUCTION" if out.identical.all() else
          f"NOT bit-exact: max |diff| = {out.max_abs_diff.max():.3g}")
    return out


def reproduction_report(ws: Workspace) -> pd.DataFrame:
    """Final locked run of this workspace vs the original notebook output (M.ORIGINAL_FINAL)."""
    if "final" not in ws.state:
        final_external(ws)
    fin, ref = ws.state["final"], M.ORIGINAL_FINAL
    folds = pd.read_csv(fin["run_root"] / "fold_metrics.csv").sort_values("fold")
    cv, ext = fin["cv"].iloc[0], fin["external"].set_index("dataset")
    rows = [(f"fold{k}_{m}", ref[f"fold_{m}"][k - 1], float(folds[m].iloc[k - 1]))
            for m in ("AUPRC", "AUROC") for k in range(1, 6)]
    rows += [(k, ref[k], float(cv[k])) for k in ("CV_AUPRC", "CV_AUPRC_SD", "CV_AUROC", "CV_F1_at_0_5",
                                                  "CV_MCC_at_0_5", "mean_best_epoch", "n_params")]
    rows.append(("OOF_tuned_threshold", ref["OOF_tuned_threshold"], float(ext["OOF_tuned_threshold"].iloc[0])))
    rows += [(f"{d}_{k}", v, float(ext.loc[d, k])) for d, metrics in ref["external"].items() for k, v in metrics.items()]
    out = pd.DataFrame(rows, columns=["metric", "original", "reproduced"])
    out["delta"] = out.reproduced - out.original
    save_table(fin["run_root"] / "FINAL_reproduction_vs_original.csv", out)
    print(out.to_string(index=False))
    return out


def baselines(ws: Workspace) -> pd.DataFrame:
    """Frozen bindEmbed21DL / bindNode24 / ESMBind / MetalBind rows (no inference)."""
    b = load_baselines(ws.paths, M.LABEL, include_transfer=True)
    if len(b):
        save_table(ws.run_dir() / "locked_baseline_metal_rows.csv", b)
    return baseline_view(b)


def manifest(ws: Workspace) -> dict:
    dec = ws.state.get("model", {})
    return write_manifest(ws.run_dir() / "FINAL_DECISION_MANIFEST.json", gear_ultimate_version=ws.gear.version,
                          device=str(ws.device), primary_metric="AUPRC",
                          development_set="DevSet1010 official 5-fold protein split",
                          external_sets=["TestSet300", "TestSetNew46"], **dec,
                          final_external_run="final" in ws.state)


def run_all(ws: Workspace, policy: str = "best_mean", graph: bool = False) -> dict:
    family_screen(ws)
    component_screen(ws)
    combination_ladder(ws, policy)
    model_family_screen(ws)
    if graph:
        graph_diagnostic(ws)
    final_external(ws)
    out = {"baselines": baselines(ws)}
    manifest(ws)
    return out


# ------------------------------------------------------------------ GEAR_ULTIMATE_METAL_BINDING grids
def ultimate_grid(ws: Workspace, which: str, models=("MLP",)):
    """Re-run one of the original GEAR_ULTIMATE grids:
    which = standardized_13 | mechanistic | final_focused | no_esm | evo_full."""
    _gear_only(ws, "GEAR_ULTIMATE grids")
    s = ws.gear.suite
    grids = {"standardized_13": (s.STANDARDIZED_13, None), "mechanistic": (s.MECHANISTIC_GRID, s.MECHANISTIC_COMPARISONS),
             "final_focused": (s.FINAL_COMPARISON_REPS, None), "no_esm": (s.NO_ESM_GRID, None),
             "evo_full": (s.EVO_GRID, None)}
    grid, comparisons = grids[which]
    run_ctx = s.clone_context_for_run(ws.ctx, run_version=M.ULTIMATE_RUN_VERSIONS[which], models=tuple(models))
    return s.run_grid(run_ctx, grid, models=tuple(models), comparisons=comparisons)


def ultimate_graph(ws: Workspace, which: str = "clean"):
    """which = maximal | clean | evo (EVO graph needs EVO on all splits)."""
    _gear_only(ws, "graph experiments")
    g = ws.gear.graph
    if which == "maximal":
        exps, cfg, rv = g.MAXIMAL_GRAPH_EXPERIMENT, g.maximal_graph_config(), M.ULTIMATE_RUN_VERSIONS["maximal_graph"]
    elif which == "clean":
        exps, cfg, rv = g.CLEAN_GRAPH_EXPERIMENTS, g.GraphConfig(), M.ULTIMATE_RUN_VERSIONS["clean_graph"]
    else:
        exps, cfg, rv = M.EVO_GRAPH_EXPERIMENTS, g.GraphConfig(), M.ULTIMATE_RUN_VERSIONS["evo_graph"]
    return g.run_graph_grid(ws.ctx.stores, ws.ctx.fold_ids, ws.paths.base, exps, run_version=rv, cfg=cfg,
                            device=ws.device)


def msa_batches(ws: Workspace, splits=("train",), batch_size: int = 25, max_batches=None):
    """Resumable ColabFold/MMseqs2 MSA generation (Dev first!). Needs `pip install colabfold[alphafold]`."""
    _gear_only(ws, "MSA / EVO")
    return ws.gear.evo.run_msa_batches(ws.ctx.stores, ws.paths.base, splits=tuple(splits), batch_size=batch_size,
                                       max_retries=4, sleep_between=8, msa_mode="mmseqs2_uniref_env",
                                       max_batches=max_batches)


def evo_dev_cv(ws: Workspace, models=("MLP",)):
    """Dev-only EVO screen (run before spending time on external MSAs)."""
    _gear_only(ws, "EVO")
    return ws.gear.suite.run_dev_cv_only(ws.ctx, ws.gear.suite.EVO_GRID, models=tuple(models),
                                         run_version=M.ULTIMATE_RUN_VERSIONS["evo_dev"])


def historical_kan(ws: Workspace):
    """Archival historical KAN protocol (label-aware Cα smoothness; not a structure-free baseline)."""
    _gear_only(ws, "historical KAN")
    if ws.gear.legacy is None or not ws.gear.has_kan:
        raise RuntimeError("gear_legacy / pykan unavailable")
    return ws.gear.legacy.run_historical_baseline(ws.gear.legacy.HistoricalConfig(base=str(ws.paths.base)),
                                                  device=ws.device)
