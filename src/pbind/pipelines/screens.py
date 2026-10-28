"""Stage: Dev-only feature screens with the SAME GEAR MLP (representation discovery + lock)."""
from __future__ import annotations

import pandas as pd

from .. import backend as B
from ..evaluation.selection import rank_vs_reference
from ..utils.cache import load_json, save_json, save_table
from .features import build_features
from .workspace import Workspace


def gear_screen(ws: Workspace, grid: dict, run_version: str, out_prefix: str | None = None,
                reference: str | None = "ESM", models=("MLP",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a representation grid through the backend (per-fold pkl reuse) and rank vs ``reference``."""
    cv, folds = B.run_dev_cv(ws.gear, ws.ctx, grid, run_version, models=models)
    ranked = rank_vs_reference(cv, reference) if reference in set(cv.representation) else \
        cv.sort_values("CV_AUPRC", ascending=False).reset_index(drop=True)
    if out_prefix:
        save_table(ws.run_dir() / f"{out_prefix}_cv.csv", cv)
        save_table(ws.run_dir() / f"{out_prefix}_folds.csv", folds)
    return ranked, folds


def _ensure_task_features(ws: Workspace) -> None:
    if not set(ws.task.new_blocks) <= ws.available:
        build_features(ws)


def family_screen(ws: Workspace) -> pd.DataFrame:
    """Stage A: ESM + one family at a time."""
    _ensure_task_features(ws)
    ranked, _ = gear_screen(ws, ws.task.family_grid(ws.available), ws.task.run_version_family, "family")
    save_table(ws.run_dir() / "family_ranked.csv", ranked)
    ws.state["family"] = ranked
    return ranked


def mechanism_ladder(ws: Workspace) -> tuple[pd.DataFrame, dict]:
    """Stage B: audit-driven mechanism combinations; locks the representation (best Dev CV AUPRC)."""
    _ensure_task_features(ws)
    grid = ws.task.mechanism_grid(ws.available)
    ranked, folds = gear_screen(ws, grid, ws.task.run_version_mechanism, "mechanism_combo")
    best = ranked.iloc[0]
    lock = {"representation": str(best.representation), "blocks": list(grid[str(best.representation)]),
            "dev_cv_auprc": float(best.CV_AUPRC), "selection_source": "DevSet1010 only"}
    save_json(ws.run_dir() / "locked_representation.json", lock)
    ws.state["mechanism"] = ranked
    ws.state["representation"] = lock
    print("LOCKED REPRESENTATION:", lock["representation"], lock["blocks"], round(lock["dev_cv_auprc"], 5))
    return ranked, lock


def representation(ws: Workspace) -> dict:
    """Locked representation: this session's ladder, else the saved ``locked_representation.json``, else the task's
    known Dev-only lock (``TaskSpec.final_representation``, printed by the original notebook), else run the ladder.
    Re-running the ladder on other hardware could pick a different winner by noise, so a known lock is preferred."""
    if "representation" not in ws.state:
        saved = load_json(ws.run_dir() / "locked_representation.json")
        name = ws.task.final_representation
        if saved:
            ws.state["representation"] = saved
        elif name:
            ws.state["representation"] = {"representation": name, "blocks": list(ws.task.mechanism_grid_base[name]),
                                          "selection_source": "original DevSet1010 mechanism ladder (TaskSpec)"}
        else:
            return mechanism_ladder(ws)[1]
        _ensure_task_features(ws)
        print("REPRESENTATION:", ws.state["representation"]["representation"], ws.state["representation"]["blocks"])
    return ws.state["representation"]
