"""Workspace = everything a pipeline stage needs (paths, splits, folds, GEAR context, device, task).

    ws = open_workspace("nuclear")      # mounts Drive in Colab, loads splits/folds/backend context
    ws.state                            # results of finished stages (filled by pipeline functions)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from .. import backend as B
from ..constants import DEV, SPLIT_SIZES, STORE_KEY
from ..data.folds import find_bindnode24, find_fold_dir, fold_residue_indices, load_official_folds, official_order
from ..data.loading import build_splits, load_raw_frames
from ..paths import Paths, mount_drive, resolve_paths
from ..tasks import TaskSpec, get_task
from ..tasks import metal as METAL
from ..utils.torch_utils import get_device


@dataclass
class Workspace:
    paths: Paths
    splits: dict[str, pd.DataFrame]
    folds: dict[int, list[str]]
    bn24: Path | None
    device: torch.device
    label: str
    task: TaskSpec | None = None
    gear: SimpleNamespace | None = None
    ctx: object | None = None
    available: set = field(default_factory=set)  # optional blocks attached (EVO, ESMIF, ...)
    state: dict = field(default_factory=dict)

    # ---------------- convenience
    @property
    def dev(self) -> pd.DataFrame:
        return self.splits[DEV]

    def store(self, split: str):
        return B.store_of(self.ctx, split)

    def run_dir(self, *parts: str) -> Path:
        root = self.task.run_root if self.task else METAL.RUN_ROOT
        return self.paths.results_dir(root, *parts)

    def results_dir(self, *parts: str) -> Path:
        return self.paths.results_dir(*parts)

    @property
    def fold_idx(self) -> dict:
        if "_fold_idx" not in self.state:
            self.state["_fold_idx"] = fold_residue_indices(self.dev, self.folds)
        return self.state["_fold_idx"]

    def require_backend(self):
        if self.ctx is None:
            raise RuntimeError("This stage needs the GEAR backend: open_workspace(..., with_backend=True)")


def open_workspace(task: str | None = "nuclear", base=None, local=None, with_backend: bool = True,
                   run_version: str | None = None, models=None, attach_evo: bool = True,
                   attach_esmif: bool = True, device=None, install_requirements: bool = True,
                   mount: bool = True, gear_dir=None, esm_batch_size: int = 8, backend: str = "native",
                   feature_groups=None, require_legacy: bool = False) -> Workspace:
    """task: "nuclear" | "small" | "metal" | None (anatomy).

    backend="native" (default): features + standardized trainer from pbind itself (no Drive package).
    backend="gear": the Drive ``gear_ultimate`` package (needed only for EVO / graph / old grids).
    feature_groups: restrict the feature store groups built/attached (default: every registered group);
    require_legacy: groups with an original cache must be imported from it entirely (exact reproduction).

    Missing backend caches (ESM-2, AA/SPATIAL/LOCAL, ATOM, CHEM, MECH1, MULTI2) are computed here
    from ``esmFold_pdb_files/``; lower ``esm_batch_size`` if ESM-2 runs out of GPU memory.
    """
    if mount:
        mount_drive()
    paths = resolve_paths(base, local)
    spec = get_task(task) if task in ("nuclear", "small") else None
    label = spec.label if spec else METAL.LABEL

    folds = load_official_folds(find_fold_dir(paths))
    bn24 = find_bindnode24(paths, clone=False)  # only an extra PDB source (AlphaFold) if present
    splits = build_splits(load_raw_frames(paths), official_order(folds))
    print("BASE:", paths.base)
    print("Fold sizes:", {k: len(v) for k, v in folds.items()})

    ws = Workspace(paths=paths, splits=splits, folds=folds, bn24=bn24, device=get_device(device),
                   label=label, task=spec)
    if not with_backend:
        return ws

    prefer = B.OTHER_FIRST if spec else B.METAL_FIRST
    ws.gear = B.load_backend(paths, backend, prefer=prefer, install_requirements=install_requirements,
                             gear_dir=gear_dir)
    if run_version is None:
        run_version = spec.gear_context_version if spec else (
            METAL.CONTEXT_VERSION if task == "metal" else "all_bindings_anatomy_context")
    if models is None:
        models = ("MLP", "KAN", "XGBoost") if task == "metal" else ("MLP",)
    extra = {}
    if backend == "native":  # feature-store options (the Drive backend has its own caches)
        extra = {"feature_groups": feature_groups, "require_legacy": require_legacy}
    ws.ctx = B.prepare_context(ws.gear, paths, label, run_version, models=models, device=ws.device,
                               esm_batch_size=esm_batch_size, **extra)
    for split, n in SPLIT_SIZES.items():
        if len(ws.ctx.stores[STORE_KEY[split]]) != n:
            raise RuntimeError(f"{split}: backend store has {len(ws.ctx.stores[STORE_KEY[split]])} proteins, expected {n}")
    B.check_alignment(ws.ctx, splits)
    print("Backend:", ws.gear.kind, ws.gear.dir or "", "version", ws.gear.version, "| device:", ws.device)

    if attach_evo:
        ok, cov = B.attach_evo(ws.gear, ws.ctx, paths)
        if ok:
            ws.available.add("EVO")
            ws.state["evo_coverage"] = cov
    if attach_esmif and B.attach_esmif_if_cached(ws.ctx, paths, splits):
        ws.available.add("ESMIF")
    print("Optional blocks attached:", sorted(ws.available) or "none")
    return ws
