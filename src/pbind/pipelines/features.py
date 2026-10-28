"""Stage: structures + task-specific feature blocks (all cached per protein on Drive)."""
from __future__ import annotations

import pandas as pd

from .. import backend as B
from ..constants import STORE_KEY
from ..data.structures import ensure_normalized_pdbs
from ..features.registry import FEATURE_FAMILIES, ensure_family
from ..groups.bank import load_bank, summarize_audit
from ..paths import resolve_paths
from ..utils.cache import save_table
from .workspace import Workspace


def ensure_structures(ws: Workspace, repack: str = "if_changed") -> dict:
    """Exact chain-A PDBs in local scratch (restored from / packed to the Drive archive)."""
    if "structures" not in ws.state:
        ws.state["structures"] = ensure_normalized_pdbs(ws.paths, ws.splits, ws.bn24, repack=repack)
    return ws.state["structures"]


def build_features(ws: Workspace, families=None, force: bool = False, attach_only: bool = False) -> pd.DataFrame:
    """Build (if missing) and attach the task's feature families to the backend stores.

    attach_only=True never parses PDBs: it requires complete caches (standalone/x-task mode).
    """
    ws.require_backend()
    families = tuple(families or ws.task.feature_families)
    item = next(iter(ws.ctx.stores["train"].values()))
    if not force and all(b in item for f in families for b in FEATURE_FAMILIES[f].dims):
        # native backend: the feature store already attached these blocks (features/store.py)
        for f in families:
            ws.available.update(FEATURE_FAMILIES[f].dims)
        return pd.DataFrame([{"family": f, "source": "feature_store"} for f in families])
    if not attach_only and any(FEATURE_FAMILIES[f].needs_structure for f in families):
        ensure_structures(ws)
    stats = [ensure_family(ws.paths, ws.splits, f, ctx=ws.ctx, force=force, build_missing=not attach_only)
             for f in families]
    for f in families:
        ws.available.update(FEATURE_FAMILIES[f].dims)
    return pd.concat(stats, ignore_index=True)


def feature_audit(ws: Workspace) -> pd.DataFrame:
    """Label-independence / completeness audit of the blocks this task uses."""
    blocks = [*ws.task.audit_blocks, *ws.task.new_blocks, *sorted(ws.available & {"EVO", "ESMIF"})]
    return B.leakage_audit(ws.gear, ws.ctx, blocks)


def group_bank(ws: Workspace, which: str = "v1", force: bool = False):
    """Patch/pocket bank: which = "v1" | "ms" | a GROUP_BANKS key."""
    name = {"v1": ws.task.bank_v1, "ms": ws.task.bank_ms}.get(which, which)
    key = f"bank:{name}"
    if key not in ws.state or force:
        ensure_structures(ws)
        bank, audit = load_bank(ws.paths, ws.splits, name, force=force)
        save_table(ws.paths.results_dir(_bank_results_root(ws, name)) / f"{bank.group}_audit.csv", audit)
        print(summarize_audit(audit).to_string(index=False))
        ws.state[key] = bank
    return ws.state[key]


def build_store(base=None, local=None, groups=None, esm_batch_size: int = 8, device=None, force: bool = False,
                legacy: bool = True):
    """Fill the residue feature store (``<BASE>/feature_store``) for the 3 splits, without opening a workspace.
    groups: names in ``pbind.features.api.FEATURE_GROUPS`` (default: all); finished groups are reused unless
    ``force``; ``legacy`` imports the original backend caches found in ``cache/spatial_revamp/``."""
    import torch

    from ..data.folds import find_fold_dir, load_official_folds
    from ..data.loading import backend_frames
    from ..features.store import build_feature_store

    paths = resolve_paths(base, local)
    folds = load_official_folds(find_fold_dir(paths))
    frames = backend_frames(paths, [p for ids in folds.values() for p in ids])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_feature_store(paths, {split: frames[key] for split, key in STORE_KEY.items()},
                               groups, device=device, esm_batch_size=esm_batch_size, force=force, legacy=legacy)


def precompute_all(tasks=("metal", "nuclear", "small"), base=None, local=None, esm_batch_size: int = 8,
                   include_ms: bool = True, backend: str = "native") -> dict:
    """Compute (or verify) every feature cache the three final models need, from raw inputs only.

    Raw inputs: the 3 CSVs + fold files + esmFold_pdb_files/. Steps, each skipped if already cached:
      1. backend  : ESM-2 650M (HuggingFace, downloaded on first use), AA/SPATIAL/LOCAL, ATOM, CHEM,
                    MECH1 (ACCESS...), MULTI2  -> cache/spatial_revamp/<split>_*.pkl
      2. nuclear  : exact chain-A PDBs, NA_* blocks, V1 + multi-scale patch banks
      3. small    : SM_* blocks, V1 + multi-scale pocket banks
    """
    import gc

    from .workspace import open_workspace

    done = {}
    for task in tasks:
        print("\n" + "=" * 80 + f"\nPRECOMPUTE: {task}\n" + "=" * 80)
        ws = open_workspace(task, base=base, local=local, mount=False, attach_evo=False, attach_esmif=False,
                            esm_batch_size=esm_batch_size, backend=backend)
        if task in ("nuclear", "small"):
            build_features(ws)
            group_bank(ws, "v1")
            if include_ms:
                group_bank(ws, "ms")
        done[task] = "ok"
        del ws
        gc.collect()
    return done


def _bank_results_root(ws: Workspace, name: str) -> str:
    t = ws.task
    if name == t.bank_v1:
        return t.run_root
    if t.v2 is not None and name == t.v2.bank:
        return t.v2.run_root
    return t.ms_run_root
