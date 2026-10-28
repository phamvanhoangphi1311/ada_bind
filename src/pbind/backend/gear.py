"""Adapter for the GEAR ``gear_ultimate`` backend that lives on Drive.

The backend owns: ESM-2 / SPATIAL / LOCAL / ATOM / CHEM / MECH1 / MULTI2 / SEQCTX
caches, EVO48 (MSA) caches, the standardized MLP/KAN/XGBoost trainer used for
all feature screens (``run_dev_cv_only`` / ``run_grid``), graph and historical
KAN experiments. pbind never re-implements those; it loads the backend from
Drive and talks to its stores through the helpers below.

Two backend copies exist on Drive:
  * ``GEAR_ULTIMATE``                — metal lineage (v1.0.0)
  * ``GEAR_ULTIMATE_OTHER_BINDINGS`` — adds ``label_column`` (nuclear / small)
"""
from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd

from ..constants import SPLITS, STORE_KEY
from ..paths import Paths

OTHER_FIRST = ("GEAR_ULTIMATE_OTHER_BINDINGS", "GEAR_ULTIMATE")
METAL_FIRST = ("GEAR_ULTIMATE", "GEAR_ULTIMATE_OTHER_BINDINGS")
_MODULES = ("gear_core", "gear_structure", "gear_mechanistic", "gear_graph",
            "gear_evolution", "gear_suite", "gear_reporting", "gear_legacy")


def find_gear_dir(paths: Paths, prefer: Iterable[str] = OTHER_FIRST) -> Path:
    roots = [paths.base, *paths.siblings()]
    for name in prefer:
        for root in roots:
            d = root / name
            if (d / "gear_ultimate").exists():
                return d
    raise FileNotFoundError(f"GEAR backend not found under {list(prefer)} in {roots}")


def load_gear(paths: Paths, prefer: Iterable[str] = OTHER_FIRST, install_requirements: bool = True,
              gear_dir: str | Path | None = None) -> SimpleNamespace:
    """Import ``gear_ultimate`` from Drive. Returns a namespace (core, suite, evo, graph, ...)."""
    gdir = Path(gear_dir) if gear_dir else find_gear_dir(paths, prefer)
    req = gdir / "requirements.txt"
    if install_requirements and req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)

    loaded = sys.modules.get("gear_ultimate")
    if loaded is not None and Path(loaded.__file__).parent.parent != gdir:
        # a different backend copy is already imported in this runtime -> drop it
        for k in [k for k in sys.modules if k == "gear_ultimate" or k.startswith("gear_ultimate.")]:
            del sys.modules[k]
    if str(gdir) in sys.path:
        sys.path.remove(str(gdir))
    sys.path.insert(0, str(gdir))

    pkg = importlib.import_module("gear_ultimate")
    ns = SimpleNamespace(dir=gdir, package=pkg)
    ns.version = getattr(pkg, "GEAR_ULTIMATE_VERSION", getattr(pkg, "__version__", "unknown"))
    for m in _MODULES:
        try:
            setattr(ns, m.replace("gear_", ""), importlib.import_module(f"gear_ultimate.{m}"))
        except ImportError as e:  # optional modules (legacy) may be missing
            setattr(ns, m.replace("gear_", ""), None)
            print(f"[gear] optional module {m} unavailable: {e!r}")
    ns.mech = ns.mechanistic
    ns.evo = ns.evolution
    ns.has_kan = bool(getattr(ns.core, "HAS_KAN", False))
    ns.has_xgb = True  # the Drive backend imports xgboost unconditionally
    ns.kind = "gear"
    return ns


def load_backend(paths: Paths, kind: str = "native", prefer: Iterable[str] = OTHER_FIRST, **kw) -> SimpleNamespace:
    """kind="native": pbind's own port (no Drive package needed); kind="gear": the Drive gear_ultimate package."""
    if kind == "native":
        from .native import load_native

        return load_native()
    if kind == "gear":
        return load_gear(paths, prefer=prefer, **kw)
    raise ValueError(f"backend must be 'native' or 'gear', got {kind!r}")


def prepare_context(gear, paths: Paths, label: str, run_version: str, models=("MLP",),
                    esm_batch_size: int = 8, device=None, **kw):
    """``suite.prepare_context`` with the notebook defaults. Backend caches are reused automatically."""
    params = dict(base=paths.base, run_version=run_version, models=tuple(models), ensure_base=True,
                  add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=esm_batch_size,
                  device=device)
    params.update(kw)
    if "label_column" in inspect.signature(gear.suite.prepare_context).parameters:
        params["label_column"] = label
    elif label != "metal_binding":
        raise RuntimeError(f"Backend at {gear.dir} has no label_column; use GEAR_ULTIMATE_OTHER_BINDINGS.")
    return gear.suite.prepare_context(**params)


# ------------------------------------------------------------------ store access
def store_of(ctx, split: str) -> dict:
    return ctx.stores[STORE_KEY.get(split, split)]


def get_record(store, pid):
    pid = str(pid)
    if pid in store:
        return store[pid]
    raise KeyError(pid)


def get_block(store, pid, block) -> np.ndarray:
    rec = get_record(store, pid)
    if isinstance(rec, dict):
        if block in rec:
            return np.asarray(rec[block])
        if isinstance(rec.get("features"), dict) and block in rec["features"]:
            return np.asarray(rec["features"][block])
    if hasattr(rec, block):
        return np.asarray(getattr(rec, block))
    for attr in ("features", "feature_dict", "blocks"):
        d = getattr(rec, attr, None)
        if isinstance(d, dict) and block in d:
            return np.asarray(d[block])
    raise KeyError(f"{pid}/{block}")


def has_block(store, pid, block) -> bool:
    try:
        get_block(store, pid, block)
        return True
    except KeyError:
        return False


def set_block(store, pid, block, arr) -> None:
    rec = get_record(store, pid)
    arr = np.asarray(arr, dtype=np.float32)
    if isinstance(rec, dict):
        if isinstance(rec.get("features"), dict):
            rec["features"][block] = arr
        else:
            rec[block] = arr
        return
    for attr in ("features", "feature_dict", "blocks"):
        d = getattr(rec, attr, None)
        if isinstance(d, dict):
            d[block] = arr
            return
    setattr(rec, block, arr)


def concat_blocks(store, pid, blocks) -> np.ndarray:
    xs = [get_block(store, pid, b).astype(np.float32, copy=False) for b in blocks]
    if len({x.shape[0] for x in xs}) != 1:
        raise RuntimeError(f"{pid}: block length mismatch")
    return np.concatenate(xs, axis=1)


def check_alignment(ctx, splits: dict[str, pd.DataFrame]) -> None:
    for split, df in splits.items():
        store_ids = set(map(str, store_of(ctx, split).keys()))
        csv_ids = set(df.prot_id.astype(str))
        if store_ids != csv_ids:
            raise RuntimeError(
                f"{split}: store/CSV mismatch. missing_in_store={sorted(csv_ids - store_ids)[:10]} "
                f"extra_in_store={sorted(store_ids - csv_ids)[:10]}"
            )


# ------------------------------------------------------------------ optional blocks
def attach_evo(gear, ctx, paths: Paths, splits=("train", "test300", "test46"), require_msa: bool = True):
    """Attach cached EVO48. Returns (ok, coverage_df_or_error)."""
    try:
        cov = gear.evo.attach_evo(ctx.stores, paths.base, splits=tuple(splits), require_msa=require_msa)
        return True, cov
    except Exception as e:  # incomplete MSA cache -> continue without EVO
        print("EVO cache unavailable/incomplete -> skip EVO:", repr(e))
        return False, e


ESMIF_ROOTS = (("gear_metal_v3", "esmif_v1"), ("nuclear_ultimate_v1", "esmif_v1"))


def attach_esmif_if_cached(ctx, paths: Paths, splits: dict[str, pd.DataFrame]) -> bool:
    """Reuse-only: attach ESM-IF (512-d) if a COMPLETE per-protein cache exists. Never computes it."""
    for parts in ESMIF_ROOTS:
        root = paths.cache.joinpath(*parts)
        dirs = {s: root / s / "esm_if" for s in SPLITS}
        if not all(d.exists() for d in dirs.values()):
            continue
        good = True
        for split, df in splits.items():
            for r in df.itertuples():
                fp = dirs[split] / f"{r.prot_id}.npy"
                if not fp.exists() or np.load(fp, mmap_mode="r").shape != (len(r.sequence), 512):
                    good = False
                    break
            if not good:
                break
        if not good:
            continue
        for split, df in splits.items():
            store = store_of(ctx, split)
            for r in df.itertuples():
                set_block(store, r.prot_id, "ESMIF", np.load(dirs[split] / f"{r.prot_id}.npy").astype(np.float32))
        print("Reused complete ESMIF cache:", root)
        return True
    return False


# ------------------------------------------------------------------ screens
def run_dev_cv(gear, ctx, grid: dict, run_version: str, models=("MLP",)):
    """Dev-only official 5-fold CV through the backend trainer.

    Resume/caching is per fold inside the backend:
    ``results/<run_version>/<label>/dev_only_predictions/<model>__<rep>__fold<k>.pkl``.
    """
    missing = {rep: [b for b in blocks if not has_block(store_of(ctx, "train"),
                                                         next(iter(store_of(ctx, "train"))), b)]
               for rep, blocks in grid.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(f"Blocks not attached to the store: {missing}")
    return gear.suite.run_dev_cv_only(ctx, grid, models=tuple(models), run_version=run_version)


def dimension_table(gear, ctx) -> pd.DataFrame:
    return gear.reporting.dimension_table(store_of(ctx, "train"))


def leakage_audit(gear, ctx, blocks) -> pd.DataFrame:
    return gear.reporting.validate_no_direct_label_features(ctx.stores, list(blocks))
