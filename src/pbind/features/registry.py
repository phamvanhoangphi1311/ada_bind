"""Feature families: how each per-protein block set is built, cached on Drive and attached to stores.

Cache unit = one file per protein:
  * fmt="npz": ``<cache>/<split>/<pid>.npz`` with one array per block
  * fmt="npy": ``<cache>/<split>/<pid>.npy`` for single-block families
A file is reused only if every block has shape (L, dim); otherwise rebuilt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..backend.gear import set_block, store_of
from ..data.structures import pdb_path
from ..paths import Paths
from ..utils.cache import cached_npy, cached_npz
from . import nuclear, small


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    cache_parts: tuple[str, ...]  # under <BASE>/cache
    dims: dict[str, int]
    build: Callable[[object, str], dict]  # (pdb_path, sequence) -> {block: (L, d)}
    fmt: str = "npz"
    needs_structure: bool = True


FEATURE_FAMILIES: dict[str, FeatureFamily] = {
    "nuclear_v1": FeatureFamily(
        "nuclear_v1", ("nuclear_ultimate_v1", "nuclear_explicit_v1"), nuclear.V1_DIMS, nuclear.build_v1_blocks),
    "nuclear_surftopo16": FeatureFamily(
        "nuclear_surftopo16", ("nuclear_v2_selective_multiscale", "surftopo16"), nuclear.SURFTOPO_DIMS,
        nuclear.build_surftopo_block, fmt="npy"),
    "small_v1": FeatureFamily(
        "small_v1", ("small_ultimate_v1", "small_explicit_v1"), small.V1_DIMS, small.build_v1_blocks),
}


def family_cache_dir(paths: Paths, family: FeatureFamily, split: str):
    return paths.cache_dir(*family.cache_parts, split)


def load_or_build_protein(paths: Paths, family: FeatureFamily, split: str, pid: str, seq: str,
                          force: bool = False, build_missing: bool = True) -> tuple[dict, bool]:
    """Blocks of one protein from cache, building (and caching) them if needed."""
    root = family_cache_dir(paths, family, split)
    L = len(seq)

    def build():
        if not build_missing:
            raise FileNotFoundError(f"{family.name}: missing/invalid cache for {split}/{pid}")
        return family.build(pdb_path(paths, split, pid), seq)

    if family.fmt == "npz":
        shapes = {k: (L, d) for k, d in family.dims.items()}
        return cached_npz(root / f"{pid}.npz", build, shapes=shapes, force=force)
    (block, d), = family.dims.items()
    arr, rebuilt = cached_npy(root / f"{pid}.npy", lambda: build()[block], shape=(L, d), force=force)
    return {block: arr}, rebuilt


def ensure_family(paths: Paths, splits: dict[str, pd.DataFrame], family: str | FeatureFamily,
                  ctx=None, force: bool = False, build_missing: bool = True, log_every: int = 250) -> pd.DataFrame:
    """Build missing caches for every protein of every split; attach to ``ctx`` stores if given.

    ``build_missing=False`` = attach-only mode (raises on a missing cache; no PDB needed),
    which is what the standalone cross-task notebooks did.
    """
    fam = FEATURE_FAMILIES[family] if isinstance(family, str) else family
    rows = []
    for split, df in splits.items():
        store = store_of(ctx, split) if ctx is not None else None
        n_built = 0
        for n, r in enumerate(df.itertuples(), start=1):
            blocks, rebuilt = load_or_build_protein(paths, fam, split, str(r.prot_id), r.sequence,
                                                    force=force, build_missing=build_missing)
            n_built += int(rebuilt)
            if store is not None:
                for k, a in blocks.items():
                    set_block(store, r.prot_id, k, np.asarray(a, dtype=np.float32))
            if log_every and n % log_every == 0:
                print(fam.name, split, n, "/", len(df))
        rows.append({"family": fam.name, "split": split, "proteins": len(df), "rebuilt": n_built,
                     "reused": len(df) - n_built})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out
