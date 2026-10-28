"""Feature API: one registry for every residue-level feature group.

A feature group is ONE function that turns one protein into named (L, d) blocks, registered with its
column names. The feature store (``features/store.py``) iterates the registry and calls these functions;
nothing else needs to know how a block is computed.

    @feature_group("MECH1", columns={"ACCESS": [...], "SHELL2": [...], ...}, needs="structure")
    def mech1(p: Protein) -> dict[str, np.ndarray]:
        return extract_mechanistic_features(p.residues)

needs = "sequence"   uses only ``p.sequence``
        "structure"  uses ``p.residues`` (PDB parsed + aligned to the CSV sequence) or ``p.pdb``
        "batch"      ``fn(proteins, opts) -> {pid: {block: arr}}`` over a whole split (e.g. ESM-2 on GPU)
structure = "esmfold"        ``p.pdb`` = the ESMFold file (GEAR blocks)
            "exact_chain_a"  ``p.pdb`` = the exact-sequence chain-A copy (nuclear / small blocks)

``legacy_cache`` maps a block to an original cache file under ``<BASE>/cache/<legacy_subdir>/``: either a per-split
pickle ``"{split}_....pkl"`` ({pid: array or {block: array}}) or a per-protein file ``".../{split}/{pid}.npz"``
(one array per block) / ``"{pid}.npy"`` (single block). Proteins found there are imported instead of recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

NEEDS = ("sequence", "structure", "batch")


@dataclass
class BuildOptions:
    """Run-time options handed to ``needs="batch"`` groups."""
    device: object = None
    max_residues: int = 1000
    esm_batch_size: int = 8


@dataclass
class Protein:
    """One protein as the feature functions see it (sequence already truncated to ``max_residues``)."""
    pid: str
    sequence: str
    split: str
    pdb: Path | None = None

    @cached_property
    def residues(self) -> list:
        """PDB residues aligned to ``sequence`` (same parsing + identity check as the GEAR backend)."""
        from ..data.structures import parse_protein_residues, validate_feature_alignment

        if self.pdb is None:
            raise ValueError(f"{self.pid}: no structure available")
        res = parse_protein_residues(self.pdb, len(self.sequence))
        n, _ = validate_feature_alignment(self.pid, self.sequence, res)
        return res[:n]


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    columns: Mapping[str, list[str]]  # block -> column names (block order = output order)
    fn: Callable
    needs: str = "structure"
    dtypes: Mapping[str, str] = field(default_factory=dict)  # block -> dtype (default float32)
    legacy_cache: Mapping[str, str] = field(default_factory=dict)  # block -> file pattern (see module docstring)
    description: str = ""
    structure: str = "esmfold"  # "esmfold" | "exact_chain_a"
    legacy_subdir: str = "spatial_revamp"  # legacy_cache patterns are relative to <BASE>/cache/<legacy_subdir>

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def dim(self, block: str) -> int:
        return len(self.columns[block])

    def dtype(self, block: str) -> np.dtype:
        return np.dtype(self.dtypes.get(block, "float32"))


FEATURE_GROUPS: dict[str, FeatureGroup] = {}


def feature_group(name: str, columns: Mapping[str, list[str]], *, needs: str = "structure", dtypes=None,
                  legacy_cache=None, description: str = "", structure: str = "esmfold",
                  legacy_subdir: str = "spatial_revamp"):
    """Decorator: register ``fn`` as the extractor of feature group ``name``."""
    if needs not in NEEDS:
        raise ValueError(f"needs must be one of {NEEDS}")
    if structure not in ("esmfold", "exact_chain_a"):
        raise ValueError("structure must be 'esmfold' or 'exact_chain_a'")

    def deco(fn: Callable) -> Callable:
        if name in FEATURE_GROUPS and FEATURE_GROUPS[name].fn is not fn:
            raise ValueError(f"feature group {name!r} registered twice")
        clash = {b for b in columns for g in FEATURE_GROUPS.values() if g.name != name and b in g.columns}
        if clash:
            raise ValueError(f"{name}: blocks {sorted(clash)} already belong to another group")
        FEATURE_GROUPS[name] = FeatureGroup(name, dict(columns), fn, needs, dict(dtypes or {}),
                                            dict(legacy_cache or {}), description or (fn.__doc__ or "").strip(),
                                            structure, legacy_subdir)
        return fn

    return deco


def group_of(block: str) -> FeatureGroup:
    for g in FEATURE_GROUPS.values():
        if block in g.columns:
            return g
    raise KeyError(f"unknown feature block {block!r}; known: {all_blocks()}")


def all_blocks() -> list[str]:
    return [b for g in FEATURE_GROUPS.values() for b in g.blocks]


# Representation names used by the notebooks (e.g. "ESM+Atomic+Chem+MULTI2").
ALIASES = {"Atomic": ["ATOM"], "Chem": ["CHEM"], "Structure": ["SPATIAL", "LOCAL"], "SeqCtx": ["SEQCTX"]}


def resolve_blocks(features) -> list[str]:
    """Blocks of a feature request: "ESM+Atomic+Chem+MULTI2", ["ESM", "MECH1"], ["ESM", "ATOM", ...].
    Tokens may be blocks, group names (expanded to their blocks) or notebook aliases (Atomic, Chem, ...);
    unknown tokens are kept as-is (blocks attached outside the store, e.g. NA_* / SM_* / EVO)."""
    tokens = features.split("+") if isinstance(features, str) else list(features)
    out = []
    for t in tokens:
        t = t.strip()
        if t in ALIASES:
            out += ALIASES[t]
        elif t in FEATURE_GROUPS and t not in FEATURE_GROUPS[t].columns:
            out += list(FEATURE_GROUPS[t].blocks)
        else:
            out.append(t)
    return list(dict.fromkeys(out))


def list_features() -> pd.DataFrame:
    """Every registered block: group, dimension, input it needs, description."""
    return pd.DataFrame([{"group": g.name, "block": b, "dim": g.dim(b), "dtype": g.dtype(b).name, "needs": g.needs,
                          "structure": g.structure if g.needs == "structure" else "",
                          "description": g.description.splitlines()[0] if g.description else ""}
                         for g in FEATURE_GROUPS.values() for b in g.blocks])
