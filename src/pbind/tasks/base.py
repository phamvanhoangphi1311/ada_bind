"""TaskSpec: everything that differs between the nuclear and small pipelines.

The pipeline code in ``pbind.pipelines`` is task-agnostic; it reads names,
grids, cache/result folders and hyper-parameters from a TaskSpec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.fusion import fusion_factory
from ..models.hypergraph import GroupHypergraph, GroupMessageV2
from ..training.group_cv import LossConfig
from ..training.loop import TrainConfig


@dataclass(frozen=True)
class GridExtra:
    """Representation added to a grid only when all of ``extra`` blocks are attached (EVO / ESMIF)."""
    name: str
    base: str  # name of an existing grid entry, or "" for a fresh list
    extra: tuple[str, ...]


@dataclass(frozen=True)
class V2Spec:
    run_root: str
    code_version: str
    bank: str
    variants: dict  # name -> {"mode": none|mean|cross, "hardneg": bool}
    complexity: tuple  # variant names, simplest first
    ablation: tuple  # (label, better, reference)
    loss: LossConfig
    train: TrainConfig
    relock_grid: dict | None = None  # nuclear re-locks the representation (adds NA_SURFTOPO16)
    relock_run_version: str = ""
    relock_feature_family: str = ""
    modality_groups: tuple = ()  # used when relocking
    name_key: str = "variant"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    gear_context_version: str
    run_root: str
    cache_root: str
    feature_families: tuple[str, ...]
    new_blocks: dict
    audit_blocks: tuple[str, ...]
    family_grid_base: dict
    family_extras: tuple[GridExtra, ...]
    mechanism_grid_base: dict
    mechanism_extras: tuple[GridExtra, ...]
    run_version_family: str
    run_version_mechanism: str
    modality_groups: tuple
    residual_key: str
    residual_prefix: str
    residual_arch: str
    moe_arch: str
    group: str  # "patch" | "pocket"
    group_family: str  # "PATCH_HYPERGRAPH" | "POCKET_HYPERGRAPH"
    bank_v1: str
    bank_ms: str
    hypergraphs: dict  # name -> {"mode": gated|attention, "moe": bool}
    v1_hg: str
    ms_hg: str
    ms_run_root: str
    ensemble_root: str
    paper_root: str
    group_state_dir: str
    local_train: TrainConfig
    group_train: TrainConfig
    latent: int = 192
    prior_runs: tuple = ()
    mechanism_report: tuple = ()
    v2: V2Spec | None = None
    xtask: dict = field(default_factory=dict)
    final_representation: str = ""  # Dev-only lock of the original ladder (a mechanism_grid_base key)
    original_final: dict = field(default_factory=dict)  # numbers printed by the original paper-ready cell

    # ---------------- grids
    @staticmethod
    def _grid(base: dict, extras, available: set) -> dict:
        grid = {k: list(v) for k, v in base.items()}
        for e in extras:
            if set(e.extra) <= available:
                grid[e.name] = [*(grid[e.base] if e.base else []), *e.extra]
        return grid

    def family_grid(self, available=frozenset()) -> dict:
        return self._grid(self.family_grid_base, self.family_extras, set(available))

    def mechanism_grid(self, available=frozenset()) -> dict:
        return self._grid(self.mechanism_grid_base, self.mechanism_extras, set(available))

    # ---------------- model factories
    def fusion_registry(self, dims: dict) -> dict:
        return fusion_factory(dims, self.residual_key, self.residual_prefix, self.residual_arch, self.moe_arch)

    def make_hypergraph(self, name: str, dims: dict, static_dim: int) -> GroupHypergraph:
        cfg = self.hypergraphs[name]
        return GroupHypergraph(dims, group=self.group, static_dim=static_dim, mode=cfg["mode"],
                               latent=self.latent, moe=cfg.get("moe", False), experts=cfg.get("experts", 4))

    def make_v2(self, variant: str, dims: dict, static_dim: int) -> GroupMessageV2:
        return GroupMessageV2(dims, group=self.group, static_dim=static_dim,
                              message_mode=self.v2.variants[variant]["mode"], latent=self.latent, heads=6, drop=0.20)
