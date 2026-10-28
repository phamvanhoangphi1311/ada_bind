"""Model registry: every trainable model is one named entry; experiments pick it by name.

kind      input                                         trained by
residue   one residue = concatenated feature blocks     GEAR standardized trainer (8-protein batches, train-only
                                                        standardizer, BCE pos_weight, early stop on val AUPRC)
tree      same as residue                               XGBoost (GEAR settings)
fusion    blocks grouped into modalities                local 5-fold CV trainer (training/local_cv.py)
group     modalities + patch/pocket groups (hypergraph) group 5-fold CV trainer (training/group_cv.py)

``build`` signature per kind:
    residue  build(d: int, cfg: GearConfig, seed: int) -> nn.Module   ((N, d) -> (N,) logit, or (N, 2) if "softmax2")
    fusion   build(dims: dict[str, int], task: TaskSpec) -> nn.Module
    group    build(dims: dict[str, int], static_dim: int, task: TaskSpec) -> nn.Module
    tree     no build (the trainer constructs the booster)

Add a model anywhere, then select it by name in ``pbind.api``:

    @register_model("WideMLP", kind="residue", description="d -> 512 -> 1")
    def wide_mlp(d, cfg, seed):
        return nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(512, 1), nn.Flatten(0))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .fusion import ConcatMLP, GatedFusion, LatentExpertMoE, ModalityAttention, ProjectedConcat, ResidualFusion
from .hypergraph import GroupHypergraph
from .residue import ResidueKAN, ResidueMLP

KINDS = ("residue", "tree", "fusion", "group")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    build: Callable | None
    output: str = "logit"  # residue: "logit" (sigmoid, BCE) | "softmax2" (2-class softmax, CE)
    seed_offset: int = 0  # GEAR fold seed = cfg.seed + fold + 1000 * seed_offset
    batch_attr: str = "proteins_per_batch_mlp"  # GearConfig field with the proteins per batch
    description: str = ""


MODELS: dict[str, ModelSpec] = {}


def register_model(name: str, kind: str, *, output: str = "logit", seed_offset: int | None = None,
                   batch_attr: str = "proteins_per_batch_mlp", description: str = ""):
    """Decorator: register ``build`` under ``name`` (see the module docstring for its signature)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if output not in ("logit", "softmax2"):
        raise ValueError("output must be 'logit' or 'softmax2'")

    def deco(build):
        if name in MODELS and MODELS[name].build is not build:
            raise ValueError(f"model {name!r} registered twice")
        offset = seed_offset if seed_offset is not None else max([m.seed_offset for m in MODELS.values()] + [0]) + 1
        MODELS[name] = ModelSpec(name, kind, build, output, offset, batch_attr,
                                 description or (getattr(build, "__doc__", None) or "").strip())
        return build

    return deco


def resolve_model(name: str, task=None) -> ModelSpec:
    """Spec of a registered model; task-specific names (e.g. NA_PatchHG_Attention, PatchResidualFusion,
    LatentNAExpertMoE, SM_PocketHG_Attention_MS) map to the generic entry they are built from."""
    if name in MODELS:
        return MODELS[name]
    if task is not None:
        if name == task.residual_arch:
            return MODELS["ResidualFusion"]
        if name == task.moe_arch:
            return MODELS["LatentExpertMoE"]
        if name == task.ms_hg:
            return MODELS["HG_Attention"]
        if name in task.hypergraphs:
            cfg = task.hypergraphs[name]
            return MODELS["HG_GatedMoE" if cfg.get("moe") else f"HG_{cfg['mode'].title()}"]
    raise KeyError(f"unknown model {name!r}; registered: {list(MODELS)}")


def get_model(name: str) -> ModelSpec:
    return resolve_model(name)


def list_models(kind: str | None = None) -> pd.DataFrame:
    rows = [{"model": m.name, "kind": m.kind, "output": m.output if m.kind == "residue" else "",
             "seed_offset": m.seed_offset, "description": m.description.splitlines()[0] if m.description else ""}
            for m in MODELS.values() if kind is None or m.kind == kind]
    return pd.DataFrame(rows)


# ====================================================================== built-in models
# residue / tree: the GEAR model families (seed offsets 1 / 2 / 3 are the original backend's)
register_model("MLP", "residue", seed_offset=1, description="GEAR ResidueMLP: d -> 256 -> 64 -> 1")(
    lambda d, cfg, seed: ResidueMLP(d, cfg.dropout))
register_model("KAN", "residue", output="softmax2", seed_offset=2, batch_attr="proteins_per_batch_kan",
               description="Linear(d, 32) -> KAN [32, 8, 2] (needs pykan)")(
    lambda d, cfg, seed: ResidueKAN(d, seed))
register_model("XGBoost", "tree", seed_offset=3, description="XGBoost hist, GEAR settings")(None)

# fusion: the nuclear/small V1 architecture screen
register_model("ConcatMLP", "fusion", description="concatenate modalities -> MLP")(lambda dims, task: ConcatMLP(dims))
register_model("ProjectedConcat", "fusion", description="per-modality projection, then concatenation")(
    lambda dims, task: ProjectedConcat(dims))
register_model("GatedFusion", "fusion", description="per-modality encoders mixed by learned gates")(
    lambda dims, task: GatedFusion(dims))
register_model("ModalityAttention", "fusion", description="attention over modality tokens")(
    lambda dims, task: ModalityAttention(dims))
register_model("LatentExpertMoE", "fusion", description="mixture of latent experts over modalities")(
    lambda dims, task: LatentExpertMoE(dims))
register_model("ResidualFusion", "fusion", description="ESM path + residual from the task's patch/pocket modality")(
    lambda dims, task: ResidualFusion(dims, residual_key=task.residual_key, prefix=task.residual_prefix))


# group: patch/pocket hypergraphs (bank "v1" = V1 model, bank "ms" = multi-scale *_MS model)
def _hypergraph(mode: str, moe: bool):
    def build(dims, static_dim, task):
        return GroupHypergraph(dims, group=task.group, static_dim=static_dim, mode=mode, latent=task.latent,
                               moe=moe, experts=4)

    return build


register_model("HG_Gated", "group", description="patch/pocket hypergraph, gated group->residue message")(
    _hypergraph("gated", False))
register_model("HG_Attention", "group", description="patch/pocket hypergraph, attention message (final V1 / MS)")(
    _hypergraph("attention", False))
register_model("HG_GatedMoE", "group", description="gated hypergraph with 4 latent experts")(
    _hypergraph("gated", True))
