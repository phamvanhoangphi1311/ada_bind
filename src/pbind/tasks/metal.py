"""Metal binding: coordination chemistry (GEAR backend owns features + trainer)."""
from ..constants import GEAR_BLOCK_DIMS, MECH1, MULTI2

LABEL = "metal_binding"
RUN_ROOT = "GEAR_METAL_ONLY_ULTIMATE_V2"
CONTEXT_VERSION = "metal_v2_context"

FAMILIES = {
    "Structure": ["SPATIAL", "LOCAL"],
    "Atomic": ["ATOM"],
    "Chem": ["CHEM"],
    "MECH1": list(MECH1),
    "MULTI2": list(MULTI2),
    "SeqCtx": ["SEQCTX"],
}
COMPONENTS = {b: [b] for b in [*MECH1, *MULTI2]}
CORE_BLOCKS = ["ESM", "ATOM", "CHEM"]

# Stage A — family screen (every candidate = ESM + one family)
FAMILY_GRID = {"ESM": ["ESM"], **{f"ESM+{k}": ["ESM", *v] for k, v in FAMILIES.items()}}

# Stage B — component screen conditioned on ESM+ATOM+CHEM
COMPONENT_GRID = {
    "Core_ESM+Atomic+Chem": CORE_BLOCKS,
    **{f"Core+{k}": [*CORE_BLOCKS, *v] for k, v in COMPONENTS.items()},
    "Core+MECH1": [*CORE_BLOCKS, *MECH1],
    "Core+MULTI2": [*CORE_BLOCKS, *MULTI2],
}

# Stage C — combination ladder (the only place the representation is locked from)
_F = FAMILIES
COMBO_GRID = {
    "ESM": ["ESM"],
    "ESM+Atomic": ["ESM", "ATOM"],
    "ESM+Chem": ["ESM", "CHEM"],
    "ESM+Atomic+Chem": CORE_BLOCKS,
    "ESM+Atomic+Chem+MECH1": [*CORE_BLOCKS, *_F["MECH1"]],
    "ESM+Atomic+Chem+MULTI2": [*CORE_BLOCKS, *_F["MULTI2"]],
    "ESM+Atomic+Chem+MECH1+MULTI2": [*CORE_BLOCKS, *_F["MECH1"], *_F["MULTI2"]],
    "ESM+Atomic+Chem+MULTI2+Structure": [*CORE_BLOCKS, *_F["MULTI2"], *_F["Structure"]],
    "ESM+Atomic+Chem+MULTI2+SeqCtx": [*CORE_BLOCKS, *_F["MULTI2"], *_F["SeqCtx"]],
    "ESM+Atomic+Chem+MECH1+MULTI2+Structure": [*CORE_BLOCKS, *_F["MECH1"], *_F["MULTI2"], *_F["Structure"]],
    "ESM+Atomic+Chem+MECH1+MULTI2+SeqCtx": [*CORE_BLOCKS, *_F["MECH1"], *_F["MULTI2"], *_F["SeqCtx"]],
    "ESM+ALL_EXPLICIT": [*CORE_BLOCKS, *_F["MECH1"], *_F["MULTI2"], *_F["Structure"], *_F["SeqCtx"]],
}

# Dev-only decision of GEAR_METAL_ONLY_ULTIMATE_V2 (combination ladder -> model family); used when the
# screens have not been re-run in this workspace.
FINAL_LOCK = {
    "locked_representation": "ESM+Atomic+Chem+MULTI2",
    "locked_blocks": [*CORE_BLOCKS, *MULTI2],
    "locked_model": "MLP",
}
# Output of the original final locked run (notebook cell "11. FINAL LOCKED EXTERNAL RUN", run hash 5c13eb618253).
ORIGINAL_FINAL = {
    "fold_AUPRC": [0.4569, 0.4593, 0.4758, 0.4805, 0.4211],
    "fold_AUROC": [0.8406, 0.9024, 0.8852, 0.8727, 0.8766],
    "CV_AUPRC": 0.458725, "CV_AUPRC_SD": 0.023387, "CV_AUROC": 0.875499, "CV_F1_at_0_5": 0.303638,
    "CV_MCC_at_0_5": 0.336113, "mean_best_epoch": 4.2, "n_params": 369793,
    "OOF_tuned_threshold": 0.9389,
    "external": {
        "TestSet300": {"AUPRC": 0.381428, "AUROC": 0.837568, "F1_at_0.5": 0.313983, "MCC_at_0.5": 0.322743,
                       "F1_at_OOF_threshold": 0.458109, "MCC_at_OOF_threshold": 0.485231},
        "TestSetNew46": {"AUPRC": 0.685333, "AUROC": 0.969393, "F1_at_0.5": 0.435088, "MCC_at_0.5": 0.480610,
                         "F1_at_OOF_threshold": 0.706767, "MCC_at_OOF_threshold": 0.712813},
    },
}

RUN_VERSIONS = {
    "family": "metal_v2_01_family_mlp",
    "component": "metal_v2_02_component_mlp",
    "combo": "metal_v2_03_combo_mlp",
    "model": "metal_v2_04_locked_model_family",
    "graph": "metal_v2_05_graph_diagnostic",
    "final": "metal_v2_06_final_locked_external",
}

# Original GEAR_ULTIMATE_METAL_BINDING notebook run versions
ULTIMATE_RUN_VERSIONS = {
    "context": "ultimate_context_v1",
    "standardized_13": "ultimate_standardized_13_v1",
    "mechanistic": "ultimate_mechanistic_grid_v1",
    "final_focused": "ultimate_final_focused_v1",
    "no_esm": "ultimate_no_esm_v1",
    "maximal_graph": "ultimate_maximal_noesm_graph_v1",
    "clean_graph": "ultimate_clean_graph_v2",
    "evo_dev": "ultimate_evo_dev_only_v1",
    "evo_full": "ultimate_evo_full_v1",
    "evo_graph": "ultimate_evo_graph_v1",
}

EVO_GRAPH_EXPERIMENTS = {
    "PureGraph_AA_Chem_EVO": {"blocks": ["AA", "CHEM", "EVO"], "architecture": "pure", "edge_mode": "rich"},
    "ResidualGraph_AA_Atomic_Chem_EVO": {"blocks": ["AA", "ATOM", "CHEM", "EVO"], "architecture": "residual",
                                         "edge_mode": "rich"},
    "ResidualGraph_DonorFocused_EVO": {"blocks": ["AA", "ATOM", "CHEM", "EVO"], "architecture": "residual",
                                       "edge_mode": "donor"},
    "ResidualGraph_Compact_EVO": {"blocks": ["AA", "ATOM", "CHEM", *MULTI2, "EVO"], "architecture": "residual",
                                  "edge_mode": "rich"},
}


def dims_of(grid: dict) -> dict:
    return {k: int(sum(GEAR_BLOCK_DIMS[b] for b in v)) for k, v in grid.items()}
