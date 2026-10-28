# Importing the feature modules registers their @feature_group functions (GEAR blocks first, then task blocks).
from . import base_structure, coordination, esm, mechanistic  # noqa: F401
from . import geometry, nuclear, small
from .api import FEATURE_GROUPS, Protein, feature_group, list_features, resolve_blocks
from .derived import attach_crosstask_block, attach_evo_access_residual, evo_access_residual
from .registry import FEATURE_FAMILIES, FeatureFamily, ensure_family, load_or_build_protein
