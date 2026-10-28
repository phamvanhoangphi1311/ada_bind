"""Dataset anatomy audit stage (ALL_BINDINGS_DATASET_ANATOMY notebook)."""
from __future__ import annotations

from ..analysis.anatomy import run_anatomy
from ..constants import DEV
from .workspace import Workspace

OUT_ROOT = ("results", "ALL_BINDINGS_DATASET_ANATOMY_V1")


def dataset_anatomy(ws: Workspace, with_feature_fingerprint: bool = True, force: bool = False) -> dict:
    """All audit tables (cached CSV). The feature fingerprint needs a backend workspace."""
    out = ws.paths.base.joinpath(*OUT_ROOT)
    ctx = ws.ctx if with_feature_fingerprint else None
    return run_anatomy(ws.splits, out, reference=DEV, ctx=ctx, force=force)
