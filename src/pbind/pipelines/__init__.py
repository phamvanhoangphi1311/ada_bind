"""Notebook stages as functions. Typical use (nuclear or small):

    from pbind.pipelines import open_workspace, features, screens, v1, v2, final
    ws = open_workspace("nuclear")
    features.build_features(ws)          # NA_* blocks (cached per protein)
    screens.family_screen(ws)            # Stage A
    screens.mechanism_ladder(ws)         # Stage B + representation lock
    v1.architecture_screen(ws)           # fusion architectures
    v1.hypergraph_screen(ws)             # patch / pocket hypergraph
    v1.global_lock(ws); v1.external_evaluation(ws)
    v2.v2_ablation(ws); v2.v2_external(ws)
    final.multiscale_screen(ws); final.paper_report(ws)

Every stage reuses caches and computes the stages it depends on if needed.
"""
from . import anatomy, crosstask, features, final, metal, screens, v1, v2
from .workspace import Workspace, open_workspace
