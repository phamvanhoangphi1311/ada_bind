#!/usr/bin/env python
"""Run pipeline stages from the command line (Colab terminal / VM). Every stage resumes from cache.

Examples
    python scripts/run_pipeline.py nuclear --stages final_models      # final model only (reuses checkpoints)
    python scripts/run_pipeline.py nuclear --stages features family mechanism
    python scripts/run_pipeline.py small --stages all
    python scripts/run_pipeline.py metal --stages all --policy one_se
    python scripts/run_pipeline.py nuclear --stages export xtask evo_hg     # cross-task add-in
    python scripts/run_pipeline.py anatomy
    PBIND_BASE=/data/Protein-binding-Phi python scripts/run_pipeline.py nuclear --stages features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.pipelines import anatomy, crosstask, features, final, metal, open_workspace, screens, v1, v2  # noqa: E402

TASK_STAGES = {
    "structures": features.ensure_structures,
    "features": features.build_features,
    "audit": features.feature_audit,
    "family": screens.family_screen,
    "mechanism": screens.mechanism_ladder,
    "arch": v1.architecture_screen,
    "hypergraph": v1.hypergraph_screen,
    "lock": v1.global_lock,
    "external": v1.external_evaluation,
    "baselines": v1.baselines,
    "insight": v1.insight_summary,
    "v2": v2.v2_ablation,
    "v2_external": v2.v2_external,
    "ms": final.multiscale_screen,
    "ms_external": final.multiscale_external,
    "paper": final.paper_report,
    "final_models": final.final_models,  # V1 + MS + ensemble only (reuses fold checkpoints)
    "manifest": v1.manifest,
    # cross-task add-in
    "export": crosstask.export_dev_oof,
    "xtask": crosstask.crosstask_screen,
    "evo_hg": crosstask.evo_hypergraph_addin,
}
ALL_TASK = ["features", "family", "mechanism", "arch", "hypergraph", "lock", "external", "baselines", "insight",
            "v2", "v2_external", "ms", "ms_external", "paper", "manifest"]
# metal has its own stage set: every public pipelines.metal function callable as fn(ws). The two that need a
# second argument (ultimate_grid, compare_fold_predictions) are deliberately not exposed on the command line.
METAL_STAGES = ("audit", "family_screen", "component_screen", "combination_ladder", "model_family_screen",
                "graph_diagnostic", "final_lock", "final_external", "reproduction_report", "baselines",
                "manifest", "run_all", "ultimate_graph", "msa_batches", "evo_dev_cv", "historical_kan")


def valid_stages(task: str) -> tuple[str, ...]:
    return ("all", *(METAL_STAGES if task == "metal" else TASK_STAGES))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages\n"
               f"    nuclear / small   {' '.join(TASK_STAGES)}\n"
               f"    metal             {' '.join(METAL_STAGES)}\n"
               "    anatomy           (no stages; runs the dataset audit)\n",
    )
    ap.add_argument("task", choices=["nuclear", "small", "metal", "anatomy"])
    ap.add_argument("--stages", nargs="+", default=["all"],
                    help="one or more stage names, or 'all' (default). See the stage list below.")
    ap.add_argument("--base", default=None, help="project root (default: Drive / $PBIND_BASE)")
    ap.add_argument("--local", default=None, help="fast scratch dir (default: /content or <base>/_local)")
    ap.add_argument("--policy", default="best_mean", choices=["best_mean", "one_se"], help="metal lock policy")
    ap.add_argument("--no-evo", action="store_true")
    a = ap.parse_args()

    # Check the stage names BEFORE opening the workspace: that step loads the splits and the feature store,
    # so a typo used to cost minutes and then end in a bare KeyError.
    if a.task != "anatomy":
        allowed = valid_stages(a.task)
        unknown = [s for s in a.stages if s not in allowed]
        if unknown:
            ap.error(f"stage khong hop le cho task {a.task!r}: {', '.join(unknown)}\n"
                     f"hop le: {', '.join(allowed)}")
        if "all" in a.stages and len(a.stages) > 1:
            ap.error("'all' khong di kem stage khac")
    elif a.stages != ["all"]:
        ap.error("task 'anatomy' khong nhan --stages")

    if a.task == "anatomy":
        ws = open_workspace(None, base=a.base, local=a.local, attach_evo=False, attach_esmif=False)
        anatomy.dataset_anatomy(ws)
        return
    ws = open_workspace(a.task, base=a.base, local=a.local, attach_evo=not a.no_evo)
    if a.task == "metal":
        if a.stages == ["all"]:
            metal.run_all(ws, policy=a.policy)
        for s in a.stages:
            if s != "all":
                getattr(metal, s)(ws)
        return
    for s in (ALL_TASK if a.stages == ["all"] else a.stages):
        print(f"\n######## {a.task}: {s}")
        out = TASK_STAGES[s](ws)
        if hasattr(out, "head"):
            print(out.head(20).to_string())


if __name__ == "__main__":
    main()
