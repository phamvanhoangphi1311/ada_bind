"""End-to-end pipeline test on a fake Drive layout + fake GEAR backend (tests/fake_gear).

Exercises the glue: workspace -> features -> screens -> v1 -> v2 -> final -> cross-task -> metal -> anatomy,
then re-runs stages to check that everything is reused from cache.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pbind.pipelines import anatomy, crosstask, features, final, metal, open_workspace, v1, v2
from conftest import SIZES

def _tiny(spec):
    return replace(spec, local_train=replace(spec.local_train, epochs=1, patience=1, batch=512),
                   group_train=replace(spec.group_train, epochs=1, patience=1),
                   v2=replace(spec.v2, train=replace(spec.v2.train, epochs=1, patience=1)))


def _ws(task, fake_base):
    base, local = fake_base
    ws = open_workspace(task, base=base, local=local, mount=False, install_requirements=False, device="cpu",
                        backend="gear")  # tests/fake_gear stands in for the Drive package
    if task in ("nuclear", "small"):
        ws.task = _tiny(ws.task)
    return ws


def test_precompute_all(fake_base):
    from pbind.pipelines.features import precompute_all

    base, local = fake_base
    assert precompute_all(base=base, local=local, backend="gear") == {"metal": "ok", "nuclear": "ok", "small": "ok"}
    for parts in (("nuclear_ultimate_v1", "nuclear_explicit_v1"), ("nuclear_ultimate_v1", "patch_hypergraph_v1"),
                  ("nuclear_v2_selective_multiscale", "selective_multiscale_patches"),
                  ("small_ultimate_v1", "small_explicit_v1"), ("small_ultimate_v1", "pocket_hypergraph_v1"),
                  ("small_multiscale_pockets_v1",)):
        assert len(list(base.joinpath("cache", *parts).rglob("*.npz"))) == sum(SIZES.values()), parts


def _mtimes(root: Path):
    return {p: p.stat().st_mtime_ns for p in root.rglob("fold*.pt")}


@pytest.mark.parametrize("task", ["nuclear", "small"])
def test_task_pipeline(task, fake_base):
    ws = _ws(task, fake_base)
    assert "EVO" in ws.available
    out = v1.run_all(ws)
    assert out["insight"]["global_model"]
    v2.v2_ablation(ws)
    v2.v2_external(ws)
    final.multiscale_screen(ws)
    final.multiscale_external(ws)
    paper = final.paper_report(ws)
    assert len(paper["headline"]) and Path(ws.paths.results_dir(ws.task.paper_root) /
                                           "FINAL_DECISION_AND_DEFINITIONS.json").exists()
    if task == "small":
        audit, _ = v2.pocket_coverage_audit(ws)
        assert len(audit) == 1

    # second session: everything must come back from cache (no checkpoint rewritten)
    before = _mtimes(ws.paths.results)
    ws2 = _ws(task, fake_base)
    v1.hypergraph_screen(ws2)
    v1.architecture_screen(ws2)
    final.paper_report(ws2)
    assert _mtimes(ws.paths.results) == before


def test_crosstask_and_evo(fake_base):
    wn, wsm = _ws("nuclear", fake_base), _ws("small", fake_base)
    for w in (wn, wsm):
        features.build_features(w, attach_only=True)
        crosstask.export_dev_oof(w)
    xn = crosstask.crosstask_screen(wn, timeout=5)
    xs = crosstask.crosstask_screen(wsm, timeout=5)
    assert "N_BASE_PLUS_EVO_ACCESS_RES_CLEANV3" in set(xn["cv"].representation)
    assert xs["decision"]["base_dev_auprc"] > 0
    e = crosstask.evo_hypergraph_addin(wn)
    assert "promote_new_final" in e["decision"]


def test_metal_and_anatomy(fake_base):
    wm = _ws("metal", fake_base)
    out = metal.run_all(wm, policy="one_se")
    assert wm.state["model"]["locked_model"] == "MLP" and "baselines" in out
    wa = _ws(None, fake_base)
    R = anatomy.dataset_anatomy(wa)
    assert {"core", "aa", "stability_verdicts", "fingerprint"} <= set(R)


def test_metal_inference(fake_base):
    from pbind.inference import run_metal_inference

    wm = _ws("metal", fake_base)
    metal.run_all(wm)  # writes the locked decision + the original run's fold prediction pkls
    first = run_metal_inference(wm, "TestSet300")
    s = first["summary"]
    assert s["folds_trained_in_this_call"] == [1, 2, 3, 4, 5]  # fake run saved no weights -> trained once
    assert len(first["residues"]) == s["n_residues"] and set(first["metrics"].threshold_rule) == {
        "fixed_0.5", "dev_oof_f1_opt"}
    assert s["agreement_with_original_run"] is not None
    out = Path(s["output_dir"])
    assert (out / "residue_predictions.csv.gz").exists() and (out / "protein_summary.csv").exists()

    again = run_metal_inference(_ws("metal", fake_base), "TestSet300")  # new session: load only
    assert again["summary"]["folds_trained_in_this_call"] == []
    assert np.allclose(again["residues"].prob, first["residues"].prob)
    assert again["summary"]["dev_oof_f1_threshold"] == s["dev_oof_f1_threshold"]
