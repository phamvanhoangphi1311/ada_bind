"""Reproduction safeguards: original caches are imported (or required), results of another config are never
overwritten, the known lock skips the ladder, and final models reuse existing fold checkpoints."""
from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("transformers")

from conftest import private_base  # noqa: E402

from pbind.data.folds import find_fold_dir, load_official_folds  # noqa: E402
from pbind.data.loading import backend_frames  # noqa: E402
from pbind.features.api import FEATURE_GROUPS  # noqa: E402
from pbind.features.store import FeatureStore, build_feature_store  # noqa: E402
from pbind.paths import resolve_paths  # noqa: E402
from pbind.pipelines import final, metal, open_workspace, screens  # noqa: E402


@pytest.fixture(scope="module")
def repro_base(fake_base, tmp_path_factory):
    return private_base(fake_base, tmp_path_factory, "repro")


def _ws(task, repro_base, **kw):
    base, local = repro_base
    ws = open_workspace(task, base=base, local=local, mount=False, device="cpu", attach_evo=False,
                        attach_esmif=False, backend="native", **kw)
    ws.ctx.cfg = replace(ws.ctx.cfg, neural_epochs=2, neural_patience=1)
    return ws


def _frames(base, local):
    paths = resolve_paths(base, local)
    folds = load_official_folds(find_fold_dir(paths))
    return paths, backend_frames(paths, [p for ids in folds.values() for p in ids])


def test_original_caches_are_imported_or_required(tiny_esm, repro_base, tmp_path, monkeypatch):
    base, local = repro_base
    paths, frames = _frames(base, local)
    df = frames["test46"]
    g = FEATURE_GROUPS["NA_V1"]
    # an "original" per-protein cache with recognisable values for every TestSetNew46 protein
    for r in df.itertuples():
        fp = paths.cache / g.legacy_subdir / "TestSetNew46" / f"{r.prot_id}.npz"
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fp, **{b: np.full((len(r.sequence), g.dim(b)), 7.0, np.float32) for b in g.blocks})
    monkeypatch.setenv("PBIND_FEATURE_STORE", str(tmp_path / "store"))
    fs = build_feature_store(paths, {"TestSetNew46": df}, ["NA_V1"], require_legacy=True)
    assert np.all(fs.matrix("TestSetNew46", list(g.blocks)) == 7.0)  # imported, not recomputed

    (paths.cache / g.legacy_subdir / "TestSetNew46" / f"{df.prot_id.iloc[0]}.npz").unlink()
    with pytest.raises(FileNotFoundError, match="missing from the original cache"):
        build_feature_store(paths, {"TestSetNew46": df}, ["NA_V1"], force=True, require_legacy=True)


def test_other_config_results_are_kept(tiny_esm, repro_base):
    ws = _ws("metal", repro_base, models=("MLP",), feature_groups=["AA", "ESM", "ATOM_CHEM", "MULTI2"])
    run = ws.gear.suite.clone_context_for_run(ws.ctx, "repro_guard", models=("MLP",))
    pred = ws.gear.core.job_pred_path(run.paths, "MLP", "ATOM+CHEM", 0)
    pred.parent.mkdir(parents=True, exist_ok=True)
    with open(pred, "wb") as f:  # a result of another config ("the original run")
        pickle.dump({"config_hash": "ORIGINAL", "val_p": np.zeros(3)}, f)
    pd.DataFrame([{"config_hash": "ORIGINAL", "model": "MLP", "representation": "ATOM+CHEM", "fold": 1,
                   "AUPRC": 0.5}]).to_csv(run.paths["run_root"] / "fold_metrics.csv", index=False)
    (run.paths["run_root"] / "cv_summary.csv").write_text("original summary\n")

    ws.gear.suite.run_grid(run, {"ATOM+CHEM": ["ATOM", "CHEM"]}, models=("MLP",))
    kept = run.paths["run_root"] / "superseded" / "ORIGINAL"
    with open(kept / pred.name, "rb") as f:
        assert pickle.load(f)["config_hash"] == "ORIGINAL"
    assert (kept / "cv_summary.csv").read_text() == "original summary\n"
    rows = pd.read_csv(run.paths["run_root"] / "fold_metrics.csv")
    assert "ORIGINAL" in set(rows.config_hash) and len(rows) == 1 + 5


def test_known_lock_and_final_models_reuse(tiny_esm, repro_base):
    ws = _ws("nuclear", repro_base)
    ws.task = replace(ws.task, group_train=replace(ws.task.group_train, epochs=1, patience=1))
    rep = screens.representation(ws)  # no ladder: the original Dev-only lock
    assert rep["representation"] == "ESM+PATCH_ORGANIZATION" and "mechanism" not in ws.state
    final.final_models(ws)
    states = sorted(ws.run_dir(ws.task.group_state_dir, ws.task.v1_hg).glob("fold*.pt"))
    stamps = {p: p.stat().st_mtime_ns for p in states}
    assert len(states) == 5

    ws2 = _ws("nuclear", repro_base)  # fresh session: every fold is reused, nothing retrained
    ws2.task = ws.task
    out = final.final_models(ws2)
    assert all(p.stat().st_mtime_ns == t for p, t in stamps.items())
    assert out["decision"]["final_method"] == "FINAL_DEV_LOCKED_ENSEMBLE"
    rep = final.reproduction_report(ws2)  # vs the original notebook numbers (fake data: only the shape is checked)
    assert {"V1_Dev_mean_AUPRC", "final_Dev_mean_AUPRC", "threshold_dev_oof_mcc_opt"} <= set(rep.metric)


def test_compare_fold_predictions_identical(tiny_esm, repro_base):
    ws = _ws("metal", repro_base, models=("MLP",), feature_groups=["AA", "ESM", "ATOM_CHEM", "MULTI2"])
    metal.final_external(ws, run_version="repro_a")
    metal.final_external(ws, run_version="repro_b")  # same config + same machine -> bit-identical
    parity = metal.compare_fold_predictions(ws, "repro_b", reference="repro_a")
    assert parity.identical.all() and len(parity) == 15
    assert FeatureStore(ws.paths.feature_store).blocks("TestSet300")
