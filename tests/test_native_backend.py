"""Native backend end-to-end: features from raw PDBs (tiny local ESM model), same-MLP screens,
final metal run with saved weights, inference that only loads, nuclear lock + hypergraph."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("transformers")

from conftest import private_base  # noqa: E402

from pbind.pipelines import metal, open_workspace, screens, v1  # noqa: E402


@pytest.fixture(scope="module")
def native_base(fake_base, tmp_path_factory):
    return private_base(fake_base, tmp_path_factory, "native")


def _ws(task, native_base):
    base, local = native_base
    ws = open_workspace(task, base=base, local=local, mount=False, device="cpu", attach_evo=False,
                        attach_esmif=False, backend="native")
    ws.ctx.cfg = replace(ws.ctx.cfg, neural_epochs=2, neural_patience=1, xgb_estimators=20)
    return ws


def test_native_features(tiny_esm, native_base):
    ws = _ws("metal", native_base)
    item = next(iter(ws.ctx.stores["train"].values()))
    dims = {k: v.shape[1] for k, v in item.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    expect = {"AA": 20, "ESM": 1280, "SPATIAL": 112, "LOCAL": 8, "ATOM": 15, "CHEM": 24, "ACCESS": 10,
              "SHELL2": 12, "ENV": 8, "CONF": 4,
              "METALSPEC": 18, "ELEC3D": 8, "TOPO": 12, "SOLV": 10, "SEQMOTIF": 10, "SEQCTX": 182}
    assert {k: dims[k] for k in expect} == expect
    for split in ("train", "test300", "test46"):
        for it in ws.ctx.stores[split].values():
            for k in expect:
                assert np.all(np.isfinite(np.asarray(it[k], np.float32))), k
    # donors exist in the fake PDBs -> chemistry blocks are not all-zero
    allv = np.concatenate([it["CHEM"] for it in ws.ctx.stores["train"].values()])
    assert allv.sum() > 0
    base, _ = native_base
    assert (base / "feature_store" / "DevSet1010" / "METALSPEC.npy").exists()


def test_feature_store_columns(tiny_esm, native_base):
    from pbind.features.api import all_blocks
    from pbind.features.store import FeatureStore, build_feature_store

    ws = _ws("metal", native_base)
    base, _ = native_base
    fs = FeatureStore(base / "feature_store")
    assert {"ESM", "ATOM", "CHEM", "METALSPEC", "SEQCTX"} <= set(fs.blocks("TestSet300"))
    X = fs.matrix("TestSet300", ["ATOM", "CHEM"])
    assert X.shape == (fs.n_residues("TestSet300"), 39) and len(fs.columns(["ATOM", "CHEM"])) == 39
    assert fs.block("DevSet1010", "ESM").dtype == np.float16
    # column view == per-protein store the trainer sees
    item = ws.ctx.stores["test300"][fs.index("TestSet300").prot_id[0]]
    np.testing.assert_array_equal(X[:len(item["y"])], np.concatenate([item["ATOM"], item["CHEM"]], axis=1))
    # second build reuses every block file
    mtimes = {b: (fs.split_dir("TestSet300") / f"{b}.npy").stat().st_mtime_ns for b in all_blocks() if fs.has("TestSet300", b)}
    build_feature_store(ws.paths, {"TestSet300": ws.ctx.frames["test300"]}, groups=("ATOM_CHEM", "MULTI2"))
    assert all((fs.split_dir("TestSet300") / f"{b}.npy").stat().st_mtime_ns == t for b, t in mtimes.items())


def test_native_metal_train_and_inference(tiny_esm, native_base):
    from pbind.inference import run_metal_inference

    ws = _ws("metal", native_base)
    metal.run_all(ws)  # family / component / ladder / model family / final (saves 5 fold weights)
    lock = ws.state["model"]
    base, _ = native_base
    weights = list((base / "trained_models" / "metal_v2_06_final_locked_external" / "metal_binding").glob("*fold*"))
    assert len([w for w in weights if not w.name.endswith(".val.npz")]) == 5
    R = run_metal_inference(_ws("metal", native_base), "TestSet300")
    assert R["summary"]["folds_trained_in_this_call"] == []  # loaded the weights saved by the final run
    assert R["summary"]["model"] == lock["locked_model"]
    agree = R["summary"]["agreement_with_original_run"]
    if lock["locked_model"] != "XGBoost":
        assert agree is not None and agree["max_abs_diff"] < 1e-5  # same models -> same ensemble


def test_native_nuclear_lock_and_hypergraph(tiny_esm, native_base):
    ws = _ws("nuclear", native_base)
    ws.task = replace(ws.task, group_train=replace(ws.task.group_train, epochs=1, patience=1))
    _, lock = screens.mechanism_ladder(ws)
    assert lock["representation"] in ws.task.mechanism_grid_base
    hg = v1.hypergraph_screen(ws, names=[ws.task.v1_hg])
    assert np.isfinite(hg["lock"]["dev_cv_auprc"])
