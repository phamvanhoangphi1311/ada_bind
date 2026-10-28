"""pbind.api: pick features + model by name, add a model through the registry (fake data, tiny ESM)."""
from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn
from sklearn.metrics import average_precision_score

pytest.importorskip("transformers")

from conftest import private_base  # noqa: E402

from pbind import api  # noqa: E402
from pbind.data.loading import n_residues  # noqa: E402

FAST_GEAR = {"neural_epochs": 2, "neural_patience": 1}
FAST_TORCH = {"epochs": 1, "patience": 1}


@pytest.fixture(scope="module")
def api_base(fake_base, tmp_path_factory):
    return private_base(fake_base, tmp_path_factory, "api")


def _ws(task, api_base):
    base, local = api_base
    return api.workspace(task, base=base, local=local, mount=False, device="cpu", backend="native")


@api.register_model("TinyMLP", kind="residue", description="test-only residue model")
def tiny_mlp(d, cfg, seed):
    return nn.Sequential(nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, 1), nn.Flatten(0))


def test_registries():
    models = api.list_models()
    assert {"MLP", "KAN", "XGBoost", "ConcatMLP", "HG_Attention", "TinyMLP"} <= set(models.model)
    feats = api.list_features()
    assert {"ESM", "ATOM", "CHEM", "METALSPEC", "SEQCTX"} <= set(feats.block)
    assert feats[feats.group == "MULTI2"].dim.sum() == 58


def test_residue_models(tiny_esm, api_base):
    ws = _ws("metal", api_base)
    cv = api.cross_validate(ws, model="TinyMLP", features="ESM+Atomic+Chem+MULTI2", **FAST_GEAR)
    assert cv.kind == "residue" and len(cv.folds) == 5 and np.isfinite(cv.dev["CV_AUPRC"])
    assert cv.blocks == ["ESM", "ATOM", "CHEM", "METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF"]
    again = api.cross_validate(ws, model="TinyMLP", features="ESM+Atomic+Chem+MULTI2", **FAST_GEAR)
    assert again.dev["CV_AUPRC"] == cv.dev["CV_AUPRC"]  # fold files reused

    fin = api.train_final(ws, model="MLP", features="ATOM+CHEM", **FAST_GEAR)
    assert len(list(fin.artifacts["weights"].glob("MLP__ATOM__CHEM__fold*.pt"))) == 5
    pred = api.predict(ws, model="MLP", features="ATOM+CHEM", split="TestSet300", **FAST_GEAR)
    ext = fin.external.set_index("dataset")
    assert abs(average_precision_score(pred.y_true, pred.probability) - ext.loc["TestSet300", "AUPRC"]) < 1e-6

    cols = api.feature_store(ws).frame("TestSetNew46", ["ATOM"], labels=["metal_binding"])
    assert cols.shape[1] == 2 + 15 + 1 and "ATOM:focal_has_donor" in cols


def test_fusion_and_group_models(tiny_esm, api_base):
    ws = _ws("nuclear", api_base)
    feats = "ESM+ACCESS+NA_PATCH3D40+NA_TOPO12"  # NA_* attached on demand
    fu = api.cross_validate(ws, model="ConcatMLP", features=feats, **FAST_TORCH)
    assert fu.kind == "fusion" and fu.oof.shape == (n_residues(ws.dev),)
    hg = api.train_final(ws, model="NA_PatchHG_Attention", features=feats, **FAST_TORCH)
    assert hg.kind == "group" and set(hg.external.dataset) == {"TestSet300", "TestSetNew46"}
    p = api.predict(ws, model="NA_PatchHG_Attention", features=feats, split="TestSetNew46", **FAST_TORCH)
    assert len(p) == n_residues(ws.splits["TestSetNew46"]) and np.isfinite(p.probability).all()


def test_registered_feature_group_reaches_the_store(tiny_esm, api_base):
    from pbind.features.api import FEATURE_GROUPS, feature_group

    @feature_group("TESTLEN", columns={"TESTLEN": ["relative_index"]}, needs="sequence")
    def testlen(p):
        L = len(p.sequence)
        return {"TESTLEN": (np.arange(L, dtype=np.float32) / max(L - 1, 1))[:, None]}

    try:
        ws = _ws("metal", api_base)
        assert "TESTLEN" in next(iter(ws.ctx.stores["train"].values()))
        r = api.cross_validate(ws, model="MLP", features="ATOM+TESTLEN", **FAST_GEAR)
        assert r.blocks == ["ATOM", "TESTLEN"] and np.isfinite(r.dev["CV_AUPRC"])
    finally:
        FEATURE_GROUPS.pop("TESTLEN", None)
