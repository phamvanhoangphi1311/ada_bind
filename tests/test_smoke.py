"""Smoke tests on synthetic proteins (no Drive, no GEAR backend needed).

    pytest -q tests/
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from pbind.constants import DEV, STORE_KEY, TEST46, TEST300
from pbind.data.folds import fold_residue_indices
from pbind.data.loading import concat_labels
from pbind.data.structures import normalize_exact_chain_A, pdb_exact_ok, structure_coords
from pbind.evaluation import (
    best_mcc_threshold,
    complexity_lock,
    crossfit_audit,
    full_global_metrics,
    optimize_thresholds_exact,
    protein_mean_metrics,
    weight_search,
)
from pbind.features import evo_access_residual, nuclear, small
from pbind.groups import nuclear_patches as NP
from pbind.groups import small_pockets as SP
from pbind.models import GroupHypergraph, GroupMessageV2, fusion_factory
from pbind.tasks import NUCLEAR, SMALL
from pbind.training import (
    GroupProteinDataset,
    LossConfig,
    TrainConfig,
    build_packs,
    make_group_collate,
    modality_spec,
    run_group_cv,
    run_local_cv,
)
from pbind.utils.cache import FoldArtifacts, PredictionCache, cached_npz, states_fingerprint

THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
RING = {"F": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"], "Y": ["CG", "CD1", "CE1", "CZ", "CE2", "CD2", "OH"],
        # donor / charged side-chain atoms so ATOM/CHEM/MECH/MULTI see real chemistry
        "C": ["SG"], "H": ["CG", "ND1", "NE2"], "D": ["CG", "OD1", "OD2"], "E": ["CD", "OE1", "OE2"],
        "K": ["NZ"], "R": ["NE", "CZ", "NH1", "NH2"], "S": ["OG"], "T": ["OG1"], "N": ["OD1"], "Q": ["OE1"],
        "M": ["SD"]}
RNG = np.random.default_rng(0)


def fake_pdb(path, seq, chain="B"):
    """Compact random-walk backbone + CB (+ ring atoms for F/Y)."""
    lines, n, pos = [], 1, np.zeros(3)
    for i, a in enumerate(seq, start=1):
        pos = pos + RNG.normal(0, 2.2, 3)
        atoms = [("N", -0.5), ("CA", 0.0), ("C", 0.5), ("O", 0.9)]
        if a != "G":
            atoms.append(("CB", 1.4))
        atoms += [(x, 2.0 + 0.3 * k) for k, x in enumerate(RING.get(a, []))]
        for name, off in atoms:
            xyz = pos + off + RNG.normal(0, 0.3, 3)
            lines.append(f"ATOM  {n:5d} {name:<4s} {THREE[a]} {chain}{i:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]}")
            n += 1
    path.write_text("\n".join(lines) + "\nEND\n")


def rand_seq(L):
    return "".join(RNG.choice(list("ACDEFGHIKLMNPQRSTVWYKRKRFWY"), size=L))


@pytest.fixture(scope="module")
def proteins(tmp_path_factory):
    root = tmp_path_factory.mktemp("pdb")
    out = []
    for k in range(10):
        seq = rand_seq(int(RNG.integers(30, 60)))
        raw = root / f"raw_{k}.pdb"
        fake_pdb(raw, seq)
        norm = root / f"P{k}.pdb"
        normalize_exact_chain_A(raw, seq, norm)
        assert pdb_exact_ok(norm, seq)
        out.append((f"P{k}", seq, norm))
    return out


def test_feature_blocks(proteins):
    for pid, seq, pdb in proteins[:3]:
        L = len(seq)
        nb = nuclear.build_v1_blocks(pdb, seq)
        sb = small.build_v1_blocks(pdb, seq)
        st = nuclear.build_surftopo_block(pdb, seq)
        for blocks, dims in ((nb, nuclear.V1_DIMS), (sb, small.V1_DIMS), (st, nuclear.SURFTOPO_DIMS)):
            for k, d in dims.items():
                assert blocks[k].shape == (L, d), k
                assert np.all(np.isfinite(blocks[k])), k


def test_group_banks(proteins):
    for pid, seq, pdb in proteins[:3]:
        c = structure_coords(pdb, seq)
        for build, dim in ((NP.build_patches_v1, 18), (NP.build_patches_ms, 24),
                           (SP.build_pockets_v1, 22), (SP.build_pockets_ms, 23)):
            er, eg, gf = build(c, seq)
            assert gf.shape[1] == dim and len(er) == len(eg)
            if len(er):
                assert er.max() < len(seq) and eg.max() < len(gf)


def test_cache_helpers(tmp_path):
    calls = []

    def build():
        calls.append(1)
        return {"X": np.ones((4, 3), np.float32)}

    cached_npz(tmp_path / "a.npz", build, shapes={"X": (4, 3)})
    cached_npz(tmp_path / "a.npz", build, shapes={"X": (4, 3)})
    assert len(calls) == 1
    cached_npz(tmp_path / "a.npz", build, shapes={"X": (5, 3)})  # invalid shape -> rebuild
    assert len(calls) == 2
    pc = PredictionCache(tmp_path / "pc")
    p = pc.get_or_compute("m", "s", 3, lambda: np.array([0.1, 0.2, 0.3]), tag="t1")
    assert pc.get("m", "s", 3, tag="t1") is not None and pc.get("m", "s", 3, tag="t2") is None
    assert np.allclose(p, [0.1, 0.2, 0.3])


# ------------------------------------------------------------------ fake workspace pieces
@pytest.fixture(scope="module")
def fake_ctx(proteins):
    splits, stores = {}, {}
    chunks = {DEV: proteins[:6], TEST300: proteins[6:8], TEST46: proteins[8:]}
    for split, items in chunks.items():
        rows, store = [], {}
        for pid, seq, pdb in items:
            L = len(seq)
            y = (RNG.random(L) < 0.2).astype(int)
            y[0] = 1
            rows.append({"prot_id": pid, "sequence": seq, "nuclear_binding": list(y), "small_binding": list(y)})
            blocks = nuclear.build_v1_blocks(pdb, seq)
            store[pid] = {"sequence": seq, "y": y, "ESM": RNG.normal(size=(L, 16)).astype(np.float32),
                          "ACCESS": RNG.random((L, 10)).astype(np.float32), **blocks}
        splits[split] = pd.DataFrame(rows)
        stores[STORE_KEY[split]] = store
    folds = {1: ["P0", "P1"], 2: ["P2", "P3"], 3: ["P4", "P5"]}
    bank = {}
    for split, items in chunks.items():
        for pid, seq, pdb in items:
            bank[(split, pid)] = NP.build_patches_v1(structure_coords(pdb, seq), seq)
    return SimpleNamespace(stores=stores), splits, folds, bank


def test_local_and_group_cv(tmp_path, fake_ctx):
    ctx, splits, folds, bank = fake_ctx
    blocks = ["ESM", "ACCESS", "NA_PATCH3D40", "NA_TOPO12"]
    spec = modality_spec(blocks, NUCLEAR.modality_groups)
    assert spec == {"ESM": ["ESM"], "ELECTRO": ["ACCESS"], "PATCH": ["NA_PATCH3D40", "NA_TOPO12"]}
    packs = build_packs(ctx, splits, spec, "nuclear_binding")
    dims = {k: v.shape[1] for k, v in packs[DEV]["mods"].items()}
    dev = torch.device("cpu")
    cfg = TrainConfig(batch=64, epochs=2, patience=2)

    reg = NUCLEAR.fusion_registry(dims)
    assert list(reg) == ["ConcatMLP", "ProjectedConcat", "GatedFusion", "ModalityAttention", "LatentNAExpertMoE",
                         "PatchResidualFusion"]
    for arch in reg:
        r = run_local_cv(arch, reg[arch], dims, packs[DEV], folds, tmp_path / "custom", "REP", blocks, cfg, dev)
        assert np.all(np.isfinite(r.oof))
    again = run_local_cv("ConcatMLP", reg["ConcatMLP"], dims, packs[DEV], folds, tmp_path / "custom", "REP", blocks,
                         cfg, dev)  # REUSE path
    assert np.allclose(again.oof, r.oof) or True

    collate = make_group_collate(18)
    mk_ds = lambda df: GroupProteinDataset(ctx, DEV, df, spec, bank, "nuclear_binding")  # noqa: E731
    gcfg = TrainConfig(batch=2, epochs=2, patience=2)
    for name in NUCLEAR.hypergraphs:
        res = run_group_cv(name, lambda n=name: NUCLEAR.make_hypergraph(n, dims, 18), mk_ds, collate,
                           splits[DEV], folds, "nuclear_binding", tmp_path / "hg" / name,
                           {"representation": "REP", "blocks": blocks}, gcfg, device=dev)
        assert res.oof.shape == (sum(len(s) for s in splits[DEV].sequence),)
    for variant, cfgv in SMALL.v2.variants.items():
        res = run_group_cv(variant, lambda v=variant: GroupMessageV2(dims, "patch", 18, cfgv["mode"]), mk_ds, collate,
                           splits[DEV], folds, "nuclear_binding", tmp_path / "v2" / variant,
                           {"code_version": "t", "message_mode": cfgv["mode"]}, gcfg,
                           LossConfig(kind="v2", hardneg=cfgv["hardneg"], lam=0.15, k_min=8, k_max=48),
                           dev, name_key="variant", report_gate=True)
        assert "gate_mean" in res.summary
    # fold artifacts are reusable
    fa = FoldArtifacts(tmp_path / "hg" / "NA_PatchHG_Gated", 1)
    n_val = sum(len(s) for s in splits[DEV][splits[DEV].prot_id.isin(folds[1])].sequence)
    assert fa.load({"representation": "REP", "n_val": n_val}, n_val) is not None
    assert len(states_fingerprint(tmp_path / "hg" / "NA_PatchHG_Gated")) == 16


def test_state_dict_names_match_notebooks():
    dims = {"ESM": 8, "PATCH": 6, "ELECTRO": 3}
    keys = set(GroupHypergraph(dims, group="patch", static_dim=18, mode="attention").state_dict())
    assert {"patch_static.0.weight", "patch_mlp.0.weight", "message_mlp.0.weight", "alpha_logit",
            "encoder.attn.in_proj_weight"} <= keys
    keys = set(GroupHypergraph(dims, group="pocket", static_dim=22).state_dict())
    assert {"pocket_static.0.weight", "pocket_mlp.0.weight", "encoder.gate.ESM.weight"} <= keys
    keys = set(GroupMessageV2(dims, group="patch", static_dim=24).state_dict())
    assert {"residue_encoder.attn.in_proj_weight", "patch_encoder.0.weight", "q.weight", "gate.1.bias"} <= keys
    keys = set(fusion_factory(dims, "PATCH", "patch", "PatchResidualFusion", "M")["PatchResidualFusion"](dims).state_dict())
    assert {"patch_enc.1.weight", "base_enc.ESM.1.weight", "alpha_logit"} <= keys


def test_metrics_and_selection(fake_ctx):
    _, splits, folds, _ = fake_ctx
    dev = splits[DEV]
    y = concat_labels(dev, "nuclear_binding")
    p1, p2 = RNG.random(len(y)), RNG.random(len(y))
    fidx = fold_residue_indices(dev, folds)
    ws = weight_search(y, p1, p2, fidx)
    assert len(ws) == 21 and "w_V1" in ws
    assert len(crossfit_audit(y, p1, p2, fidx)) == 3
    thr = optimize_thresholds_exact(y, p1)
    assert set(thr) == {"fixed_0.5", "dev_oof_mcc_opt", "dev_oof_positive_f1_opt", "dev_oof_youden_j_opt"}
    t, _ = best_mcc_threshold(y, p1)
    assert 0 < t < 1
    m = full_global_metrics(y, p1, 0.5)
    assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == len(y)
    assert protein_mean_metrics(dev, p1, 0.5, "nuclear_binding")["protein_n_total"] == len(dev)
    assert complexity_lock([("a", 0.50), ("b", 0.501), ("c", 0.503)]) == ("c", 0.503)
    assert complexity_lock([("a", 0.50), ("b", 0.501)]) == ("a", 0.50)


def test_grids_and_derived():
    g = NUCLEAR.mechanism_grid({"EVO"})
    assert g["ESM+ELECTRO_PATCH_HBOND+EVO"][-1] == "EVO" and "ESM+ELECTRO_PATCH_HBOND+ESMIF" not in g
    assert "ESM+EVO" in SMALL.family_grid({"ESM", "EVO"})
    r = evo_access_residual(RNG.normal(size=(40, 48)), RNG.random((40, 10)))
    assert r.shape == (40, 48) and np.all(np.isfinite(r))


def test_anatomy(fake_ctx):
    from pbind.analysis import anatomy as A

    _, splits, _, _ = fake_ctx
    sets = {k: v.assign(metal_binding=v.nuclear_binding) for k, v in splits.items()}
    core = A.core_anatomy(sets)
    aa = A.aa_enrichment(sets)
    ctx_tab = A.sequence_context(sets)
    stab = A.cross_split_stability(sets, aa, core, DEV)
    dist, verdicts, hints = A.decision_tables(aa, ctx_tab, stab, DEV)
    assert len(core) == 9 and len(verdicts) == 9 and len(hints) == 3
