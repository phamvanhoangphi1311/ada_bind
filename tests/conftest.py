import sys
from pathlib import Path

# make `pbind` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------- shared fake Drive layout
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from pbind import constants  # noqa: E402

HERE = Path(__file__).parent
SIZES = {"DevSet1010": 30, "TestSet300": 10, "TestSetNew46": 6}


def _labels(L, rng, p):
    y = (rng.random(L) < p).astype(int)
    y[int(rng.integers(0, L))] = 1
    return str(list(map(int, y)))


@pytest.fixture(scope="session")
def fake_base(tmp_path_factory):
    old = dict(constants.SPLIT_SIZES)
    constants.SPLIT_SIZES.update(SIZES)  # same dict object is imported everywhere
    from test_smoke import fake_pdb, rand_seq

    base = tmp_path_factory.mktemp("drive") / "Protein-binding-Phi"
    local = tmp_path_factory.mktemp("content")
    rng = np.random.default_rng(3)
    (base / "esmFold_pdb_files").mkdir(parents=True)
    frames, dev_ids = {}, []
    for split, n in (("train", SIZES["DevSet1010"] + 2), ("test300", 10), ("test46", 6)):
        rows = []
        for k in range(n):
            pid = f"{ {'train': 'DV', 'test300': 'TA', 'test46': 'TB'}[split] }{k:03d}"
            seq = rand_seq(int(rng.integers(20, 40)))
            fake_pdb(base / "esmFold_pdb_files" / f"{pid}.pdb", seq, chain="B")
            rows.append({"prot_id": pid, "sequence": seq, "metal_binding": _labels(len(seq), rng, .05),
                         "nuclear_binding": _labels(len(seq), rng, .1), "small_binding": _labels(len(seq), rng, .1),
                         "any_ligand_binding_sites": _labels(len(seq), rng, .2)})
        frames[split] = pd.DataFrame(rows)
    dev_ids = list(frames["train"].prot_id[:SIZES["DevSet1010"]])  # last 2 = "excluded" raw rows
    for rel, key in (("data/development_set/full_grouped_train_binding_sites_df.csv", "train"),
                     ("data/development_set/full_grouped_test_binding_sites_df.csv", "test300"),
                     ("data/independent_set/grouped_test_46_new_binding_sites.csv", "test46")):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(base / rel, index=False)
    fd = base / "repos" / "bindnode24" / "dataset" / "cv_splits" / "predicted_structures"
    fd.mkdir(parents=True)
    for k in range(5):
        (fd / f"ids_split{k + 1}.txt").write_text("\n".join(dev_ids[k::5]))
    for name in ("GEAR_ULTIMATE_OTHER_BINDINGS", "GEAR_ULTIMATE"):
        shutil.copytree(HERE / "fake_gear", base / name)
    yield base, local
    constants.SPLIT_SIZES.clear()
    constants.SPLIT_SIZES.update(old)




# ---------------------------------------------------------------- native backend: tiny ESM + private data roots
ESM_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q",
             "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]


@pytest.fixture(scope="session")
def tiny_esm(tmp_path_factory):
    """1-layer ESM with the real 1280-d width, used instead of the 650M model (PBIND_ESM_MODEL)."""
    pytest.importorskip("transformers")
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    d = tmp_path_factory.mktemp("tiny_esm")
    (d / "vocab.txt").write_text("\n".join(ESM_VOCAB))
    EsmTokenizer(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    cfg = EsmConfig(vocab_size=len(ESM_VOCAB), hidden_size=1280, num_hidden_layers=1, num_attention_heads=4,
                    intermediate_size=64, max_position_embeddings=1026, pad_token_id=1, mask_token_id=32,
                    position_embedding_type="rotary")
    EsmModel(cfg).save_pretrained(d)
    mp = pytest.MonkeyPatch()
    mp.setenv("PBIND_ESM_MODEL", str(d))
    yield d
    mp.undo()


def private_base(fake_base, tmp_path_factory, name):
    """Private copy of the fake Drive root (raw inputs only), so caches don't leak between test modules."""
    base, _ = fake_base
    dst = tmp_path_factory.mktemp(name) / "Protein-binding-Phi"
    shutil.copytree(base, dst, ignore=shutil.ignore_patterns("cache", "results", "trained_models", "GEAR_*"))
    return dst, tmp_path_factory.mktemp(name + "_local")
