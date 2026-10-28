"""ESM-IF1 encoder embeddings on the ESMFold chain-A structures (the `ESMIF_EF` block of the V3 models).

NOTE — there are two different ESM-IF caches in this project, and they are not interchangeable:

    ESMIF      <BASE>/cache/{gear_metal_v3,nuclear_ultimate_v1}/esmif_v1/<split>/esm_if/<pid>.npy
               float32, attached by ``backend.gear.attach_esmif_if_cached``; used by the metal screens.
    ESMIF_EF   <BASE>/cache/gear_metal_v3/esmif_esmfold_v1/<split>/esm_if/<pid>.npy
               float16, the block the nuclear / small V3 members were trained on (cell V3-3 of
               notebooks/09_* and 10_*). "EF" = the ESMFold chain-A structures, which is what this module builds.

Fed to the trained models as float32 after the float16 round-trip (``inference/final.py`` "dtype_notes").

Computing it needs the fair-esm environment of ``docs/FOUNDATION_MODELS.md`` section 5 (python 3.10,
torch 2.4.1, fair-esm 2.0.0, biotite 0.41.2, torch-scatter, torch-geometric), which is deliberately NOT a
dependency of this package. In the normal environment the block is imported from the cache above; if it is
missing, build it in the ESM-IF environment with ``scripts/esmif_embed.py`` and drop the .npy files in.

Weights: ``<repo>/model_weights/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt`` (or ``PBIND_ESMIF_WEIGHTS``).
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch

from .api import feature_group

ESMIF_DIM = 512
ESMIF_DIR = "esm_if1"  # folder name inside model_weights/ (same as inference/predictor.py)
ESMIF_FILE = "esm_if1_gvp4_t16_142M_UR50.pt"
LEGACY_SUBDIR = "gear_metal_v3/esmif_esmfold_v1"
_REPO = Path(__file__).resolve().parents[3]


def esmif_weights() -> Path | None:
    """The local ESM-IF checkpoint, or None to let fair-esm download it."""
    env = os.environ.get("PBIND_ESMIF_WEIGHTS")
    if env:
        return Path(env)
    for d in (_REPO / "model_weights" / ESMIF_DIR, _REPO / "data" / "foundation_models" / ESMIF_DIR):
        if (d / ESMIF_FILE).exists():
            return d / ESMIF_FILE
    return None


def load_esmif(device=None):
    """(model, alphabet, CoordBatchConverter). Raises with instructions when fair-esm is not installed."""
    try:
        import esm
        import esm.inverse_folding
        from esm.inverse_folding.util import CoordBatchConverter
    except Exception as e:
        raise RuntimeError(
            "ESM-IF needs the fair-esm environment (docs/FOUNDATION_MODELS.md section 5), which this package "
            "does not depend on. Either copy the cache to <BASE>/cache/" + LEGACY_SUBDIR +
            "/<split>/esm_if/<prot_id>.npy, or build it there with scripts/esmif_embed.py."
        ) from e
    fp = esmif_weights()
    if fp is not None:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(fp))
    else:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    return model.eval().to(device), alphabet, CoordBatchConverter


@torch.inference_mode()
def embed_structure(chain_pdb, sequence: str, model, alphabet, conv, device) -> np.ndarray:
    """Encoder output for one chain-A PDB, BOS/EOS dropped — same call as inference/predictor.py."""
    import esm.inverse_folding

    structure = esm.inverse_folding.util.load_structure(str(chain_pdb), "A")
    coords, pdb_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    if pdb_seq != sequence:
        raise ValueError(f"chain A of {chain_pdb} does not match the sequence "
                         f"({len(pdb_seq)} vs {len(sequence)} residues)")
    c, conf, _, _, pad = conv(alphabet)([(coords, None, None)], device=device)
    enc = model.encoder.forward(c, pad, conf, return_all_hiddens=False)
    return enc["encoder_out"][0][1:-1, 0].cpu().numpy().astype(np.float16)


@feature_group("ESMIF_EF", columns={"ESMIF_EF": [f"esmif_{k}" for k in range(ESMIF_DIM)]}, needs="batch",
               dtypes={"ESMIF_EF": "float16"}, structure="exact_chain_a", legacy_subdir=LEGACY_SUBDIR,
               legacy_cache={"ESMIF_EF": "{split}/esm_if/{pid}.npy"}, optional=True)
def esmif_ef(proteins, opts) -> dict:
    """ESM-IF1 encoder embedding (512-d, fp16) on the exact-sequence chain-A copy of the ESMFold structure."""
    from tqdm.auto import tqdm

    device = opts.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, conv = load_esmif(device)
    out = {}
    try:
        for p in tqdm(proteins, desc="ESM-IF"):
            if p.pdb is None:
                raise ValueError(f"{p.pid}: no chain-A structure available")
            arr = embed_structure(p.pdb, p.sequence, model, alphabet, conv, device)
            if arr.shape != (len(p.sequence), ESMIF_DIM):
                raise ValueError(f"{p.pid}: ESM-IF alignment mismatch {arr.shape}")
            out[p.pid] = {"ESMIF_EF": arr}
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
