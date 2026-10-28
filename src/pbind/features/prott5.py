"""ProtT5-XL half encoder embeddings (the `PROTT5` block of the V3 nuclear / small models).

Rostlab/prot_t5_xl_half_uniref50-enc, encoder only, fp16 on GPU / fp32 on CPU, ONE protein per forward
(no padding), ``</s>`` dropped, stored as float16 (L, 1024). Every setting here is the one that produced the
training features: cell V3-3 of ``notebooks/09_*`` and ``notebooks/10_*``, and ``docs/FOUNDATION_MODELS.md``
section 4. The same code path runs in ``inference/predictor.py::Predictor.prott5`` for a single protein.

Weights are not downloaded by this module beyond the HuggingFace fallback. Resolution order:
    1. ``PBIND_PROTT5_MODEL``                                   (path or hub id)
    2. ``<repo>/model_weights/prot_t5_xl_half_uniref50-enc``    (foundation_models.zip unpacked here)
    3. ``<repo>/data/foundation_models/prot_t5_xl_half_uniref50-enc``
    4. the hub id ``Rostlab/prot_t5_xl_half_uniref50-enc``      (downloaded on first use)

Original cache (imported by the feature store when present, so nothing is recomputed):
    <BASE>/cache/prott5_xl_half_v1/<split>/<prot_id>.npy        (L, 1024) float16
"""
from __future__ import annotations

import gc
import os
import re
from pathlib import Path

import numpy as np
import torch

from .api import feature_group

PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
PROTT5_DIM = 1024
PROTT5_DIR = "prot_t5_xl_half_uniref50-enc"  # folder name inside model_weights/ (same as inference/predictor.py)
_REPO = Path(__file__).resolve().parents[3]


def prott5_model_name() -> str:
    """Local folder if the weights were unpacked next to the repo, else the hub id."""
    env = os.environ.get("PBIND_PROTT5_MODEL")
    if env:
        return env
    for d in (_REPO / "model_weights" / PROTT5_DIR, _REPO / "data" / "foundation_models" / PROTT5_DIR):
        if (d / "config.json").exists():
            return str(d)
    return PROTT5_MODEL


def load_prott5(model_name: str | None = None, device=None):
    """(ids_fn, unk_token_id, model). ``ids_fn(clean_sequence) -> token ids``, one id per residue plus ``</s>``."""
    from transformers import T5EncoderModel

    name = model_name or prott5_model_name()
    try:
        from transformers import T5Tokenizer

        tok = T5Tokenizer.from_pretrained(name, do_lower_case=False)
        ids_fn = lambda clean: tok(" ".join(clean), add_special_tokens=True)["input_ids"]  # noqa: E731
        unk = tok.unk_token_id
    except Exception:  # transformers without a usable slow tokenizer: the SentencePiece model directly
        import sentencepiece as spm

        src = Path(name) / "spiece.model"
        if not src.exists():
            from huggingface_hub import hf_hub_download

            src = hf_hub_download(name, "spiece.model")
        sp = spm.SentencePieceProcessor(model_file=str(src))
        ids_fn = lambda clean: [sp.piece_to_id(f"▁{a}") for a in clean] + [sp.eos_id()]  # noqa: E731
        unk = sp.unk_id()

    model = T5EncoderModel.from_pretrained(name).to(device).eval()
    # Rostlab usage, and what the training features were computed with: half precision on GPU, fp32 on CPU.
    model = model.half() if torch.device(device).type == "cuda" else model.float()
    return ids_fn, unk, model


@torch.inference_mode()
def embed_sequence(seq: str, ids_fn, unk, model, device) -> np.ndarray:
    """One protein, no padding — batching would change the result through the attention mask."""
    clean = re.sub(r"[UZOB]", "X", seq)
    ids = ids_fn(clean)
    if len(ids) != len(seq) + 1 or unk in ids[:-1]:
        raise ValueError(f"ProtT5 tokenisation gave {len(ids)} tokens for {len(seq)} residues "
                         f"(expected one per residue plus </s>, and no unknown token)")
    t = torch.tensor(ids, dtype=torch.long)[None].to(device)
    h = model(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state
    return h[0, :len(seq)].float().cpu().numpy().astype(np.float16)  # drop </s>, store fp16


@feature_group("PROTT5", columns={"PROTT5": [f"prott5_{k}" for k in range(PROTT5_DIM)]}, needs="batch",
               dtypes={"PROTT5": "float16"}, legacy_subdir="prott5_xl_half_v1",
               legacy_cache={"PROTT5": "{split}/{pid}.npy"}, optional=True)
def prott5(proteins, opts) -> dict:
    """ProtT5-XL half encoder embedding (1024-d, fp16). Only nuclear / small V3 use it."""
    from tqdm.auto import tqdm

    device = opts.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids_fn, unk, model = load_prott5(device=device)
    out = {}
    try:
        # shortest first: the peak allocation is one sequence, so this only makes the progress bar honest
        for p in tqdm(sorted(proteins, key=lambda q: len(q.sequence)), desc="ProtT5"):
            arr = embed_sequence(p.sequence, ids_fn, unk, model, device)
            if arr.shape != (len(p.sequence), PROTT5_DIM):
                raise ValueError(f"{p.pid}: ProtT5 alignment mismatch {arr.shape}")
            out[p.pid] = {"PROTT5": arr}
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
