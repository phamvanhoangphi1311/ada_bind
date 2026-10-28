"""ESM-2 residue embeddings (port of GEAR ``gear_structure.ensure_esm_cache``).

facebook/esm2_t33_650M_UR50D (HuggingFace, downloaded on first use), fp32 forward, mean of the last two
hidden layers, BOS/EOS removed, stored as float16 (L, 1280). Sequences are truncated to ``max_residues``.
``PBIND_ESM_MODEL`` overrides the model (path or hub id; the tests use a tiny local model).
Original backend cache (imported by the feature store when present):
    <BASE>/cache/spatial_revamp/<split>_ESM2_t33_mean_last2_1280_fp16.pkl   {prot_id: (L, 1280) fp16}
"""
from __future__ import annotations

import gc
import os

import numpy as np
import torch

from .api import feature_group

ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM_LAST_N_LAYERS = 2
ESM_DIM = 1280


def esm_model_name() -> str:
    return os.environ.get("PBIND_ESM_MODEL", ESM_MODEL)


def load_esm_model(model_name: str = ESM_MODEL, device=None):
    from transformers import EsmModel, EsmTokenizer

    tok = EsmTokenizer.from_pretrained(model_name)
    # fp32 forward like the original (newer transformers default to the checkpoint dtype)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True).to(device).float().eval()
    return tok, model


@torch.inference_mode()
def embed_sequences(items, tok, model, device, max_residues: int = 1000, batch_size: int = 8,
                    dim: int = ESM_DIM, progress: bool = True) -> dict:
    """items = [(prot_id, sequence)] (already truncated). Returns {prot_id: (L, dim) float16}."""
    from tqdm.auto import tqdm

    out = {}
    starts = range(0, len(items), batch_size)
    for start in (tqdm(starts, desc="ESM") if progress else starts):
        batch = items[start:start + batch_size]
        t = tok([s for _, s in batch], return_tensors="pt", padding=True, truncation=True,
                max_length=max_residues + 2).to(device)
        hidden = model(**t).hidden_states
        h = torch.stack(hidden[-ESM_LAST_N_LAYERS:], dim=0).mean(0)
        for bi, (pid, seq) in enumerate(batch):
            arr = h[bi, 1:len(seq) + 1].float().cpu().numpy().astype(np.float16)  # drop BOS/EOS
            if arr.shape != (len(seq), dim):
                raise ValueError(f"{pid}: ESM alignment mismatch {arr.shape}")
            out[pid] = arr
        del t, hidden, h
    return out


@feature_group("ESM", columns={"ESM": [f"esm_{k}" for k in range(ESM_DIM)]}, needs="batch",
               dtypes={"ESM": "float16"}, legacy_cache={"ESM": "{split}_ESM2_t33_mean_last2_1280_fp16.pkl"})
def esm2(proteins, opts) -> dict:
    """ESM-2 650M embedding: mean of the last two layers (1280-d, fp16)."""
    device = opts.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_esm_model(esm_model_name(), device)
    emb = embed_sequences([(p.pid, p.sequence) for p in proteins], tok, model, device, opts.max_residues,
                          opts.esm_batch_size)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {pid: {"ESM": a} for pid, a in emb.items()}
