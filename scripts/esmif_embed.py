#!/usr/bin/env python
"""ESM-IF1 embedding exactly as computed for the training features: ESMBind multi_modal_binding/
get_esm_if_embedding.py (commit 5d81fb5b) with the device fix of get_encoder_output. Standalone (no pbind import):
run it in the ESM-IF environment (python 3.10, torch 2.4.1, fair-esm 2.0.0, biotite 0.41.2, torch-scatter 2.1.2,
torch-geometric 2.6.1; see docs/FOUNDATION_MODELS.md section 5).

    python esmif_embed.py --pdb query.pdb --sequence MKT... --out query.ESMIF.npy
    python esmif_embed.py --golden-dir golden_set --out-dir my_embeddings     # -> <prot_id>.ESMIF.npy for
                                                                             #    verify_installation.py --embeddings-dir
Weights: --weights <model_weights>/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt (or the file in ~/.cache/torch/hub/checkpoints/).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import esm
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter


def load(weights=None, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if weights:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights))
    else:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    return model.eval().to(device), alphabet, device


def get_encoder_output(model, alphabet, coords, device):
    batch = [(coords, None, None)]
    coords, confidence, _, _, padding_mask = CoordBatchConverter(alphabet)(batch, device=device)
    encoder_out = model.encoder.forward(coords, padding_mask, confidence, return_all_hiddens=False)
    return encoder_out["encoder_out"][0][1:-1, 0]  # drop BOS / EOS


def embed(model, alphabet, device, pdb, seq) -> np.ndarray:
    structure = esm.inverse_folding.util.load_structure(str(pdb), "A")
    coords, pdb_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    if pdb_seq != seq:
        raise SystemExit(f"{pdb}: chain A sequence differs from the input sequence")
    with torch.no_grad():
        return get_encoder_output(model, alphabet, coords, device).cpu().numpy()  # float32 (L, 512)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb")
    ap.add_argument("--sequence")
    ap.add_argument("--out")
    ap.add_argument("--golden-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--weights")
    ap.add_argument("--device")
    a = ap.parse_args()
    model, alphabet, device = load(a.weights, a.device)
    if a.golden_dir:
        out = Path(a.out_dir or "golden_embeddings")
        out.mkdir(parents=True, exist_ok=True)
        for e in json.loads((Path(a.golden_dir) / "index.json").read_text())["proteins"]:
            arr = embed(model, alphabet, device, Path(a.golden_dir) / f"{e['prot_id']}.pdb", e["sequence"])
            np.save(out / f"{e['prot_id']}.ESMIF.npy", arr)
            print(e["prot_id"], arr.shape)
    else:
        np.save(a.out, embed(model, alphabet, device, a.pdb, a.sequence))
        print("saved", a.out)


if __name__ == "__main__":
    main()
