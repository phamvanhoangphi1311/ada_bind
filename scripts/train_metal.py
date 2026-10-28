#!/usr/bin/env python
"""Retrain the final metal model on this machine, end to end (every step resumes from its cache):

    1. feature store   : ESM-2 / AA / SPATIAL / LOCAL / ATOM / CHEM / MECH1 / MULTI2 / SEQCTX, computed once
                         -> <BASE>/feature_store/<split>/<BLOCK>.npy
    2. final locked run: MLP on ESM+ATOM+CHEM+MULTI2, 5 official folds, exactly as notebook cell
                         "11. FINAL LOCKED EXTERNAL RUN" -> weights in <BASE>/trained_models/<run-version>/metal_binding/
    3. report          : reproduced vs original numbers -> results/<run-version>/metal_binding/
                         FINAL_reproduction_vs_original.csv
    4. inference       : TestSet300 + TestSetNew46 with these weights -> results/METAL_INFERENCE/.../<run-version>/

The final model itself is NOT retrained: it lives in <BASE>/final_model/metal/mlp (notebooks/07, Colab T4,
bit-exact) and scripts/infer_metal.py uses it. A run here gives the same numbers only on the same GPU/software.

    python scripts/train_metal.py
    python scripts/train_metal.py --esm-batch-size 2      # if ESM-2 runs out of GPU memory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.constants import EXTERNAL_SPLITS  # noqa: E402
from pbind.inference import run_metal_inference  # noqa: E402
from pbind.pipelines import metal, open_workspace  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="data root (default: <repo>/data, then $PBIND_BASE / Drive)")
    ap.add_argument("--local", default=None)
    ap.add_argument("--esm-batch-size", type=int, default=8)
    ap.add_argument("--run-version", default="metal_v2_06_final_locked_external_local",
                    help="output folder of this retrain (never final_model/)")
    a = ap.parse_args()

    ws = open_workspace("metal", base=a.base, local=a.local, attach_evo=False, attach_esmif=False,
                        models=("MLP",), esm_batch_size=a.esm_batch_size)
    metal.final_external(ws, run_version=a.run_version)
    metal.reproduction_report(ws)
    for split in EXTERNAL_SPLITS:
        s = run_metal_inference(ws, split, run_version=a.run_version)["summary"]
        print(f"{split}: AUPRC {s['AUPRC']:.4f} | AUROC {s['AUROC']:.4f} | Dev-OOF threshold "
              f"{s['dev_oof_f1_threshold']:.4f} | folds trained now {s['folds_trained_in_this_call']} -> {s['output_dir']}")


if __name__ == "__main__":
    main()
