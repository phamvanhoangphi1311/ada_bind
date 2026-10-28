#!/usr/bin/env python
"""Metal-binding inference with the locked GEAR model (MLP, ESM+ATOM+CHEM+MULTI2, 5-fold ensemble).
Default weights: <BASE>/final_model/metal/mlp (no training).

    python scripts/infer_metal.py                        # TestSet300
    python scripts/infer_metal.py --split TestSetNew46
    python scripts/infer_metal.py --run-version metal_v2_06_final_locked_external_local   # weights of train_metal.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.inference import run_metal_inference  # noqa: E402
from pbind.pipelines import open_workspace  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="TestSet300", choices=["TestSet300", "TestSetNew46"])
    ap.add_argument("--base", default=None, help="data root (default: <repo>/data, then $PBIND_BASE / Drive)")
    ap.add_argument("--local", default=None)
    ap.add_argument("--out", default=None, help="output dir (default: results/METAL_INFERENCE/<rep>/<split>)")
    ap.add_argument("--force-retrain", action="store_true", help="retrain the folds of --run-version")
    ap.add_argument("--force-predict", action="store_true")
    ap.add_argument("--run-version", default=None,
                    help="fold weights in trained_models/<run-version>/ instead of final_model/")
    a = ap.parse_args()

    ws = open_workspace("metal", base=a.base, local=a.local, attach_evo=False, attach_esmif=False)
    R = run_metal_inference(ws, a.split, force_retrain=a.force_retrain, force_predict=a.force_predict, out_dir=a.out,
                            run_version=a.run_version)
    print(json.dumps(R["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
