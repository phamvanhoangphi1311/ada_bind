#!/usr/bin/env python
"""Compute every feature the final models need, from raw inputs only (no Drive cache needed).

Needs under the data root (default <repo>/data, same layout as Drive):
    data/development_set/*.csv + ids_split1..5.txt, data/independent_set/*.csv, esmFold_pdb_files/
    (no Drive backend code needed: backend="native"; folds fall back to a bindNode24 git clone)
ESM-2 650M weights are downloaded from HuggingFace on first use (~2.5 GB).

Residue blocks go to the feature store <BASE>/feature_store/<split>/<BLOCK>.npy (computed once, then only read);
nuclear/small add their NA_*/SM_* families and patch/pocket banks under <BASE>/cache/.

    python scripts/precompute_features.py                                  # store + nuclear + small
    python scripts/precompute_features.py --store-only                     # residue feature store only
    python scripts/precompute_features.py --store-only --groups ATOM_CHEM MULTI2
    python scripts/precompute_features.py --esm-batch-size 2               # if ESM-2 runs out of GPU memory
    python scripts/precompute_features.py --store-only --force             # rebuild the store from the ORIGINAL
                                       # Drive caches copied into <BASE>/cache/spatial_revamp/ (exact original values)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.features.store import FEATURE_GROUPS  # noqa: E402
from pbind.pipelines.features import build_store, precompute_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["metal", "nuclear", "small"], choices=["metal", "nuclear", "small"])
    ap.add_argument("--base", default=None, help="data root (default: <repo>/data, then $PBIND_BASE / Drive)")
    ap.add_argument("--local", default=None, help="fast scratch dir (default: <base>/_local)")
    ap.add_argument("--esm-batch-size", type=int, default=8)
    ap.add_argument("--store-only", action="store_true", help="only the residue feature store")
    ap.add_argument("--groups", nargs="+", default=None, choices=list(FEATURE_GROUPS),
                    help="feature groups (default: all registered except the optional ones, PROTT5 / ESMIF_EF)")
    ap.add_argument("--force", action="store_true", help="rewrite the chosen groups even if already in the store")
    ap.add_argument("--no-legacy", action="store_true",
                    help="never import original backend caches (cache/spatial_revamp/*.pkl); always recompute")
    ap.add_argument("--no-ms", action="store_true", help="skip the multi-scale patch/pocket banks")
    a = ap.parse_args()
    fs = build_store(a.base, a.local, a.groups, a.esm_batch_size, force=a.force, legacy=not a.no_legacy)
    print("Feature store:", fs.root, {s: fs.blocks(s) for s in fs.manifest["splits"]})
    if not a.store_only:
        print(precompute_all(a.tasks, a.base, a.local, a.esm_batch_size, include_ms=not a.no_ms))


if __name__ == "__main__":
    main()
