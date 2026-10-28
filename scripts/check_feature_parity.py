#!/usr/bin/env python
"""Check the registered feature functions against caches produced by the ORIGINAL GEAR backend.

Download from Drive  Protein-binding-Phi/cache/spatial_revamp/  the <split>_*.pkl files of one split
(e.g. TestSetNew46_atomic_coordination_v1.pkl, _chemistry_v1_24d.pkl, _mechanistic_v1.pkl, _multisignal_v2.pkl,
_spatial_local_v2.pkl, optionally _ESM2_t33_mean_last2_1280_fp16.pkl) into some folder, then:

    python scripts/check_feature_parity.py --ref /path/to/that/folder
    python scripts/check_feature_parity.py --ref ... --esm          # also ESM-2 (downloads the 650M model)

Every feature group with a ``legacy_cache`` is recomputed with its registered function and compared block by
block. Every block should show max_abs_diff ~0 (ESM: small fp16/GPU noise, ~1e-2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.constants import STORE_KEY  # noqa: E402
from pbind.data.folds import find_fold_dir, load_official_folds  # noqa: E402
from pbind.data.loading import backend_frames  # noqa: E402
from pbind.features.store import FEATURE_GROUPS, BuildOptions, _legacy_blocks, compute_group  # noqa: E402
from pbind.paths import resolve_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="folder with the original <split>_*.pkl caches")
    ap.add_argument("--split", default="TestSetNew46", choices=list(STORE_KEY))
    ap.add_argument("--base", default=None)
    ap.add_argument("--esm", action="store_true")
    ap.add_argument("--max-proteins", type=int, default=None, help="default: all (50 for DevSet1010)")
    a = ap.parse_args()

    paths = resolve_paths(a.base)
    folds = load_official_folds(find_fold_dir(paths))
    df = backend_frames(paths, [p for ids in folds.values() for p in ids])[STORE_KEY[a.split]]
    n_max = a.max_proteins or (50 if a.split == "DevSet1010" else None)
    if n_max:
        df = df.head(n_max)
    lengths = {str(r.prot_id): len(str(r.sequence)[:1000]) for r in df.itertuples()}
    opts = BuildOptions()

    rows = []
    for g in FEATURE_GROUPS.values():
        if not g.legacy_cache or (g.needs == "batch" and not a.esm):
            continue
        ref = _legacy_blocks(g, Path(a.ref), a.split, lengths)
        if not ref:
            print(f"{g.name}: no reference cache in {a.ref}")
            continue
        ours = compute_group(g, df[df.prot_id.astype(str).isin(ref)], a.split, paths, opts)  # recompute, no import
        for pid, blocks in ref.items():
            for b, arr in blocks.items():
                d = np.abs(np.asarray(ours[pid][b], np.float64) - np.asarray(arr, np.float64))
                rows.append({"group": g.name, "block": b, "prot_id": pid, "max_abs_diff": float(d.max()),
                             "frac_equal_1e-5": float((d <= 1e-5).mean())})
    t = pd.DataFrame(rows)
    if t.empty:
        print("Nothing compared: check --ref and the split name.")
        return
    print(t.groupby(["group", "block"]).agg(proteins=("prot_id", "nunique"), max_abs_diff=("max_abs_diff", "max"),
                                            mean_frac_equal=("frac_equal_1e-5", "mean")).to_string())


if __name__ == "__main__":
    main()
