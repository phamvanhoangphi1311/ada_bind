#!/usr/bin/env python
"""First command to run after cloning: is the project root found, and is the data in the right place?

    python scripts/check_setup.py [--base DIR]

Exit code 0 when everything the pipelines need is present, 1 otherwise. Optional items are reported but
never fail the check: caches and the feature store are rebuilt on demand, and the baseline reproduction is
only needed to redraw the published comparison tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.paths import check_inputs, resolve_paths  # noqa: E402

# item name in check_inputs() -> why it is needed
REQUIRED = {
    "train_csv": "DevSet1010 labels",
    "test300_csv": "TestSet300 labels",
    "test46_csv": "TestSetNew46 labels",
    "esmFold_pdb_files": "structures every structural feature is built from",
}
OPTIONAL = {
    "cache": "GEAR caches; rebuilt by scripts/precompute_features.py when absent",
    "shared_exact_pdb_archive": "chain-A copies matching each sequence; rebuilt on demand",
    "experimental_pdb": "only for the experimental-structure ablation",
    "baseline_reproduction_master_v6": "bindEmbed21DL / bindNode24 predictions, for the comparison tables",
    "backend:GEAR_ULTIMATE_OTHER_BINDINGS": "original backend; the native port is the default and needs none",
    "backend:GEAR_ULTIMATE": "original backend; the native port is the default and needs none",
}
# extra paths check_inputs() does not cover but the folds and the final models do
EXTRA_REQUIRED = {
    "fold_ids": ("data/development_set/ids_split1.txt", "official bindNode24 5-fold split"),
}
EXTRA_OPTIONAL = {
    "final_model": ("final_model/manifest.json", "frozen weights; only needed for inference, not training"),
    "feature_store": ("feature_store/manifest.json", "columnar features; rebuilt by precompute_features.py"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="project root (default: PBIND_BASE, then <repo>/data, then Drive)")
    a = ap.parse_args()

    try:
        paths = resolve_paths(a.base)
    except FileNotFoundError as e:
        print("KHONG tim thay project root.\n")
        print(e)
        return 1

    print(f"project root : {paths.base}")
    print(f"scratch      : {paths.local}\n")

    inv = {r.item: (Path(r.path), bool(r.exists)) for r in check_inputs(paths).itertuples()}
    for name, (rel, why) in EXTRA_REQUIRED.items():
        inv[name] = (paths.base / rel, (paths.base / rel).exists())
    for name, (rel, why) in EXTRA_OPTIONAL.items():
        inv[name] = (paths.base / rel, (paths.base / rel).exists())

    missing = []
    for title, group in (("BAT BUOC", {**REQUIRED, **{k: v[1] for k, v in EXTRA_REQUIRED.items()}}),
                         ("TUY CHON", {**OPTIONAL, **{k: v[1] for k, v in EXTRA_OPTIONAL.items()}})):
        print(f"--- {title} ---")
        for name, why in group.items():
            if name not in inv:
                continue
            path, ok = inv[name]
            mark = " co  " if ok else ("THIEU" if title == "BAT BUOC" else "khong")
            print(f"  [{mark}] {name:36s} {why}")
            if not ok:
                print(f"           {path}")
                if title == "BAT BUOC":
                    missing.append(name)
        print()

    if missing:
        print(f"THIEU {len(missing)} thu bat buoc: {', '.join(missing)}")
        print("Tai tu Drive 'Protein-binding-Phi/' vao thu muc tren, giu nguyen layout.")
        return 1
    print("Du input bat buoc. Buoc tiep theo:")
    print("  python -m pytest -q tests/                       # 29 test, khong can GPU")
    print("  python scripts/precompute_features.py            # dung feature store (chi tinh phan con thieu)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
