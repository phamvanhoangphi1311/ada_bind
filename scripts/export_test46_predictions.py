#!/usr/bin/env python
"""Per-residue predictions of the final models on TestSetNew46, for the case-study figure and Figure 2.

Examples
    python scripts/export_test46_predictions.py --task small --weights-dir /content/weights
    python scripts/export_test46_predictions.py --task metal --base /data/Protein-binding-Phi --out metal.csv

Writes one row per residue: prot_id, resid (1-based), aa, prob, threshold, label, plddt. At the end the pooled
AUPRC over all residues is compared with the value reported in the paper; the script exits with status 1 when
they differ by more than --tol, so a wrong profile or wrong structures cannot feed the figures.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pbind.evaluation.metrics import ap  # noqa: E402
from pbind.inference.predictor import Predictor  # noqa: E402
from pbind.paths import resolve_paths  # noqa: E402

LABEL_COLUMN = {"metal": "metal_binding", "small": "small_binding", "nuclear": "nuclear_binding"}
REPORTED_AUPRC = {"metal": 0.682, "small": 0.472}  # TestSetNew46, Table 5 of the manuscript


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(LABEL_COLUMN), default="small")
    parser.add_argument("--base", default=None, help="project root (default: PBIND_BASE or the Drive folder)")
    parser.add_argument("--final-model-dir", default=None, help="default: <base>/final_model")
    parser.add_argument("--weights-dir", required=True, help="folder with the ESM-2 / ProtT5 / ESM-IF1 weights")
    parser.add_argument("--pdb-dir", default=None,
                        help="ESMFold PDBs named <prot_id>.pdb (default: <base>/esmFold_pdb_files)")
    parser.add_argument("--labels", default=None, help="TestSetNew46 csv (default: the one under <base>)")
    parser.add_argument("--profile", default="final")
    parser.add_argument("--out", default=None, help="default: test46_<task>_residues.csv")
    parser.add_argument("--tol", type=float, default=0.0015, help="allowed deviation from the reported AUPRC")
    args = parser.parse_args()

    paths = resolve_paths(args.base)
    labels = pd.read_csv(args.labels or paths.test46_csv)
    pdb_dir = Path(args.pdb_dir or paths.esmfold_pdb)
    predictor = Predictor(args.final_model_dir or paths.final_model, args.weights_dir, profile=args.profile)

    rows = []
    for rec in labels.itertuples(index=False):
        y = np.asarray(ast.literal_eval(getattr(rec, LABEL_COLUMN[args.task])), dtype=int)
        pdb = pdb_dir / f"{rec.prot_id}.pdb"
        if not pdb.exists():
            print(f"missing structure: {pdb}", file=sys.stderr)
            return 1
        res = predictor.predict(rec.sequence, args.task, pdb=str(pdb), return_pdb=False)
        if res["length"] != len(y):
            print(f"{rec.prot_id}: {res['length']} residues predicted, {len(y)} labels", file=sys.stderr)
            return 1
        for i, (p, lab, pl) in enumerate(zip(res["probability"], y, res["structure"]["plddt"])):
            rows.append((rec.prot_id, i + 1, res["sequence"][i], p, res["threshold"], int(lab), pl))
        print(f"{rec.prot_id}: {len(y)} residues, {int(y.sum())} positives", flush=True)

    df = pd.DataFrame(rows, columns=["prot_id", "resid", "aa", "prob", "threshold", "label", "plddt"])
    out = Path(args.out or f"test46_{args.task}_residues.csv")
    df.to_csv(out, index=False)

    pooled = ap(df["label"], df["prob"])
    print(f"wrote {out}: {len(df)} residues, {df['prot_id'].nunique()} proteins; pooled AUPRC {pooled:.4f}")
    expected = REPORTED_AUPRC.get(args.task)
    if expected is not None and abs(pooled - expected) > args.tol:
        print(f"pooled AUPRC {pooled:.4f} differs from the reported {expected:.3f}: check profile and structures",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
