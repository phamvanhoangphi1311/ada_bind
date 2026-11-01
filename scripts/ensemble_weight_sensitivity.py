#!/usr/bin/env python
"""Development AUPRC of every ensemble weighting on the grid used to choose the final weights (notebooks 09/10 cell V3-6,
12/13 cell LX-4): simplex grid, step 0.1 for four members and 0.05 for pairs, fold-mean AUPRC over the five official folds,
cross-fitted score = weights chosen on four folds and scored on the fifth.

    python scripts/ensemble_weight_sensitivity.py [--base data/drive_inputs] [--out reports/finalize_results/metrics]

Gate: the grid optimum must reproduce the weights of Table 2, and the small-molecule and merged-label scores the values saved
in notebooks 09/10 (in-sample and cross-fitted, within 1e-4).
"""
import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finalize_metrics import dna_rna_labels, load_member, load_splits  # noqa: E402

TASKS = {
    "small": ("SMALL_V3_IMPROVE", ["V1_s0", "MS_s0", "V1+ESMIF_EF+PROTT5_bag3", "MS+ESMIF_EF+PROTT5_bag3"], 0.1),
    "nuclear": ("NUCLEAR_V3_IMPROVE", ["V1_s0", "MS_s0", "V1+ESMIF_EF+PROTT5_bag3", "MS+ESMIF_EF+PROTT5_bag3"], 0.1),
    "dna": ("NUCLEAR_V3_IMPROVE", ["dna_V1_bag", "dna_MS_bag"], 0.05),
    "rna": ("NUCLEAR_V3_IMPROVE", ["rna_V1_bag", "rna_MS_bag"], 0.05),
}
CHOSEN = {"small": [0.1, 0.1, 0.4, 0.4], "nuclear": [0.2, 0.0, 0.4, 0.4], "dna": [0.5, 0.5], "rna": [0.65, 0.35]}
SAVED = {"small": {"in_sample": 0.4865, "crossfit": 0.4852, "pair_crossfit": 0.4825},
         "nuclear": {"in_sample": 0.5581, "crossfit": 0.5571, "pair_crossfit": 0.5499}}


def simplex(k, step):
    n = int(round(1 / step))
    return [np.array(c) / n for c in product(range(n + 1), repeat=k) if sum(c) == n]


def fold_mean(y, p, folds, use):
    return float(np.mean([average_precision_score(y[folds[f]], p[folds[f]]) for f in use]))


def search(P, y, folds, step, use):
    best_w, best_s = None, -1.0
    for w in simplex(len(P), step):
        s = fold_mean(y, sum(wi * pi for wi, pi in zip(w, P)), folds, use)
        if s > best_s + 1e-12:
            best_w, best_s = w, s
    return best_w, best_s


def crossfit(P, y, folds, step):
    out = []
    for f in folds:
        w, _ = search(P, y, folds, step, [g for g in folds if g != f])
        out.append(fold_mean(y, sum(wi * pi for wi, pi in zip(w, P)), folds, [f]))
    return float(np.mean(out))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(root / "data" / "drive_inputs"))
    ap.add_argument("--out", default=str(root / "reports" / "finalize_results" / "metrics"))
    args = ap.parse_args()
    base = Path(args.base)
    split_df, y, prot, fold_idx = load_splits(base)
    yt, keep, _ = dna_rna_labels(split_df, y, prot, base / "data/development_set/dna_rna_kinds_notebook16.json")
    kd = keep["DevSet1010"]

    grid, summary, ok = [], [], True
    for task, (folder, names, step) in TASKS.items():
        P = [load_member(base / f"results/{folder}/member_preds", n)["DevSet1010"] for n in names]
        if task in ("dna", "rna"):
            yy, folds = yt[task]["DevSet1010"], {f: np.intersect1d(i, kd) for f, i in fold_idx.items()}
        else:
            yy, folds = y["small_binding" if task == "small" else "nuclear_binding"]["DevSet1010"], fold_idx
        use = list(folds)
        scores = []
        for w in simplex(len(P), step):
            s = fold_mean(yy, sum(wi * pi for wi, pi in zip(w, P)), folds, use)
            scores.append(s)
            grid.append({"task": task, **{f"w{i + 1}": float(v) for i, v in enumerate(w)}, "dev_auprc": s})
        w_best, s_best = search(P, yy, folds, step, use)
        s_cf = crossfit(P, yy, folds, step)
        equal = fold_mean(yy, sum(P) / len(P), folds, use)
        singles = [fold_mean(yy, p, folds, use) for p in P]
        row = {"task": task, "members": " + ".join(names), "chosen_weights": list(map(float, w_best)),
               "in_sample": s_best, "crossfit": s_cf, "optimism": s_best - s_cf, "equal_weights": equal,
               "best_single_member": max(singles), "n_grid": len(scores),
               "share_within_0.005": float(np.mean(np.asarray(scores) >= s_best - 0.005))}
        if len(P) == 4:
            pair = [P[2], P[3]]
            row["pair_crossfit"] = crossfit(pair, yy, folds, 0.05)
            row["pair_equal_weights"] = fold_mean(yy, (pair[0] + pair[1]) / 2, folds, use)
        summary.append(row)
        if not np.allclose(w_best, CHOSEN[task]):
            print(f"GATE: {task} grid optimum {w_best} != Table 2 {CHOSEN[task]}", file=sys.stderr)
            ok = False
        for k, v in SAVED.get(task, {}).items():
            if abs(row[k] - v) > 1e-4:
                print(f"GATE: {task} {k} {row[k]:.4f} != saved {v}", file=sys.stderr)
                ok = False
        print(task, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "members"}, flush=True)
    if not ok:
        print("GATE FAILED: nothing written", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(out / "ensemble_weight_grid.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "ensemble_weight_summary.csv", index=False)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
