#!/usr/bin/env python
"""Extract the paired AUPRC differences (AdaBind minus comparator) for Table S-delta, on TestSet300 and on the two
test sets pooled (346 proteins), directly from notebook 16's own bootstrap output -- the same file
scripts/finalize_metrics.py gates Tables 3-4 against, so no new download or bootstrap run is needed.

    python scripts/export_delta_auprc.py [--base data/drive_inputs] [--out reports/finalize_results/metrics]
"""
import argparse
from pathlib import Path

import pandas as pd

ORDER = {"metal": ["MetalBind_13ion_max_TRANSFER", "bindEmbed21DL", "bindNode24", "ESMBind_7ion_max_TRANSFER"],
         "small": ["bindEmbed21DL", "bindNode24"], "dna": ["bindEmbed21DL", "bindNode24"],
         "rna": ["bindNode24", "bindEmbed21DL"]}
NAME = {"MetalBind_13ion_max_TRANSFER": "MetalBind", "ESMBind_7ion_max_TRANSFER": "ESMBind",
        "bindEmbed21DL": "bindEmbed21DL", "bindNode24": "bindNode24"}

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(root / "data" / "drive_inputs"))
    ap.add_argument("--out", default=str(root / "reports" / "finalize_results" / "metrics"))
    args = ap.parse_args()

    ref = pd.read_csv(Path(args.base) / "results/CI_BOOTSTRAP_ALL/CI_ALL_8_METRICS_B10000.csv")
    ref.columns = [c.strip() for c in ref.columns]
    ref = ref[(ref.metric == "AUPRC") & (ref.dataset.isin(["TestSet300", "TestSetNew46", "POOLED_TEST"]))]

    rows = []
    for task, comparators in ORDER.items():
        for m in comparators:
            for split in ("TestSet300", "TestSetNew46", "POOLED_TEST"):
                r = ref[(ref.task == task) & (ref.dataset == split) & (ref.ref_model == m)]
                assert len(r) == 1, (task, m, split)
                r = r.iloc[0]
                rows.append({"task": task, "comparator": NAME[m], "split": split, "delta": r.delta,
                             "ci_lo": r.delta_CI95_lo, "ci_hi": r.delta_CI95_hi})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "delta_auprc.csv", index=False)
    print("wrote", out / "delta_auprc.csv")
