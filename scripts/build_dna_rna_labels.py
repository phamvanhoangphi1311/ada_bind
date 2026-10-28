#!/usr/bin/env python
"""Split the existing ``nuclear`` labels into a DNA part and an RNA part, using BioLiP2 as the only new input.

The per-residue nuclear labels are NOT changed. For every protein that has nuclear binding sites we only ask BioLiP
*which kind* of nucleic acid it binds (ligand_id "dna" / "rna", column 05 of the BioLiP table), and decompose:

    protein is dna  -> y_DNA = nuclear,  y_RNA = 0
    protein is rna  -> y_RNA = nuclear,  y_DNA = 0
    protein is both -> y_DNA = y_RNA = nuclear        (conservative; a handful of proteins)
    unknown         -> masked out of the auxiliary loss (y stays nuclear for the main task)

so ``y_DNA | y_RNA == nuclear`` by construction: the auxiliary task can never contradict the labels the models were
selected on. Residue-level BioLiP evidence is only reported as an audit column, never used to relabel.

    python scripts/build_dna_rna_labels.py [--snapshot DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind.paths import resolve_paths  # noqa: E402

SNAPSHOT = "/var/lib/docker/biolip_snapshots/BioLiP2_full_2026-09-29"
TABLE = "derived_tables/interaction_sites_res3_uniprot.parquet"
SPLITS = ("DevSet1010", "TestSet300", "TestSetNew46")


def nuclear_frames(P):
    out = {}
    for split, fp in zip(SPLITS, (P.train_csv, P.test300_csv, P.test46_csv)):
        df = pd.read_csv(fp)
        df["prot_id"] = df.prot_id.astype(str)
        df["y"] = df.nuclear_binding.apply(lambda s: np.asarray(ast.literal_eval(s), np.int8))
        out[split] = df[df.y.apply(lambda a: int(a.sum())) > 0].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    P = resolve_paths()
    out_fp = Path(a.out) if a.out else P.base / "data" / "development_set" / "nuclear_dna_rna_labels.csv"
    frames = nuclear_frames(P)
    seqs = {r.prot_id: r.sequence for df in frames.values() for r in df.itertuples()}

    t = pq.read_table(Path(a.snapshot) / TABLE,
                      columns=["ligand_id", "uniprot_id_final", "binding_residues_seq_raw", "receptor_sequence"]
                      ).to_pandas()
    na = t[t.ligand_id.isin(["dna", "rna"]) & t.uniprot_id_final.isin(seqs)]
    kinds = na.groupby("uniprot_id_final").ligand_id.agg(lambda s: frozenset(s)).to_dict()

    # audit only: BioLiP binding residues mapped onto our sequence by exact substring of the PDB chain
    mapped = {}
    for pid, grp in na.groupby("uniprot_id_final"):
        seq = seqs[pid]
        hit = {"dna": np.zeros(len(seq), bool), "rna": np.zeros(len(seq), bool)}
        for r in grp.itertuples():
            off = seq.find(r.receptor_sequence or "\0")
            if off < 0:
                continue
            for m in re.finditer(r"[A-Z](\d+)", r.binding_residues_seq_raw or ""):
                i = off + int(m.group(1)) - 1
                if 0 <= i < len(seq):
                    hit[r.ligand_id][i] = True
        mapped[pid] = hit

    rows = []
    for split, df in frames.items():
        for r in df.itertuples():
            k = kinds.get(r.prot_id)
            kind = "unknown" if k is None else ("both" if len(k) == 2 else next(iter(k)))
            h = mapped.get(r.prot_id, {"dna": np.zeros(0, bool), "rna": np.zeros(0, bool)})
            pos = r.y > 0
            agree = [float((pos[h[g]].mean())) if h[g].any() else np.nan for g in ("dna", "rna")]
            rows.append({"prot_id": r.prot_id, "split": split, "length": len(r.sequence),
                         "n_nuclear": int(r.y.sum()), "ligand_kind": kind,
                         "n_biolip_dna_residues": int(h["dna"].sum()), "n_biolip_rna_residues": int(h["rna"].sum()),
                         "audit_dna_in_nuclear": agree[0], "audit_rna_in_nuclear": agree[1]})
    out = pd.DataFrame(rows).sort_values(["split", "prot_id"])
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_fp, index=False)

    meta = {"source": "BioLiP2 snapshot " + Path(a.snapshot).name,
            "table": TABLE + " (resolution <= 3.0 A; uniprot_id_final = BioLiP column 18, filled/corrected via SIFTS)",
            "rule": "protein-level ligand kind only; y_DNA | y_RNA == nuclear by construction; "
                    "unknown proteins are masked out of the auxiliary loss",
            "counts": {s: out[out.split == s].ligand_kind.value_counts().to_dict() for s in SPLITS},
            "positives": {s: out[out.split == s].groupby("ligand_kind").n_nuclear.sum().to_dict() for s in SPLITS}}
    (out_fp.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(out.groupby(["split", "ligand_kind"]).agg(proteins=("prot_id", "count"), positives=("n_nuclear", "sum")
                                                     ).to_string())
    a_ = out[["audit_dna_in_nuclear", "audit_rna_in_nuclear"]].stack()
    print(f"\naudit: BioLiP-mapped residues that are nuclear positives: median {a_.median():.3f} over {len(a_)} checks")
    print("wrote", out_fp, "and", out_fp.with_suffix(".json"))


if __name__ == "__main__":
    main()
