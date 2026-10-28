#!/usr/bin/env python
"""Golden set for verify_installation.py: test proteins (<= 400 aa, positives of every binding type) with the PDB and
embeddings used at training time plus two reference outputs per profile and task:

    pipeline_<profile>_<task>  Predictor output on THESE inputs (checks the installed pbind code + checkpoints)
    training_<profile>_<task>  the training-time prediction of the same files (end-to-end reference)

    python scripts/build_golden_set.py            # -> <BASE>/golden_set/
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbind import Predictor  # noqa: E402
from pbind.paths import resolve_paths  # noqa: E402

TASKS = ("metal", "nuclear", "small")
LABEL = {"metal": "metal_binding", "nuclear": "nuclear_binding", "small": "small_binding"}
PER_TASK, TOTAL, MAX_LEN = 3, 12, 400


def main():
    P = resolve_paths()
    out = P.base / "golden_set"
    out.mkdir(exist_ok=True)
    man_raw = (P.final_model / "manifest.json").read_bytes()
    man = json.loads(man_raw)
    csv = {"TestSet300": P.test300_csv, "TestSetNew46": P.test46_csv}
    frames = {}
    for s, fp in csv.items():
        df = pd.read_csv(fp)
        df["prot_id"] = df.prot_id.astype(str)
        df["start"] = np.r_[0, np.cumsum(df.sequence.str.len())[:-1]]
        for t in TASKS:
            df[t] = df[LABEL[t]].apply(lambda v: int(np.sum(eval(v))))
        frames[s] = df

    # selection: >= PER_TASK proteins with positives per binding type, some from TestSetNew46, all <= MAX_LEN aa
    rng = np.random.default_rng(2026)
    picked = []
    for s, n_new in (("TestSetNew46", 1), ("TestSet300", PER_TASK - 1)):
        df = frames[s][frames[s].sequence.str.len().between(60, MAX_LEN)]
        for t in TASKS:
            pool = df[(df[t] > 0) & ~df.prot_id.isin([p for _, p in picked])]
            picked += [(s, p) for p in rng.choice(pool.prot_id, n_new, replace=False)]
    df = frames["TestSet300"][frames["TestSet300"].sequence.str.len().between(60, MAX_LEN)]
    rest = df[~df.prot_id.isin([p for _, p in picked])]
    picked += [("TestSet300", p) for p in rng.choice(rest.prot_id, TOTAL - len(picked), replace=False)]

    # training-time predictions of every profile / task (whole split, CSV order)
    ref = {"final": {}, "lite": {}}
    for prof, rv, rep in (("final", "metal_v4_06_final_mlp_esmif_esmfold", "ESM__Atomic__Chem__MULTI2__ESMIF"),
                          ("lite", "metal_v2_06_final_locked_external_saved", "ESM__Atomic__Chem__MULTI2")):
        Z = [pickle.load(open(P.results / rv / "metal_binding/fold_predictions" / f"MLP__{rep}__fold{k}.pkl", "rb"))
             for k in range(1, 6)]
        ref[prof]["metal"] = {s: np.mean(np.stack([np.asarray(z[f"{k}_p"], float) for z in Z]), axis=0)
                              for s, k in (("TestSet300", "test300"), ("TestSetNew46", "test46"))}
    for task in ("nuclear", "small"):
        M = {f.stem: np.load(f) for f in (P.results / f"{task.upper()}_V3_IMPROVE/member_preds").glob("*.npz")}

        def name(m):
            n = m.split("/")[1]
            return {"v1": "V1_s0", "ms": "MS_s0"}.get(n) or n.replace("v1_esmif_prott5", "V1+ESMIF_EF+PROTT5").replace(
                "ms_esmif_prott5", "MS+ESMIF_EF+PROTT5")
        for prof in ("final", "lite"):
            terms = man["profiles"][prof][task]["terms"]
            ref[prof][task] = {s: sum(t["weight"] * np.mean([M[name(m)][s] for m in t["members"]], axis=0)
                                      for t in terms if t["weight"]) for s in csv}

    esm = {}
    for s in csv:
        idx = pd.read_csv(P.feature_store / s / "index.csv")
        esm[s] = (idx.set_index(idx.prot_id.astype(str)), np.load(P.feature_store / s / "ESM.npy", mmap_mode="r"))
    predictors = {prof: Predictor(P.final_model, None, prof) for prof in ("final", "lite")}
    index = []
    for s, pid in picked:
        r = frames[s].set_index("prot_id").loc[pid]
        seq, a, L = r.sequence, int(r.start), len(r.sequence)
        idx, arr = esm[s]
        emb = {"ESM": np.asarray(arr[int(idx.loc[pid, "start"]):int(idx.loc[pid, "end"])], np.float16),
               "ESMIF": np.load(P.cache / "gear_metal_v3/esmif_esmfold_v1" / s / "esm_if" / f"{pid}.npy").astype(np.float32),
               "PROTT5": np.load(P.cache / "prott5_xl_half_v1" / s / f"{pid}.npy").astype(np.float16)}
        pdb = (P.esmfold_pdb / f"{pid}.pdb").read_text()
        (out / f"{pid}.pdb").write_text(pdb)
        arrays = {k: v for k, v in emb.items()}
        for prof, pred in predictors.items():
            for t in TASKS:
                res = pred.predict(seq, t, pdb, embeddings=emb, return_pdb=False)
                arrays[f"pipeline_{prof}_{t}"] = np.asarray(res["probability"], np.float64)
                arrays[f"training_{prof}_{t}"] = np.asarray(ref[prof][t][s][a:a + L], np.float64)
        np.savez_compressed(out / f"{pid}.npz", **arrays)
        index.append({"prot_id": pid, "split": s, "length": L, "sequence": seq,
                      "positives": {t: int(r[t]) for t in TASKS}})
        print(f"{pid} ({s}, {L} aa) positives {[int(r[t]) for t in TASKS]}")
    meta = {"model_version": hashlib.sha1(man_raw).hexdigest()[:12],
            "embeddings": {"ESM": "ESM-2 650M, mean of last 2 layers, fp16 (L, 1280)",
                           "ESMIF": "ESM-IF1 encoder, float32 (L, 512), exact training cache",
                           "PROTT5": "ProtT5-XL half, fp16 (L, 1024), exact training cache"},
            "pdb": "ESMFold v1 files used at training time (<prot_id>.pdb)",
            "proteins": index}
    (out / "index.json").write_text(json.dumps(meta, indent=2))
    print("golden set:", out, len(index), "proteins")


if __name__ == "__main__":
    main()
