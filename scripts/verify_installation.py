#!/usr/bin/env python
"""Check an installation of the binding-site predictor against the golden set (PASS / FAIL, exit code 0 / 1).

    python verify_installation.py --final-model-dir D:/.../final_model --weights-dir D:/.../model_weights \
                                  --golden-dir golden_set [--embeddings-dir DIR] [--device cpu] [--profile final|lite|all]

[1] files       final_model checksums (manifest.json) and foundation-model sha256 (MANIFEST.json)
[2] pipeline    pbind code + checkpoints on the golden inputs (training PDB + training embeddings): must equal the
                reference to 1e-4
[3] embeddings  every embedding the application will use, compared with the training one. Embeddings your own code
                computes go in --embeddings-dir as <prot_id>.<ESM|ESMIF|PROTT5>.npy (from golden_set/<prot_id>.pdb and
                the sequence in golden_set/index.json); missing ones are computed by pbind (ESM, PROTT5) or, for ESMIF,
                in-process when fair-esm is installed
[4] end-to-end  predictions from the embeddings of [3] vs the training-time predictions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TASKS = ("metal", "nuclear", "small")
TOL_PIPELINE = 1e-4          # same inputs, different hardware: float noise only
TOL_COSINE = 0.999           # per-residue cosine of a recomputed embedding vs the training one
TOL_E2E_MAX = 0.03           # end-to-end: max |probability difference| per residue
TOL_E2E_FLIPS = 0.01         # end-to-end: fraction of predicted binding residues whose call may flip

RESULTS = []


def report(step, name, ok, detail):
    status = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    RESULTS.append(status)
    print(f"  [{status}] {step} {name}: {detail}")


def sha(fp: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(fp, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_files(a, golden_meta):
    print("[1] files")
    man_fp = Path(a.final_model_dir) / "manifest.json"
    version = hashlib.sha1(man_fp.read_bytes()).hexdigest()[:12]
    report("1", "manifest.json version", version == golden_meta["model_version"],
           f"{version} (golden set built with {golden_meta['model_version']})")
    man = json.loads(man_fp.read_text())
    bad, n = [], 0
    for member, info in man["members"].items():
        for f, want in info["sha1_16"].items():
            n += 1
            fp = Path(a.final_model_dir) / member / f
            if not fp.exists() or sha(fp, "sha1")[:16] != want:
                bad.append(f"{member}/{f}")
    report("1", "final_model files", not bad, f"{n - len(bad)}/{n} match" + (f", wrong: {bad[:5]}" if bad else ""))
    if a.weights_dir and not a.skip_sha256:
        wm = Path(a.weights_dir) / "MANIFEST.json"
        if not wm.exists():
            report("1", "model_weights", False, f"{wm} missing")
            return
        for name, e in json.loads(wm.read_text()).items():
            for f, want in e["sha256"].items():
                fp = Path(a.weights_dir) / name / f
                if not fp.exists():
                    report("1", f"{name}/{f}", None if name == "esmfold_v1" else False, "missing")
                    continue
                report("1", f"{name}/{f}", sha(fp, "sha256") == want, "sha256")


def cos_min(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(c.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-model-dir", required=True)
    ap.add_argument("--weights-dir", default=None, help="foundation models (skip [3]/[4] when omitted)")
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--embeddings-dir", default=None,
                    help="<prot_id>.<ESM|ESMIF|PROTT5>.npy embeddings computed by your own code")
    ap.add_argument("--device", default=None)
    ap.add_argument("--profile", default="final", choices=["final", "lite", "all"])
    ap.add_argument("--skip-sha256", action="store_true", help="skip hashing the large foundation-model files")
    a = ap.parse_args()

    from pbind import Predictor

    gdir = Path(a.golden_dir)
    meta = json.loads((gdir / "index.json").read_text())
    profiles = ["final", "lite"] if a.profile == "all" else [a.profile]
    check_files(a, meta)

    print("[2] pipeline (golden inputs)")
    preds = {p: Predictor(a.final_model_dir, a.weights_dir, p, a.device) for p in profiles}
    golden = {e["prot_id"]: (e, dict(np.load(gdir / f"{e['prot_id']}.npz")), (gdir / f"{e['prot_id']}.pdb").read_text())
              for e in meta["proteins"]}
    for prof in profiles:
        for t in TASKS:
            worst = 0.0
            for pid, (e, z, pdb) in golden.items():
                emb = {k: z[k] for k in ("ESM", "ESMIF", "PROTT5")}
                p = np.asarray(preds[prof].predict(e["sequence"], t, pdb, embeddings=emb, return_pdb=False)["probability"])
                worst = max(worst, float(np.abs(p - z[f"pipeline_{prof}_{t}"]).max()))
            report("2", f"{prof}/{t}", worst <= TOL_PIPELINE, f"max |diff| {worst:.2e} over {len(golden)} proteins")

    if not a.weights_dir:
        print("[3]/[4] skipped (no --weights-dir)")
    else:
        print("[3] embeddings")
        P0 = preds[profiles[0]]
        edir = Path(a.embeddings_dir) if a.embeddings_dir else None
        mine, source = {}, {}
        for name in ("ESM", "PROTT5", "ESMIF"):
            worst, t0, done = 1.0, time.time(), 0
            for pid, (e, z, pdb) in golden.items():
                seq = e["sequence"]
                own = edir / f"{pid}.{name}.npy" if edir else None
                if own is not None and own.exists():
                    arr, source[name] = np.load(own), "your --embeddings-dir"
                elif name == "ESM":
                    arr, source[name] = P0.esm2(seq), "pbind"
                elif name == "PROTT5":
                    arr, source[name] = P0.prott5(seq), "pbind"
                else:
                    try:
                        import tempfile
                        from pbind.data.structures import normalize_exact_chain_A
                        with tempfile.TemporaryDirectory() as tmp:
                            raw, chain = Path(tmp) / "q.pdb", Path(tmp) / "qA.pdb"
                            raw.write_text(pdb)
                            normalize_exact_chain_A(raw, seq, chain)
                            arr, source[name] = P0.esmif(chain, seq), "pbind in-process"
                    except RuntimeError:
                        report("3", "ESMIF", None, "not provided (put <prot_id>.ESMIF.npy in --embeddings-dir); "
                                                   "[4] uses the training ESM-IF")
                        break
                if np.shape(arr) != np.shape(z[name]):
                    report("3", f"{name} {pid}", False, f"shape {np.shape(arr)} != {np.shape(z[name])}")
                    break
                mine.setdefault(pid, {})[name] = arr
                worst = min(worst, cos_min(arr, z[name]))
                done += 1
            else:
                report("3", name, worst >= TOL_COSINE, f"min per-residue cosine {worst:.5f} over {done} proteins, "
                                                       f"source: {source[name]} ({time.time() - t0:.0f}s)")
        esmif_mode = source.get("ESMIF", "training")

        print("[4] end-to-end (your embeddings vs training-time predictions)")
        for prof in profiles:
            for t in TASKS:
                worst, flips, calls = 0.0, 0, 0
                for pid, (e, z, pdb) in golden.items():
                    emb = {k: mine.get(pid, {}).get(k, z[k]) for k in ("ESM", "ESMIF", "PROTT5")}
                    out = preds[prof].predict(e["sequence"], t, pdb, embeddings=emb, return_pdb=False)
                    p, ref = np.asarray(out["probability"]), z[f"training_{prof}_{t}"]
                    worst = max(worst, float(np.abs(p - ref).max()))
                    flips += int(((p >= out["threshold"]) != (ref >= out["threshold"])).sum())
                    calls += int((ref >= out["threshold"]).sum())
                ok = worst <= TOL_E2E_MAX and flips <= max(1, TOL_E2E_FLIPS * calls)
                report("4", f"{prof}/{t}", ok, f"max |diff| {worst:.4f}, calls flipped {flips}/{calls} "
                                               f"(ESM-IF: {esmif_mode})")

    n_fail = RESULTS.count("FAIL")
    print(f"\n{'ALL PASS' if not n_fail else f'{n_fail} FAIL'}  ({RESULTS.count('PASS')} pass, "
          f"{RESULTS.count('SKIP')} skip)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
