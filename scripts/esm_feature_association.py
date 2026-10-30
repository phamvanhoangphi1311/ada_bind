#!/usr/bin/env python
"""Do the hand-crafted descriptor blocks add information beyond the protein language-model embedding?

For every descriptor block of the feature store (DevSet1010) and every label, against two baselines
(ESM-2 alone, ESM-2 + ESM-IF1), the script reports

  * probe R^2   : mean out-of-fold R^2 of a ridge regression of the block on the baseline (five folds split by
                  protein). High = the block is linearly predictable from the embedding.
  * LRT         : likelihood-ratio test of a protein-stratified (conditional) logistic model, baseline PCs vs
                  baseline PCs + block; Benjamini-Hochberg FDR across blocks within a baseline and label.
  * delta AUPRC : out-of-fold AUPRC of a logistic model on baseline PCs + block minus the same model on baseline
                  PCs alone (grouped five-fold CV by protein), with a protein-level bootstrap 95% interval
                  (percentiles). It is an effect size in the paper's own metric and also works for metal ions.
  The LRT is by far the slowest part; --no-lrt skips it.

Baseline = 32 unsupervised principal components of the embedding; a block wider than 6 columns enters as its first
6 principal components (labels are never used to build either). Labels: metal, small, nucleic (merged), dna, rna
(DNA/RNA from data/development_set/nuclear_dna_rna_labels.csv, proteins of unknown kind dropped).

    python scripts/esm_feature_association.py --out figures/data/esm_assoc.csv          # Colab / GPU-less is fine
    python scripts/esm_feature_association.py --selftest                                # synthetic check, no data
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPLIT = "DevSet1010"
N_BASE_PCS = 32
BLOCK_PC_CAP = 6
PROBE_SPLITS = 5
PROBE_ALPHAS = np.logspace(-1, 5, 8)
HIGH_R2 = 0.50
FDR_ALPHA = 0.05
SEED = 2026
EMBEDDINGS = {"ESM", "ESMIF_EF", "PROTT5"}
LABEL_COLUMN = {"metal": "metal_binding", "small": "small_binding", "nucleic": "nuclear_binding",
                "dna": "nuclear_binding", "rna": "nuclear_binding"}
FAMILY = {"metal": "metal", "small": "small", "nucleic": "nucleic", "dna": "nucleic", "rna": "nucleic"}
METAL_BLOCKS = {"ATOM", "CHEM", "METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF"}


def designed_for(block: str) -> str:
    if block.startswith("SM_"):
        return "small"
    if block.startswith("NA_"):
        return "nucleic"
    return "metal" if block in METAL_BLOCKS else "generic"


# ------------------------------------------------------------------------------------------ statistics
def ridge_weights(x, y, alpha):
    gram = (x.T @ x).astype(np.float64)
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    return np.linalg.solve(gram, (x.T @ y).astype(np.float64))


def probe_r2(base, features, groups):
    """Out-of-fold R^2 of every column of ``features`` regressed on ``base`` (alpha chosen on an inner split)."""
    n_cols = features.shape[1]
    oof = np.full(features.shape, np.nan)
    for tr, te in GroupKFold(n_splits=min(PROBE_SPLITS, len(np.unique(groups)))).split(base, groups=groups):
        inner = GroupKFold(n_splits=min(5, len(np.unique(groups[tr]))))
        inner_tr, inner_va = next(inner.split(base[tr], groups=groups[tr]))
        fit, val = tr[inner_tr], tr[inner_va]
        sc = StandardScaler().fit(base[fit])
        xf, xv = sc.transform(base[fit]).astype(np.float32), sc.transform(base[val]).astype(np.float32)
        mu = features[fit].mean(axis=0)
        mse = np.vstack([np.mean((features[val] - (xv @ ridge_weights(xf, features[fit] - mu, a).astype(np.float32)
                                                   + mu)) ** 2, axis=0) for a in PROBE_ALPHAS])
        chosen = PROBE_ALPHAS[np.argmin(mse, axis=0)]
        sc = StandardScaler().fit(base[tr])
        xt, xe = sc.transform(base[tr]).astype(np.float32), sc.transform(base[te]).astype(np.float32)
        mu = features[tr].mean(axis=0)
        pred = np.empty((len(te), n_cols))
        for a in np.unique(chosen):
            cols = np.flatnonzero(chosen == a)
            pred[:, cols] = xe @ ridge_weights(xt, features[tr][:, cols] - mu[cols], a).astype(np.float32) + mu[cols]
        oof[te] = pred
    ss_tot = np.sum((features - features.mean(axis=0)) ** 2, axis=0)
    r2 = np.full(n_cols, np.nan)
    ok = ss_tot > 1e-8
    r2[ok] = 1.0 - np.sum((features - oof) ** 2, axis=0)[ok] / ss_tot[ok]
    return r2


def design(values, cap):
    """Standardise; raw columns when the block has at most ``cap`` columns, otherwise its first ``cap`` PCs."""
    z = StandardScaler().fit_transform(values)
    if z.shape[1] <= cap:
        keep = np.nanstd(z, axis=0) > 1e-8
        if not keep.any():
            raise RuntimeError("block has no variance")
        return z[:, keep]
    return PCA(n_components=min(cap, z.shape[0] - 1), svd_solver="randomized", random_state=SEED).fit_transform(z)


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    raw, m = p[ok], int(ok.sum())
    if m:
        order = np.argsort(raw)
        adj = np.minimum.accumulate((raw[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        res = np.empty(m)
        res[order] = np.clip(adj, 0, 1)
        out[ok] = res
    return out


def conditional_lrt(y, groups, base, block):
    """Protein-stratified logistic model: baseline vs baseline + block. Falls back to BFGS if Newton fails."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    keep = np.zeros(len(y), bool)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) >= 2 and y[idx].min() != y[idx].max():
            keep[idx] = True
    order = np.argsort(groups[keep], kind="mergesort")
    yk, gk = y[keep][order], groups[keep][order]
    codes = pd.factorize(gk, sort=False)[0]
    b0, b1 = base[keep][order], block[keep][order]
    last = "not run"
    for method, maxiter in (("newton", 100), ("bfgs", 500)):
        try:
            null = ConditionalLogit(yk, b0, groups=codes).fit(disp=0, method=method, maxiter=maxiter)
            alt = ConditionalLogit(yk, np.concatenate([b0, b1], axis=1), groups=codes).fit(
                disp=0, method=method, maxiter=maxiter)
            lr = float(2.0 * (alt.llf - null.llf))
            if np.isfinite(lr) and lr >= 0:
                return {"lr_stat": lr, "df": int(b1.shape[1]), "p_value": float(stats.chi2.sf(lr, b1.shape[1])),
                        "lrt_note": method, "lrt_proteins": int(len(np.unique(codes)))}
            last = f"{method}: negative or non-finite statistic"
        except Exception as exc:  # noqa: BLE001 - statsmodels raises several types on separation
            last = f"{method}: {type(exc).__name__}"
    return {"lr_stat": np.nan, "df": int(b1.shape[1]), "p_value": np.nan, "lrt_note": "failed (" + last + ")",
            "lrt_proteins": int(len(np.unique(codes)))}


def oof_probability(x, y, groups):
    p = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        sc = StandardScaler().fit(x[tr])
        model = LogisticRegression(max_iter=300).fit(sc.transform(x[tr]), y[tr])
        p[te] = model.predict_proba(sc.transform(x[te]))[:, 1]
    return p


def weighted_ap(y_sorted, w):
    """Average precision of a ranking given in descending score order, with non-negative residue weights."""
    tp = np.cumsum(w * y_sorted)
    seen = np.cumsum(w)
    prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    total = float(w @ y_sorted)
    return float((w * y_sorted * prec).sum() / total) if total > 0 else np.nan


def delta_auprc(y, pid_idx, p_base, p_full, boot, rng):
    """AUPRC(full) - AUPRC(base) with a protein-level bootstrap interval (resampled proteins = integer weights)."""
    o0, o1 = np.argsort(-p_base, kind="stable"), np.argsort(-p_full, kind="stable")
    y0, y1, g0, g1 = y[o0], y[o1], pid_idx[o0], pid_idx[o1]
    ones = np.ones(len(y))
    point = weighted_ap(y1, ones) - weighted_ap(y0, ones)
    n_prot = int(pid_idx.max()) + 1
    draws = np.empty(boot)
    for b in range(boot):
        counts = rng.multinomial(n_prot, np.full(n_prot, 1.0 / n_prot)).astype(np.float64)
        draws[b] = weighted_ap(y1, counts[g1]) - weighted_ap(y0, counts[g0])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ------------------------------------------------------------------------------------------ driver
def analyse(label, y, groups, base_name, base_design, blocks, r2_table, boot, log=print, lrt=True):
    """One row per applicable block for a label and a baseline."""
    rng = np.random.default_rng(SEED)
    _, pid_idx = np.unique(groups, return_inverse=True)
    t0 = time.time()
    p_base = oof_probability(base_design, y, groups)
    rows = []
    for name, values in blocks.items():
        family = designed_for(name)
        if family not in ("generic", "metal", FAMILY[label]):
            continue
        blk = design(values, BLOCK_PC_CAP)
        row = {"baseline": base_name, "label": label, "block": name, "designed_for": family,
               "block_dim": int(values.shape[1]), "n_residues": int(len(y)),
               "n_proteins": int(len(np.unique(groups))),
               "r2_mean": float(np.nanmean(r2_table[name])), "r2_median": float(np.nanmedian(r2_table[name]))}
        row.update(conditional_lrt(y, groups, base_design, blk) if lrt else
                   {"lr_stat": np.nan, "df": int(blk.shape[1]), "p_value": np.nan, "lrt_note": "skipped",
                    "lrt_proteins": 0})
        p_full = oof_probability(np.concatenate([base_design, blk], axis=1), y, groups)
        row["delta_auprc"], row["ci_lo"], row["ci_hi"] = delta_auprc(y, pid_idx, p_base, p_full, boot, rng)
        rows.append(row)
        log(f"  {base_name:12s} {label:7s} {name:16s} R2 {row['r2_mean']:.2f}  LR {row['lr_stat']:.1f}  "
            f"dAUPRC {row['delta_auprc']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  ({time.time() - t0:.0f}s)")
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out


def baselines(esm, esmif, n_pcs=N_BASE_PCS):
    """name -> (raw matrix for the probe, unsupervised PCs for the models).

    A baseline of only a few PCs discards part of the embedding, so a block that is a perfect linear copy of the
    embedding can still look "informative"; compare --base-pcs 32 (as in the paper) with a larger value."""
    raw = {"ESM-2": esm}
    if esmif is not None:
        raw["ESM-2+ESM-IF1"] = np.concatenate([StandardScaler().fit_transform(esm),
                                               StandardScaler().fit_transform(esmif)], axis=1)
    return {k: (v, PCA(n_components=n_pcs, svd_solver="randomized", random_state=SEED)
                .fit_transform(StandardScaler().fit_transform(v))) for k, v in raw.items()}


def run(blocks, esm, esmif, labels, groups, boot, log=print, n_pcs=N_BASE_PCS, lrt=True):
    """labels: name -> (y, keep mask). Returns the long result table."""
    frames = []
    for bname, (raw, pcs) in baselines(esm, esmif, n_pcs).items():
        log(f"probe R2 against {bname} ...")
        r2 = {n: probe_r2(raw, v.astype(np.float32), groups) for n, v in blocks.items()}
        for label, (y, keep) in labels.items():
            frames.append(analyse(label, y[keep], groups[keep], bname, pcs[keep],
                                  {n: v[keep] for n, v in blocks.items()}, r2, boot, log, lrt))
    return pd.concat(frames, ignore_index=True)


def load_store(args):
    from pbind.features.store import FeatureStore
    from pbind.paths import resolve_paths

    paths = resolve_paths(args.base)
    fs = FeatureStore(paths.feature_store)
    idx = fs.index(SPLIT)
    if len(idx) != 1010:
        raise SystemExit(f"{SPLIT}: {len(idx)} proteins in the feature store, expected 1010")
    groups = np.repeat(idx.prot_id.to_numpy(str), (idx.end - idx.start).to_numpy())
    names = [b for b in fs.blocks(SPLIT) if b not in EMBEDDINGS]
    blocks = {n: np.asarray(fs.block(SPLIT, n), np.float32) for n in names}
    esm = np.asarray(fs.block(SPLIT, "ESM"), np.float32)
    esmif = np.asarray(fs.block(SPLIT, "ESMIF_EF"), np.float32) if fs.has(SPLIT, "ESMIF_EF") else None
    if esmif is None:
        print("ESMIF_EF is not in the feature store: the ESM-2 + ESM-IF1 control is skipped")
    kinds = None
    csv = paths.base / "data" / "development_set" / "nuclear_dna_rna_labels.csv"
    if csv.exists():
        table = pd.read_csv(csv, dtype={"prot_id": str})
        table = table[table.split == SPLIT]
        kinds = dict(zip(table.prot_id, table.ligand_kind))
    labels = {}
    for label in args.labels:
        y = np.asarray(fs.labels(SPLIT, LABEL_COLUMN[label]), np.int8)
        keep = np.ones(len(y), bool)
        if label in ("dna", "rna"):
            if kinds is None:
                print(f"{csv} not found (python scripts/build_dna_rna_labels.py): {label} skipped")
                continue
            kind = pd.Series(groups).map(kinds).fillna("none").to_numpy()
            keep = kind != "unknown"
            wanted = ("dna", "both") if label == "dna" else ("rna", "both")
            y = np.where(np.isin(kind, wanted), y, 0).astype(np.int8)
        labels[label] = (y, keep)
    return blocks, esm, esmif, labels, groups


def selftest():
    """Planted structure: a copy of the embedding, an independent signal and pure noise."""
    rng = np.random.default_rng(0)
    n_prot, length, dim = 60, 40, 12
    groups = np.repeat([f"p{i}" for i in range(n_prot)], length)
    esm = rng.normal(size=(n_prot * length, dim)).astype(np.float32)
    signal = rng.normal(size=len(groups))
    logit = -2.2 + 0.8 * esm[:, 0] + 1.6 * signal
    y = (rng.random(len(groups)) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    blocks = {"AA": esm[:, :3] @ rng.normal(size=(3, 5)) + 0.01 * rng.normal(size=(len(groups), 5)),
              "ATOM": signal[:, None] + 0.1 * rng.normal(size=(len(groups), 1)),
              "SOLV": rng.normal(size=(len(groups), 4))}
    blocks = {k: v.astype(np.float32) for k, v in blocks.items()}
    out = run(blocks, esm, None, {"metal": (y, np.ones(len(y), bool))}, groups, boot=200, log=lambda *_: None, n_pcs=dim)
    r = out.set_index("block")
    print(out[["block", "r2_mean", "lr_stat", "fdr", "delta_auprc", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    from sklearn.metrics import average_precision_score
    p = rng.random(len(y))
    o = np.argsort(-p, kind="stable")
    assert abs(weighted_ap(y[o], np.ones(len(y))) - average_precision_score(y, p)) < 1e-9
    assert r.loc["AA", "r2_mean"] > 0.95, "copy of the embedding must be fully recoverable"
    assert abs(r.loc["AA", "delta_auprc"]) < 0.02, "a copy adds nothing"
    assert r.loc["ATOM", "r2_mean"] < 0.1 and r.loc["ATOM", "delta_auprc"] > 0.05 and r.loc["ATOM", "ci_lo"] > 0
    assert r.loc["ATOM", "fdr"] < 0.05, "independent signal must pass the LRT"
    assert abs(r.loc["SOLV", "delta_auprc"]) < 0.02 and r.loc["SOLV", "ci_lo"] <= 0 <= r.loc["SOLV", "ci_hi"]
    # a baseline that keeps too few PCs makes a perfect copy look informative (why --base-pcs exists)
    short = run({"AA": blocks["AA"]}, esm, None, {"metal": (y, np.ones(len(y), bool))}, groups, boot=50,
                log=lambda *_: None, n_pcs=3, lrt=False)
    assert short.delta_auprc.iloc[0] > r.loc["AA", "delta_auprc"] + 0.03, "truncated baseline should inflate a copy"
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="project root (default: PBIND_BASE or the Drive folder)")
    ap.add_argument("--labels", nargs="+", default=["metal", "small", "nucleic", "dna", "rna"],
                    choices=sorted(LABEL_COLUMN))
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates (paper: 10000; ~10x slower)")
    ap.add_argument("--base-pcs", type=int, default=N_BASE_PCS,
                    help="principal components kept from the embedding (paper: 32; also try 128 or 256)")
    ap.add_argument("--no-lrt", action="store_true",
                    help="skip the conditional-logit LRT (statsmodels, hours on DevSet1010); keeps probe R2 and dAUPRC")
    ap.add_argument("--out", default="esm_assoc.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    blocks, esm, esmif, labels, groups = load_store(args)
    table = run(blocks, esm, esmif, labels, groups, args.boot, n_pcs=args.base_pcs, lrt=not args.no_lrt)
    table["base_pcs"] = args.base_pcs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(table)} rows; {int(table.fdr.lt(FDR_ALPHA).sum())} tests with FDR < {FDR_ALPHA}")


if __name__ == "__main__":
    main()
