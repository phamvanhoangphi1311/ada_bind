#!/usr/bin/env python
"""Final metrics for Tables 3-5, Figure 2 and Section 3.5, recomputed from the saved predictions of the final models.

The predictions are the ones notebook 16 (16_BOOTSTRAP_CI_ALL_colab) assembles from Drive: metal ensemble CSVs, the
member .npz files of the hypergraph models combined with the locked weights of FINAL_DECISION.json, and the reproduced
baseline predictions. Copy them under --base with the Drive layout (data/, results/, baseline_reproduction_master_v6/).

    python scripts/finalize_metrics.py [--base data/drive_inputs] [--out reports/finalize_results] [--boot 10000]

Two gates stop the run before anything is written:
  1. every AUPRC, AUROC and protein-mean F1/MCC (point value and 95% interval) must equal notebook 16
     (results/CI_BOOTSTRAP_ALL/CI_ALL_8_METRICS_B10000.csv), so the inputs and the protein resampling are the same;
  2. per-protein precision and recall of bindEmbed21DL and bindNode24 (metal, small, TestSet300) must match the values
     the two papers publish, so the averaging rule is the one used in the manuscript tables.
Uncertainty is the SD of the 10,000 protein-bootstrap replicates (seed 2026, notebook 16's resampling).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

SPLITS = ("DevSet1010", "TestSet300", "TestSetNew46")
TESTS = ("TestSet300", "TestSetNew46")
CSV = {"DevSet1010": "data/development_set/full_grouped_train_binding_sites_df.csv",
       "TestSet300": "data/development_set/full_grouped_test_binding_sites_df.csv",
       "TestSetNew46": "data/independent_set/grouped_test_46_new_binding_sites.csv"}
LABEL_COLS = ("metal_binding", "small_binding", "nuclear_binding")
EXCLUDED_DEV = {"C8BD48", "P84801", "Q00277", "Q9NZV6"}  # ESMFold does not reproduce their sequence
NPZ_KEY = {"DevSet1010": "oof", "TestSet300": "TestSet300", "TestSetNew46": "TestSetNew46"}
METAL_STEM = "MLP__ESM__Atomic__Chem__MULTI2__ESMIF"
METAL_THRESHOLD = 0.9148463010787964  # OOF_tuned_threshold of external_summary.csv (notebook 16, THR['metal'])
BASELINES = {"metal": ["bindEmbed21DL", "bindNode24", "ESMBind_7ion_max_TRANSFER", "MetalBind_13ion_max_TRANSFER"],
             "small": ["bindEmbed21DL", "bindNode24"], "dna": ["bindEmbed21DL", "bindNode24"],
             "rna": ["bindEmbed21DL", "bindNode24"]}
LABEL_OF = {"metal": "metal_binding", "small": "small_binding", "dna": "nuclear_binding", "rna": "nuclear_binding"}
DISPLAY = {"ours": "AdaBind", "ESMBind_7ion_max_TRANSFER": "ESMBind", "MetalBind_13ion_max_TRANSFER": "MetalBind"}
# Per-protein precision / recall at 0.5 published by the predictors themselves (TestSet300, mean over proteins):
# bindEmbed21DL, Littmann et al. 2021, Suppl. Table S1; bindNode24, Erckert et al. 2025, Suppl. Table S5.
PUBLISHED = {("metal", "bindEmbed21DL"): (0.23, 0.25), ("small", "bindEmbed21DL"): (0.34, 0.37),
             ("metal", "bindNode24"): (0.30, 0.27), ("small", "bindNode24"): (0.40, 0.40)}
METRICS = ("AUPRC", "AUROC", "precision", "recall", "F1", "MCC")


# ---------------------------------------------------------------- data (notebook 16, cells 2 and 4)
def load_splits(base: Path):
    folds, dev_order = {}, []
    for k in range(1, 6):
        ids = [x for x in (base / f"data/development_set/ids_split{k}.txt").read_text().split() if x not in EXCLUDED_DEV]
        folds[k] = ids
        dev_order += ids
    split_df = {}
    for s, rel in CSV.items():
        d = pd.read_csv(base / rel)
        d["prot_id"] = d.prot_id.astype(str)
        for c in LABEL_COLS:
            d[c] = d[c].apply(lambda v: np.asarray(ast.literal_eval(v) if isinstance(v, str) else v, np.int8))
        split_df[s] = d.set_index("prot_id", drop=False).loc[dev_order].reset_index(drop=True) if s == "DevSet1010" else d
    y = {c: {s: np.concatenate(df[c].tolist()).astype(np.int64) for s, df in split_df.items()} for c in LABEL_COLS}
    prot = {s: np.repeat(np.arange(len(df)), df.sequence.str.len().to_numpy()) for s, df in split_df.items()}
    offset = {}
    for s, df in split_df.items():
        cur, o = 0, {}
        for r in df.itertuples():
            o[r.prot_id] = (cur, cur + len(r.sequence))
            cur += len(r.sequence)
        offset[s] = o
    fold_idx = {f: np.concatenate([np.arange(*offset["DevSet1010"][p]) for p in ids]) for f, ids in folds.items()}
    assert [len(v) for v in folds.values()] == [202, 203, 203, 202, 200]
    assert {s: len(y["nuclear_binding"][s]) for s in SPLITS} == {"DevSet1010": 170181, "TestSet300": 62689,
                                                                  "TestSetNew46": 6227}
    return split_df, y, prot, fold_idx


def dna_rna_labels(split_df, y, prot, kinds_json: Path):
    code = {"D": "dna", "R": "rna", "B": "both", "U": "unknown"}
    kind = {k: code[v] for k, v in json.loads(kinds_json.read_text())["kinds"].items()}
    yt, keep, prot_k = {"dna": {}, "rna": {}}, {}, {}
    for s, df in split_df.items():
        mask, per = [], {"dna": [], "rna": []}
        for r in df.itertuples():
            nuc = np.asarray(r.nuclear_binding, np.int8)
            k = "negative" if nuc.sum() == 0 else kind.get(r.prot_id, "unknown")
            for t in per:
                per[t].append(nuc if k in (t, "both") else np.zeros_like(nuc))
            mask.append(np.zeros(len(nuc), bool) if k == "unknown" else np.ones(len(nuc), bool))
        for t in per:
            yt[t][s] = np.concatenate(per[t]).astype(np.int64)
        keep[s] = np.flatnonzero(np.concatenate(mask))
        prot_k[s] = pd.factorize(prot[s][keep[s]])[0]
        assert np.array_equal((yt["dna"][s] | yt["rna"][s])[keep[s]], y["nuclear_binding"][s][keep[s]])
    return yt, keep, prot_k


# ---------------------------------------------------------------- predictions (notebook 16, cells 9 and 12)
def load_member(pred_dir: Path, name: str) -> dict:
    if "_bag" in name:
        seeds = [load_member(pred_dir, f"{name.rsplit('_bag', 1)[0]}_s{i}") for i in range(3)]
        return {s: np.mean([g[s] for g in seeds], axis=0) for s in SPLITS}
    z = np.load(pred_dir / f"{name}.npz")
    return {s: np.asarray(z[NPZ_KEY[s]], np.float64) for s in SPLITS}


def assemble(decision: Path, pred_dir: Path):
    dec = json.loads(decision.read_text())
    terms = [(m, w) for m, w in zip(dec["members"], dec["weights"]) if w]
    preds = {m: load_member(pred_dir, m) for m, _ in terms}
    return {s: sum(w * preds[m][s] for m, w in terms) for s in SPLITS}, float(dec["dev_mcc_threshold"])


def load_pred_csv(fp: Path, df: pd.DataFrame) -> np.ndarray:
    d = pd.read_csv(fp)
    d["prot_id"] = d.prot_id.astype(str)
    col = next(c for c in ("probability", "prob", "score", "raw_score", "prob_FINAL") if c in d.columns)
    g = {p: sub.sort_values("residue_index")[col].to_numpy() for p, sub in d.groupby("prot_id")}
    out = []
    for r in df.itertuples():
        a = g[r.prot_id]
        assert len(a) == len(r.sequence), f"{fp.name}/{r.prot_id}"
        out.append(a)
    return np.concatenate(out).astype(np.float64)


# ---------------------------------------------------------------- metrics
def prep(p):
    order = np.argsort(-p, kind="mergesort")
    return order, np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]


def curve_metrics(y, w, pr) -> tuple[float, float]:
    """Weighted AUPRC and AUROC, identical to sklearn's with sample_weight (notebook 16, _metrics)."""
    order, last = pr
    tp = np.cumsum((w * y)[order])[last]
    fp = np.cumsum((w * (1 - y))[order])[last]
    rec = np.r_[0.0, tp / tp[-1]]
    auprc = float(np.sum(np.diff(rec) * tp / np.maximum(tp + fp, 1e-300)))
    fpr = np.r_[0.0, fp / max(fp[-1], 1e-300)]
    return auprc, float(np.sum(np.diff(fpr) * (rec[1:] + rec[:-1]) / 2))


def protein_counts(y, pred, prot, n):
    c = {k: np.bincount(prot, weights=v.astype(float), minlength=n) for k, v in
         {"tp": (y == 1) & pred, "fp": (y == 0) & pred, "fn": (y == 1) & ~pred, "tn": (y == 0) & ~pred}.items()}
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pre = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(pre + rec > 0, 2 * pre * rec / (pre + rec), 0.0)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return {"precision": pre, "recall": rec, "F1": f1, "MCC": mcc, "inc": (tp + fp + fn > 0).astype(float)}


def protein_means(c, cnt) -> dict:
    """Mean over proteins with TP+FP+FN > 0, each protein counted as often as it was resampled."""
    wi = cnt * c["inc"]
    s = wi.sum()
    return {k: float(wi @ c[k] / s) if s else np.nan for k in ("precision", "recall", "F1", "MCC")}


def score(y, prot, models: dict, boot: int, seed: int):
    """Point value, bootstrap SD and percentile 95% interval of the six metrics for every model."""
    n = int(prot.max()) + 1
    y = np.asarray(y, float)
    pre = {m: prep(p) for m, (p, _) in models.items()}
    cnts = {m: protein_counts(y, p >= t, prot, n) for m, (p, t) in models.items()}
    one = np.ones(n)
    point = {m: dict(zip(("AUPRC", "AUROC"), curve_metrics(y, one[prot], pre[m])), **protein_means(cnts[m], one))
             for m in models}
    reps = {m: {k: [] for k in METRICS} for m in models}
    rng = np.random.default_rng(seed)
    for _ in range(boot):
        cnt = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        w = cnt[prot]
        if np.sum(w * y) == 0:  # same skip rule as notebook 16
            continue
        for m in models:
            a, r = curve_metrics(y, w, pre[m])
            reps[m]["AUPRC"].append(a)
            reps[m]["AUROC"].append(r)
            for k, v in protein_means(cnts[m], cnt).items():
                reps[m][k].append(v)
    rows = []
    for m, (p, t) in models.items():
        for k in METRICS:
            r = np.asarray(reps[m][k], float)
            rows.append({"method": m, "metric": k, "value": point[m][k], "sd": float(np.nanstd(r, ddof=1)),
                         "ci_lo": float(np.nanpercentile(r, 2.5)), "ci_hi": float(np.nanpercentile(r, 97.5)),
                         "n_boot": int(np.isfinite(r).sum()), "threshold": t,
                         "n_protein_included": int(cnts[m]["inc"].sum()) if k not in ("AUPRC", "AUROC") else n})
    return rows


def thin(x, y, keep=1200):
    """Drop repeated vertices and cap the number of points for plotting (areas are computed before thinning)."""
    pts = pd.DataFrame({"x": np.round(x, 5), "y": np.round(y, 5)}).drop_duplicates().to_numpy()
    if len(pts) > keep:
        pts = pts[np.unique(np.r_[0, np.linspace(0, len(pts) - 1, keep).astype(int), len(pts) - 1])]
    return pts


def curves_and_counts(y, models: dict):
    """ROC and precision-recall curves plus pooled counts at each model's threshold; returns (curve rows, count rows, areas)."""
    cur, cnt, area = [], [], {}
    for m, (p, thr) in models.items():
        fpr, tpr, _ = roc_curve(y, p)
        pre, rec, _ = precision_recall_curve(y, p)
        area[m] = {"AUROC": float(auc(fpr, tpr)), "AUPRC": float(average_precision_score(y, p))}
        for name, (a, b) in {"roc": (fpr, tpr), "pr": (rec, pre)}.items():
            cur += [{"method": m, "curve": name, "x": u, "y": v} for u, v in thin(a, b)]
        pred = p >= thr
        cnt.append({"method": m, "threshold": thr, "n_residue": len(y), "n_positive": int(y.sum()),
                    "n_predicted_positive": int(pred.sum()), "tp": int((pred & (y == 1)).sum()),
                    "fp": int((pred & (y == 0)).sum()), "fn": int((~pred & (y == 1)).sum())})
    return cur, cnt, area


def per_protein(y, prot, p, thr, ids) -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(ids):
        sel = prot == i
        yi, pi = y[sel].astype(bool), p[sel]
        pred = pi >= thr
        tp, fp = int((yi & pred).sum()), int((~yi & pred).sum())
        fn, tn = int((yi & ~pred).sum()), int((~yi & ~pred).sum())
        den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        rows.append({"prot_id": pid, "n_pos": int(yi.sum()),
                     "auroc": float(roc_auc_score(yi, pi)) if 0 < yi.sum() < len(yi) else np.nan,
                     "mcc": (tp * tn - fp * fn) / den if den > 0 else 0.0,
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "recall": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(root / "data" / "drive_inputs"))
    ap.add_argument("--out", default=str(root / "reports" / "finalize_results"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)

    split_df, y, prot, fold_idx = load_splits(base)
    yt, keep, prot_k = dna_rna_labels(split_df, y, prot, base / "data/development_set/dna_rna_kinds_notebook16.json")

    ours, thr = {}, {}
    metal_dir = base / "results/metal_v4_06_final_mlp_esmif_esmfold/metal_binding/ensemble_predictions"
    ours["metal"] = {s: load_pred_csv(metal_dir / f"{METAL_STEM}__{s}.csv.gz", split_df[s]) for s in TESTS}
    thr["metal"] = METAL_THRESHOLD
    ours["small"], thr["small"] = assemble(base / "results/SMALL_V3_IMPROVE/FINAL_DECISION.json",
                                           base / "results/SMALL_V3_IMPROVE/member_preds")
    for t in ("dna", "rna"):
        ours[t], thr[t] = assemble(base / f"results/NUCLEAR_{t.upper()}_LINE/FINAL_DECISION.json",
                                   base / "results/NUCLEAR_V3_IMPROVE/member_preds")
    pred_dir = base / "baseline_reproduction_master_v6/predictions"
    refs = {t: {m: {s: load_pred_csv(pred_dir / f"{m}__{s}__{LABEL_OF[t]}.csv.gz", split_df[s]) for s in TESTS}
                for m in ms} for t, ms in BASELINES.items()}

    rows, fig2, curves, ops, areas = [], [], {}, [], {}
    for t in ("metal", "small", "dna", "rna"):
        for s in TESTS:
            if t in ("dna", "rna"):
                k = keep[s]
                yy, pp, sub = yt[t][s][k], prot_k[s], (lambda a, k=k: np.asarray(a)[k])
                ids = pd.unique(split_df[s].prot_id.to_numpy()[prot[s][k]])
            else:
                yy, pp, sub = y[LABEL_OF[t]][s], prot[s], (lambda a: np.asarray(a))
                ids = split_df[s].prot_id.to_numpy()
            models = {"ours": (sub(ours[t][s]), thr[t])}
            models.update({m: (sub(refs[t][m][s]), 0.5) for m in BASELINES[t]})
            for r in score(yy, pp, models, args.boot, args.seed):
                rows.append({"task": t, "split": s, **r})
            cur, cnt, area = curves_and_counts(yy, models)
            curves[(t, s)] = cur
            ops += [{"task": t, "split": s, **c} for c in cnt]
            areas[(t, s)] = area
            if s == "TestSetNew46":
                for m, (p, th) in models.items():
                    d = per_protein(yy, pp, p, th, ids)
                    d.insert(0, "method", DISPLAY.get(m, m))
                    d.insert(0, "task", t)
                    fig2.append(d)
            print(f"{t:5s} {s:12s} done", flush=True)
    tab = pd.DataFrame(rows)

    # gate 1: notebook 16
    ref = pd.read_csv(base / "results/CI_BOOTSTRAP_ALL/CI_ALL_8_METRICS_B10000.csv")
    ref.columns = [c.strip() for c in ref.columns]
    name = {"AUPRC": "AUPRC", "AUROC": "AUROC", "F1": "protein_mean_F1", "MCC": "protein_mean_MCC"}
    gate1 = []
    for r in tab[tab.metric.isin(name)].itertuples():
        nb = ref[(ref.task == r.task) & (ref.dataset == r.split) & (ref.metric == name[r.metric])]
        if r.method == "ours":
            nb = nb.iloc[0]
            v, lo, hi = nb.new, nb.new_CI95_lo, nb.new_CI95_hi
        else:
            nb = nb[nb.ref_model == r.method].iloc[0]
            v, lo, hi = nb.ref, nb.ref_CI95_lo, nb.ref_CI95_hi
        gate1.append(max(abs(r.value - v), abs(r.ci_lo - lo), abs(r.ci_hi - hi)))
    gate1_max = float(np.max(gate1))
    # gate 2: published per-protein precision / recall
    gate2 = {}
    for (t, m), (p_pub, r_pub) in PUBLISHED.items():
        g = tab[(tab.task == t) & (tab.split == "TestSet300") & (tab.method == m)].set_index("metric").value
        gate2[f"{t}/{m}"] = {"precision": [round(g["precision"], 4), p_pub], "recall": [round(g["recall"], 4), r_pub]}
    gate2_ok = all(abs(a - b) <= 0.01 for v in gate2.values() for a, b in v.values())
    # gate 3: curve areas equal the AUROC/AUPRC printed in Tables 3-4
    gate3 = max(abs(areas[(r.task, r.split)][r.method][r.metric] - r.value)
                for r in tab[tab.metric.isin(["AUPRC", "AUROC"])].itertuples())
    print(f"gate 3 (curve areas vs table values): max |difference| = {gate3:.2e}")
    print(f"gate 1 (notebook 16): max |difference| = {gate1_max:.2e}")
    print("gate 2 (published per-protein precision/recall):", json.dumps(gate2))

    # Table 5(c): development fold means, hypergraph with broad candidates, without and with ESM-IF1
    t5 = []
    for task, label in (("small", "small_binding"), ("nucleic_acids_undivided", "nuclear_binding")):
        d = base / f"results/{'SMALL' if task == 'small' else 'NUCLEAR'}_V3_IMPROVE/member_preds"
        yd = y[label]["DevSet1010"]
        folds = {m: np.array([average_precision_score(yd[i], load_member(d, m)["DevSet1010"][i])
                              for i in fold_idx.values()]) for m in ("V1_s0", "V1+ESMIF_EF_s0")}
        diff = folds["V1+ESMIF_EF_s0"] - folds["V1_s0"]
        for m, f in [*folds.items(), ("difference", diff)]:
            t5.append({"task": task, "config": m, "fold_mean": f.mean(), "fold_sd": f.std(ddof=1),
                       **{f"fold{i + 1}": v for i, v in enumerate(f)}})
    t5 = pd.DataFrame(t5)
    print(t5.round(4).to_string(index=False))

    if gate1_max > 1e-6 or not gate2_ok or gate3 > 1e-6:
        print("GATE FAILED: nothing written", file=sys.stderr)
        return 1
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    for (t, s), cur in curves.items():
        pd.DataFrame(cur).to_csv(out / "curves" / f"{t}__{s}.csv", index=False)
    pd.DataFrame(ops).to_csv(out / "metrics" / "operating_points.csv", index=False)
    tab.to_csv(out / "metrics" / "table34_mean_sd.csv", index=False)
    t5.to_csv(out / "metrics" / "table5_foldmean.csv", index=False)
    pd.concat(fig2, ignore_index=True).to_csv(out / "metrics" / "test46_per_protein.csv", index=False)
    df46 = split_df["TestSetNew46"]
    res = pd.DataFrame({"prot_id": np.repeat(df46.prot_id.to_numpy(), df46.sequence.str.len().to_numpy()),
                        "resid": np.concatenate([np.arange(1, len(q) + 1) for q in df46.sequence]),
                        "aa": list("".join(df46.sequence)), "prob": ours["small"]["TestSetNew46"],
                        "threshold": thr["small"], "label": y["small_binding"]["TestSetNew46"]})
    assert abs(average_precision_score(res.label, res.prob) - 0.471567) < 1e-5
    res.to_csv(out / "predictions" / "test46_small_residues.csv", index=False)
    sources = json.loads((base / "SOURCES.json").read_text()) if (base / "SOURCES.json").exists() else {}
    manifest = {"script": "scripts/finalize_metrics.py", "bootstrap_B": args.boot, "seed": args.seed,
                "uncertainty": "SD (ddof=1) of protein-bootstrap replicates; ci = percentile 95%",
                "per_protein_rule": "per-protein value, mean over proteins with TP+FP+FN>0; precision 0 when no "
                                    "residue is predicted (as bindEmbed21DL/bindNode24)",
                "thresholds": {**thr, "baselines": 0.5}, "gate1_max_abs_diff_vs_notebook16": gate1_max,
                "gate2_published_vs_ours": gate2,
                "outputs": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(out.rglob("*.csv"))},
                "inputs_from_drive": sources}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
