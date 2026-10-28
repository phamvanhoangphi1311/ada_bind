"""Pre-model dataset anatomy audit for metal / nuclear / small (no predictor is trained).

Every table is cached as CSV under results/ALL_BINDINGS_DATASET_ANATOMY_V1/ and only
recomputed with ``force=True``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import fisher_exact

from ..backend.gear import get_block, store_of
from ..constants import AA20, TASKS
from ..utils.cache import cached_table

PAIRS = [("metal_binding", "nuclear_binding"), ("metal_binding", "small_binding"),
         ("nuclear_binding", "small_binding")]
CONTEXT_GROUPS = {
    "basic_KRH": set("KRH"), "strong_basic_KR": set("KR"), "acidic_DE": set("DE"),
    "polar_NQSTY": set("NQSTY"), "aromatic_FWYH": set("FWYH"), "hydrophobic_AVILMFWY": set("AVILMFWY"),
    "sulfur_CM": set("CM"), "gly_pro_GP": set("GP"),
}
CONTEXT_RADII = [2, 4, 7, 10]
FINGERPRINT_BLOCKS = ["SPATIAL", "LOCAL", "ATOM", "CHEM", "ACCESS", "SHELL2", "ENV", "CONF",
                      "METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF", "SEQCTX"]


def _y(x):
    return np.asarray(x, dtype=np.int8)


# ------------------------------------------------------------------ 2. core anatomy
def core_anatomy(sets: dict[str, pd.DataFrame], tasks=TASKS) -> pd.DataFrame:
    rows = []
    for dataset, df in sets.items():
        for task in tasks:
            y_all = np.concatenate([_y(x) for x in df[task]])
            per = np.asarray([int(np.sum(x)) for x in df[task]])
            pos = per > 0
            rows.append({
                "dataset": dataset, "task": task, "n_proteins": len(df), "n_residues": len(y_all),
                "n_positive_residues": int(y_all.sum()), "residue_prevalence": float(y_all.mean()),
                "n_positive_proteins": int(pos.sum()), "positive_protein_fraction": float(pos.mean()),
                "median_positive_residues_per_positive_protein": float(np.median(per[pos])) if pos.any() else 0,
                "mean_positive_residues_per_positive_protein": float(np.mean(per[pos])) if pos.any() else 0,
                "max_positive_residues_one_protein": int(per.max()),
                "median_protein_length": float(np.median(df.sequence.str.len())),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. overlap
def pairwise_overlap(sets, pairs=PAIRS) -> pd.DataFrame:
    rows = []
    for dataset, df in sets.items():
        for a, b in pairs:
            ya = np.concatenate([np.asarray(x, dtype=bool) for x in df[a]])
            yb = np.concatenate([np.asarray(x, dtype=bool) for x in df[b]])
            pa = np.array([np.sum(x) > 0 for x in df[a]], dtype=bool)
            pb = np.array([np.sum(x) > 0 for x in df[b]], dtype=bool)
            inter, union = int((ya & yb).sum()), int((ya | yb).sum())
            rows.append({
                "dataset": dataset, "task_a": a, "task_b": b, "residue_intersection": inter,
                "residue_jaccard": inter / max(union, 1),
                "fraction_a_positive_residues_also_b": inter / max(int(ya.sum()), 1),
                "fraction_b_positive_residues_also_a": inter / max(int(yb.sum()), 1),
                "protein_intersection": int((pa & pb).sum()),
                "protein_jaccard": int((pa & pb).sum()) / max(int((pa | pb).sum()), 1),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. AA enrichment
def aa_enrichment(sets, tasks=TASKS) -> pd.DataFrame:
    rows = []
    for dataset, df in sets.items():
        for task in tasks:
            pos, neg, allseq = [], [], []
            for r in df.itertuples():
                y = _y(getattr(r, task))
                allseq.extend(r.sequence)
                pos.extend(a for a, yy in zip(r.sequence, y) if yy)
                neg.extend(a for a, yy in zip(r.sequence, y) if not yy)
            ac, pc, nc = Counter(allseq), Counter(pos), Counter(neg)
            nall, npos, nneg = sum(ac.values()), sum(pc.values()), sum(nc.values())
            for aa in AA20:
                a, c = pc[aa], nc[aa]
                odds, p = fisher_exact([[a, npos - a], [c, nneg - c]])
                fa, fp = ac[aa] / max(nall, 1), a / max(npos, 1)
                rows.append({"dataset": dataset, "task": task, "aa": aa, "global_frequency": fa,
                             "positive_frequency": fp, "log2_enrichment": float(np.log2((fp + 1e-8) / (fa + 1e-8))),
                             "odds_ratio": float(odds), "fisher_p": float(p), "positive_count": int(a)})
    return pd.DataFrame(rows)


def enrichment_pivot(aa: pd.DataFrame, task: str) -> pd.DataFrame:
    p = aa[aa.task.eq(task)].pivot(index="aa", columns="dataset", values="log2_enrichment")
    p["mean"] = p.mean(axis=1)
    return p.sort_values("mean", ascending=False)


# ------------------------------------------------------------------ 5. task similarity
def task_signature_similarity(aa: pd.DataFrame, pairs=PAIRS) -> pd.DataFrame:
    rows = []
    for dataset in aa.dataset.unique():
        corr = aa[aa.dataset.eq(dataset)].pivot(index="aa", columns="task", values="log2_enrichment").corr()
        for a, b in pairs:
            rows.append({"dataset": dataset, "task_a": a, "task_b": b, "AA_enrichment_correlation": float(corr.loc[a, b])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6. sequence context
def sequence_context(sets, tasks=TASKS, groups=CONTEXT_GROUPS, radii=CONTEXT_RADII) -> pd.DataFrame:
    rows = []
    for dataset, df in sets.items():
        # window fractions are label-independent: compute once per protein
        fr = {}
        for r in df.itertuples():
            s = r.sequence
            L = len(s)
            for g, chars in groups.items():
                ind = np.fromiter((x in chars for x in s), dtype=np.float64, count=L)
                cs = np.r_[0.0, np.cumsum(ind)]
                for rad in radii:
                    i = np.arange(L)
                    lo, hi = np.maximum(0, i - rad), np.minimum(L, i + rad + 1)
                    fr[(r.prot_id, g, rad)] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
        for task in tasks:
            ys = {r.prot_id: _y(getattr(r, task)).astype(bool) for r in df.itertuples()}
            for g in groups:
                for rad in radii:
                    v = np.concatenate([fr[(pid, g, rad)] for pid in ys])
                    y = np.concatenate(list(ys.values()))
                    p, n = v[y], v[~y]
                    sp = np.sqrt((p.var() + n.var()) / 2 + 1e-12)
                    rows.append({"dataset": dataset, "task": task, "feature": g, "radius": rad,
                                 "positive_mean": float(p.mean()), "negative_mean": float(n.mean()),
                                 "standardized_mean_difference": float((p.mean() - n.mean()) / sp)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 7. clustering
def _clusters(pos, max_gap):
    pos = np.asarray(pos, dtype=int)
    if len(pos) == 0:
        return []
    out = [[int(pos[0])]]
    for x in pos[1:]:
        if x - out[-1][-1] <= max_gap:
            out[-1].append(int(x))
        else:
            out.append([int(x)])
    return out


def positive_clusters(sets, tasks=TASKS, gaps=(1, 2, 3, 5)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset, df in sets.items():
        for task in tasks:
            for r in df.itertuples():
                pos = np.flatnonzero(_y(getattr(r, task)))
                for gap in gaps:
                    cc = _clusters(pos, gap)
                    rows.append({"dataset": dataset, "task": task, "prot_id": str(r.prot_id), "max_gap": gap,
                                 "n_positive": len(pos), "n_clusters": len(cc),
                                 "largest_cluster": max((len(c) for c in cc), default=0),
                                 "fraction_positive_in_cluster_ge2":
                                     sum(len(c) for c in cc if len(c) >= 2) / max(len(pos), 1)})
    per = pd.DataFrame(rows)
    summary = (per[per.n_positive.gt(0)].groupby(["dataset", "task", "max_gap"])
               .agg(positive_proteins=("prot_id", "nunique"), mean_largest_cluster=("largest_cluster", "mean"),
                    mean_fraction_positive_in_cluster_ge2=("fraction_positive_in_cluster_ge2", "mean"))
               .reset_index())
    return per, summary


# ------------------------------------------------------------------ 8. cross-split stability
def _aa_dist(df, task=None, positive_only=False):
    c = Counter()
    for r in df.itertuples():
        if positive_only:
            c.update(a for a, yy in zip(r.sequence, _y(getattr(r, task))) if yy)
        else:
            c.update(r.sequence)
    v = np.asarray([c[a] for a in AA20], dtype=float)
    return v / v.sum() if v.sum() else np.zeros_like(v)


def cross_split_stability(sets, aa: pd.DataFrame, core: pd.DataFrame, reference: str, tasks=TASKS) -> pd.DataFrame:
    ref_df = sets[reference]
    ref_global = _aa_dist(ref_df)

    def enr(ds, t):
        return aa[aa.dataset.eq(ds) & aa.task.eq(t)].set_index("aa").loc[list(AA20), "log2_enrichment"].to_numpy()

    def prev(ds, t):
        return float(core[core.dataset.eq(ds) & core.task.eq(t)].residue_prevalence.iloc[0])

    rows = []
    for task in tasks:
        ref_pos, ref_e, ref_prev = _aa_dist(ref_df, task, True), enr(reference, task), prev(reference, task)
        for dataset, df in sets.items():
            p = prev(dataset, task)
            rows.append({
                "task": task, "dataset": dataset,
                "JS_global_AA_vs_Dev": float(jensenshannon(ref_global, _aa_dist(df), base=2)),
                "JS_positive_AA_vs_Dev": float(jensenshannon(ref_pos, _aa_dist(df, task, True), base=2)),
                "AA_enrichment_correlation_vs_Dev": float(np.corrcoef(ref_e, enr(dataset, task))[0, 1]),
                "residue_prevalence": p, "prevalence_ratio_vs_Dev": p / max(ref_prev, 1e-12),
            })
    return pd.DataFrame(rows)


def stability_flag(row, reference: str) -> str:
    if row["dataset"] == reference:
        return "REFERENCE"
    corr, js, pr = row["AA_enrichment_correlation_vs_Dev"], row["JS_positive_AA_vs_Dev"], row["prevalence_ratio_vs_Dev"]
    if corr >= 0.75 and js <= 0.25 and 0.5 <= pr <= 2.0:
        return "STABLE"
    if corr >= 0.5 and js <= 0.4:
        return "MODERATE_SHIFT"
    return "STRONG_SHIFT"


# ------------------------------------------------------------------ 9. cached-feature fingerprint
def feature_fingerprint(ctx, sets, blocks=FINGERPRINT_BLOCKS, tasks=TASKS) -> pd.DataFrame:
    """Standardized positive-minus-negative effect per GEAR feature block (descriptive, not performance)."""
    rows = []
    for dataset, df in sets.items():
        store = store_of(ctx, dataset)
        for b in blocks:
            try:
                xs = [get_block(store, r.prot_id, b).astype(np.float32, copy=False) for r in df.itertuples()]
            except KeyError:
                print("Skip unavailable block:", dataset, b)
                continue
            if any(x.shape[0] != len(r.sequence) for x, r in zip(xs, df.itertuples())):
                print("Skip length-mismatched block:", dataset, b)
                continue
            X = np.concatenate(xs, axis=0)
            Z = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
            for t in tasks:
                y = np.concatenate([_y(x) for x in df[t]]).astype(bool)
                if y.sum() == 0 or (~y).sum() == 0:
                    continue
                eff = Z[y].mean(axis=0) - Z[~y].mean(axis=0)
                rows.append({"dataset": dataset, "task": t, "block": b, "dim": X.shape[1],
                             "effect_rms": float(np.sqrt(np.mean(eff ** 2))),
                             "effect_abs_mean": float(np.mean(np.abs(eff))),
                             "effect_signed_mean": float(np.mean(eff))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 10. decision tables
def decision_tables(aa, seqctx, stability, reference: str, tasks=TASKS):
    dist = []
    for dataset in aa.dataset.unique():
        corr = aa[aa.dataset.eq(dataset)].pivot(index="aa", columns="task", values="log2_enrichment").corr()
        vals = [corr.loc[a, b] for i, a in enumerate(tasks) for b in tasks[i + 1:]]
        dist.append({"dataset": dataset, "mean_pairwise_task_AA_signature_correlation": float(np.mean(vals)),
                     "max_pairwise_task_AA_signature_correlation": float(np.max(vals))})
    stab = stability.copy()
    stab["stability_flag"] = stab.apply(stability_flag, axis=1, reference=reference)
    hints = []
    for task in tasks:
        p = aa[aa.task.eq(task)].pivot(index="aa", columns="dataset", values="log2_enrichment")
        p["min"] = p.min(axis=1)
        consistent = p[p["min"] > 0].sort_values("min", ascending=False)
        c = (seqctx[seqctx.task.eq(task)].groupby(["feature", "radius"]).standardized_mean_difference
             .agg(["mean", "min", "max"]).reset_index())
        top = c.iloc[c["mean"].abs().argmax()] if len(c) else None
        hints.append({"task": task, "consistently_enriched_AA": ",".join(consistent.index.tolist()[:8]),
                      "strongest_context_feature": str(top["feature"]) if top is not None else "",
                      "strongest_context_mean_SMD": float(top["mean"]) if top is not None else np.nan})
    return pd.DataFrame(dist), stab, pd.DataFrame(hints)


# ------------------------------------------------------------------ orchestration
def run_anatomy(sets: dict[str, pd.DataFrame], out_dir: Path, reference: str, ctx=None,
                force: bool = False) -> dict[str, pd.DataFrame]:
    """Full audit; every table cached in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    R = {}
    R["core"] = cached_table(out_dir / "core_task_anatomy.csv", lambda: core_anatomy(sets), force)
    R["overlap"] = cached_table(out_dir / "pairwise_label_overlap.csv", lambda: pairwise_overlap(sets), force)
    R["aa"] = cached_table(out_dir / "aa_enrichment_all_tasks.csv", lambda: aa_enrichment(sets), force)
    R["task_similarity"] = cached_table(out_dir / "task_signature_similarity.csv",
                                        lambda: task_signature_similarity(R["aa"]), force)
    R["seqctx"] = cached_table(out_dir / "sequence_context_all_tasks.csv", lambda: sequence_context(sets), force)
    clusters_fp = out_dir / "positive_cluster_per_protein_all_tasks.csv"
    summary_fp = out_dir / "positive_cluster_summary_all_tasks.csv"
    if force or not (clusters_fp.exists() and summary_fp.exists()):
        per, summ = positive_clusters(sets)
        per.to_csv(clusters_fp, index=False)
        summ.to_csv(summary_fp, index=False)
    R["clusters"], R["cluster_summary"] = pd.read_csv(clusters_fp), pd.read_csv(summary_fp)
    R["stability"] = cached_table(out_dir / "cross_split_stability_all_tasks.csv",
                                  lambda: cross_split_stability(sets, R["aa"], R["core"], reference), force)
    if ctx is not None:
        R["fingerprint"] = cached_table(out_dir / "cached_feature_mechanism_fingerprint.csv",
                                        lambda: feature_fingerprint(ctx, sets), force)
    dist, stab, hints = decision_tables(R["aa"], R["seqctx"], R["stability"], reference)
    for name, df, fn in (("task_distinctiveness", dist, "task_distinctiveness.csv"),
                         ("stability_verdicts", stab, "stability_verdicts.csv"),
                         ("mechanism_hints", hints, "mechanism_hints.csv")):
        df.to_csv(out_dir / fn, index=False)
        R[name] = df
    return R
