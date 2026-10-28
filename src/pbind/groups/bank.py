"""Group banks = per-protein cached (edge_res, edge_group, group_feat) used by hypergraph models.

On-disk format is exactly the original one, so existing Drive caches are reused:
  nuclear: npz keys edge_res / edge_patch / patch_feat
  small:   npz keys edge_res / edge_pocket / pocket_feat
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..data.structures import pdb_path, structure_coords
from ..paths import Paths
from ..utils.cache import cached_npz
from . import nuclear_patches as NP
from . import small_pockets as SP


@dataclass(frozen=True)
class GroupBankSpec:
    name: str
    group: str  # "patch" | "pocket"  (npz key / state_dict prefix)
    cache_parts: tuple[str, ...]
    feat_dim: int
    build: Callable[[np.ndarray, str], tuple]  # (coords, seq) -> (edge_res, edge_group, feat)
    params: dict = field(default_factory=dict)


GROUP_BANKS: dict[str, GroupBankSpec] = {
    "nuclear_patch_v1": GroupBankSpec(
        "nuclear_patch_v1", "patch", ("nuclear_ultimate_v1", "patch_hypergraph_v1"), NP.V1_FEAT_DIM,
        NP.build_patches_v1, {"radius": 10.0, "max_patches": 128}),
    "nuclear_patch_ms": GroupBankSpec(
        "nuclear_patch_ms", "patch", ("nuclear_v2_selective_multiscale", "selective_multiscale_patches"),
        NP.MS_FEAT_DIM, NP.build_patches_ms, {"max_patches": 48, "nms": 0.65}),
    "small_pocket_v1": GroupBankSpec(
        "small_pocket_v1", "pocket", ("small_ultimate_v1", "pocket_hypergraph_v1"), SP.V1_FEAT_DIM,
        SP.build_pockets_v1, {"radius": 8.0, "max_pockets": 160}),
    "small_pocket_ms": GroupBankSpec(
        "small_pocket_ms", "pocket", ("small_multiscale_pockets_v1",), SP.MS_FEAT_DIM,
        SP.build_pockets_ms, {"max_pockets": 64, "nms": 0.65}),
}


class GroupBank(dict):
    """(split, pid) -> (edge_res, edge_group, group_feat), plus its spec."""

    def __init__(self, spec: GroupBankSpec):
        super().__init__()
        self.spec = spec

    @property
    def feat_dim(self) -> int:
        return self.spec.feat_dim

    @property
    def group(self) -> str:
        return self.spec.group


def _keys(group: str):
    return "edge_res", f"edge_{group}", f"{group}_feat"


def load_bank(paths: Paths, splits: dict[str, pd.DataFrame], bank: str | GroupBankSpec,
              force: bool = False, log_every: int = 250) -> tuple[GroupBank, pd.DataFrame]:
    """Load (building missing/invalid proteins) a bank for all splits. Returns (bank, audit_df)."""
    spec = GROUP_BANKS[bank] if isinstance(bank, str) else bank
    k_res, k_grp, k_feat = _keys(spec.group)

    def valid(z):
        return z[k_feat].ndim == 2 and z[k_feat].shape[1] == spec.feat_dim and len(z[k_res]) == len(z[k_grp])

    out, audit = GroupBank(spec), []
    for split, df in splits.items():
        root = paths.cache_dir(*spec.cache_parts, split)
        for n, r in enumerate(df.itertuples(), start=1):
            pid = str(r.prot_id)

            def build(pid=pid, seq=r.sequence, split=split):
                er, eg, gf = spec.build(structure_coords(pdb_path(paths, split, pid), seq), seq, **spec.params)
                return {k_res: er, k_grp: eg, k_feat: gf}

            z, _ = cached_npz(root / f"{pid}.npz", build, check=valid, force=force)
            er = z[k_res].astype(np.int64)
            eg = z[k_grp].astype(np.int64)
            gf = z[k_feat].astype(np.float32)
            out[(split, pid)] = (er, eg, gf)
            L = len(r.sequence)
            audit.append({
                "dataset": split, "prot_id": pid, "n_residues": L, "n_groups": len(gf),
                "n_memberships": len(er), "coverage": (len(np.unique(er)) if len(er) else 0) / max(L, 1),
            })
            if log_every and n % log_every == 0:
                print(spec.name, split, n, "/", len(df))
    return out, pd.DataFrame(audit)


def summarize_audit(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.groupby("dataset").agg(
        proteins=("prot_id", "nunique"),
        mean_groups=("n_groups", "mean"),
        median_groups=("n_groups", "median"),
        mean_memberships=("n_memberships", "mean"),
        mean_coverage=("coverage", "mean"),
        median_coverage=("coverage", "median"),
    ).reset_index()


def coverage_enrichment(bank: GroupBank, df: pd.DataFrame, label: str, split: str):
    """Does the label-free generator actually enrich true binding residues? (Dev only)

    Returns (global audit row DataFrame, per-positive-protein recall DataFrame).
    """
    all_y, all_cov, per = [], [], []
    for r in df.itertuples():
        pid = str(r.prot_id)
        y = np.asarray(getattr(r, label), dtype=np.int8)
        er = bank[(split, pid)][0]
        cov = np.zeros(len(y), dtype=bool)
        if len(er):
            cov[np.unique(er)] = True
        all_y.append(y)
        all_cov.append(cov)
        if y.sum() > 0:
            per.append({"prot_id": pid, "n_positive": int(y.sum()),
                        "positive_recall": float(cov[y == 1].mean()), "overall_coverage": float(cov.mean())})
    y, cov = np.concatenate(all_y), np.concatenate(all_cov)
    prev, pos_cov, neg_cov = float(y.mean()), float(cov[y == 1].mean()), float(cov[y == 0].mean())
    cov_prev = float(y[cov].mean())
    row = {
        "dataset": split,
        "global_prevalence": prev,
        "overall_residue_coverage": float(cov.mean()),
        "positive_residue_coverage": pos_cov,
        "negative_residue_coverage": neg_cov,
        "P_positive_given_covered": cov_prev,
        "covered_enrichment_vs_global": cov_prev / max(prev, 1e-12),
        "coverage_odds_ratio_pos_vs_neg": (pos_cov / max(1 - pos_cov, 1e-12)) / (neg_cov / max(1 - neg_cov, 1e-12)),
    }
    return pd.DataFrame([row]), pd.DataFrame(per)
