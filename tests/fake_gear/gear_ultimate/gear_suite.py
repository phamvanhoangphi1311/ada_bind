from __future__ import annotations

import ast
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from . import gear_core as core
from .gear_core import Config

DIMS = {"AA": 20, "ESM": 16, "SPATIAL": 6, "LOCAL": 4, "ATOM": 5, "CHEM": 5, "ACCESS": 10, "SHELL2": 3, "ENV": 3,
        "CONF": 2, "METALSPEC": 4, "ELEC3D": 8, "TOPO": 12, "SOLV": 4, "SEQMOTIF": 4, "SEQCTX": 6}
MULTI2 = ["METALSPEC", "ELEC3D", "TOPO", "SOLV", "SEQMOTIF"]
STANDARDIZED_13 = MECHANISTIC_GRID = FINAL_COMPARISON_REPS = NO_ESM_GRID = EVO_GRID = {"ESM": ["ESM"]}
MECHANISTIC_COMPARISONS = []


@dataclass
class Context:
    cfg: Config
    paths: dict
    frames: dict
    fold_ids: list
    stores: dict
    device: object


def _frame(fp):
    df = pd.read_csv(fp)
    df["prot_id"] = df.prot_id.astype(str)
    return df


def prepare_context(base, run_version="x", models=("MLP",), label_column="metal_binding", ensure_base=True,
                    add_mech=True, add_multi=True, add_seqctx=True, esm_batch_size=8, device=None):
    base = Path(base)
    fold_dir = base / "repos" / "bindnode24" / "dataset" / "cv_splits" / "predicted_structures"
    folds = [[x for x in (fold_dir / f"ids_split{k}.txt").read_text().split() if x] for k in range(1, 6)]
    official = set().union(*map(set, folds))
    frames = {"train": _frame(base / "data/development_set/full_grouped_train_binding_sites_df.csv"),
              "test300": _frame(base / "data/development_set/full_grouped_test_binding_sites_df.csv"),
              "test46": _frame(base / "data/independent_set/grouped_test_46_new_binding_sites.csv")}
    frames["train"] = frames["train"][frames["train"].prot_id.isin(official)].reset_index(drop=True)
    stores = {}
    for key, df in frames.items():
        st = {}
        for r in df.itertuples():
            L = len(r.sequence)
            rng = np.random.default_rng(abs(hash(r.prot_id)) % 2**32)
            y = np.asarray(ast.literal_eval(getattr(r, label_column)), dtype=np.int8)
            rec = {"sequence": r.sequence, "y": y}
            for b, d in DIMS.items():
                x = rng.normal(size=(L, d)).astype(np.float32)
                x[:, 0] += 1.5 * y  # weak signal so AUPRC is meaningful
                rec[b] = x
            st[r.prot_id] = rec
        stores[key] = st
    cfg = Config(base=str(base), label_column=label_column, run_version=run_version, models=tuple(models))
    return Context(cfg, core.make_paths(cfg), frames, folds, stores, device)


def audit_context(ctx):
    return pd.DataFrame([{"split": k, "proteins": len(s)} for k, s in ctx.stores.items()])


def clone_context_for_run(ctx, run_version, models=("MLP",), **kw):
    cfg = Config(base=ctx.cfg.base, label_column=ctx.cfg.label_column, run_version=run_version, models=tuple(models))
    return Context(cfg, core.make_paths(cfg), ctx.frames, ctx.fold_ids, ctx.stores, ctx.device)


def _xy(store, pids, blocks):
    X = np.concatenate([np.concatenate([np.asarray(store[p][b]) for b in blocks], axis=1) for p in pids])
    y = np.concatenate([store[p]["y"] for p in pids])
    return X, y


def run_dev_cv_only(ctx, representations, models=("MLP",), run_version="dev"):
    root = Path(ctx.cfg.base) / "results" / run_version / ctx.cfg.label_column
    root.mkdir(parents=True, exist_ok=True)
    store, rows = ctx.stores["train"], []
    for m in models:
        for rep, blocks in representations.items():
            for k, va in enumerate(ctx.fold_ids):
                tr = [p for p in store if p not in set(va)]
                Xt, yt = _xy(store, tr, blocks)
                Xv, yv = _xy(store, va, blocks)
                p = LogisticRegression(max_iter=200).fit(Xt, yt).predict_proba(Xv)[:, 1]
                rows.append({"model": m, "representation": rep, "fold": k + 1,
                             "AUPRC": average_precision_score(yv, p), "AUROC": roc_auc_score(yv, p)})
    f = pd.DataFrame(rows)
    cv = f.groupby(["model", "representation"], as_index=False).agg(
        CV_AUPRC=("AUPRC", "mean"), CV_AUPRC_SD=("AUPRC", "std"), CV_AUROC=("AUROC", "mean"))
    return cv, f


def run_grid(ctx, representations, models=None, comparisons=None):
    """Writes fold_predictions/*.pkl with test300_p / test46_p like the real backend (random probs)."""
    cv, _ = run_dev_cv_only(ctx, representations, models or ctx.cfg.models, ctx.cfg.run_version)
    for m in (models or ctx.cfg.models):
        for rep in representations:
            for k in range(5):
                rng = np.random.default_rng(k)
                z = {key + "_p": rng.random(sum(len(r["y"]) for r in ctx.stores[key].values())).astype(np.float32)
                     for key in ("test300", "test46")}
                with open(core.job_pred_path(ctx.paths, m, rep, k), "wb") as f:
                    pickle.dump(z, f)
    ext = pd.DataFrame([{"model": m, "representation": r, "dataset": d, "AUPRC": 0.5}
                        for m in (models or ctx.cfg.models) for r in representations for d in ("TestSet300", "TestSetNew46")])
    return cv, ext, pd.DataFrame()
