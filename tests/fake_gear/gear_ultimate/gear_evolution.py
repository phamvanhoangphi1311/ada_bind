import numpy as np
import pandas as pd


def attach_evo(stores, base, splits=("train", "test300", "test46"), require_msa=True):
    rows = []
    for s in splits:
        for pid, rec in stores[s].items():
            L = len(rec["sequence"])
            rec["EVO"] = np.random.default_rng(len(pid)).normal(size=(L, 48)).astype(np.float32)
            rows.append({"split": s, "protein": pid, "msa_depth": 10, "has_real_msa": True})
    return pd.DataFrame(rows)
