import numpy as np
import pandas as pd


def dimension_table(store):
    rec = next(iter(store.values()))
    return pd.DataFrame([{"block": k, "dim": v.shape[1]} for k, v in rec.items()
                         if isinstance(v, np.ndarray) and v.ndim == 2]).sort_values("block")


def validate_no_direct_label_features(stores, blocks):
    return pd.DataFrame([{"split": s, "block": b, "missing_proteins": sum(b not in r for r in st.values())}
                         for s, st in stores.items() for b in blocks])
