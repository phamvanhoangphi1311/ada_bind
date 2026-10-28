"""Cache helpers. Every expensive step in pbind goes through one of these.

Rules (same as the original notebooks, made explicit):
  * a cached artifact is reused only if it loads AND passes validation
    (shapes, lengths, finiteness, meta keys); otherwise it is rebuilt;
  * ``force=True`` always rebuilds;
  * writes are atomic (tmp file + os.replace) so an interrupted Colab
    session never leaves a half-written cache that later looks valid.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ atomic I/O
def _tmp_for(fp: Path) -> Path:
    # keep the real suffix last so numpy/pandas infer the format correctly
    return fp.with_name(f".{fp.stem}.tmp{''.join(fp.suffixes) or ''}")


def atomic_write(fp: str | Path, writer: Callable[[Path], None]) -> Path:
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(fp)
    writer(tmp)
    os.replace(tmp, fp)
    return fp


def save_npz(fp, **arrays) -> Path:
    return atomic_write(fp, lambda t: np.savez_compressed(t, **arrays))


def save_npy(fp, arr) -> Path:
    return atomic_write(fp, lambda t: np.save(t, arr))


def save_json(fp, obj) -> Path:
    return atomic_write(fp, lambda t: t.write_text(json.dumps(obj, indent=2, default=_json_default)))


def save_table(fp, df: pd.DataFrame) -> Path:
    fp = Path(fp)
    comp = "gzip" if fp.suffix == ".gz" else None
    return atomic_write(fp, lambda t: df.to_csv(t, index=False, compression=comp))


def save_pickle(fp, obj) -> Path:
    def _w(t):
        with open(t, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return atomic_write(fp, _w)


def load_json(fp, default=None):
    fp = Path(fp)
    if not fp.exists():
        return default
    try:
        return json.loads(fp.read_text())
    except Exception:
        return default


def load_pickle_dict(fp) -> dict | None:
    """Pickled ``{prot_id: ...}`` cache (keys forced to str); None if missing/unreadable."""
    fp = Path(fp)
    if not fp.exists():
        return None
    try:
        with open(fp, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"Cache load failed ({fp.name}): {e!r}")
        return None
    return {str(k): v for k, v in obj.items()} if isinstance(obj, dict) else obj


def read_table(fp) -> pd.DataFrame:
    """CSV reader that tolerates `.gz` files that are actually plain text."""
    fp = Path(fp)
    if fp.suffix == ".gz":
        with open(fp, "rb") as f:
            if f.read(2) != b"\x1f\x8b":
                return pd.read_csv(fp, compression=None)
    return pd.read_csv(fp)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------ validated load
def load_npz_valid(fp, shapes: Mapping[str, tuple] | None = None,
                   check: Callable[[dict], bool] | None = None) -> dict | None:
    """Load an .npz; return None if missing, unreadable or invalid."""
    fp = Path(fp)
    if not fp.exists():
        return None
    try:
        with np.load(fp, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
    except Exception:
        return None
    if shapes is not None:
        for k, shp in shapes.items():
            if k not in data or tuple(data[k].shape) != tuple(shp):
                return None
    if check is not None:
        try:
            if not check(data):
                return None
        except Exception:
            return None
    return data


def load_npy_valid(fp, shape: tuple | None = None, finite: bool = False) -> np.ndarray | None:
    fp = Path(fp)
    if not fp.exists():
        return None
    try:
        a = np.load(fp, allow_pickle=False)
    except Exception:
        return None
    if shape is not None and tuple(a.shape) != tuple(shape):
        return None
    if finite and not np.all(np.isfinite(a)):
        return None
    return a


# ------------------------------------------------------------------ cached builders
def cached_npz(fp, build: Callable[[], Mapping[str, np.ndarray]],
               shapes: Mapping[str, tuple] | None = None,
               check: Callable[[dict], bool] | None = None,
               force: bool = False) -> tuple[dict, bool]:
    """Return (arrays, rebuilt?)."""
    if not force:
        data = load_npz_valid(fp, shapes, check)
        if data is not None:
            return data, False
    data = {k: np.asarray(v) for k, v in build().items()}
    save_npz(fp, **data)
    return data, True


def cached_npy(fp, build: Callable[[], np.ndarray], shape: tuple | None = None,
               force: bool = False, dtype=np.float32) -> tuple[np.ndarray, bool]:
    if not force:
        a = load_npy_valid(fp, shape)
        if a is not None:
            return a, False
    a = np.asarray(build(), dtype=dtype)
    save_npy(fp, a)
    return a, True


def cached_table(fp, build: Callable[[], pd.DataFrame], force: bool = False) -> pd.DataFrame:
    fp = Path(fp)
    if fp.exists() and not force:
        try:
            return read_table(fp)
        except Exception:
            pass
    df = build()
    save_table(fp, df)
    return df


def cached_json(fp, build: Callable[[], Any], force: bool = False):
    fp = Path(fp)
    if not force:
        obj = load_json(fp)
        if obj is not None:
            return obj
    obj = build()
    save_json(fp, obj)
    return obj


# ------------------------------------------------------------------ fold artifacts
class FoldArtifacts:
    """``fold{k}.pt`` + ``fold{k}_val.npy`` + ``fold{k}.json`` (layout of the original notebooks).

    A fold is reused only if all three files exist, the meta matches ``expect``
    on every given key, and the validation prediction has the expected length
    and is finite.
    """

    def __init__(self, root: str | Path, fold: int):
        self.root = Path(root)
        self.fold = int(fold)
        self.state_fp = self.root / f"fold{self.fold}.pt"
        self.pred_fp = self.root / f"fold{self.fold}_val.npy"
        self.meta_fp = self.root / f"fold{self.fold}.json"

    def load(self, expect: Mapping[str, Any], n_val: int):
        """Return (state_dict, val_pred) or None."""
        import torch

        if not (self.state_fp.exists() and self.pred_fp.exists() and self.meta_fp.exists()):
            return None
        meta = load_json(self.meta_fp)
        if not isinstance(meta, dict):
            return None
        if any(meta.get(k) != v for k, v in expect.items()):
            return None
        pred = load_npy_valid(self.pred_fp, (n_val,), finite=True)
        if pred is None:
            return None
        try:
            state = torch.load(self.state_fp, map_location="cpu")
        except Exception:
            return None
        return state, pred.astype(np.float32)

    def save(self, state, pred: np.ndarray, meta: Mapping[str, Any]) -> None:
        import torch

        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write(self.state_fp, lambda t: torch.save(state, t))
        save_npy(self.pred_fp, np.asarray(pred, dtype=np.float32))
        save_json(self.meta_fp, dict(meta))

    def load_val_pred(self, n_val: int) -> np.ndarray | None:
        """Prediction only (used to rebuild OOF without loading the model)."""
        return load_npy_valid(self.pred_fp, (n_val,), finite=True)


def states_fingerprint(state_dir: str | Path) -> str:
    """Identity of a set of fold checkpoints (name/size/mtime of fold*.pt).

    Used as the tag of cached external predictions so they are invalidated when
    the fold models are retrained.
    """
    import hashlib

    h = hashlib.sha1()
    for fp in sorted(Path(state_dir).glob("fold*.pt")):
        st = fp.stat()
        h.update(f"{fp.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


class PredictionCache:
    """Per-(method, split) probability vectors: ``<root>/<method>__<split>.npy``.

    Optional ``tag`` (e.g. ``states_fingerprint``) is stored in a sidecar json and must match.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, method: str, split: str) -> Path:
        return self.root / f"{method}__{split}.npy"

    def get(self, method: str, split: str, n: int, tag: str | None = None) -> np.ndarray | None:
        if tag is not None:
            meta = load_json(self.path(method, split).with_suffix(".json"), {})
            if meta.get("tag") != tag:
                return None
        a = load_npy_valid(self.path(method, split), (n,), finite=True)
        return None if a is None else a.astype(np.float64)

    def put(self, method: str, split: str, p, tag: str | None = None) -> None:
        save_npy(self.path(method, split), np.asarray(p, dtype=np.float32))
        if tag is not None:
            save_json(self.path(method, split).with_suffix(".json"), {"tag": tag})

    def get_or_compute(self, method: str, split: str, n: int, fn: Callable[[], np.ndarray],
                       force: bool = False, tag: str | None = None) -> np.ndarray:
        if not force:
            p = self.get(method, split, n, tag)
            if p is not None:
                return p
        p = np.asarray(fn(), dtype=np.float64)
        if p.shape != (n,):
            raise RuntimeError(f"{method}/{split}: prediction shape {p.shape} != {(n,)}")
        self.put(method, split, p, tag)
        return p
