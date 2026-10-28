"""Where everything lives.

All inputs, caches and results are addressed relative to ONE project root
(``BASE``), which by default is the Google Drive folder used by the original
notebooks. The Drive layout is kept byte-for-byte so existing caches
(ESM, structures, NA_*/SM_* features, patch/pocket banks, fold states) are reused.

Resolution order for BASE:
  1. explicit ``base=`` argument
  2. environment variable ``PBIND_BASE``
  3. ``<repo>/data`` (local copy downloaded from Drive) if it contains the training CSV
  4. the first default Drive folder that contains the training CSV

A local mirror (e.g. downloaded from Drive) works the same way: point
``PBIND_BASE`` at a folder with the same sub-layout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DRIVE_MOUNT = Path("/content/drive")
REPO_DATA = Path(__file__).resolve().parents[2] / "data"  # <repo>/data, same layout as the Drive root
DEFAULT_BASES = (
    Path("/content/drive/MyDrive/Protein-binding-Phi"),
    Path("/content/drive/MyDrive/Protein-binding"),
)

TRAIN_CSV_REL = "data/development_set/full_grouped_train_binding_sites_df.csv"
TEST300_CSV_REL = "data/development_set/full_grouped_test_binding_sites_df.csv"
TEST46_CSV_REL = "data/independent_set/grouped_test_46_new_binding_sites.csv"


@dataclass(frozen=True)
class Paths:
    base: Path
    local: Path  # fast scratch disk (Colab: /content). Never the source of truth.

    # ---------------- inputs ----------------
    @property
    def train_csv(self) -> Path:
        return self.base / TRAIN_CSV_REL

    @property
    def test300_csv(self) -> Path:
        return self.base / TEST300_CSV_REL

    @property
    def test46_csv(self) -> Path:
        return self.base / TEST46_CSV_REL

    @property
    def esmfold_pdb(self) -> Path:
        return self.base / "esmFold_pdb_files"

    @property
    def experimental_pdb(self) -> Path:
        return self.base / "experimental_pdb"

    @property
    def baseline_work(self) -> Path:
        return self.base / "baseline_reproduction_master_v6"

    @property
    def baseline_results(self) -> Path:
        return self.baseline_work / "results"

    @property
    def gear_dirs(self) -> dict[str, Path]:
        return {
            "GEAR_ULTIMATE_OTHER_BINDINGS": self.base / "GEAR_ULTIMATE_OTHER_BINDINGS",
            "GEAR_ULTIMATE": self.base / "GEAR_ULTIMATE",
        }

    # ---------------- outputs ----------------
    @property
    def cache(self) -> Path:
        return self.base / "cache"

    @property
    def final_model(self) -> Path:
        """The three final models (the only physical copy of their weights); see ``pbind.inference.final``."""
        return self.base / "final_model"

    @property
    def feature_store(self) -> Path:
        """<BASE>/feature_store, or $PBIND_FEATURE_STORE (e.g. /content/feature_store: fast local disk on Colab)."""
        return Path(os.environ.get("PBIND_FEATURE_STORE") or self.base / "feature_store")

    @property
    def results(self) -> Path:
        return self.base / "results"

    def cache_dir(self, *parts: str, create: bool = True) -> Path:
        p = self.cache.joinpath(*parts)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def results_dir(self, *parts: str, create: bool = True) -> Path:
        p = self.results.joinpath(*parts)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def local_dir(self, *parts: str, create: bool = True) -> Path:
        p = self.local.joinpath(*parts)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    def siblings(self) -> list[Path]:
        """Other default project roots that exist (the notebooks looked there for repos/backends)."""
        return [b for b in DEFAULT_BASES if b != self.base and b.exists()]


def mount_drive(force_remount: bool = False) -> bool:
    """Mount Google Drive when running in Colab. Returns False outside Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    drive.mount(str(DRIVE_MOUNT), force_remount=force_remount)
    return True


def resolve_paths(base: str | Path | None = None, local: str | Path | None = None) -> Paths:
    if base is None:
        base = os.environ.get("PBIND_BASE")
    if base is None:
        base = next((b for b in (REPO_DATA, *DEFAULT_BASES) if (b / TRAIN_CSV_REL).exists()), None)
    if base is None:
        raise FileNotFoundError(
            f"Project root not found. Put the Drive files under {REPO_DATA}, mount Drive "
            "(pbind.paths.mount_drive()), or set PBIND_BASE / pass base= to a folder containing " + TRAIN_CSV_REL
        )
    base = Path(base)

    if local is None:
        local = os.environ.get("PBIND_LOCAL")
    if local is None:
        local = Path("/content") if Path("/content").exists() else base / "_local"
    local = Path(local)
    local.mkdir(parents=True, exist_ok=True)
    return Paths(base=base, local=local)


def check_inputs(paths: Paths) -> pd.DataFrame:
    """Inventory of the inputs the pipelines expect, with existence flags."""
    items = {
        "train_csv": paths.train_csv,
        "test300_csv": paths.test300_csv,
        "test46_csv": paths.test46_csv,
        "esmFold_pdb_files": paths.esmfold_pdb,
        "experimental_pdb": paths.experimental_pdb,
        "baseline_reproduction_master_v6": paths.baseline_work,
        "cache": paths.cache,
        "shared_exact_pdb_archive": paths.cache / "shared_exact_pdb_v1.tar.gz",
        **{f"backend:{k}": v for k, v in paths.gear_dirs.items()},
    }
    return pd.DataFrame(
        [{"item": k, "path": str(v), "exists": v.exists()} for k, v in items.items()]
    )
