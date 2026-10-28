#!/usr/bin/env python
"""Bundle for the application team: pbind source, verify script, golden set, manifest, docs -> one zip.

    python scripts/make_app_bundle.py            # -> <BASE>/delivery/pbind_app_bundle_<date>.zip
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pbind.paths import resolve_paths  # noqa: E402

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")


def main():
    P = resolve_paths()
    name = f"pbind_app_bundle_{dt.date.today():%Y%m%d}"
    out = P.base / "delivery" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "pbind_src").mkdir(parents=True)
    shutil.copytree(REPO / "src" / "pbind", out / "pbind_src" / "src" / "pbind", ignore=IGNORE)
    shutil.copy2(REPO / "pyproject.toml", out / "pbind_src" / "pyproject.toml")
    (out / "pbind_src" / "README.md").write_text("pbind: residue-level binding-site prediction. See ../README.md.\n")
    shutil.copy2(REPO / "requirements-app.txt", out / "requirements-app.txt")
    for f in ("verify_installation.py", "esmif_embed.py"):
        shutil.copy2(REPO / "scripts" / f, out / f)
    shutil.copytree(P.base / "golden_set", out / "golden_set")
    shutil.copy2(P.final_model / "manifest.json", out / "manifest.json")
    (out / "docs").mkdir()
    for f in ("APP_INTEGRATION.md", "FOUNDATION_MODELS.md"):
        shutil.copy2(REPO / "docs" / f, out / "docs" / f)
    shutil.copy2(REPO / "docs" / "APP_BUNDLE_README.md", out / "README.md")
    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(out.rglob("*")):
            if fp.is_file():
                z.write(fp, Path(name) / fp.relative_to(out))
    n = sum(1 for fp in out.rglob("*") if fp.is_file())
    print(f"{zp}  ({n} files, {zp.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
