"""final_model/ is the only physical copy of the final weights: pipelines see it through symlinks and can never
write through them.

    pytest -q tests/test_final_models.py
"""
from __future__ import annotations

import os

from pbind.inference import final_members, install_final_models, write_manifest
from pbind.inference.final import PROFILES, profile_members
from pbind.paths import Paths


def _paths(tmp_path) -> Paths:
    return Paths(base=tmp_path, local=tmp_path / "_local")


def _fill(paths, member, payload=b"final"):
    d = paths.final_model / member.task / member.name
    d.mkdir(parents=True, exist_ok=True)
    for f in member.files:
        (d / f).write_bytes(payload + f.encode())
    return d


def test_members_cover_three_tasks():
    ms = final_members()
    names = {t: [m.name for m in ms if m.task == t] for t in ("metal", "nuclear", "small")}
    assert names["metal"] == ["mlp", "mlp_esmif"]
    for task in ("nuclear", "small"):
        assert names[task][:2] == ["v1", "ms"] and len(names[task]) == 8
    for prof, tasks in PROFILES.items():
        for task, (terms, thr, rule) in tasks.items():
            assert abs(sum(w for _, w in terms) - 1.0) < 1e-9 and 0 < thr < 1
            assert set(n for folders, _ in terms for n in folders) <= set(names[task])
    n_ckpt = {prof: sum(len(profile_members(prof, t)) * 5 for t in ("metal", "nuclear", "small")) for prof in PROFILES}
    assert n_ckpt == {"final": 80, "lite": 25}


def test_install_links_and_never_writes_through(tmp_path):
    paths = _paths(tmp_path)
    metal = final_members()[0]
    src = _fill(paths, metal)

    out = install_final_models(paths, verbose=False)
    st = dict(zip(out.member + "/" + out.task, out.status))
    assert st["mlp/metal"] == "linked"
    assert all(s.startswith("skipped") for k, s in st.items() if k != "mlp/metal")  # nuclear/small folders absent

    dst = paths.base.joinpath(*metal.pipeline_dir)
    f = metal.files[0]
    assert (dst / f).is_symlink() and (dst / f).resolve() == (src / f).resolve()
    assert install_final_models(paths, ["metal"], verbose=False).status.iloc[0] == "linked"  # idempotent

    # a pipeline write (tmp + os.replace, as the trainers do) replaces the link, not the final weights
    tmp = dst / f".tmp.{f}"
    tmp.write_bytes(b"retrained")
    os.replace(tmp, dst / f)
    assert (src / f).read_bytes() == b"final" + f.encode()
    assert not (dst / f).is_symlink()

    # ... and installing again keeps that file instead of overwriting it
    st = install_final_models(paths, ["metal"], verbose=False).status.iloc[0]
    assert "CONFLICT" in st and (dst / f).read_bytes() == b"retrained"


def test_manifest_flags_incomplete_members(tmp_path):
    paths = _paths(tmp_path)
    ms = {(m.task, m.name): m for m in final_members()}
    metal, nuc_v1 = ms["metal", "mlp"], ms["nuclear", "v1"]
    _fill(paths, metal)
    (paths.final_model / "nuclear" / "v1").mkdir(parents=True)
    (paths.final_model / "nuclear" / "v1" / nuc_v1.files[0]).write_bytes(b"x")

    man = write_manifest(paths)
    done = {k: v["complete"] for k, v in man["members"].items()}
    assert done["metal/mlp"] and not done["nuclear/v1"] and not done["small/ms"]
    assert (paths.final_model / "manifest.json").exists()
    assert man["profiles"]["lite"]["metal"]["n_checkpoints"] == 5 and man["profiles"]["final"]["nuclear"]["n_checkpoints"] == 1
    assert [t["weight"] for t in man["profiles"]["final"]["small"]["terms"]] == [0.1, 0.1, 0.4, 0.4]
