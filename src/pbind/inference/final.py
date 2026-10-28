"""The final models in ONE place: ``<BASE>/final_model/`` holds the only physical copy of their weights.

    final_model/
      manifest.json                       profiles (final / lite), members, weights, thresholds, sha1 of every file
      metal/mlp_esmif/                    MLP on ESM+ATOM+CHEM+MULTI2+ESM-IF, 5 folds            final
      metal/mlp/                          MLP on ESM+ATOM+CHEM+MULTI2, 5 folds                    lite
      nuclear|small/v1/, ms/              original V1 / MS hypergraphs, fold{1..5}.pt + _val.npy + .json
      nuclear|small/{v1,ms}_esmif_prott5_s{0,1,2}/   V1 / MS hypergraph + ESMIF_EF + PROTT5, 3 seeds

Profiles (residue probability = sum_i w_i * mean over the term's seeds of the mean over its 5 folds, terms in order):

    final  metal   1.0 mlp_esmif                                                       80 checkpoints in total
           nuclear 0.2 v1 + 0.4 v1_esmif_prott5[s0-2] + 0.4 ms_esmif_prott5[s0-2]       (ms has weight 0 -> dropped)
           small   0.1 v1 + 0.1 ms + 0.4 v1_esmif_prott5[s0-2] + 0.4 ms_esmif_prott5[s0-2]
    lite   metal 1.0 mlp | nuclear 0.7 v1 + 0.3 ms | small 0.6 v1 + 0.4 ms                    25 checkpoints

Sources: metal/mlp = Colab T4 rerun of GEAR_METAL_ONLY_ULTIMATE_V2 (bit-identical to the original run); metal/mlp_esmif =
notebook 08 (Dev-only lock + final external run); nuclear|small v1/ms = original Drive checkpoints; *_esmif_prott5_* =
notebooks 09 / 10 (V3). Dev / test numbers below were recomputed from the saved predictions of exactly these files.

The pipelines keep reading their usual folders; ``install_final_models`` points those at ``final_model/`` with symlinks
(copies where links are impossible). Every pipeline write replaces the link itself (atomic ``os.replace``).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..paths import Paths
from ..tasks import get_task
from ..tasks import metal as M
from ..utils.cache import save_json

N_FOLDS = 5
SEEDS = (0, 1, 2)
GROUP_FILES = tuple(f"fold{k}{s}" for k in range(1, N_FOLDS + 1) for s in (".pt", "_val.npy", ".json"))
METAL_ESMIF_RUN = "metal_v4_06_final_mlp_esmif_esmfold"
METAL_ESMIF_REP = "ESM+Atomic+Chem+MULTI2+ESMIF"
NEW_MODS = ("ESMIF_EF", "PROTT5")
GROUP = {  # task -> (V1 arch, MS arch, V3 run folder, original modalities)
    "nuclear": ("NA_PatchHG_Attention", "NA_PatchHG_Attention_MS", "NUCLEAR_V3_IMPROVE",
                {"ESM": ["ESM"], "ELECTRO": ["ACCESS"], "PATCH": ["NA_PATCH3D40", "NA_TOPO12"]}),
    "small": ("SM_PocketHG_Attention", "SM_PocketHG_Attention_MS", "SMALL_V3_IMPROVE",
              {"ESM": ["ESM"], "POCKET": ["ACCESS", "SM_POCKET3D44", "SM_TOPO16"]}),
}


@dataclass(frozen=True)
class FinalMember:
    task: str  # metal | nuclear | small
    name: str  # folder under final_model/<task>/
    model: str  # architecture
    modalities: tuple  # metal: input blocks in order; hypergraphs: (modality, blocks) pairs (encoder tokens = sorted names)
    files: tuple  # file names expected in the folder
    pipeline_dir: tuple  # folder (relative to BASE) where the pipelines look for these files


def _seeds(prefix):
    return tuple(f"{prefix}_s{s}" for s in SEEDS)


PROFILES = {  # task -> (terms [(folders averaged together, weight)], Dev-OOF threshold, threshold rule)
    "final": {
        "metal": ([(("mlp_esmif",), 1.0)], 0.9148463010787964, "dev_oof_f1_opt"),
        "nuclear": ([(("v1",), 0.2), (_seeds("v1_esmif_prott5"), 0.4), (_seeds("ms_esmif_prott5"), 0.4)],
                    0.7078787878787879, "dev_oof_mcc_grid199"),
        "small": ([(("v1",), 0.1), (("ms",), 0.1), (_seeds("v1_esmif_prott5"), 0.4), (_seeds("ms_esmif_prott5"), 0.4)],
                  0.4901010101010101, "dev_oof_mcc_grid199"),
    },
    "lite": {
        "metal": ([(("mlp",), 1.0)], 0.9389004707336426, "dev_oof_f1_opt"),
        "nuclear": ([(("v1",), 0.7000000000000001), (("ms",), 0.29999999999999993)], 0.7177777777777777,
                    "dev_oof_mcc_grid199"),
        "small": ([(("v1",), 0.6000000000000001), (("ms",), 0.3999999999999999)], 0.5148484848484849,
                  "dev_oof_mcc_grid199"),
    },
}

REPORTED = {  # AUPRC of each profile, recomputed from the saved predictions of these exact files
    "final": {"metal": {"Dev_fold_mean": 0.469899, "TestSet300": 0.385955, "TestSetNew46": 0.681657, "pooled_test": 0.4085},
              "nuclear": {"Dev_fold_mean": 0.5581, "Dev_crossfit": 0.5571, "TestSet300": 0.4799, "TestSetNew46": 0.1182,
                          "pooled_test": 0.3757},
              "small": {"Dev_fold_mean": 0.4865, "Dev_crossfit": 0.4852, "TestSet300": 0.5041, "TestSetNew46": 0.4716,
                        "pooled_test": 0.5010}},
    "lite": {"metal": {"Dev_fold_mean": 0.458725, "TestSet300": 0.381428, "TestSetNew46": 0.685333, "pooled_test": 0.4049},
             "nuclear": {"Dev_fold_mean": 0.508961, "TestSet300": 0.432839, "TestSetNew46": 0.124501, "pooled_test": 0.3485},
             "small": {"Dev_fold_mean": 0.409283, "TestSet300": 0.446015, "TestSetNew46": 0.427383, "pooled_test": 0.4439}},
}


def final_members() -> list[FinalMember]:
    blocks = M.FINAL_LOCK["locked_blocks"]
    model = M.FINAL_LOCK["locked_model"]
    metal_files = lambda rep: tuple(f"{model}__{rep.replace('+', '__')}__fold{k}.pt" for k in range(1, N_FOLDS + 1))
    out = [FinalMember("metal", "mlp", model, tuple(blocks), metal_files(M.FINAL_LOCK["locked_representation"]),
                       ("trained_models", M.RUN_VERSIONS["final"], M.LABEL)),
           FinalMember("metal", "mlp_esmif", model, (*blocks, "ESMIF"), metal_files(METAL_ESMIF_REP),
                       ("trained_models", METAL_ESMIF_RUN, M.LABEL))]
    for task, (a_v1, a_ms, v3_root, mods) in GROUP.items():
        t = get_task(task)
        base = tuple((m, tuple(b)) for m, b in mods.items())
        new = base + tuple((m, (m,)) for m in NEW_MODS)
        out.append(FinalMember(task, "v1", a_v1, base, GROUP_FILES, ("results", t.run_root, t.group_state_dir, t.v1_hg)))
        out.append(FinalMember(task, "ms", a_ms, base, GROUP_FILES, ("results", t.ms_run_root, "states", t.ms_hg)))
        for s in SEEDS:
            for kind, arch in (("v1", a_v1), ("ms", a_ms)):
                out.append(FinalMember(task, f"{kind}_esmif_prott5_s{s}", arch, new, GROUP_FILES,
                                       ("results", v3_root, "states", f"{kind.upper()}+ESMIF_EF+PROTT5_s{s}", arch)))
    return out


def profile_members(profile: str, task: str) -> list[str]:
    return [name for folders, w in PROFILES[profile][task][0] if w for name in folders]


def member_dir(paths: Paths, m: FinalMember) -> Path:
    return paths.final_model / m.task / m.name


def _link(src: Path, dst: Path) -> str:
    """Point ``dst`` at ``src``; an existing different regular file is never overwritten."""
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return "linked"
        dst.unlink()
    elif dst.exists():
        same = dst.stat().st_size == src.stat().st_size and dst.read_bytes() == src.read_bytes()
        return "identical-file" if same else "CONFLICT (kept existing file)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
        return "linked"
    except OSError:  # e.g. a filesystem without symlinks
        shutil.copy2(src, dst)
        return "copied"


def install_final_models(paths: Paths, tasks=None, verbose: bool = True) -> pd.DataFrame:
    """Make the pipelines use final_model/ (no training). Members whose folder is incomplete are skipped."""
    rows = []
    for m in final_members():
        if tasks is not None and m.task not in tasks:
            continue
        src_dir = member_dir(paths, m)
        missing = [f for f in m.files if not (src_dir / f).exists()]
        if missing:
            rows.append({"task": m.task, "member": m.name, "status": f"skipped: {len(missing)} files missing"})
            continue
        dst_dir = paths.base.joinpath(*m.pipeline_dir)
        status = [_link(src_dir / f, dst_dir / f) for f in m.files]
        rows.append({"task": m.task, "member": m.name, "model": m.model,
                     "status": ", ".join(sorted(set(status))), "pipeline_dir": str(dst_dir)})
    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(out[["task", "member", "status"]].to_string(index=False))
    return out


def write_manifest(paths: Paths) -> dict:
    """final_model/manifest.json: profiles, members, weights, thresholds, sources, file checksums, reported numbers."""
    members = {}
    for m in final_members():
        d = member_dir(paths, m)
        files = {f: hashlib.sha1((d / f).read_bytes()).hexdigest()[:16] for f in m.files if (d / f).exists()}
        members[f"{m.task}/{m.name}"] = {
            "model": m.model, "complete": len(files) == len(m.files),
            "inputs" if m.task == "metal" else "modalities": (list(m.modalities) if m.task == "metal"
                                                               else {k: list(v) for k, v in m.modalities}),
            "n_checkpoints": sum(f.endswith(".pt") for f in files), "sha1_16": files}
    profiles = {}
    for prof, tasks in PROFILES.items():
        profiles[prof] = {}
        for task, (terms, thr, rule) in tasks.items():
            used = [n for folders, w in terms if w for n in folders]
            profiles[prof][task] = {
                "terms": [{"members": [f"{task}/{n}" for n in folders], "weight": w} for folders, w in terms],
                "threshold": thr, "threshold_rule": rule,
                "n_checkpoints": sum(members[f"{task}/{n}"]["n_checkpoints"] for n in used),
                "reported_AUPRC": REPORTED[prof][task]}
        profiles[prof]["n_checkpoints_total"] = sum(v["n_checkpoints"] for v in profiles[prof].values())
    manifest = {
        "description": "Residue-level binding-site models (metal / nuclear / small), 5 official DevSet1010 folds each; "
                       "external sets TestSet300 / TestSetNew46. Use profile 'final' (best) or 'lite' (no ESM-IF/ProtT5).",
        "combine": "p(residue) = sum_i weight_i * mean over the term's members (seeds) of the mean over their 5 fold "
                   "checkpoints; terms in the listed order; binary call: p >= threshold",
        "foundation_models": {"final": ["ESMFold v1", "ESM-2 650M", "ESM-IF1", "ProtT5-XL half (nuclear, small)"],
                              "lite": ["ESMFold v1", "ESM-2 650M"],
                              "how_to": "docs/FOUNDATION_MODELS.md (exact settings, pinned files, sha256)"},
        "dtype_notes": {"ESM": "stored float16, fed as float32",
                        "PROTT5": "stored float16, fed as float32",
                        "ESMIF (metal)": "float32 as produced by ESM-IF",
                        "ESMIF_EF (nuclear, small)": "rounded through float16, fed as float32"},
        "hypergraph_notes": {"encoder": "one token per modality, tokens in sorted modality-name order; mode=attention, latent=192",
                             "v1_bank": {"nuclear": "surface patches r=10 A, <=128/protein (18-d static)",
                                         "small": "pockets r=8 A around WFYHCM seeds, <=160/protein (22-d static)"},
                             "ms_bank": {"nuclear": "selective multi-scale patches r=6/8/10/12 A, <=48, NMS 0.65 (24-d)",
                                         "small": "multi-scale pockets r=6/8/10/12 A, <=64, NMS 0.65 (23-d)"}},
        "sources": {"metal/mlp": "Colab T4 rerun of GEAR_METAL_ONLY_ULTIMATE_V2 (save_models=True); 15/15 predictions "
                                 "bit-identical to the original run",
                    "metal/mlp_esmif": f"notebook 08: Dev-only lock + final external run ({METAL_ESMIF_RUN})",
                    "nuclear|small/v1, ms": "original Drive checkpoints of the V1 / MS runs",
                    "nuclear|small/*_esmif_prott5_s*": "notebooks 09 / 10 (V3), results/<TASK>_V3_IMPROVE",
                    "selection": "DevSet1010 only (cross-fitted for the ensembles); tests only confirm; bootstrap B=10000"},
        "profiles": profiles,
        "members": members,
    }
    save_json(paths.final_model / "manifest.json", manifest)
    return manifest
