# AdaBind

**Ligand-adaptive prediction of binding residues for metal ions, small molecules, DNA and RNA.**

AdaBind predicts, for every residue of a protein, the probability that it binds a given class of ligand. It adapts
both the residue representation and the network architecture to the ligand class:

- **Metal ions:** a multilayer perceptron on atom-level coordination descriptors (geometry and chemistry of the
  side-chain donor atoms around each residue), combined with ESM-2 and ESM-IF1 embeddings.
- **Small molecules, DNA and RNA:** label-free candidate pockets (small molecules) or surface patches (nucleic acids)
  are proposed from the predicted structure, and a hypergraph neural network passes information from each candidate
  to its member residues. Small-molecule and nucleic-acid models additionally receive ProtT5 embeddings.

All models start from the amino-acid sequence alone: the structure is predicted with ESMFold.

> **Naming.** The project is called AdaBind. The Python package, the import name and the command-line scripts keep the
> original name **`pbind`** (`import pbind`). In the code, the merged DNA/RNA label of the benchmark is called
> `nuclear`; DNA and RNA are additionally available as separate tasks (see [Reproducing the results](#6-reproducing-the-results)).

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data and model weights](#4-data-and-model-weights)
5. [Feature store](#5-feature-store)
6. [Reproducing the results](#6-reproducing-the-results)
7. [Single-protein inference](#7-single-protein-inference)
8. [Extending the code](#8-extending-the-code)
9. [Notes and known limitations](#9-notes-and-known-limitations)
10. [Testing and license](#10-testing-and-license)

---

## 1. Results at a glance

Model selection uses **DevSet1010 only** (the five official bindNode24 folds; primary metric: AUPRC). TestSet300 and
TestSetNew46 are used for confirmation, after the models have been fixed. The numbers below are those stored in
`final_model/manifest.json` and are recomputed from the saved predictions (see [Tier 1](#tier-1-re-evaluate-the-released-checkpoints-no-training)).

**Profile `final`** (needs ESMFold, ESM-2, ESM-IF1 and ProtT5; 80 checkpoints). AUPRC:

| Task | Model | Dev | TestSet300 | TestSetNew46 | Pooled test | Threshold |
|---|---|---|---|---|---|---|
| Metal | MLP on ESM + ATOM + CHEM + MULTI2 + ESM-IF1 (1,889 features), 5 folds | 0.470 | 0.386 | 0.682 | 0.409 | 0.9148 (F1) |
| Nuclear (merged DNA/RNA) | 0.2 V1 + 0.4 V1(+ESM-IF1, ProtT5; 3 seeds) + 0.4 MS(+ESM-IF1, ProtT5; 3 seeds); 35 checkpoints | 0.558 | 0.480 | 0.118 | 0.376 | 0.7079 (MCC) |
| Small molecule | 0.1 V1 + 0.1 MS + 0.4 V1(+...; 3 seeds) + 0.4 MS(+...; 3 seeds); 40 checkpoints | 0.487 | 0.504 | 0.472 | 0.501 | 0.4901 (MCC) |

V1 and MS denote the single-radius and multi-scale hypergraph variants.

**Profile `lite`** (needs only ESMFold and ESM-2; 25 checkpoints). AUPRC on Dev / TestSet300 / TestSetNew46:
metal MLP 0.459 / 0.381 / 0.685; nuclear 0.7 V1 + 0.3 MS 0.509 / 0.433 / 0.125; small 0.6 V1 + 0.4 MS
0.409 / 0.446 / 0.427.

Compared with `lite`, `final` is significantly better on the pooled test sets for nuclear (+0.027, 95% CI
[+0.009, +0.048]) and small molecules (+0.057, [+0.044, +0.070]). For metal ions it is better on Dev but not
significantly so on the test sets (paired protein-level bootstrap, B = 10,000; notebooks 08-10).

---

## 2. Repository layout

```text
AdaBind/
├── src/pbind/                  # Python package
│   ├── api.py                  # one-call API: cross_validate / train_final / predict (features and models by name)
│   ├── paths.py, constants.py  # data root resolution, fixed dataset and task constants
│   ├── data/                   # CSV splits, bindNode24 folds, exact-sequence chain-A structures
│   ├── features/               # @feature_group registry (ESM-2, AA/SPATIAL/LOCAL, ATOM/CHEM, MECH1, MULTI2, NA_*, SM_*, ...)
│   │   └── store.py            # feature store: computed once, one file per block
│   ├── groups/                 # candidate patches (nucleic acids) and pockets (small molecules) for the hypergraphs
│   ├── models/                 # @register_model: residue MLP/KAN, fusion models, hypergraph networks, losses
│   ├── training/               # trainers (MLP/KAN/XGBoost), resume-safe cross-validation
│   ├── evaluation/, reporting/ # metrics, thresholds (selected on Dev OOF only), ensembles, reports
│   ├── tasks/                  # TaskSpec per ligand class (grids, names, directories, original reference numbers)
│   ├── backend/                # native.py (default, port) | gear.py (adapter for the original gear_ultimate package)
│   ├── pipelines/              # each notebook stage as a function (screens, v1, v2, final, metal, crosstask, ...)
│   └── inference/              # final.py (final_model/ layout, profiles, thresholds), predictor.py, metal.py
├── scripts/                    # command-line entry points (see Section 6)
├── notebooks/                  # 00-05: thin notebooks calling the package; 07-15: Colab experiment notebooks
├── tests/                      # pytest suite on synthetic data (no downloads required)
├── docs/                       # foundation-model settings, application integration, app bundle
├── legacy/                     # original notebooks (with outputs) and the original backend sources, for reference
└── data/                       # not tracked; same layout as the data release (Section 4)
```

Correspondence between the original notebooks (`legacy/notebooks/`) and the package:

| Original notebook | Notebook here | Code |
|---|---|---|
| `ALL_BINDINGS_DATASET_ANATOMY_DEV1010_300_46` | `00_dataset_anatomy` | `analysis/anatomy.py` |
| `GEAR_ULTIMATE_METAL_BINDING`, `GEAR_METAL_ONLY_ULTIMATE_V2` | `01_metal`, `05_metal_inference`, `07` | `pipelines/metal.py`, `tasks/metal.py`, `inference/metal.py` |
| `NUCLEAR_ULTIMATE_V1_DATA_DRIVEN_PATCH` | `02_nuclear` | `features/nuclear.py`, `groups/nuclear_patches.py`, `pipelines/{screens,v1,v2,final}.py` |
| `SMALL_ULTIMATE_V1_DATA_DRIVEN_POCKET` | `03_small` | `features/small.py`, `groups/small_pockets.py`, same pipelines |
| `NUCLEAR/SMALL_STANDALONE_*_ADDIN` | `04_crosstask_evo` | `pipelines/crosstask.py`, `features/derived.py` |
| `GEAR_METAL_V4_FULL_INSIGHT_HYPERGRAPH` (never completed) | `08` (fixed) | n/a |
| n/a | `09`, `10` (V3 nuclear / small) | `pipelines/v3`, `final.py` |
| n/a | `12`, `13`, `15` (DNA / RNA lines) | `scripts/build_dna_rna_labels.py` |

---

## 3. Installation

Python 3.10 or newer. A CUDA GPU is recommended for feature computation and training and required for ESMFold.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
# or, without uv:
python -m venv .venv && source .venv/bin/activate && pip install -e ".[all,dev]"
```

| Extra | Adds | Needed for |
|---|---|---|
| `esm` | `transformers` | ESM-2 650M embeddings (weights are downloaded from HuggingFace on first use) |
| `structure` | `mdtraj` | the LOCAL block (DSSP, torsions, solvent accessibility) |
| `xgboost`, `kan` | `xgboost`, `pykan` | optional model families in the metal screens; not needed by the final models |
| `app` | `transformers`, `sentencepiece`, `protobuf` | single-protein inference with `pbind.Predictor` |
| `all` | `esm` + `structure` + `xgboost` | everything except `kan` and `app` |
| `dev` | `pytest` | tests |

`requirements.txt` mirrors `pyproject.toml`; `requirements-app.txt` lists the versions verified for application use.
The `numpy>=2` requirement is strict: the metal checkpoints store `numpy._core` arrays.

**Environment for the manuscript analyses and figures.** The additional analyses of Section 6 (Tier 4) and the
manuscript figures were produced in a conda environment named `AdaBind` (Python 3.11: numpy, pandas, scipy,
scikit-learn, statsmodels, matplotlib, biopython, pymol-open-source). If an analysis needs `torch` or
`transformers`, install them into that same environment.

---

## 4. Data and model weights

The repository contains **code only**. Data, checkpoints and foundation-model weights are distributed separately and
placed next to the code. The data root is resolved in this order: the `base=` argument, the `PBIND_BASE` environment
variable, `<repo>/data/`, then a mounted Google Drive.

### 4.1 Data release

| Archive | Extracts to | Required | Content | Download |
|---|---|---|---|---|
| `pbind_data.zip` | `data/data/` | **yes** | label CSVs for DevSet1010, TestSet300, TestSetNew46; `ids_split1..5.txt` (official folds) | TBD |
| `pbind_esmfold_pdb.zip` | `data/esmFold_pdb_files/` | **yes** | ESMFold v1 structures (711 MB) | TBD |
| `pbind_final_model.zip` | `data/final_model/` | for inference and Tier 1 | 80 checkpoints and `manifest.json` (291 MB) | TBD |
| `pbind_feature_store.zip` | `data/feature_store/` | no | precomputed feature blocks (1.3 GB) | TBD |
| `pbind_cache.zip` | `data/cache/` | no | per-family caches (1.8 GB) | TBD |
| `foundation_models.zip` | `model_weights/` | only for new proteins | ESMFold, ESM-2, ProtT5, ESM-IF1 (about 15 GB) | TBD |

Only the **five required items** (three CSVs, `ids_split1..5.txt` and `esmFold_pdb_files/`) are needed to rebuild
everything from raw inputs; adding `final_model/` enables inference without any training. The feature store and cache
only save recomputation and are rebuilt on demand.

Each archive contains its destination folder at the top level. Unpack all of them with:

```bash
bash scripts/setup_data.sh <folder-with-the-downloaded-archives>        # add --dry-run to preview, --force to overwrite
```

The script checks the top-level folder name of each archive before extracting, skips what already exists, accepts
`--base DIR` for another data root, and runs `scripts/check_setup.py` at the end.

Expected layout:

```text
data/                                      (or any folder pointed to by PBIND_BASE)
├── data/
│   ├── development_set/
│   │   ├── full_grouped_train_binding_sites_df.csv     required   DevSet1010 labels (1,014 rows, 1,010 used)
│   │   ├── full_grouped_test_binding_sites_df.csv      required   TestSet300 labels
│   │   ├── ids_split1..5.txt                           required   official bindNode24 folds
│   │   └── nuclear_dna_rna_labels.csv                  optional   DNA/RNA protein classes from BioLiP2
│   └── independent_set/
│       └── grouped_test_46_new_binding_sites.csv       required   TestSetNew46 labels
├── esmFold_pdb_files/                     required   structures from which all structural features are computed
├── final_model/                           inference  metal/ nuclear/ small/ and manifest.json
├── feature_store/                         optional   rebuilt by scripts/precompute_features.py
├── cache/                                 optional   rebuilt on demand
├── baseline_reproduction_master_v6/       optional   baseline predictions, only to redraw the comparison tables
└── results/                               generated
```

Verify the installation of the data before running anything else:

```bash
python scripts/check_setup.py        # exit code 0 = all required inputs found; reports what is missing and where
```

### 4.2 Foundation models

Two usage modes need different weights.

**A. Computing features or retraining.** Only **ESM-2 650M** is needed (block `ESM`). It is downloaded automatically
from HuggingFace (about 2.6 GB). On a machine without internet access, download it beforehand and point to it:

```bash
export PBIND_ESM_MODEL=/path/to/esm2_t33_650M_UR50D      # default: facebook/esm2_t33_650M_UR50D
```

ESMFold is not needed because the structures are provided. ProtT5 and ESM-IF1 are optional feature groups
(Section 5).

**B. Predicting new proteins with `pbind.Predictor`.** All four models are required, placed offline with exactly
these folder names:

```text
model_weights/
├── MANIFEST.json                     source, revision and sha256 of each file
├── esmfold_v1/                       8.44 GB   facebook/esmfold_v1 @ 75a3841e
├── esm2_t33_650M_UR50D/              2.61 GB   facebook/esm2_t33_650M_UR50D @ 08e4846e
├── prot_t5_xl_half_uniref50-enc/     2.42 GB   Rostlab/prot_t5_xl_half_uniref50-enc @ 94a6abc0
└── esm_if1/                          1.70 GB   esm_if1_gvp4_t16_142M_UR50.pt (fair-esm)
```

The exact settings used to create every embedding (precision, layers, tokenisation) and the full revisions and
checksums are in [`docs/FOUNDATION_MODELS.md`](docs/FOUNDATION_MODELS.md). Wrong settings do not raise errors but
produce wrong features, so do not change them.

---

## 5. Feature store

Every feature group is a function of one protein, registered with `@feature_group`. The store evaluates each group
once per protein and saves one file per block:

```text
feature_store/
├── manifest.json                       # block -> dimension, dtype, group, column names
└── <DevSet1010|TestSet300|TestSetNew46>/
    ├── index.csv   labels.npz          # prot_id, start, end, sequence | metal / nuclear / small labels (int8)
    └── <BLOCK>.npy                     # one row per residue; ESM in float16, other blocks float32
```

```python
from pbind.features.store import FeatureStore
fs = FeatureStore("data/feature_store")
X = fs.matrix("TestSet300", ["ESM", "ATOM", "CHEM"])                  # residues x columns
df = fs.frame("TestSet300", ["ATOM"], labels=["metal_binding"])       # named columns
```

```bash
python scripts/precompute_features.py                       # all groups plus patch and pocket banks (only what is missing)
python scripts/precompute_features.py --store-only          # residue feature store only
python scripts/precompute_features.py --groups PROTT5       # optional embedding groups (need their weights)
python scripts/precompute_features.py --groups ESMIF_EF     # run in an environment with fair-esm installed
python scripts/precompute_features.py --store-only --force  # rebuild from the original caches in cache/spatial_revamp/
```

- **Structures.** GEAR blocks are computed from the ESMFold files; `NA_*` and `SM_*` blocks from the chain-A copy
  matched to the sequence, as in the original notebooks.
- **Optional embeddings.** `PROTT5` (1,024 features) and `ESMIF_EF` (512 features) are registered as optional groups,
  so a plain run builds the other 29 blocks without their weights. Two different ESM-IF1 caches exist and are not
  interchangeable: `ESMIF` (float32, used by the metal screens) and `ESMIF_EF` (float16, used by the final models).
- **Original caches.** If the original per-protein caches are available they are imported directly;
  `--store-only --force` rebuilds the store from them for an exact match. Compare recomputed features with the original
  ones using `python scripts/check_feature_parity.py --ref <folder-with-original-pkl-files>`.
- **Row safety.** Each block must have exactly one row per residue; a mismatch raises an error instead of silently
  shifting values.

---

## 6. Reproducing the results

The steps are ordered by cost and by how much of the original pipeline they exercise. Each tier states what it
reproduces and what it does not. Commands assume the virtual environment of Section 3 is active.

### Tier 0: sanity checks

```bash
python scripts/check_setup.py                                   # inputs present and in the right place
python -m pytest -q tests/                                      # test suite on synthetic data, no downloads
python scripts/verify_installation.py --final-model-dir data/final_model \
       --weights-dir model_weights --golden-dir golden_set     # checksums, golden-set predictions (PASS / FAIL)
```

`verify_installation.py` checks four things: file checksums, the pipeline on the golden inputs (must match the
reference to 1e-4), every embedding the application will use, and end-to-end predictions. Without `--weights-dir` only
the first two run. The golden set is built by `scripts/build_golden_set.py`.

### Tier 1: re-evaluate the released checkpoints (no training)

Needs: the required data, `final_model/`, and the feature store (or ESM-2 to build it).

```bash
python scripts/precompute_features.py                           # skip if you downloaded the feature store
python scripts/infer_metal.py --split TestSet300                # metal, final models; also --split TestSetNew46
python scripts/run_pipeline.py nuclear --stages final_models    # V1 + MS ensembles from the saved checkpoints
python scripts/run_pipeline.py small   --stages final_models
```

`final_models` does not train: it links the checkpoints of `final_model/<task>/` into the pipeline folders and reuses
them. Reports and the comparison with the original numbers are written under `results/`, for example
`FINAL_reproduction_vs_original.csv`.

Expected agreement with the table in Section 1: development AUPRC of nuclear and small molecules matches exactly (the
saved out-of-fold predictions are used); thresholds match to 1e-7; external AUPRC agrees within 5e-4; metal TestSet300
gives 0.38144 because the ESM-2 features are recomputed on the local machine.

### Tier 2: rebuild the features from raw inputs

```bash
python scripts/precompute_features.py                           # Section 5
python scripts/check_feature_parity.py --ref <original-cache-folder> --split TestSetNew46    # optional audit
```

### Tier 3: retrain

```bash
python scripts/train_metal.py                                   # metal, profile `lite`, end to end
python scripts/run_pipeline.py nuclear --stages all             # screens, ladders, hypergraph, V2, multi-scale, report
python scripts/run_pipeline.py small   --stages all
```

Every stage resumes from its cache. Training is deterministic for the same GPU and software versions; another GPU or
torch version can change fold-level AUPRC by about 0.01. For results identical to the original runs use a Colab T4 with
torch 2.11 and the original caches. Re-running a stage with a different configuration moves the old results to
`superseded/<hash>/` instead of overwriting them.

**What the package cannot rebuild yet.** The V3 stage (three-seed bagging of two architectures with ESM-IF1 and ProtT5,
and cross-fitted ensemble weights on Dev) still lives in the notebooks:

| Member of profile `final` | Weight | Rebuilt by `src/pbind`? |
|---|---|---|
| `metal/mlp_esmif` | 1.0 | no, notebook 08 (needs ESM-IF1) |
| `nuclear/v1`, `small/v1`, `small/ms` | 0.2 | yes, `run_pipeline.py <task> --stages hypergraph ms` |
| `nuclear\|small/{v1,ms}_esmif_prott5_s0..2` | 0.8 | no, notebooks 09 and 10 (need ESM-IF1 and ProtT5) |

In other words, about 80% of the ensemble weight for nuclear and small molecules, and the whole final metal model,
come from the notebooks; the checkpoints are provided in `final_model/`. The provenance of every member is recorded
under `sources` in `final_model/manifest.json`. Porting V3 to a `pipelines/` function and a `v3` stage of
`run_pipeline.py` is the remaining work for a fully closed pipeline.

### Tier 4: analyses of the manuscript

| Manuscript item | Command | Needs | Status |
|---|---|---|---|
| DNA and RNA as separate tasks | `python scripts/build_dna_rna_labels.py` then notebooks `12`, `13`, `15` | a BioLiP2 snapshot (`--snapshot`) | reproducible; proteins of unknown class are masked |
| Per-residue predictions of the final models on TestSetNew46 (case-study figure, per-protein comparison) | `python scripts/export_test46_predictions.py --task small --weights-dir model_weights` | `final_model/`, all four foundation models, ESMFold structures of TestSetNew46 | the script stops if the pooled AUPRC differs from the reported value (0.472 small, 0.682 metal) |
| Do hand-crafted descriptors add information beyond the embeddings? | `python scripts/esm_feature_association.py --out figures/data/esm_assoc.csv --no-lrt` | the feature store with `ESM` and, for the second baseline, `ESMIF_EF` | see below |
| Paired protein-level bootstrap (B = 10,000) of AUPRC differences | notebooks `08`, `09`, `10` | predictions of the compared methods | notebooks only |
| Baseline comparison tables | `baseline_reproduction_master_v6/` | stored baseline predictions | inputs not part of this repository |

`esm_feature_association.py` compares each descriptor block with ESM-2 alone and with ESM-2 plus ESM-IF1. For each block
and label it reports the out-of-fold R-squared of a ridge regression of the block on the embedding, the gain in AUPRC
when the block is added to a logistic model on the embedding (with a protein-level bootstrap interval), and optionally
a protein-stratified likelihood-ratio test (`--no-lrt` skips it; it is the slowest step). Run
`python scripts/esm_feature_association.py --selftest` for a check on synthetic data. The embedding enters as
`--base-pcs` principal components (default 32); because a baseline that keeps few components can make a perfect
linear copy of the embedding look informative, repeat the analysis with a larger value (for example 128 or 256).

The manuscript figure scripts are kept in the manuscript repository and are not part of this repository.

---

## 7. Single-protein inference

```python
from pbind import Predictor

predictor = Predictor("data/final_model", weights_dir="model_weights", profile="final")   # load once
result = predictor.predict(sequence, "metal", pdb=esmfold_pdb_text)                        # "metal" | "nuclear" | "small"
result["probability"]          # one probability per residue
result["binding_residues"]     # residues above the stored threshold
result["warnings"]             # for example low mean pLDDT, or a length above the longest training protein
```

Input rules: the sequence must contain only the 20 standard amino acids (sequences with `X`, `U`, `B`, `Z` or `O` are
rejected) and be at most 1,000 residues long (the longest training protein has 813). Structures must be ESMFold v1
models in the training format (chain A, pLDDT on a 0-1 scale); fold with the original settings documented in
[`docs/FOUNDATION_MODELS.md`](docs/FOUNDATION_MODELS.md) and `legacy/notebooks/protein_folding_ESMFold_v1.ipynb`.
Do not fold sequences containing `X` or `U`: the original loop silently wrote the previous protein's structure when
folding failed, which affected four proteins (excluded from the 1,014 -> 1,010 Dev set, as in bindNode24).

Integration into an application (API contract, backend steps) is described in
[`docs/APP_INTEGRATION.md`](docs/APP_INTEGRATION.md) and [`docs/APP_BUNDLE_README.md`](docs/APP_BUNDLE_README.md).

---

## 8. Extending the code

One-call API (each call uses the original runner, cache, seed and fold files, so a call to a registered MLP matches the
pipeline bit for bit):

```python
from pbind import api
ws = api.workspace("metal")
api.list_features(); api.list_models()
api.cross_validate(ws, model="MLP", features="ESM+Atomic+Chem+MULTI2")             # Dev only
api.train_final(ws, model="MLP", features="ESM+MECH1", neural_epochs=20)           # saves weights, evaluates 300 / 46
api.predict(ws, model="NA_PatchHG_Attention", features="...", split="TestSet300")
```

- **Add a feature group:** write a function decorated with `@feature_group("NAME", columns={...}, needs="sequence" | "structure")`;
  the store computes it and `features="ESM+NAME"` works immediately. See `features/api.py`.
- **Add a model:** write a function decorated with `@api.register_model("NAME", kind="residue" | "fusion" | "group")`.
  See `models/registry.py`.

---

## 9. Notes and known limitations

- **Selection protocol.** Features, models and thresholds are chosen on DevSet1010 (thresholds on Dev out-of-fold
  predictions only). TestSet300 and TestSetNew46 are used only to confirm choices that were fixed on Dev.
- **DNA and RNA.** When split, the ligand class enters through the **label**, not the input, and every model is scored
  on all 300 / 46 proteins, which keeps the comparison fair. Combining the two specialised models back into a single
  nuclear prediction is an oracle (routing a protein to the DNA model already asserts that it binds DNA) and must not be
  reported. The RNA line is weak: the lower bound of its AUPRC difference is +0.0016 on TestSet300 and it loses on
  TestSetNew46.
- **External benchmark (GraphBind).** `DNA-129_Test` overlaps DevSet1010 (for example `5wx9_A` is identical to
  `Q9LSX0`), so a model trained on DevSet1010 cannot be evaluated on it without leakage. Notebook 14 blocks this and
  documents that a fair comparison requires retraining on `DNA-573_Train` / `RNA-495_Train`. Positive density also differs
  (about 6% there against 2.08% here), so AUPRC values are not comparable across the two benchmarks.
- **`CONF` block.** The original code divides pLDDT by 100 although ESMFold stores it on a 0-1 scale, so the block is very
  small. This is kept as in the original; no final model uses it.
- **Metal caches.** The ESM, structure, ATOM/CHEM and MECH1/MULTI2 caches in `cache/spatial_revamp/` of a fresh run are
  recomputed locally. For exact original values copy the original caches and run
  `scripts/precompute_features.py --store-only --force`.
- **Descriptors come from predicted monomer structures,** which may not reproduce the ligand-bound conformation or
  interfaces with other chains.

---

## 10. Testing and license

```bash
python -m pytest -q tests/
```

The tests run on synthetic data and need no downloads. Released under the [MIT License](LICENSE).
