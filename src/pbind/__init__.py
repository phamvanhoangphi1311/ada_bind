"""pbind — residue-level binding-site prediction (metal / nuclear / small).

Start here: ``from pbind import api`` — ``api.cross_validate(ws, model="MLP", features="ESM+Atomic+Chem+MULTI2")``.

Layout
------
api                    one-call experiments: pick features + model by name (cross_validate / train_final / predict)
paths, constants       where data lives, fixed dataset/task constants
utils/                 cache helpers (every expensive step goes through them), torch helpers
data/                  CSV splits, official bindNode24 folds, exact-sequence PDB normalization
backend/               native port of the GEAR backend (default) + adapter for the Drive `gear_ultimate` package
features/              feature registry (@feature_group) + feature store; GEAR blocks, nuclear NA_*, small SM_*
groups/                residue groups for hypergraphs (nuclear surface patches, small pockets)
models/                model registry (@register_model): residue MLP/KAN, fusion MLPs, hypergraphs, losses
training/              modality packs, datasets, resume-safe CV trainers
evaluation/            metrics, thresholds, ensembles, Dev-only selection rules
reporting/             baselines, prediction export, paper-ready report, manifests
analysis/              dataset anatomy audit
tasks/                 TaskSpec per binding type (grids, names, hyper-parameters)
pipelines/             notebook stages as functions: call these from notebooks/
inference/             frozen final models: final.py (final_model/ layout, profiles, thresholds),
                       predictor.py (single-protein API, exposed lazily as ``pbind.Predictor``)
"""

__version__ = "0.1.0"


def __getattr__(name):  # lazy: ``import pbind`` stays light (no torch) until the predictor is used
    if name in ("Predictor", "predict", "InputError"):
        from .inference import predictor

        return getattr(predictor, name)
    raise AttributeError(name)
