"""Inference with locked models (no model selection happens here)."""
from .metal import DEFAULT_LOCK, load_or_train_fold, locked_config, run_metal_inference
from .final import final_members, install_final_models, write_manifest
