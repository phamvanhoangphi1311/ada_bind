"""Device / seeding / memory helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def get_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    return dev


def seed_everything(seed: int) -> None:
    """Same seeding as the notebooks (torch + numpy), plus python's random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def free_gpu(device: torch.device | None = None) -> None:
    if (device is None or device.type == "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def to_device(obj, device: torch.device):
    """Move tensors in (nested) dicts to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj
