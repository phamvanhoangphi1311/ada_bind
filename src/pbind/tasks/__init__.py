from . import metal
from .base import GridExtra, TaskSpec, V2Spec
from .nuclear import NUCLEAR
from .small import SMALL

TASK_SPECS = {"nuclear": NUCLEAR, "small": SMALL}


def get_task(name: str) -> TaskSpec:
    try:
        return TASK_SPECS[name]
    except KeyError:
        raise KeyError(f"Unknown task {name!r}; choose from {list(TASK_SPECS)} (metal uses pbind.pipelines.metal)")
