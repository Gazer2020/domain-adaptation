"""
Methods package with lazy solver loading.

Only the requested solver module is imported when get_solver(name) is called.
This avoids import-time side effects from unrelated methods.
"""

from importlib import import_module

from methods.registry import get_solver as _get_solver_from_registry
from methods.registry import register_solver


_SOLVER_MODULES = {
    "sourceonly": "methods.base_solver",
    "ros": "methods.ros",
    "mic": "methods.mic",
    "cad": "methods.cad",
    "cosda": "methods.cosda",
    "rtda": "methods.rtda",
    "dcfm": "methods.dcfm",
    "factda": "methods.factda",
    "dare": "methods.dare",
    "rvtc": "methods.rvtc",
    "prc": "methods.prc",
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _load_solver_module(name: str) -> None:
    module_name = _SOLVER_MODULES.get(name)
    if module_name is None:
        available = list(_SOLVER_MODULES.keys())
        raise KeyError(f"Solver '{name}' not found. Available solvers: {available}")
    import_module(module_name)


def get_solver(name: str):
    """
    Get a solver class by name with lazy module import.
    """
    normalized = _normalize_name(name)
    _load_solver_module(normalized)
    return _get_solver_from_registry(normalized)


def list_solvers() -> list:
    """
    List available solver names without importing all solver modules.
    """
    return sorted(_SOLVER_MODULES.keys())


__all__ = [
    "get_solver",
    "register_solver",
    "list_solvers",
]
