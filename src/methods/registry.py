"""
Method registry for automatic solver registration and lookup.
"""

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from methods.base_solver import BaseSolver

# Global registry
_SOLVER_REGISTRY: Dict[str, Type["BaseSolver"]] = {}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def register_solver(name: str):
    """
    Decorator to register a solver class.
    
    Usage:
        @register_solver("dann")
        class DANNSolver(BaseSolver):
            ...
    """
    def decorator(cls: Type["BaseSolver"]) -> Type["BaseSolver"]:
        normalized = _normalize_name(name)
        if normalized in _SOLVER_REGISTRY:
            raise ValueError(f"Solver '{normalized}' is already registered.")
        _SOLVER_REGISTRY[normalized] = cls
        return cls
    return decorator


def get_solver(name: str) -> Type["BaseSolver"]:
    """
    Get a solver class by name.
    
    Args:
        name: The name of the solver (e.g., "ros", "mic", "sourceonly")
        
    Returns:
        The solver class
        
    Raises:
        KeyError: If the solver is not registered
    """
    normalized = _normalize_name(name)
    if normalized not in _SOLVER_REGISTRY:
        available = list_solvers()
        raise KeyError(
            f"Solver '{normalized}' not found. Available solvers: {available}"
        )
    return _SOLVER_REGISTRY[normalized]


def list_solvers() -> list:
    """Return a list of all registered solver names."""
    return sorted(_SOLVER_REGISTRY.keys())
