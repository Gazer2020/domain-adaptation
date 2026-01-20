"""
Methods package for domain adaptation solvers.

All solvers are automatically registered via the @register_solver decorator.
Use get_solver(name) to retrieve a solver class by name.
"""

from methods.registry import get_solver, register_solver, list_solvers

# Import all solvers to trigger registration
from methods.base_solver import BaseSolver
from methods.ros import RotationSolver
from methods.mic import MICSolver  # Renamed from MaskSolver
from methods.cad import CADSolver

__all__ = [
    "get_solver",
    "register_solver",
    "list_solvers",
    "BaseSolver",
    "RotationSolver",
    "MICSolver",  # Renamed from MaskSolver
    "CADSolver",
]
