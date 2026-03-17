"""
Methods package for domain adaptation solvers.

All solvers are automatically registered via the @register_solver decorator.
Use get_solver(name) to retrieve a solver class by name.
"""

from methods.registry import get_solver, register_solver, list_solvers

# Import all solvers to trigger registration
from methods.base_solver import BaseSolver
from methods.ros import RotationSolver
from methods.mic import MICSolver
from methods.cad import CADSolver

from methods.cosda import COSDASolver
from methods.rtda import RTDASolver
from methods.dcfm import DCFMSolver
from methods.odcfm import ODCFMSolver
from methods.mdcfm import MDCFMSolver

__all__ = [
    # Registry
    "get_solver",
    "register_solver",
    "list_solvers",
    # Solvers
    "BaseSolver",
    "RotationSolver",
    "MICSolver",
    "CADSolver",

    "COSDASolver",
    "RTDASolver",
    "DCFMSolver",
    "ODCFMSolver",
    "MDCFMSolver",
]


