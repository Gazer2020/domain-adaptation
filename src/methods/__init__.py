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
from methods.tod import TODSolver
from methods.oid_gda import OIDGDASolver
from methods.gad import GADSolver
from methods.dga import DGASolver
from methods.mic_simsiam import MICSimSiamSolver
from methods.mic_gmm import MICGMMSolver

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
    "TODSolver",
    "OIDGDASolver",
    "GADSolver",
    "DGASolver",
    "MICSimSiamSolver",
    "MICGMMSolver",
]


