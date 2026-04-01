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
from methods.dcfm_cs import DCFMColorSpaceSolver
from methods.odcfm import ODCFMSolver
from methods.mdcfm import MDCFMSolver
from methods.dare import DARESolver
from methods.rvtc import RVTCSolver
from methods.trajuda import TrajUDASolver
from methods.fact_da import FactDASolver
from methods.clipfilm import CLIPFiLMSolver
from methods.clipfilm_domain import CLIPFiLMDomainSolver
from methods.embraceda_lite import EmbraceDALiteSolver

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
    "DCFMColorSpaceSolver",
    "ODCFMSolver",
    "MDCFMSolver",
    "DARESolver",
    "RVTCSolver",
    "TrajUDASolver",
    "FactDASolver",
    "CLIPFiLMSolver",
    "CLIPFiLMDomainSolver",
    "EmbraceDALiteSolver",
]

