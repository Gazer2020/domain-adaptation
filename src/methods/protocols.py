"""
Protocols for solver type checking and testing validation.

These protocols define the expected interfaces for domain adaptation solvers,
enabling static type checking and runtime validation.
"""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class SolverProtocol(Protocol):
    """
    Protocol defining the core solver interface.
    
    All domain adaptation solvers should satisfy this protocol.
    Use isinstance(solver, SolverProtocol) for runtime validation.
    
    Example:
        solver = get_solver("cad")(config, loaders, class_info)
        assert isinstance(solver, SolverProtocol)
    """
    
    config: object
    device: torch.device
    num_classes: int
    
    def build_model(self) -> None:
        """Build the network architecture."""
        ...
    
    def train(self) -> None:
        """Execute the training procedure."""
        ...
    
    def evaluate(self) -> float:
        """Evaluate on target test set. Returns accuracy or H-score."""
        ...
    
    def forward_for_eval(self, imgs: Tensor) -> Tensor:
        """Forward pass for evaluation. Returns logits."""
        ...
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to path."""
        ...
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint from path."""
        ...


def validate_solver(solver: object) -> bool:
    """
    Validate that an object satisfies the SolverProtocol.
    
    Args:
        solver: Object to validate
        
    Returns:
        True if solver satisfies protocol, False otherwise
        
    Raises:
        TypeError: If solver is missing required attributes/methods
    """
    if not isinstance(solver, SolverProtocol):
        missing = []
        for attr in ['build_model', 'train', 'evaluate', 'forward_for_eval', 
                     'save_checkpoint', 'load_checkpoint']:
            if not hasattr(solver, attr) or not callable(getattr(solver, attr)):
                missing.append(attr)
        if missing:
            raise TypeError(
                f"Solver {type(solver).__name__} is missing required methods: {missing}"
            )
    return True


def validate_solver_output(solver: object, sample_input: Tensor) -> bool:
    """
    Validate solver produces correct output shapes.
    
    Args:
        solver: Solver to validate
        sample_input: Sample input tensor [B, C, H, W]
        
    Returns:
        True if outputs are valid
        
    Raises:
        ValueError: If outputs have unexpected shapes
    """
    if not hasattr(solver, 'forward_for_eval'):
        raise TypeError("Solver missing forward_for_eval method")
    
    with torch.no_grad():
        output = solver.forward_for_eval(sample_input)
    
    batch_size = sample_input.size(0)
    if output.size(0) != batch_size:
        raise ValueError(
            f"Output batch size {output.size(0)} doesn't match input {batch_size}"
        )
    
    if hasattr(solver, 'num_classes') and output.size(1) != solver.num_classes:
        raise ValueError(
            f"Output classes {output.size(1)} doesn't match num_classes {solver.num_classes}"
        )
    
    return True
