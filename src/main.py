"""
Domain Adaptation Training Entry Point

This module provides the main entry point for training domain adaptation models.
Methods are automatically discovered via the registry pattern - just add a new
method file with @register_solver decorator and a config file.
"""

import logging
import random
from pathlib import Path

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

from datasets.loader import get_dataloader
from methods import get_solver, list_solvers


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # 1. Log config
    logger.debug(OmegaConf.to_yaml(cfg))
    logger.info(f"Available solvers: {list_solvers()}")

    # 2. Set seed
    set_seed(cfg.get("seed", 42))

    # 3. Get dataLoaders
    source_loader, target_loader, target_test_loader, class_info = get_dataloader(cfg)
    loaders = (source_loader, target_loader, target_test_loader)

    logger.info(
        f"Data loaded. Source: {cfg.dataset.source}, Target: {cfg.dataset.target}"
    )
    logger.info(
        f"Setting: {class_info['setting']}, "
        f"Num classes: {class_info['num_classes']}, "
        f"Shared: {len(class_info['shared_classes'])}, "
        f"Unknown label: {class_info['unknown_label']}"
    )

    # 4. Initialize Solver via registry
    method_name = cfg.method.name.lower()
    try:
        solver_cls = get_solver(method_name)
    except KeyError as e:
        logger.error(str(e))
        raise

    solver = solver_cls(cfg, loaders, class_info)
    logger.info(f"Initialized solver: {solver_cls.__name__} for method '{method_name}'")

    # 5. Train
    logger.info("Starting training...")
    solver.train()

    # 6. Save Model
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{cfg.exp_name}.pth"
    solver.save_checkpoint(save_path)

    logger.info(f"Model saved to: {save_path.absolute()}")


if __name__ == "__main__":
    main()
