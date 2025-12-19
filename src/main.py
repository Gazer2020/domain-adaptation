import logging
import random
from pathlib import Path

import hydra
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

from datasets.loader import get_dataloader
from methods.base_solver import BaseSolver
from methods.ros import RotationSolver


logger = logging.getLogger(__name__)


def set_seed(seed):
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
    # 1. Log config
    logger.debug(OmegaConf.to_yaml(cfg))

    # 2. Set seed
    set_seed(cfg.get("seed", 42))

    # 3. Get dataLoaders
    source_loader, target_loader, target_test_loader = get_dataloader(cfg)

    logger.info(
        f"Data loaded. Source: {cfg.dataset.source}, Target: {cfg.dataset.target}, ",
    )

    # 4. Initialize Solver
    loaders = (source_loader, target_loader, target_test_loader)
    name = cfg.method.name
    if name == "ros":
        solver = RotationSolver(cfg, loaders)
    elif name == "sourceonly":
        solver = BaseSolver(cfg, loaders)
    else:
        raise NotImplementedError(f"Unknown method {name}")

    logger.info(f"Initialized solver for method: {name}")

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
