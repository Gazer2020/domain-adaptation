import sys
import random
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from utils.config import get_config
from datasets.loader import get_dataloader
from methods.base_solver import BaseSolver


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


def main():
    # 1. Load Config
    try:
        config = get_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    logger.info(
        f"Loaded config for dataset: {config.dataset.name}, method: {config.method.name}"
    )

    # 2. Set Seed
    set_seed(config.get("seed", 42))

    # 3. Get DataLoaders
    source_loader, target_loader, target_test_loader = get_dataloader(config)

    logger.info(
        f"Data loaded. Source: {config.dataset.source}, Target: {config.dataset.target}, "
    )

    # 4. Initialize Solver
    loaders = (source_loader, target_loader, target_test_loader)

    # In the future, use a factory pattern here based on config.method.name
    method_name = config.method.get("name", "SourceOnly")
    if method_name == "SourceOnly":
        solver = BaseSolver(config, loaders)
    else:
        logger.warning(
            f"Unknown method {method_name}, falling back to BaseSolver (SourceOnly)"
        )
        solver = BaseSolver(config, loaders)

    # 5. Train
    logger.info("Starting training...")
    solver.train()

    # 6. Save Model
    # Determine save path
    # Assuming we want to save in a 'checkpoints' dir in project root
    # or relative to main.py
    # Let's use config.exp_name to create a unique file
    src_dir = Path(__file__).resolve()
    project_root = src_dir.parent.parent
    save_dir = (
        project_root
        / "checkpoints"
        / config.dataset.name
        / config.dataset.setting
    )
    save_name = f"{config.exp_name}.pth"
    save_path = save_dir / save_name

    solver.save_checkpoint(save_path)
    logger.info("Experiment finished successfully.")


if __name__ == "__main__":
    main()
