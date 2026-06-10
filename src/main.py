"""Training entrypoint for domain adaptation experiments."""

from pathlib import Path
import logging

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict

from datasets.loader import get_dataloader
from methods import get_solver, list_solvers
from utils import (
    cfg_get,
    cleanup_distributed,
    configure_torch_runtime,
    get_distributed_context,
    initialize_distributed,
    is_truthy,
    log_dataset_summary,
    log_runtime_summary,
    register_resolvers,
    set_seed,
    shutdown_dataloader_workers,
    validate_config,
)


logger = logging.getLogger(__name__)


register_resolvers(src_dir=Path(__file__).resolve().parent)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    logger.debug(OmegaConf.to_yaml(cfg))
    available_solvers = list_solvers()
    logger.info("Available solvers: %s", available_solvers)
    validate_config(cfg, available_solvers=available_solvers)

    distributed = initialize_distributed(cfg)
    loaders = ()
    try:
        if distributed.enabled and distributed.backend == "nccl":
            with open_dict(cfg):
                cfg.device = f"cuda:{distributed.local_rank}"

        perf = cfg_get(cfg, "performance", {})
        base_seed = int(cfg.get("seed", 42))
        process_seed = base_seed + distributed.rank
        runtime_cfg = configure_torch_runtime(cfg)
        set_seed(
            process_seed,
            deterministic=is_truthy(cfg_get(perf, "deterministic", False)),
            benchmark=is_truthy(cfg_get(perf, "benchmark", True)),
        )
        log_runtime_summary(logger, runtime_cfg, process_seed)

        source_loader, target_loader, target_test_loader, class_info = get_dataloader(cfg)
        loaders = (source_loader, target_loader, target_test_loader)
        log_dataset_summary(logger, cfg, class_info)

        method_name = cfg.method.name.lower()
        supported_settings = getattr(cfg.method, "supported_settings", None)
        if supported_settings is not None and class_info["setting"] not in list(supported_settings):
            logger.error(
                f"Method '{method_name}' does not support setting='{class_info['setting']}'. "
                f"supported_settings={list(supported_settings)}"
            )
            raise ValueError(f"Unsupported setting for method '{method_name}'")
        try:
            solver_cls = get_solver(method_name)
        except KeyError as e:
            logger.error(str(e))
            raise

        solver = solver_cls(cfg, loaders, class_info)
        logger.info(f"Initialized solver: {solver_cls.__name__} for method '{method_name}'")

        logger.info("Starting training...")
        solver.train()

        if get_distributed_context().is_main_process:
            best_path = getattr(solver, "_best_ckpt_path", None)
            best_saved = is_truthy(getattr(solver, "_best_saved", False))
            if best_saved and best_path is not None and Path(best_path).exists():
                logger.info(f"Best model saved to: {Path(best_path).absolute()}")
            else:
                logger.warning(
                    "No checkpoint was saved. This can happen when total epochs < "
                    "method.save_start_epoch "
                    f"(current save_start_epoch={getattr(solver, '_save_start_epoch', 'N/A')})."
                )
    finally:
        shutdown_dataloader_workers(loaders)
        cleanup_distributed()


if __name__ == "__main__":
    main()
