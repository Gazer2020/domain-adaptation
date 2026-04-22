"""
Runtime helpers shared by the training entrypoint and solver implementations.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from utils.config import cfg_get, is_truthy


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    """Set Python, NumPy, and Torch RNG state with consistent backend flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark) and (not bool(deterministic))
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def configure_torch_runtime(cfg) -> dict[str, object]:
    """
    Apply project-wide Torch runtime settings and return the resolved values.
    """
    perf = cfg_get(cfg, "performance", {})
    allow_tf32 = is_truthy(cfg_get(perf, "allow_tf32", True))
    matmul_precision = str(cfg_get(perf, "matmul_precision", "high")).lower()
    deterministic = is_truthy(cfg_get(perf, "deterministic", False))
    benchmark = is_truthy(cfg_get(perf, "benchmark", True))

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    return {
        "allow_tf32": allow_tf32,
        "matmul_precision": matmul_precision,
        "deterministic": deterministic,
        "benchmark": benchmark and (not deterministic),
    }


def log_runtime_summary(logger: logging.Logger, runtime_cfg: dict[str, object], seed: int) -> None:
    """Emit a compact summary of the active runtime defaults."""
    logger.info(
        "Runtime configured | seed=%s deterministic=%s benchmark=%s allow_tf32=%s matmul_precision=%s",
        seed,
        runtime_cfg["deterministic"],
        runtime_cfg["benchmark"],
        runtime_cfg["allow_tf32"],
        runtime_cfg["matmul_precision"],
    )


def log_dataset_summary(logger: logging.Logger, cfg, class_info: dict) -> None:
    """Emit the domain/task summary in one place for a consistent entrypoint log."""
    if class_info["setting"] == "msda":
        logger.info(
            "Data loaded | setting=%s sources=%s target=%s",
            class_info["setting"],
            list(cfg.dataset.sources),
            cfg.dataset.target,
        )
    else:
        logger.info(
            "Data loaded | setting=%s source=%s target=%s",
            class_info["setting"],
            cfg.dataset.source,
            cfg.dataset.target,
        )
    logger.info(
        "Class info | num_classes=%s shared=%s unknown_label=%s",
        class_info["num_classes"],
        len(class_info["shared_classes"]),
        class_info["unknown_label"],
    )
