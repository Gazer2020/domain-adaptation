"""
Runtime helpers shared by the training entrypoint and solver implementations.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

import numpy as np
import torch

from utils.config import cfg_get, is_truthy


def configure_faiss_runtime(cfg) -> int:
    """Apply the shared FAISS CPU thread limit and return the resolved value."""
    perf = cfg_get(cfg, "performance", {})
    threads = max(1, int(cfg_get(perf, "faiss_threads", 1)))

    import faiss

    faiss.omp_set_num_threads(threads)
    return threads


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
    tf32_api = "unavailable"
    if torch.cuda.is_available():
        precision = "tf32" if allow_tf32 else "ieee"
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if hasattr(torch.backends.cuda.matmul, "fp32_precision") and hasattr(
            cudnn_conv, "fp32_precision"
        ):
            torch.backends.cuda.matmul.fp32_precision = precision
            cudnn_conv.fp32_precision = precision
            tf32_api = "fp32_precision"
        else:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            tf32_api = "allow_tf32"

    return {
        "allow_tf32": allow_tf32,
        "tf32_api": tf32_api,
        "matmul_precision": matmul_precision,
        "deterministic": deterministic,
        "benchmark": benchmark and (not deterministic),
    }


def log_runtime_summary(logger: logging.Logger, runtime_cfg: dict[str, object], seed: int) -> None:
    """Emit a compact summary of the active runtime defaults."""
    logger.info(
        "Runtime configured | seed=%s deterministic=%s benchmark=%s allow_tf32=%s "
        "tf32_api=%s matmul_precision=%s",
        seed,
        runtime_cfg["deterministic"],
        runtime_cfg["benchmark"],
        runtime_cfg["allow_tf32"],
        runtime_cfg["tf32_api"],
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


def _record_stream_recursive(batch: Any, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(batch):
        if batch.is_cuda:
            batch.record_stream(stream)
        return
    if isinstance(batch, (list, tuple)):
        for value in batch:
            _record_stream_recursive(value, stream)
        return
    if isinstance(batch, dict):
        for value in batch.values():
            _record_stream_recursive(value, stream)


class CudaBatchPrefetcher:
    """Prefetch the next batch to a CUDA stream while the current batch computes."""

    def __init__(
        self,
        iterator,
        load_fn: Callable[[Any], Any],
        enabled: bool,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        self.iterator = iterator
        self.load_fn = load_fn
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.stream = stream if self.enabled else None
        if self.enabled and self.stream is None:
            self.stream = torch.cuda.Stream()
        self._next_batch = None
        self._preload()

    def _preload(self) -> None:
        try:
            raw = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            return

        if not self.enabled or self.stream is None:
            self._next_batch = self.load_fn(raw)
            return

        with torch.cuda.stream(self.stream):
            self._next_batch = self.load_fn(raw)

    def pop(self):
        if self._next_batch is None:
            raise StopIteration

        if self.enabled and self.stream is not None:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.stream)
            _record_stream_recursive(self._next_batch, current_stream)

        batch = self._next_batch
        self._preload()
        return batch

    def close(self) -> None:
        self._next_batch = None
