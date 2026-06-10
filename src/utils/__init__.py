"""
Shared utility surface for configuration, runtime, and small generic helpers.
"""

from .config import (
    cfg_get,
    is_truthy,
    parse_range,
    register_resolvers,
    resolve_auto_bool,
    resolve_int_or_auto,
    resolve_optional_auto_bool,
)
from .constants import IMAGENET_MEAN, IMAGENET_STD
from .distributed import (
    average_module_buffers,
    broadcast_modules,
    cleanup_distributed,
    distributed_barrier,
    get_distributed_context,
    gather_objects_to_main,
    initialize_distributed,
    synchronize_optimizer_gradients,
)
from .runtime import (
    CudaBatchPrefetcher,
    configure_faiss_runtime,
    configure_torch_runtime,
    log_dataset_summary,
    log_runtime_summary,
    set_seed,
    shutdown_dataloader_workers,
)
from .utils import AverageMeter, GpuLossAccumulator, cycle, get_device
from .validation import validate_config

__all__ = [
    "AverageMeter",
    "GpuLossAccumulator",
    "CudaBatchPrefetcher",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "cfg_get",
    "average_module_buffers",
    "broadcast_modules",
    "cleanup_distributed",
    "distributed_barrier",
    "configure_faiss_runtime",
    "configure_torch_runtime",
    "cycle",
    "get_device",
    "get_distributed_context",
    "gather_objects_to_main",
    "initialize_distributed",
    "is_truthy",
    "log_dataset_summary",
    "log_runtime_summary",
    "parse_range",
    "register_resolvers",
    "resolve_auto_bool",
    "resolve_int_or_auto",
    "resolve_optional_auto_bool",
    "set_seed",
    "shutdown_dataloader_workers",
    "synchronize_optimizer_gradients",
    "validate_config",
]
