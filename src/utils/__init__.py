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
)
from .constants import IMAGENET_MEAN, IMAGENET_STD
from .runtime import (
    CudaBatchPrefetcher,
    configure_faiss_runtime,
    configure_torch_runtime,
    log_dataset_summary,
    log_runtime_summary,
    set_seed,
)
from .utils import AverageMeter, cycle, get_device

__all__ = [
    "AverageMeter",
    "CudaBatchPrefetcher",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "cfg_get",
    "configure_faiss_runtime",
    "configure_torch_runtime",
    "cycle",
    "get_device",
    "is_truthy",
    "log_dataset_summary",
    "log_runtime_summary",
    "parse_range",
    "register_resolvers",
    "resolve_auto_bool",
    "resolve_int_or_auto",
    "set_seed",
]
