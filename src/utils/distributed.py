"""Small torch.distributed runtime used by supported solvers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import torch
import torch.distributed as dist

from utils.config import cfg_get, resolve_auto_bool


logger = logging.getLogger(__name__)

_SUPPORTED_SOLVERS = {
    "cad",
    "dcfm",
    "factda",
    "mic",
    "ros",
    "rvtc",
    "sourceonly",
}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    gradient_bucket_bytes: int = 25 * 1024 * 1024

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


_CONTEXT = DistributedContext()


def get_distributed_context() -> DistributedContext:
    return _CONTEXT


def initialize_distributed(cfg) -> DistributedContext:
    """Initialize an env:// process group when requested by torchrun."""
    global _CONTEXT

    distributed_cfg = cfg_get(cfg, "distributed", {})
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = resolve_auto_bool(
        cfg_get(distributed_cfg, "enabled", "auto"),
        auto_value=world_size > 1,
    )
    if not enabled:
        _CONTEXT = DistributedContext()
        return _CONTEXT
    if world_size <= 1:
        raise ValueError(
            "distributed.enabled=True requires torchrun with WORLD_SIZE > 1"
        )
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build")

    method_name = str(cfg_get(cfg_get(cfg, "method", {}), "name", "")).lower()
    if method_name not in _SUPPORTED_SOLVERS:
        raise ValueError(
            f"Distributed training is not supported for method={method_name!r}. "
            f"Supported methods: {sorted(_SUPPORTED_SOLVERS)}. Methods with global "
            "prototype/memory state require method-specific synchronization."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    backend_cfg = str(cfg_get(distributed_cfg, "backend", "auto")).lower()
    backend = (
        "nccl"
        if backend_cfg == "auto" and torch.cuda.is_available()
        else "gloo"
        if backend_cfg == "auto"
        else backend_cfg
    )
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise ValueError("distributed.backend=nccl requires CUDA")
        torch.cuda.set_device(local_rank)

    timeout_seconds = int(cfg_get(distributed_cfg, "timeout_seconds", 1800))
    gradient_bucket_mb = int(cfg_get(distributed_cfg, "gradient_bucket_mb", 25))
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )
    _CONTEXT = DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        gradient_bucket_bytes=gradient_bucket_mb * 1024 * 1024,
    )
    logger.info(
        "Distributed runtime initialized | backend=%s rank=%d local_rank=%d world_size=%d",
        backend,
        rank,
        local_rank,
        world_size,
    )
    return _CONTEXT


def cleanup_distributed() -> None:
    global _CONTEXT
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _CONTEXT = DistributedContext()


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_objects_to_main(value):
    """Gather one picklable object per rank and return them on rank zero."""
    context = get_distributed_context()
    if not context.enabled:
        return [value]
    gathered = [None] * context.world_size if context.is_main_process else None
    dist.gather_object(value, gathered, dst=0)
    return gathered


@torch.no_grad()
def broadcast_modules(modules: Iterable[torch.nn.Module], src: int = 0) -> None:
    """Broadcast unique module parameters and buffers from one rank."""
    context = get_distributed_context()
    if not context.enabled:
        return
    seen: set[int] = set()
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            dist.broadcast(tensor, src=src)


@torch.no_grad()
def average_module_buffers(modules: Iterable[torch.nn.Module]) -> None:
    """Average floating buffers and broadcast integral buffers before evaluation."""
    context = get_distributed_context()
    if not context.enabled:
        return
    seen: set[int] = set()
    for module in modules:
        for buffer in module.buffers():
            identity = id(buffer)
            if identity in seen:
                continue
            seen.add(identity)
            if torch.is_floating_point(buffer):
                dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
                buffer.div_(context.world_size)
            else:
                dist.broadcast(buffer, src=0)


@torch.no_grad()
def synchronize_optimizer_gradients(optimizer) -> None:
    """Average optimizer gradients across ranks in dtype/device buckets."""
    context = get_distributed_context()
    if not context.enabled:
        return

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    if not parameters:
        return

    device = parameters[0].device
    presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.uint8,
        device=device,
    )
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)

    buckets: dict[tuple[torch.device, torch.dtype], list[torch.nn.Parameter]] = {}
    for parameter, globally_used in zip(parameters, presence.tolist()):
        if not globally_used:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise RuntimeError("Sparse gradients are not supported by distributed training")
        buckets.setdefault(
            (parameter.grad.device, parameter.grad.dtype), []
        ).append(parameter)

    for bucket in buckets.values():
        chunks: list[list[torch.Tensor]] = []
        current: list[torch.Tensor] = []
        current_bytes = 0
        for parameter in bucket:
            gradient = parameter.grad
            gradient_bytes = gradient.numel() * gradient.element_size()
            if current and current_bytes + gradient_bytes > context.gradient_bucket_bytes:
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(gradient)
            current_bytes += gradient_bytes
        if current:
            chunks.append(current)

        for gradients in chunks:
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(context.world_size)
            offset = 0
            for gradient in gradients:
                size = gradient.numel()
                gradient.copy_(flat[offset : offset + size].view_as(gradient))
                offset += size
