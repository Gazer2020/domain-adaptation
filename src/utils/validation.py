"""Central startup validation for shared experiment configuration."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from utils.config import cfg_get


_AUTO_BOOL_VALUES = {"auto", "true", "false", "1", "0", "yes", "no", "on", "off"}
_SETTINGS = {"csda", "osda", "pda", "unida", "msda"}


def _normalized(value: Any) -> str:
    return str(value).strip().lower()


def _require_choice(errors: list[str], path: str, value: Any, choices: set[str]) -> None:
    normalized = _normalized(value)
    if normalized not in choices:
        errors.append(f"{path}={value!r}; expected one of {sorted(choices)}")


def _require_int_at_least(errors: list[str], path: str, value: Any, minimum: int) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{path}={value!r}; expected an integer >= {minimum}")
        return
    if parsed < minimum:
        errors.append(f"{path}={parsed}; expected >= {minimum}")


def _validate_worker_value(errors: list[str], path: str, value: Any) -> None:
    if _normalized(value) == "auto":
        return
    _require_int_at_least(errors, path, value, 0)


def validate_config(cfg: Any, available_solvers: Iterable[str] | None = None) -> None:
    """Fail fast on invalid shared runtime and method configuration."""
    errors: list[str] = []

    _require_int_at_least(errors, "batch_size", cfg_get(cfg, "batch_size", None), 1)
    _require_int_at_least(errors, "num_workers", cfg_get(cfg, "num_workers", None), 0)
    device = _normalized(cfg_get(cfg, "device", "auto"))
    if device not in {"auto", "cpu", "cuda", "mps"} and not re.fullmatch(
        r"cuda:\d+", device
    ):
        errors.append(
            f"device={device!r}; expected auto, cpu, cuda, cuda:<index>, or mps"
        )

    method = cfg_get(cfg, "method", {})
    method_name = _normalized(cfg_get(method, "name", ""))
    if not method_name:
        errors.append("method.name must be a non-empty string")
    elif available_solvers is not None:
        known_solvers = {_normalized(name) for name in available_solvers}
        if method_name not in known_solvers:
            errors.append(
                f"method.name={method_name!r}; expected one of {sorted(known_solvers)}"
            )

    setting = _normalized(cfg_get(method, "setting", ""))
    _require_choice(errors, "method.setting", setting, _SETTINGS)
    supported_settings = cfg_get(method, "supported_settings", None)
    if supported_settings is not None:
        normalized_supported = {_normalized(item) for item in supported_settings}
        invalid_settings = normalized_supported - _SETTINGS
        if invalid_settings:
            errors.append(
                "method.supported_settings contains invalid values: "
                f"{sorted(invalid_settings)}"
            )
        if setting and setting not in normalized_supported:
            errors.append(
                f"method.setting={setting!r} is not listed in method.supported_settings"
            )

    epochs = cfg_get(method, "epochs", None)
    if epochs is not None:
        _require_int_at_least(errors, "method.epochs", epochs, 1)
    _require_choice(
        errors,
        "method.target_aug_backend",
        cfg_get(method, "target_aug_backend", "dataset"),
        {"dataset", "tensor_v2"},
    )

    perf = cfg_get(cfg, "performance", {})
    _require_choice(
        errors,
        "performance.matmul_precision",
        cfg_get(perf, "matmul_precision", "high"),
        {"highest", "high", "medium"},
    )
    for name in (
        "deterministic",
        "benchmark",
        "allow_tf32",
        "pin_memory",
        "channels_last",
        "non_blocking_transfer",
        "zero_grad_set_to_none",
    ):
        _require_choice(
            errors,
            f"performance.{name}",
            cfg_get(perf, name, "auto"),
            _AUTO_BOOL_VALUES,
        )
    _require_int_at_least(
        errors,
        "performance.faiss_threads",
        cfg_get(perf, "faiss_threads", 1),
        1,
    )

    amp = cfg_get(perf, "amp", {})
    _require_choice(
        errors,
        "performance.amp.enabled",
        cfg_get(amp, "enabled", "auto"),
        _AUTO_BOOL_VALUES,
    )
    _require_choice(
        errors,
        "performance.amp.dtype",
        cfg_get(amp, "dtype", "bf16"),
        {"bf16", "bfloat16", "fp16", "float16", "half"},
    )

    compile_cfg = cfg_get(perf, "compile", {})
    _require_choice(
        errors,
        "performance.compile.enabled",
        cfg_get(compile_cfg, "enabled", False),
        _AUTO_BOOL_VALUES,
    )
    _require_choice(
        errors,
        "performance.compile.dynamic",
        cfg_get(compile_cfg, "dynamic", "auto"),
        _AUTO_BOOL_VALUES,
    )
    _require_choice(
        errors,
        "performance.compile.fullgraph",
        cfg_get(compile_cfg, "fullgraph", False),
        _AUTO_BOOL_VALUES - {"auto"},
    )
    _require_choice(
        errors,
        "performance.compile.mode",
        cfg_get(compile_cfg, "mode", "default"),
        {
            "auto",
            "default",
            "none",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        },
    )

    dataloader = cfg_get(perf, "dataloader", {})
    _require_choice(
        errors,
        "performance.dataloader.storage_backend",
        cfg_get(dataloader, "storage_backend", "files"),
        {"files", "lmdb"},
    )
    for name in (
        "persistent_workers",
        "persistent_workers_source",
        "persistent_workers_target",
        "persistent_workers_test",
        "limit_worker_threads",
    ):
        value = cfg_get(dataloader, name, None)
        if value is not None:
            _require_choice(
                errors,
                f"performance.dataloader.{name}",
                value,
                _AUTO_BOOL_VALUES - {"auto"},
            )
    for name in ("num_workers_source", "num_workers_target", "num_workers_test"):
        _validate_worker_value(
            errors,
            f"performance.dataloader.{name}",
            cfg_get(dataloader, name, "auto"),
        )
    for name in (
        "prefetch_factor",
        "prefetch_factor_source",
        "prefetch_factor_target",
        "prefetch_factor_test",
    ):
        value = cfg_get(dataloader, name, None)
        if value is not None:
            _require_int_at_least(errors, f"performance.dataloader.{name}", value, 1)
    _require_int_at_least(
        errors,
        "performance.dataloader.worker_threads",
        cfg_get(dataloader, "worker_threads", 1),
        1,
    )

    resume = cfg_get(cfg, "resume", {})
    _require_int_at_least(
        errors,
        "resume.save_every_epochs",
        cfg_get(resume, "save_every_epochs", 0),
        0,
    )

    distributed = cfg_get(cfg, "distributed", {})
    _require_choice(
        errors,
        "distributed.enabled",
        cfg_get(distributed, "enabled", "auto"),
        _AUTO_BOOL_VALUES,
    )
    _require_choice(
        errors,
        "distributed.backend",
        cfg_get(distributed, "backend", "auto"),
        {"auto", "nccl", "gloo"},
    )
    _require_int_at_least(
        errors,
        "distributed.timeout_seconds",
        cfg_get(distributed, "timeout_seconds", 1800),
        1,
    )
    _require_int_at_least(
        errors,
        "distributed.gradient_bucket_mb",
        cfg_get(distributed, "gradient_bucket_mb", 25),
        1,
    )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid configuration:\n{details}")
