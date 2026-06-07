"""
Shared configuration helpers.

This module keeps OmegaConf resolver registration and common config parsing
utilities in one place so the rest of the codebase can share the same rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def parse_range(range_str):
    """
    Parse a range string into a list of integers.

    Examples:
        ``"0-30"`` -> ``[0, 1, ..., 30]``
        ``"1,3,5-7"`` -> ``[1, 3, 5, 6, 7]``
    """
    result = []
    parts = str(range_str).split(",")
    for part in parts:
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result


def cfg_get(cfg: Any, key: str, default: Any):
    """Safely read a possibly nested OmegaConf-like value."""
    value = cfg.get(key, default) if hasattr(cfg, "get") else default
    return default if value is None else value


def is_truthy(value) -> bool:
    """Parse booleans consistently across bool/int/string config values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_auto_bool(value, auto_value: bool) -> bool:
    """Resolve ``auto``/bool-like config values with a caller-supplied default."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return bool(auto_value)
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return is_truthy(value)


def resolve_optional_auto_bool(value) -> bool | None:
    """Resolve a bool-like value while preserving ``auto`` as ``None``."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return is_truthy(value)


def resolve_int_or_auto(value, auto_value: int) -> int:
    """Resolve integer config values that may also be the string ``auto``."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return int(auto_value)
    return int(value)


def register_resolvers(*, src_dir: str | Path | None = None) -> None:
    """
    Register the repository's shared OmegaConf resolvers.

    ``src_dir`` is optional because some scripts only need the ``range``
    resolver, while the training entrypoint also relies on ``src_dir``.
    """
    OmegaConf.register_new_resolver("range", parse_range, replace=True)
    if src_dir is not None:
        resolved_src_dir = str(Path(src_dir).resolve())
        OmegaConf.register_new_resolver("src_dir", lambda: resolved_src_dir, replace=True)
