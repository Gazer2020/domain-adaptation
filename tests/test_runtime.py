from __future__ import annotations

import torch
from omegaconf import OmegaConf

from methods.base_solver import BaseSolver
from utils.config import resolve_optional_auto_bool
from utils.runtime import configure_torch_runtime, shutdown_dataloader_workers


class _CompileTestSolver(BaseSolver):
    def build_model(self):
        pass

    def train(self):
        pass


def test_optional_auto_bool_preserves_torch_compile_auto_mode():
    assert resolve_optional_auto_bool("auto") is None
    assert resolve_optional_auto_bool("true") is True
    assert resolve_optional_auto_bool(False) is False


def test_runtime_configures_matmul_precision_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = OmegaConf.create(
        {
            "performance": {
                "allow_tf32": True,
                "matmul_precision": "medium",
                "deterministic": False,
                "benchmark": True,
            }
        }
    )

    previous = torch.get_float32_matmul_precision()
    try:
        resolved = configure_torch_runtime(cfg)
        assert torch.get_float32_matmul_precision() == "medium"
        assert resolved["tf32_api"] == "unavailable"
        assert resolved["benchmark"] is True
    finally:
        torch.set_float32_matmul_precision(previous)


def test_compile_auto_dynamic_uses_pytorch_default(monkeypatch):
    solver = object.__new__(_CompileTestSolver)
    solver.compile_enabled = True
    solver.device = torch.device("cpu")
    solver.compile_backend = None
    solver.compile_mode = None
    solver.compile_dynamic = None
    solver.compile_fullgraph = False
    captured = {}

    def fake_compile(fn, **kwargs):
        captured.update(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    fn = lambda value: value + 1

    assert solver._compile_callable(fn, "test") is fn
    assert "dynamic" not in captured
    assert captured["fullgraph"] is False


def test_shutdown_dataloader_workers_releases_persistent_iterator():
    class _Iterator:
        def __init__(self):
            self.shutdown = False

        def _shutdown_workers(self):
            self.shutdown = True

    class _Loader:
        def __init__(self):
            self._iterator = _Iterator()

    loader = _Loader()
    iterator = loader._iterator
    shutdown_dataloader_workers([loader])

    assert iterator.shutdown is True
    assert loader._iterator is None
