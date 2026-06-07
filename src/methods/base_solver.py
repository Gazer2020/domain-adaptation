"""
Base solver class for domain adaptation methods.

All domain adaptation methods should inherit from BaseSolver and implement
the required abstract methods: build_model() and train().
"""

import io
import logging
import random
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from models.backbones import get_backbone
from utils import (
    average_module_buffers,
    broadcast_modules,
    distributed_barrier,
    gather_objects_to_main,
    get_device,
    get_distributed_context,
    synchronize_optimizer_gradients,
)
from utils.config import cfg_get, is_truthy, resolve_auto_bool, resolve_optional_auto_bool

from methods.registry import register_solver


logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """
    Abstract base solver class for domain adaptation.
    
    Subclasses must implement:
        - build_model(): Setup model architecture
        - train(): Full training procedure
        
    Subclasses may optionally override:
        - forward_for_eval(): Customize inference logic
        - evaluate(): Customize evaluation metrics
        - save_checkpoint() / load_checkpoint(): Customize checkpointing
    """

    def __init__(self, config, loaders: Tuple[DataLoader, DataLoader, DataLoader], 
                 class_info: dict = None):
        """
        Initialize the solver with config and data loaders.

        Args:
            config: OmegaConf configuration object
            loaders: Tuple of (source_loader, target_loader, target_test_loader)
            class_info: Dict containing class metadata for OSDA handling:
                - src_classes: List of source class indices
                - tgt_classes: List of target class indices
                - shared_classes: List of shared class indices
                - num_classes: Number of classifier output classes
                - unknown_label: Label for unknown classes (None for CSDA)
                - setting: DA setting string
        """
        self.config = config
        self.source_loader, self.target_loader, self.target_test_loader = loaders
        
        # Store class info for OSDA handling
        self.class_info = class_info if class_info else {}
        
        # Setup device (auto-detect if needed)
        device_str = get_device(config.device)
        self.device = torch.device(device_str)
        logger.info(f"Using device: {self.device}")
        self._setup_performance_runtime()
        
        # Setup number of classes based on setting
        self._setup_num_classes()
        
        # Build model (must be implemented by subclass)
        self.build_model()
        self.distributed = get_distributed_context()
        broadcast_modules(self._solver_modules())
        
        # Default loss function (can be overridden or unused)
        self.criterion = nn.CrossEntropyLoss()
        self._save_start_epoch = int(self.config.method.get("save_start_epoch", 10))
        self._best_metric = float("-inf")
        exp_name = str(self.config.get("exp_name", "experiment"))
        self._best_ckpt_path = Path("checkpoints") / f"{exp_name}.pth"
        self._best_saved = False
        self._training_state_objects: dict[str, Any] = {}
        self._pending_training_state: dict[str, Any] = {}
        self._resume_epoch = 0
        self._training_global_step = 0

        resume_cfg = self._cfg_get(self.config, "resume", {})
        resume_path = self._cfg_get(resume_cfg, "path", None)
        self._resume_path = (
            None
            if resume_path is None or str(resume_path).strip().lower() in {"", "none"}
            else Path(str(resume_path))
        )
        save_path = self._cfg_get(resume_cfg, "save_path", "auto")
        self._training_ckpt_path = (
            Path("checkpoints") / f"{exp_name}.resume.pth"
            if str(save_path).strip().lower() in {"", "auto"}
            else Path(str(save_path))
        )
        self._training_save_every = max(
            0,
            int(self._cfg_get(resume_cfg, "save_every_epochs", 0)),
        )
        if self._resume_path is not None:
            self._load_training_checkpoint(self._resume_path)

    @property
    def solver_name(self) -> str:
        method_cfg = getattr(self.config, "method", None)
        configured_name = getattr(method_cfg, "name", None) if method_cfg is not None else None
        if configured_name:
            return str(configured_name).strip().lower()
        return self._solver_display_name().lower()

    def _solver_display_name(self) -> str:
        class_name = self.__class__.__name__
        if class_name.endswith("Solver"):
            return class_name[:-6]
        return class_name

    @staticmethod
    def _cfg_get(cfg, key, default):
        return cfg_get(cfg, key, default)

    @staticmethod
    def _is_truthy(value) -> bool:
        return is_truthy(value)

    @classmethod
    def _resolve_auto_bool(cls, value, auto_value: bool) -> bool:
        return resolve_auto_bool(value, auto_value)

    def _setup_performance_runtime(self):
        perf = self._cfg_get(self.config, "performance", {})
        amp_cfg = self._cfg_get(perf, "amp", {})
        compile_cfg = self._cfg_get(perf, "compile", {})

        self.non_blocking_transfer = self._is_truthy(self._cfg_get(perf, "non_blocking_transfer", True)) and self.device.type == "cuda"
        self.zero_grad_set_to_none = self._is_truthy(self._cfg_get(perf, "zero_grad_set_to_none", True))
        self.channels_last = self._resolve_auto_bool(self._cfg_get(perf, "channels_last", False), auto_value=False) and self.device.type == "cuda"

        amp_enabled_cfg = str(self._cfg_get(amp_cfg, "enabled", "auto")).lower()
        if amp_enabled_cfg == "auto":
            self.amp_enabled = self.device.type == "cuda"
        else:
            self.amp_enabled = amp_enabled_cfg in {"1", "true", "yes", "on"}

        amp_dtype_cfg = str(self._cfg_get(amp_cfg, "dtype", "bf16")).lower()
        if amp_dtype_cfg in {"fp16", "float16", "half"}:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.bfloat16

        self.use_grad_scaler = self.amp_enabled and self.device.type == "cuda" and self.amp_dtype == torch.float16
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        self._amp_probe_done = False

        compile_enabled_cfg = str(self._cfg_get(compile_cfg, "enabled", "false")).lower()
        if compile_enabled_cfg == "auto":
            self.compile_enabled = self.device.type == "cuda"
        else:
            self.compile_enabled = compile_enabled_cfg in {"1", "true", "yes", "on"}
        compile_backend_cfg = str(self._cfg_get(compile_cfg, "backend", "inductor")).strip().lower()
        if compile_backend_cfg in {"", "none", "default", "auto"}:
            self.compile_backend = None
        else:
            self.compile_backend = compile_backend_cfg
        compile_mode_cfg = str(self._cfg_get(compile_cfg, "mode", "default")).strip().lower()
        if compile_mode_cfg in {"", "none", "default", "auto"}:
            self.compile_mode = None
        else:
            self.compile_mode = compile_mode_cfg
        self.compile_dynamic = resolve_optional_auto_bool(
            self._cfg_get(compile_cfg, "dynamic", "auto")
        )
        self.compile_fullgraph = self._is_truthy(self._cfg_get(compile_cfg, "fullgraph", False))
        logger.info(
            "Performance runtime | amp=%s dtype=%s non_blocking=%s set_to_none=%s channels_last=%s compile=%s",
            self.amp_enabled,
            str(self.amp_dtype).replace("torch.", ""),
            self.non_blocking_transfer,
            self.zero_grad_set_to_none,
            self.channels_last,
            self.compile_enabled,
        )

    def _auto_cast(self):
        if self.amp_enabled:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return nullcontext()

    def _compile_callable(self, fn: Callable[..., Any], name: str) -> Callable[..., Any]:
        if not self.compile_enabled:
            return fn
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile requested for %s but is unavailable; fallback to eager mode.", name)
            return fn
        if self.device.type == "mps":
            logger.warning("torch.compile requested for %s on MPS; fallback to eager mode.", name)
            return fn

        compile_kwargs = {"fullgraph": self.compile_fullgraph}
        if self.compile_dynamic is not None:
            compile_kwargs["dynamic"] = self.compile_dynamic
        if self.compile_backend is not None:
            compile_kwargs["backend"] = self.compile_backend
        if self.compile_mode is not None:
            compile_kwargs["mode"] = self.compile_mode

        try:
            compiled_fn = torch.compile(fn, **compile_kwargs)
            logger.info(
                "torch.compile enabled for %s | backend=%s mode=%s dynamic=%s fullgraph=%s",
                name,
                self.compile_backend if self.compile_backend is not None else "default",
                self.compile_mode if self.compile_mode is not None else "default",
                self.compile_dynamic,
                self.compile_fullgraph,
            )
            return compiled_fn
        except Exception as e:
            logger.warning("torch.compile unavailable for %s, fallback to eager mode: %s", name, e)
            return fn

    def _compile_module(self, module: nn.Module, name: str) -> Callable[..., Any]:
        return self._compile_callable(module, name)

    def _probe_amp_tensor(
        self,
        tensor: torch.Tensor,
        name: str = "tensor",
        *,
        warn_on_float32: bool = True,
    ):
        if self._amp_probe_done or (not torch.is_tensor(tensor)):
            return
        self._amp_probe_done = True
        if not self.amp_enabled:
            return

        dtype_str = str(tensor.dtype).replace("torch.", "")
        logger.info("AMP probe | %s dtype=%s", name, dtype_str)
        if self.device.type == "cuda" and warn_on_float32 and tensor.dtype == torch.float32:
            logger.warning(
                "AMP is enabled but %s remains float32. Check autocast coverage and operator support.",
                name,
            )

    def _to_device(self, x):
        if torch.is_tensor(x):
            memory_format = torch.channels_last if self.channels_last and x.ndim == 4 else torch.preserve_format
            return x.to(
                self.device,
                non_blocking=self.non_blocking_transfer,
                memory_format=memory_format,
            )
        if isinstance(x, (list, tuple)):
            converted = [self._to_device(v) for v in x]
            return type(x)(converted)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        return x

    def _zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)

    def _backward(self, loss: torch.Tensor):
        if self.use_grad_scaler:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, optimizer, *, synchronize: bool = True):
        self._register_automatic_optimizer(optimizer)
        if synchronize:
            synchronize_optimizer_gradients(optimizer)
        if self.use_grad_scaler:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()
        self._training_global_step += 1

    def _optimizer_step_with_optional_clip(self, loss, optimizer, clip_params=None, clip_max_norm=None):
        self._backward(loss)
        if clip_params is not None and clip_max_norm is not None:
            if self.use_grad_scaler:
                self.grad_scaler.unscale_(optimizer)
            synchronize_optimizer_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=clip_max_norm)
            self._optimizer_step(optimizer, synchronize=False)
            return
        self._optimizer_step(optimizer)

    @staticmethod
    def _format_log_value(value: Any) -> str:
        fmt = None
        if isinstance(value, tuple) and len(value) == 2:
            value, fmt = value

        if torch.is_tensor(value):
            if value.ndim == 0:
                value = value.item()
            else:
                return str(value)

        if isinstance(value, float):
            return format(value, fmt or ".4f")
        if isinstance(value, int):
            return format(value, fmt or "d")
        return str(value)

    def _format_log_fields(self, fields: Mapping[str, Any] | None) -> str:
        if not fields:
            return ""
        return " ".join(
            f"{name}={self._format_log_value(value)}"
            for name, value in fields.items()
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        *,
        metrics: Mapping[str, Any] | None = None,
        extras: Mapping[str, Any] | None = None,
        score: float | None = None,
        best_score: float | None = None,
        score_name: str = "Acc",
        prefix: str | None = None,
    ):
        parts = [f"{prefix or self._solver_display_name()} {epoch}/{total_epochs}"]
        metric_text = self._format_log_fields(metrics)
        if metric_text:
            parts.append(f"| {metric_text}")
        extra_text = self._format_log_fields(extras)
        if extra_text:
            parts.append(f"| {extra_text}")
        if score is not None:
            score_text = f"{score_name}={score:.2f}%"
            if best_score is not None:
                score_text += f" (best={best_score:.2f}%)"
            parts.append(f"| {score_text}")
        logger.info(" ".join(parts))

    def _load_checkpoint_file(self, path):
        load_kwargs = {"map_location": self.device}
        try:
            return torch.load(path, weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(path, **load_kwargs)

    def _build_checkpoint_payload(
        self,
        *,
        modules: Mapping[str, nn.Module] | None = None,
        extra_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {"method": self.solver_name}
        for name, module in (modules or {}).items():
            payload[name] = module.state_dict()
        if extra_state:
            payload.update(extra_state)
        return payload

    def register_training_state(self, **objects: Any) -> None:
        """Register optimizer/scheduler-like objects for resumable training."""
        for name, value in objects.items():
            if value is None:
                continue
            if not hasattr(value, "state_dict") or not hasattr(value, "load_state_dict"):
                raise TypeError(
                    f"Training state object '{name}' must implement state_dict/load_state_dict"
                )
            self._training_state_objects[name] = value
            pending = self._pending_training_state.pop(name, None)
            if pending is not None:
                value.load_state_dict(pending)
                logger.info("Restored training state object: %s", name)

    def _register_automatic_optimizer(self, optimizer: Any) -> None:
        if any(value is optimizer for value in self._training_state_objects.values()):
            return
        index = sum(name.startswith("optimizer_") for name in self._training_state_objects)
        self.register_training_state(**{f"optimizer_{index}": optimizer})

    def _capture_model_checkpoint(self) -> bytes:
        buffer = io.BytesIO()
        self.save_checkpoint(buffer)
        return buffer.getvalue()

    def _restore_model_checkpoint(self, payload: bytes) -> None:
        self.load_checkpoint(io.BytesIO(payload))

    @staticmethod
    def _rng_state_dict() -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _load_rng_state(state: Mapping[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def extra_training_state_dict(self) -> dict[str, Any]:
        """Hook for method-specific non-module state."""
        return {}

    def load_extra_training_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore method-specific non-module state."""

    def save_training_checkpoint(
        self,
        path,
        *,
        epoch: int,
        global_step: int | None = None,
    ) -> None:
        context = get_distributed_context()
        rng_by_rank = gather_objects_to_main(self._rng_state_dict())
        if not context.is_main_process:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "method": self.solver_name,
            "epoch": int(epoch),
            "global_step": int(
                self._training_global_step if global_step is None else global_step
            ),
            "best_metric": float(self._best_metric),
            "model_checkpoint": self._capture_model_checkpoint(),
            "training_objects": {
                name: value.state_dict()
                for name, value in self._training_state_objects.items()
            },
            "grad_scaler": self.grad_scaler.state_dict(),
            "rng": rng_by_rank[0],
            "rng_by_rank": rng_by_rank,
            "extra_state": self.extra_training_state_dict(),
        }
        torch.save(payload, path)
        logger.info("Training checkpoint saved to %s", path)

    def _load_training_checkpoint(self, path) -> None:
        path = Path(path)
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("method") != self.solver_name:
            raise ValueError(
                f"Resume checkpoint method={checkpoint.get('method')!r} does not match "
                f"solver={self.solver_name!r}"
            )
        self._restore_model_checkpoint(checkpoint["model_checkpoint"])
        self._resume_epoch = int(checkpoint.get("epoch", 0))
        self._training_global_step = int(checkpoint.get("global_step", 0))
        self._best_metric = float(checkpoint.get("best_metric", float("-inf")))
        self._pending_training_state = dict(checkpoint.get("training_objects", {}))
        grad_scaler_state = checkpoint.get("grad_scaler")
        if grad_scaler_state:
            self.grad_scaler.load_state_dict(grad_scaler_state)
        rng_by_rank = checkpoint.get("rng_by_rank")
        if rng_by_rank:
            rank = min(get_distributed_context().rank, len(rng_by_rank) - 1)
            self._load_rng_state(rng_by_rank[rank])
        else:
            self._load_rng_state(checkpoint.get("rng", {}))
        self.load_extra_training_state_dict(checkpoint.get("extra_state", {}))
        logger.info(
            "Training checkpoint loaded from %s | epoch=%d global_step=%d pending=%s",
            path,
            self._resume_epoch,
            self._training_global_step,
            sorted(self._pending_training_state),
        )

    def _epoch_range(self, total_epochs: int, *, offset: int = 0):
        """Return the remaining local epochs after a resume checkpoint."""
        local_start = max(0, self._resume_epoch - int(offset))
        for local_epoch in range(
            min(local_start, int(total_epochs)),
            int(total_epochs),
        ):
            self._set_dataloader_epoch(int(offset) + local_epoch)
            yield local_epoch

    def _set_dataloader_epoch(self, epoch: int) -> None:
        for loader in (
            self.source_loader,
            self.target_loader,
            self.target_test_loader,
        ):
            sampler = getattr(loader, "sampler", None)
            batch_sampler = getattr(loader, "batch_sampler", None)
            for candidate in (sampler, batch_sampler):
                set_epoch = getattr(candidate, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(int(epoch))

    def _solver_modules(self) -> list[nn.Module]:
        modules = []
        seen: set[int] = set()
        for value in self.__dict__.values():
            if isinstance(value, nn.Module) and id(value) not in seen:
                seen.add(id(value))
                modules.append(value)
        return modules

    def _synchronize_model_buffers(self) -> None:
        average_module_buffers(self._solver_modules())

    def _maybe_save_training_checkpoint(self, epoch: int) -> bool:
        if self._training_save_every <= 0:
            return False
        if int(epoch) % self._training_save_every != 0:
            return False
        self.save_training_checkpoint(self._training_ckpt_path, epoch=int(epoch))
        return True

    def _save_named_modules_checkpoint(
        self,
        path,
        *,
        modules: Mapping[str, nn.Module],
        extra_state: Mapping[str, Any] | None = None,
    ):
        payload = self._build_checkpoint_payload(modules=modules, extra_state=extra_state)
        torch.save(payload, path)
        logger.info("%s checkpoint saved to %s", self._solver_display_name(), path)

    def _load_named_modules_checkpoint(
        self,
        path,
        *,
        modules: Mapping[str, nn.Module],
        strict: bool = True,
        fallback_key: str | None = None,
    ):
        checkpoint = self._load_checkpoint_file(path)

        if isinstance(checkpoint, dict):
            for name, module in modules.items():
                state_dict = None
                if name in checkpoint:
                    state_dict = checkpoint[name]
                elif len(modules) == 1:
                    if fallback_key is not None and fallback_key in checkpoint:
                        state_dict = checkpoint[fallback_key]
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    else:
                        state_dict = checkpoint
                if state_dict is None:
                    raise ValueError(f"Checkpoint '{path}' is missing key '{name}'.")
                module.load_state_dict(state_dict, strict=strict)
        else:
            if len(modules) != 1:
                raise ValueError(
                    f"Checkpoint '{path}' does not contain named module states required by {self._solver_display_name()}."
                )
            next(iter(modules.values())).load_state_dict(checkpoint, strict=strict)

        logger.info("%s checkpoint loaded from %s", self._solver_display_name(), path)
        return checkpoint

    def _log_best_checkpoint_loaded(self, metric_name: str = "Score"):
        logger.info(
            "Loaded best %s checkpoint from %s with %s=%.2f%%",
            self._solver_display_name(),
            self._best_ckpt_path,
            metric_name,
            self._best_metric,
        )

    def _log_training_complete(self, *, best_score: float | None = None, score_name: str = "Acc"):
        if best_score is None:
            logger.info("%s training finished.", self._solver_display_name())
            return
        logger.info(
            "%s training finished. Best %s=%.2f%%",
            self._solver_display_name(),
            score_name,
            best_score,
        )

    def _setup_num_classes(self):
        """
        Setup number of classes based on class_info or config.
        
        For OSDA/UniDA: num_classes = len(src_classes) + 1 (includes unknown class)
        For CSDA/PDA: num_classes = len(src_classes)
        """
        if self.class_info and "num_classes" in self.class_info:
            base_num_classes = self.class_info["num_classes"]
            self.unknown_label = self.class_info.get("unknown_label")
            self.shared_classes = self.class_info.get("shared_classes", [])
            self.setting = self.class_info.get("setting", "csda")
            
            # For OSDA/UniDA, add 1 class for unknown
            if self.setting in ["osda", "unida"] and self.unknown_label is not None:
                self.num_classes = base_num_classes + 1
            else:
                self.num_classes = base_num_classes
        else:
            # Fallback for backward compatibility
            self.setting = self.config.method.get("setting", "csda")
            self.num_classes = self.config.dataset.num_classes
            self.unknown_label = None
            self.shared_classes = []
        
        # Unknown rejection threshold (for confidence-based rejection)
        self.unknown_threshold = self.config.method.get("unknown_threshold", 0.5)

    @abstractmethod
    def build_model(self):
        """
        Build the network architecture.
        
        Must be implemented by subclasses.
        Should set self.net or appropriate model attributes.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Execute the full training procedure.
        
        Must be implemented by subclasses.
        Each method defines its own training loop, optimizer, and losses.
        """
        pass

    def _set_train_mode(self):
        """Set model to training mode. Override for multi-component models."""
        if hasattr(self, 'net'):
            self.net.train()

    def _set_eval_mode(self):
        """Set model to evaluation mode. Override for multi-component models."""
        if hasattr(self, 'net'):
            self.net.eval()

    def forward_for_eval(self, imgs):
        """
        Forward pass for evaluation.
        
        Override this if your model has a different inference path.
        
        Args:
            imgs: Input images
            
        Returns:
            outputs: Model outputs (logits)
        """
        if hasattr(self, 'net'):
            return self.net(imgs)
        raise NotImplementedError(
            "Subclass must either set self.net or override forward_for_eval()"
        )

    def evaluate(self):
        """
        Evaluate on target test set.
        
        For OSDA/UniDA settings, computes:
        - Known Accuracy (OS*): Accuracy on shared classes
        - Unknown Accuracy: Rate of predicting unknown for target-private classes
        - H-score: Harmonic mean of known and unknown accuracy
        
        For CSDA, computes standard accuracy.
        
        Returns:
            acc: Overall accuracy (or H-score for OSDA)
        """
        self._set_eval_mode()
        self._synchronize_model_buffers()
        
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.inference_mode():
            for imgs, labels in self.target_test_loader:
                imgs = self._to_device(imgs)
                with self._auto_cast():
                    outputs = self.forward_for_eval(imgs)
                
                probs = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, dim=1)
                
                all_preds.append(predicted.cpu())
                all_labels.append(labels)
                all_probs.append(max_probs.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            # Apply rejection mechanism
            final_preds = self.predict_with_rejection(all_preds, all_probs)
            return self._compute_osda_metrics(final_preds, all_labels)
        else:
            correct = (all_preds == all_labels).sum().item()
            total = len(all_labels)
            acc = 100 * correct / total if total > 0 else 0
            return acc
    
    def predict_with_rejection(self, preds: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Apply unknown class rejection strategy.
        
        Default implementation uses confidence thresholding.
        Subclasses can override this for custom rejection methods (e.g. entropy, extensive/proto).
        
        Args:
            preds: Base predictions [N]
            probs: Confidence scores [N] or [N, C]
            
        Returns:
            final_preds: Predictions with unknown label assigned to rejected samples
        """
        # If probs is [N, C], take max
        if probs.ndim > 1:
            probs, _ = torch.max(probs, dim=1)
            
        rejected_mask = probs < self.unknown_threshold
        final_preds = preds.clone()
        final_preds[rejected_mask] = self.unknown_label
        
        return final_preds

    def _compute_osda_metrics(self, preds, labels):
        """
        Compute OSDA metrics: Known Accuracy, Unknown Accuracy, and H-score.
        
        Args:
            preds: Predictions (already processed with rejection)
            labels: Ground truth labels
        """
        unknown_label = self.unknown_label
        
        # preds already have rejection applied
        preds_with_rejection = preds

        
        known_mask = labels != unknown_label
        unknown_mask = labels == unknown_label
        
        # Known accuracy
        if known_mask.sum() > 0:
            known_preds = preds_with_rejection[known_mask]
            known_labels = labels[known_mask]
            known_correct = (known_preds == known_labels).sum().item()
            known_total = known_mask.sum().item()
            known_acc = known_correct / known_total
        else:
            known_acc = 0.0
        
        # Unknown accuracy
        if unknown_mask.sum() > 0:
            unknown_preds = preds_with_rejection[unknown_mask]
            unknown_correct = (unknown_preds == unknown_label).sum().item()
            unknown_total = unknown_mask.sum().item()
            unknown_acc = unknown_correct / unknown_total
        else:
            unknown_acc = 0.0
        
        # H-score
        if known_acc + unknown_acc > 0:
            hscore = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        else:
            hscore = 0.0
        
        logger.info(
            f"OSDA Metrics - Known Acc: {100*known_acc:.2f}%, "
            f"Unknown Acc: {100*unknown_acc:.2f}%, H-score: {100*hscore:.2f}%"
        )
        
        return 100 * hscore

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Override if you have multiple components to save.
        """
        if hasattr(self, 'net'):
            self._save_named_modules_checkpoint(path, modules={"model": self.net})
        else:
            raise NotImplementedError(
                "Subclass must either set self.net or override save_checkpoint()"
            )

    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        
        Override if you have multiple components to load.
        """
        if hasattr(self, 'net'):
            self._load_named_modules_checkpoint(
                path,
                modules={"model": self.net},
                strict=True,
                fallback_key="model",
            )
        else:
            raise NotImplementedError(
                "Subclass must either set self.net or override load_checkpoint()"
            )

    def _maybe_save_best(self, metric: float, epoch: int) -> bool:
        """
        Save checkpoint only when:
        - epoch >= self._save_start_epoch
        - metric strictly improves
        """
        saved_best = False
        if (
            int(epoch) >= int(self._save_start_epoch)
            and float(metric) > float(self._best_metric)
        ):
            self._best_metric = float(metric)
            if get_distributed_context().is_main_process:
                self._best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(self._best_ckpt_path)
            self._best_saved = True
            saved_best = True
        self._maybe_save_training_checkpoint(epoch)
        distributed_barrier()
        return saved_best

    def _load_best_checkpoint_if_available(self) -> bool:
        if self._best_saved:
            if get_distributed_context().is_main_process:
                if not self._best_ckpt_path.exists():
                    return False
                self.load_checkpoint(self._best_ckpt_path)
            distributed_barrier()
            broadcast_modules(self._solver_modules())
            return True
        return False


@register_solver("sourceonly")
class SourceOnlySolver(BaseSolver):
    """
    Source-only baseline solver.
    
    Trains only on source domain data without any domain adaptation.
    """

    def build_model(self):
        """Build a simple classification network."""
        backbone = get_backbone(self.config.method.get("backbone", "resnet18"))
        
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Linear(backbone.fc.in_features, self.num_classes)
        
        self.net = backbone.to(self.device)

    def train(self):
        """Train on source domain only."""
        import torch.optim as optim
        from utils import AverageMeter
        
        max_epochs = self.config.method.epochs
        lr = self.config.method.lr
        
        optimizer = optim.SGD(
            self.net.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.register_training_state(optimizer=optimizer)
        
        logger.info("%s training | epochs=%d", self._solver_display_name(), max_epochs)
        best_acc = self._best_metric
        
        for epoch in self._epoch_range(max_epochs):
            self.net.train()
            loss_meter = AverageMeter()
            
            for batch in self.source_loader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    src_imgs, src_labels = batch[0], batch[1]
                else:
                    raise ValueError("Source-only solver expects source batches to provide at least images and labels")
                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)
                
                self._zero_grad(optimizer)
                with self._auto_cast():
                    logits = self.net(src_imgs)
                    loss = self.criterion(logits, src_labels)
                self._optimizer_step_with_optional_clip(loss, optimizer)
                
                loss_meter.update(loss.item())
            
            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            self._maybe_save_best(acc, epoch + 1)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics={"loss": loss_meter.avg},
                score=acc,
                best_score=best_acc,
                score_name="Acc",
            )
        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")
