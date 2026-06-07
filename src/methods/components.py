"""Small reusable components shared by compatible solver implementations."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

from utils import IMAGENET_MEAN, IMAGENET_STD
from utils.config import is_truthy, resolve_auto_bool


class TargetViewBuilder:
    """Build weak/strong target views using dataset or tensor-v2 augmentation."""

    def __init__(
        self,
        *,
        config,
        device: torch.device,
        to_device: Callable,
        logger: logging.Logger,
        display_name: str,
    ):
        self.device = device
        self.to_device = to_device
        self.enabled = False
        self.weak_augment = None
        self.strong_augment = None

        backend = str(
            getattr(config.method, "target_aug_backend", "dataset")
        ).strip().lower()
        perf = getattr(config, "performance", None)
        augmentation = getattr(perf, "augmentation", None) if perf is not None else None
        requested = (
            getattr(augmentation, "target_tensor_v2", "auto")
            if augmentation is not None
            else "auto"
        )
        enabled = resolve_auto_bool(
            requested,
            auto_value=(device.type == "cuda" and backend == "tensor_v2"),
        )
        if not enabled or backend != "tensor_v2":
            return
        if not is_truthy(getattr(config.method, "strong_aug", False)):
            logger.warning(
                "%s tensor_v2 augmentation requires method.strong_aug=True; "
                "using dataset transforms.",
                display_name,
            )
            return
        color_space = getattr(config.method, "color_space", None)
        if color_space is not None and is_truthy(
            getattr(color_space, "enabled", False)
        ):
            logger.warning(
                "%s tensor_v2 augmentation is incompatible with color_space; "
                "using dataset transforms.",
                display_name,
            )
            return

        target_aug = getattr(config.method, "target_aug", None)
        num_ops = (
            int(getattr(target_aug, "randaugment_num_ops", 2))
            if target_aug is not None
            else 2
        )
        magnitude = (
            int(getattr(target_aug, "randaugment_magnitude", 10))
            if target_aug is not None
            else 10
        )
        mean = list(IMAGENET_MEAN)
        std = list(IMAGENET_STD)
        self.weak_augment = transforms_v2.Compose(
            [
                transforms_v2.RandomCrop(224),
                transforms_v2.RandomHorizontalFlip(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean, std),
            ]
        )
        self.strong_augment = transforms_v2.Compose(
            [
                transforms_v2.RandomCrop(224),
                transforms_v2.RandomHorizontalFlip(),
                transforms_v2.RandAugment(
                    num_ops=num_ops,
                    magnitude=magnitude,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean, std),
            ]
        )
        self.enabled = True
        logger.info(
            "%s target tensor augmentation enabled: RandAugment(%d,%d)",
            display_name,
            num_ops,
            magnitude,
        )

    @staticmethod
    def to_uint8(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.uint8:
            return tensor
        if torch.is_floating_point(tensor):
            if tensor.max() <= 1.0 and tensor.min() >= 0.0:
                tensor = tensor * 255.0
            return tensor.round().clamp(0.0, 255.0).to(torch.uint8)
        return tensor.clamp(0, 255).to(torch.uint8)

    def prepare(self, target_images):
        if isinstance(target_images, (tuple, list)) and len(target_images) >= 2:
            return (
                self.to_device(target_images[0]),
                self.to_device(target_images[1]),
            )
        if isinstance(target_images, (tuple, list)) and len(target_images) == 1:
            target_images = target_images[0]
        if not self.enabled:
            return self.to_device(target_images), self.to_device(target_images)

        base = self.to_uint8(self.to_device(target_images))
        return self.weak_augment(base), self.strong_augment(base)


@torch.no_grad()
def update_ema_model(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    """Update teacher parameters and copy buffers from the student."""
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters()
    ):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def linear_ema_decay(
    step: int,
    total_steps: int,
    start: float,
    end: float,
) -> float:
    progress = min(1.0, step / max(1, total_steps))
    return start + (end - start) * progress
