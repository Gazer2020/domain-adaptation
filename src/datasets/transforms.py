"""Reusable image transforms for domain adaptation datasets."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ColorSpaceToTensorStack:
    """Convert one image into a normalized stack of color-space views."""

    def __init__(
        self,
        spaces: List[str],
        mean: List[float],
        std: List[float],
        random_erasing_p: float = 0.0,
    ):
        if not spaces:
            raise ValueError("spaces must be a non-empty list")
        self.spaces = [str(space).lower() for space in spaces]
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.random_erasing = (
            transforms.RandomErasing(p=float(random_erasing_p))
            if random_erasing_p > 0
            else None
        )

    @staticmethod
    def _convert(image: Image.Image, space: str) -> Image.Image:
        if space == "rgb":
            return image.convert("RGB")
        if space == "lab":
            return image.convert("LAB")
        if space == "hsv":
            return image.convert("HSV")
        if space == "ycbcr":
            return image.convert("YCbCr")
        if space == "yuv":
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            y = 0.299 * red + 0.587 * green + 0.114 * blue
            u = -0.14713 * red - 0.28886 * green + 0.436 * blue + 128.0
            v = 0.615 * red - 0.51499 * green - 0.10001 * blue + 128.0
            yuv = np.clip(np.stack([y, u, v], axis=-1), 0.0, 255.0)
            return Image.fromarray(yuv.astype(np.uint8), mode="RGB")
        if space == "gray":
            return image.convert("L").convert("RGB")
        raise ValueError(f"Unsupported color space: {space}")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        views = []
        for space in self.spaces:
            tensor = self.to_tensor(self._convert(image, space))
            tensor = self.normalize(tensor)
            if self.random_erasing is not None:
                tensor = self.random_erasing(tensor)
            views.append(tensor)
        return torch.stack(views, dim=0)


class WeakStrongAugment:
    """Return weak and strong augmented views of the same input."""

    def __init__(self, weak, strong):
        self.weak = weak
        self.strong = strong

    def __call__(self, image):
        return self.weak(image), self.strong(image)
