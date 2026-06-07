from __future__ import annotations

import logging

import torch
from omegaconf import OmegaConf

from methods.components import TargetViewBuilder


def _config(*, enabled, backend="tensor_v2"):
    return OmegaConf.create(
        {
            "method": {
                "target_aug_backend": backend,
                "strong_aug": True,
                "target_aug": {
                    "randaugment_num_ops": 1,
                    "randaugment_magnitude": 5,
                },
            },
            "performance": {
                "augmentation": {"target_tensor_v2": enabled},
            },
        }
    )


def test_target_view_builder_accepts_single_view_list_when_disabled():
    builder = TargetViewBuilder(
        config=_config(enabled=False),
        device=torch.device("cpu"),
        to_device=lambda value: value,
        logger=logging.getLogger(__name__),
        display_name="test",
    )
    image = torch.rand(2, 3, 224, 224)

    weak, strong = builder.prepare([image])

    assert weak is image
    assert strong is image


def test_target_view_builder_tensor_v2_produces_normalized_views():
    builder = TargetViewBuilder(
        config=_config(enabled=True),
        device=torch.device("cpu"),
        to_device=lambda value: value,
        logger=logging.getLogger(__name__),
        display_name="test",
    )
    images = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8)

    weak, strong = builder.prepare(images)

    assert builder.enabled is True
    assert weak.shape == strong.shape == (2, 3, 224, 224)
    assert weak.dtype == strong.dtype == torch.float32
