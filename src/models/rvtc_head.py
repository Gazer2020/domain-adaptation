"""
Kinematic heads for RVTC. Task logits come from the orthodox ResNet fc, not from this module.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KinematicHeadV2(nn.Module):
    """
    Predicts layer4 pooled direction from layer3 pooled direction (1024 -> 2048 for ResNet-50).
    Output is L2-normalized to match v4_dir.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 2048, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, v3_dir: Tensor) -> Tensor:
        out = self.net(v3_dir)
        return F.normalize(out, p=2, dim=1, eps=1e-6)
