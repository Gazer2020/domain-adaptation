"""
Kinematic heads for RVTC. Task logits come from the orthodox ResNet fc, not from this module.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
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


class KinematicHead(nn.Module):
    """Legacy: 16-block trajectory + GRU (kept for optional reuse)."""

    def __init__(
        self,
        channel_dims: Sequence[int],
        traj_dim: int,
        kin_hidden: int = 256,
    ):
        super().__init__()
        if len(channel_dims) != 16:
            raise ValueError(f"Expected 16 blocks, got {len(channel_dims)}")
        self.traj_dim = traj_dim
        self.projs = nn.ModuleList([nn.Linear(c, traj_dim) for c in channel_dims])
        self.kin_gru = nn.GRU(traj_dim, kin_hidden, batch_first=True, num_layers=1)
        self.kin_fc = nn.Linear(kin_hidden, traj_dim)

    def velocities_to_trajectory(self, velocities: List[Tensor]) -> Tensor:
        outs: List[Tensor] = []
        for v, proj in zip(velocities, self.projs):
            g = F.adaptive_avg_pool2d(v, 1).flatten(1)
            g = F.normalize(g, dim=1, eps=1e-6)
            u = proj(g)
            u = F.normalize(u, dim=1, eps=1e-6)
            outs.append(u)
        return torch.stack(outs, dim=1)

    def next_state_loss(self, v: Tensor) -> Tensor:
        seq_in = v[:, :-1, :]
        gru_out, _ = self.kin_gru(seq_in)
        pred = self.kin_fc(gru_out)
        pred = F.normalize(pred, dim=-1, eps=1e-6)
        tgt = v[:, 1:16, :]
        return (1.0 - (pred * tgt).sum(dim=-1)).mean()

    def alignment_loss(self, v_src: Tensor, v_tgt: Tensor) -> Tensor:
        ms = v_src.mean(dim=0)
        mt = v_tgt.mean(dim=0)
        return F.mse_loss(ms, mt)


RVTCHead = KinematicHead
