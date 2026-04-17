"""
Models package for domain adaptation.

Includes backbones, classification heads, and auxiliary networks.
"""

from models.backbones import get_backbone, get_resnet18, get_resnet50, get_resnet101
from models.heads import RotationHead, SemanticHead, ChannelGatingModule

__all__ = [
    "get_backbone",
    "get_resnet18",
    "get_resnet50",
    "get_resnet101",
    "RotationHead",
    "SemanticHead",
    "ChannelGatingModule",
]
