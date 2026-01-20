"""
Classification heads and auxiliary networks for domain adaptation.

These modules can be used by various DA methods for specialized tasks
like rotation prediction, semantic classification, etc.
"""

import torch
import torch.nn as nn


class RotationHead(nn.Module):
    """
    Rotation prediction head for self-supervised learning.
    
    Takes two feature vectors (original and rotated) and predicts
    the rotation angle class (0°, 90°, 180°, 270°).
    """
    
    def __init__(self, in_features: int, num_classes: int = 4):
        """
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of rotation classes (default: 4)
        """
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feat1: Original image features
            feat2: Rotated image features
            
        Returns:
            Rotation class logits
        """
        x = torch.cat((feat1, feat2), dim=1)
        return self.classifier(x)


class SemanticHead(nn.Module):
    """
    Semantic classification head.
    
    Simple MLP classifier for semantic (class) predictions.
    """
    
    def __init__(self, in_features: int, num_classes: int):
        """
        Args:
            in_features: Number of input features
            num_classes: Number of semantic classes
        """
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feat: Input features
            
        Returns:
            Class logits
        """
        return self.classifier(feat)


class DomainHead(nn.Module):
    """
    Domain discriminator head for adversarial domain adaptation.
    
    Used in methods like DANN, CDAN, etc.
    """
    
    def __init__(self, in_features: int, hidden_dim: int = 256):
        """
        Args:
            in_features: Number of input features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feat: Input features
            
        Returns:
            Domain prediction logits
        """
        return self.discriminator(feat)


class ChannelSelector(nn.Module):
    """
    Learnable channel selection/weighting module.
    
    Uses SE-style (Squeeze-and-Excitation) attention mechanism to learn
    which channels are most discriminative for separating known vs unknown.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Args:
            in_channels: Number of input channels (e.g., 2048 for ResNet50)
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        self.in_channels = in_channels
        
        # SE-style channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, channel_acts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            channel_acts: Channel activations [B, C, H, W]
            
        Returns:
            Channel weights [B, C] in range (0, 1)
        """
        return self.attention(channel_acts)

