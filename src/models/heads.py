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
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_features: Number of input features
            num_classes: Number of semantic classes
            hidden_dim: Hidden layer dimension (default: 256)
            dropout: Dropout probability for MLP blocks (default: 0.1)
        """
        super().__init__()
        self.num_classes = num_classes

        self.pre_norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.shortcut = (
            nn.Identity() if in_features == hidden_dim else nn.Linear(in_features, hidden_dim, bias=False)
        )
        self.classifier = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feat: Input features
            
        Returns:
            Class logits
        """
        x = self.pre_norm(feat)
        residual = self.shortcut(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = x + residual
        return self.classifier(x)


class ChannelGatingModule(nn.Module):
    """
    Channel Gating Module for feature recalibration.
    
    A simple FC network with Sigmoid that outputs gate values 
    in (0, 1) for each feature dimension. Used to learn class-consistent
    channel activation patterns for open set domain adaptation.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = None):
        """
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden layer dimension (default: feature_dim // 4)
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        if hidden_dim is None:
            hidden_dim = feature_dim // 4
        
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),  # Output in (0, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute gate values for input features.
        
        Args:
            features: [B, D] feature vectors
            
        Returns:
            gate: [B, D] gate values in (0, 1)
        """
        return self.gate(features)
