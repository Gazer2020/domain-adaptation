import torch
import torch.nn as nn
import torchvision.models as models

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Layer.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        attn = self.fc(y) # [B, C]
        self.last_attn_weights = attn
        
        y = attn.view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNetAttnFingerprint(nn.Module):
    def __init__(self, num_classes=31, reduction=16, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Standard layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Wrapper for SE Blocks
        # ResNet50: layer3 out=1024, layer4 out=2048
        self.se3 = SELayer(1024, reduction=reduction)
        self.se4 = SELayer(2048, reduction=reduction)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dim
        self.feature_dim = 2048
        
        # Classifier
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.se3(x)
        attn3 = self.se3.last_attn_weights # [B, 1024]
        
        x = self.layer4(x)
        x = self.se4(x)
        attn4 = self.se4.last_attn_weights # [B, 2048]
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1) # [B, 2048]
        
        logits = self.fc(features)
        
        attn_fingerprint = torch.cat([attn3, attn4], dim=1) # [B, 3072]
        
        return logits, features, attn_fingerprint

# Helper to build it
def resnet50_attn_fingerprint(**kwargs):
    return ResNetAttnFingerprint(**kwargs)
