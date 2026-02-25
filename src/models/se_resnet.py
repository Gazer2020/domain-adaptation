"""
SE-ResNet: ResNet with Squeeze-and-Excitation Channel Attention.

This module provides SE-enhanced ResNet variants for the MIC-SimSiam framework.
Key features:
- SELayer: Channel attention module (squeeze-excitation)
- SEBottleneck: ResNet Bottleneck with SE attention
- build_se_resnet50: Load pretrained ResNet50, freeze Layer 1-2, 
                     replace Layer 3-4 with SE-enhanced blocks
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Layer.
    
    Computes channel-wise attention weights via:
    1. Global Average Pooling (squeeze)
    2. FC -> ReLU -> FC -> Sigmoid (excitation)
    3. Channel-wise multiplication (scale)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input/output channels
            reduction: Reduction ratio for the bottleneck FC layer
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            Attention-weighted feature map [B, C, H, W]
        """
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        
        # Store SE weights for sparsity regularization (detached)
        self.last_se_weights = y.detach().squeeze()  # [B, C]
        
        # Scale: channel-wise multiplication
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    """
    ResNet Bottleneck block with SE attention at the end.
    
    Structure: conv1x1 -> conv3x3 -> conv1x1 -> SE -> residual add -> ReLU
    
    This replaces the standard Bottleneck in Layer 3/4 of ResNet50.
    """
    
    expansion = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: nn.Module = None,
        reduction: int = 16
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.0)) * groups
        
        # Standard Bottleneck layers
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, bias=False, dilation=dilation
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        # SE attention layer
        self.se = SELayer(planes * self.expansion, reduction)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Bottleneck forward
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # SE attention (before residual add)
        out = self.se(out)
        
        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


def _copy_bottleneck_weights(src_block: Bottleneck, dst_block: SEBottleneck):
    """
    Copy weights from standard Bottleneck to SEBottleneck.
    
    The SE layer is left with its random initialization.
    """
    # Copy conv and bn weights
    dst_block.conv1.load_state_dict(src_block.conv1.state_dict())
    dst_block.bn1.load_state_dict(src_block.bn1.state_dict())
    dst_block.conv2.load_state_dict(src_block.conv2.state_dict())
    dst_block.bn2.load_state_dict(src_block.bn2.state_dict())
    dst_block.conv3.load_state_dict(src_block.conv3.state_dict())
    dst_block.bn3.load_state_dict(src_block.bn3.state_dict())
    
    # Copy downsample if exists
    if src_block.downsample is not None and dst_block.downsample is not None:
        dst_block.downsample.load_state_dict(src_block.downsample.state_dict())


def _create_se_layer(
    original_layer: nn.Sequential,
    inplanes: int,
    planes: int,
    reduction: int = 16
) -> nn.Sequential:
    """
    Create an SE-enhanced layer by replacing Bottleneck blocks with SEBottleneck.
    
    Args:
        original_layer: Original ResNet layer (e.g., layer3 or layer4)
        inplanes: Input channels for the first block
        planes: Output channels (before expansion)
        reduction: SE reduction ratio
        
    Returns:
        New Sequential layer with SEBottleneck blocks
    """
    blocks = []
    current_inplanes = inplanes
    
    for i, block in enumerate(original_layer):
        # Get stride and downsample info from original block
        stride = block.stride if hasattr(block, 'stride') else 1
        if i == 0:
            # First block may have stride > 1 and downsample
            stride = 2 if inplanes != planes * 4 else block.conv2.stride[0]
        else:
            stride = 1
        
        # Create downsample if needed
        downsample = None
        if i == 0 and (stride != 1 or current_inplanes != planes * SEBottleneck.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(
                    current_inplanes, planes * SEBottleneck.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * SEBottleneck.expansion),
            )
        
        # Create SEBottleneck
        se_block = SEBottleneck(
            inplanes=current_inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            reduction=reduction
        )
        
        # Copy weights from original block
        _copy_bottleneck_weights(block, se_block)
        
        blocks.append(se_block)
        current_inplanes = planes * SEBottleneck.expansion
    
    return nn.Sequential(*blocks)


def build_se_resnet50(
    freeze_early: bool = True,
    reduction: int = 16
) -> nn.Module:
    """
    Build SE-ResNet50 with pretrained weights.
    
    Process:
    1. Load standard pretrained ResNet50
    2. Optionally freeze Layer 1-2 (early layers)
    3. Replace Layer 3-4 blocks with SE-enhanced versions
    4. Copy original weights to new SE blocks
    
    Args:
        freeze_early: Whether to freeze Layer 1-2 parameters
        reduction: SE reduction ratio (default: 16)
        
    Returns:
        SE-enhanced ResNet50 model (without fc layer)
    """
    # Load pretrained ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Get layer info for SE replacement
    # Layer 3: 1024 output channels (256 * 4), input from layer2 is 512
    # Layer 4: 2048 output channels (512 * 4), input from layer3 is 1024
    
    # Replace Layer 3 with SE version
    se_layer3 = _create_se_layer(resnet.layer3, inplanes=512, planes=256, reduction=reduction)
    
    # Replace Layer 4 with SE version
    se_layer4 = _create_se_layer(resnet.layer4, inplanes=1024, planes=512, reduction=reduction)
    
    # Create new model
    resnet.layer3 = se_layer3
    resnet.layer4 = se_layer4
    
    # Remove fc layer (we'll add our own classifier)
    resnet.fc = nn.Identity()
    
    # Freeze early layers if requested
    if freeze_early:
        # Freeze conv1, bn1, layer1, layer2
        for name, param in resnet.named_parameters():
            if any(name.startswith(layer) for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                param.requires_grad = False
    
    return resnet


def get_se_resnet50_feature_dim() -> int:
    """Return the feature dimension of SE-ResNet50 (before fc layer)."""
    return 2048
