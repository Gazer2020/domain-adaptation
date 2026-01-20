"""
Channel Activation-based Domain Adaptation (CAD) Solver.

This method leverages the observation that known and unknown classes
have different channel activation patterns in deep CNN features.
It uses a learnable channel selector to emphasize discriminative channels.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import ChannelSelector, SemanticHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


class FeatureExtractorWithHook(nn.Module):
    """
    ResNet backbone with hook to extract layer4 channel activations.
    """
    
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.channel_activations = None
        
        # Register hook on layer4
        self.backbone.layer4.register_forward_hook(self._hook_fn)
        
        # Remove original fc layer
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    
    def _hook_fn(self, module, input, output):
        """Capture channel activations (before GAP)."""
        # output shape: [B, C, H, W]
        self.channel_activations = output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            features: Global average pooled features [B, D]
            channel_acts: Channel activations [B, C, H, W]
        """
        features = self.backbone(x)  # [B, D] after fc=Identity
        return features, self.channel_activations


@register_solver("cad")
class CADSolver(BaseSolver):
    """
    Channel Activation-based Domain Adaptation solver.
    
    Two-stage training:
    1. Pretrain: Standard classification on source domain
    2. Adaptation: Fine-tune with channel selection and unknown rejection
    
    The key insight is that discriminative channels help separate
    known classes from unknown classes in the target domain.
    """
    
    def build_model(self):
        """Build feature extractor, channel selector, and classifier."""
        backbone_name = self.config.method.get("backbone", "resnet50")
        
        # Feature extractor with hook
        backbone = get_backbone(backbone_name)
        self.feature_extractor = FeatureExtractorWithHook(backbone).to(self.device)
        
        # Get feature dimensions
        self.feature_dim = self.feature_extractor.feature_dim
        
        # Determine channel count based on backbone
        if "resnet18" in backbone_name or "resnet34" in backbone_name:
            self.channel_dim = 512
        else:
            self.channel_dim = 2048
        
        # Channel selector (learnable channel weighting)
        reduction = self.config.method.get("channel_reduction", 16)
        self.channel_selector = ChannelSelector(
            in_channels=self.channel_dim, 
            reduction=reduction
        ).to(self.device)
        
        # Classification head
        self.classifier = SemanticHead(
            in_features=self.feature_dim,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Feature fusion weight
        self.fusion_weight = self.config.method.get("fusion_weight", 0.1)
        
        logger.info(f"Built CAD model: backbone={backbone_name}, "
                    f"feature_dim={self.feature_dim}, channels={self.channel_dim}, "
                    f"fusion_weight={self.fusion_weight}")

    def _get_trainable_params(self):
        """Get all trainable parameters."""
        params = list(self.feature_extractor.parameters())
        params += list(self.channel_selector.parameters())
        params += list(self.classifier.parameters())
        return params

    def _set_train_mode(self):
        """Set all components to training mode."""
        self.feature_extractor.train()
        self.channel_selector.train()
        self.classifier.train()

    def _set_eval_mode(self):
        """Set all components to evaluation mode."""
        self.feature_extractor.eval()
        self.channel_selector.eval()
        self.classifier.eval()

    def train(self):
        """Two-stage training: pretrain + adaptation."""
        pretrain_epochs = self.config.method.get("pretrain_epochs", 10)
        adapt_epochs = self.config.method.get("adapt_epochs", 10)
        
        # Stage 1: Pretrain on source
        logger.info(f"Stage 1: Pretraining for {pretrain_epochs} epochs...")
        self._train_pretrain_stage(pretrain_epochs)
        
        # Compute channel statistics on target domain
        logger.info("Computing channel activation statistics...")
        self._compute_channel_stats()
        
        # Stage 2: Adaptation with channel selection
        logger.info(f"Stage 2: Adaptation for {adapt_epochs} epochs...")
        self._train_adaptation_stage(adapt_epochs)
        
        logger.info("Training finished.")

    def _train_pretrain_stage(self, max_epochs: int):
        """Stage 1: Standard classification on source domain."""
        for epoch in range(max_epochs):
            self._set_train_mode()
            loss_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Pretrain {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward
                features, channel_acts = self.feature_extractor(src_imgs)
                logits = self.classifier(features)
                
                # Cross-entropy loss
                loss = self.criterion(logits, src_labels)
                
                loss.backward()
                self.optimizer.step()
                
                loss_meter.update(loss.item())
                pbar.set_postfix({"loss": loss_meter.avg})
            
            acc = self.evaluate()
            logger.info(f"Pretrain Epoch {epoch+1}: Loss={loss_meter.avg:.4f}, Acc={acc:.2f}%")

    @torch.no_grad()
    def _compute_channel_stats(self):
        """
        Compute channel activation statistics on target domain.
        
        This helps identify which channels are most discriminative
        for separating known vs unknown classes.
        """
        self._set_eval_mode()
        
        all_channel_acts = []
        all_confidences = []
        
        for imgs, _ in tqdm(self.target_loader, desc="Computing stats"):
            imgs = imgs.to(self.device)
            
            features, channel_acts = self.feature_extractor(imgs)
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]
            
            # Global average pooling on channel activations
            channel_acts_gap = channel_acts.mean(dim=[2, 3])  # [B, C]
            
            all_channel_acts.append(channel_acts_gap.cpu())
            all_confidences.append(max_probs.cpu())
        
        all_channel_acts = torch.cat(all_channel_acts, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        # Samples with high confidence are likely known classes
        # Samples with low confidence are likely unknown classes
        threshold = all_confidences.median()
        
        high_conf_mask = all_confidences > threshold
        low_conf_mask = all_confidences <= threshold
        
        high_conf_acts = all_channel_acts[high_conf_mask]
        low_conf_acts = all_channel_acts[low_conf_mask]
        
        # Channel importance: difference between high/low confidence samples
        high_mean = high_conf_acts.mean(dim=0)
        low_mean = low_conf_acts.mean(dim=0)
        
        self.channel_importance = (high_mean - low_mean).abs().to(self.device)
        self.channel_importance = self.channel_importance / self.channel_importance.max()
        
        logger.info(f"Channel stats computed. High-conf samples: {high_conf_mask.sum()}, "
                    f"Low-conf samples: {low_conf_mask.sum()}")

    def _train_adaptation_stage(self, max_epochs: int):
        """Stage 2: Fine-tune with channel-weighted features."""
        lambda_channel = self.config.method.get("lambda_channel", 0.1)
        lambda_entropy = self.config.method.get("lambda_entropy", 0.1)
        
        for epoch in range(max_epochs):
            self._set_train_mode()
            
            tgt_iter = cycle(self.target_loader)
            cls_loss_meter = AverageMeter()
            chan_loss_meter = AverageMeter()
            ent_loss_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Adapt {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)
                
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Source forward with channel selection
                src_features, src_channel_acts = self.feature_extractor(src_imgs)
                src_channel_weights = self.channel_selector(src_channel_acts)  # [B, C]
                
                # Apply channel weighting on the channel activations
                # src_channel_acts: [B, C, H, W]
                # src_channel_weights: [B, C]
                # We apply weights channel-wise then do global average pooling
                src_channel_acts_weighted = src_channel_acts * src_channel_weights.unsqueeze(-1).unsqueeze(-1)
                src_weighted_features = src_channel_acts_weighted.mean(dim=[2, 3])  # [B, C]
                
                # Combine original features with channel-weighted features
                # Concatenate or add them based on dimensionality
                if src_weighted_features.size(1) == src_features.size(1):
                    # Same dimension: weighted addition
                    src_combined = src_features + self.fusion_weight * src_weighted_features
                else:
                    # Different dimension: use original features (channel_dim != feature_dim)
                    src_combined = src_features
                
                src_logits = self.classifier(src_combined)
                cls_loss = self.criterion(src_logits, src_labels)
                
                # Target forward for entropy minimization
                tgt_features, tgt_channel_acts = self.feature_extractor(tgt_imgs)
                tgt_channel_weights = self.channel_selector(tgt_channel_acts)  # [B, C]
                
                # Apply channel weighting
                tgt_channel_acts_weighted = tgt_channel_acts * tgt_channel_weights.unsqueeze(-1).unsqueeze(-1)
                tgt_weighted_features = tgt_channel_acts_weighted.mean(dim=[2, 3])  # [B, C]
                
                if tgt_weighted_features.size(1) == tgt_features.size(1):
                    tgt_combined = tgt_features + self.fusion_weight * tgt_weighted_features
                else:
                    tgt_combined = tgt_features
                    
                tgt_logits = self.classifier(tgt_combined)
                
                # Entropy minimization on target (encourage confident predictions)
                tgt_probs = F.softmax(tgt_logits, dim=1)
                ent_loss = -torch.mean(torch.sum(tgt_probs * torch.log(tgt_probs + 1e-8), dim=1))
                
                # Channel diversity loss: encourage using discriminative channels
                chan_loss = -torch.mean(tgt_channel_weights * self.channel_importance)
                
                # Total loss
                loss = cls_loss + lambda_entropy * ent_loss + lambda_channel * chan_loss
                
                loss.backward()
                self.optimizer.step()
                
                cls_loss_meter.update(cls_loss.item())
                chan_loss_meter.update(chan_loss.item())
                ent_loss_meter.update(ent_loss.item())
                
                pbar.set_postfix({
                    "cls": cls_loss_meter.avg,
                    "chan": chan_loss_meter.avg,
                    "ent": ent_loss_meter.avg
                })
            
            acc = self.evaluate()
            logger.info(f"Adapt Epoch {epoch+1}: Cls={cls_loss_meter.avg:.4f}, "
                        f"Chan={chan_loss_meter.avg:.4f}, Acc={acc:.2f}%")

    def compute_loss(self, src_imgs, src_labels, tgt_imgs):
        """Compute combined loss (for compatibility with base class)."""
        src_features, _ = self.feature_extractor(src_imgs)
        logits = self.classifier(src_features)
        return self.criterion(logits, src_labels)

    def forward_for_eval(self, imgs):
        """Forward pass for evaluation."""
        features, _ = self.feature_extractor(imgs)
        return self.classifier(features)

    def save_checkpoint(self, path):
        """Save all model components."""
        torch.save({
            "method": "cad",
            "feature_extractor": self.feature_extractor.state_dict(),
            "channel_selector": self.channel_selector.state_dict(),
            "classifier": self.classifier.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """Load all model components."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        # Handle both old and new formats
        if "feature_extractor" in checkpoint:
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            self.channel_selector.load_state_dict(checkpoint["channel_selector"])
            self.classifier.load_state_dict(checkpoint["classifier"])
        else:
            # Old format compatibility (if needed)
            logger.warning("Loading from old checkpoint format")
            
        logger.info(f"Model loaded from {path}")
