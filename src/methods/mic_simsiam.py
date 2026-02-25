"""
MIC-SimSiam: Multi-Dimensional Robustness Training for OSDA.

A self-supervised framework combining:
- Masked Image Consistency (MIC): "Predict complete from incomplete" task
- SE-Block Channel Attention: Auto-suppress local textures, amplify global semantics
- SimSiam Architecture: Avoid negative sample collapse in small batches
- Entropy Reweighting: High weight for known classes, low for uncertain samples

Key Design:
- Asymmetric task: masked image → predict full image features
- Soft masking: 25-40% (vs MAE's 75%) for small dataset stability
- Stop-gradient on full image branch (SimSiam style)
"""

import logging
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.se_resnet import build_se_resnet50, get_se_resnet50_feature_dim
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


# =============================================================================
# Masking Module
# =============================================================================

class RandomPatchMasker:
    """
    Random Patch Masking for soft image masking.
    
    Randomly masks square patches with configurable ratio (25-40%).
    Much gentler than MAE's 75% for small dataset stability.
    """
    
    def __init__(
        self,
        mask_ratio_min: float = 0.25,
        mask_ratio_max: float = 0.40,
        patch_size: int = 32,
        mask_value: float = 0.0
    ):
        """
        Args:
            mask_ratio_min: Minimum masking ratio
            mask_ratio_max: Maximum masking ratio
            patch_size: Size of each masked patch
            mask_value: Value to fill masked regions (0 = black)
        """
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.patch_size = patch_size
        self.mask_value = mask_value
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random patch masking.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            masked: Masked images [B, C, H, W]
            mask: Binary mask [B, 1, H, W] (1 = masked, 0 = visible)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Random masking ratio per batch
        ratio = torch.empty(B).uniform_(self.mask_ratio_min, self.mask_ratio_max)
        
        # Calculate grid size
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Create mask for each sample
        masks = []
        for b in range(B):
            # Number of patches to mask
            num_mask = int(num_patches * ratio[b].item())
            
            # Random patch indices to mask
            perm = torch.randperm(num_patches, device=device)
            mask_indices = perm[:num_mask]
            
            # Create patch-level mask
            patch_mask = torch.zeros(num_patches, device=device)
            patch_mask[mask_indices] = 1.0
            patch_mask = patch_mask.view(num_patches_h, num_patches_w)
            
            # Upsample to image size
            mask = patch_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, ph, pw]
            mask = F.interpolate(mask, size=(H, W), mode='nearest')  # [1, 1, H, W]
            masks.append(mask.squeeze(0))  # [1, H, W]
        
        mask = torch.stack(masks, dim=0)  # [B, 1, H, W]
        
        # Apply mask
        masked = x * (1 - mask) + self.mask_value * mask
        
        return masked, mask


# =============================================================================
# SimSiam Architecture Components
# =============================================================================

class Projector(nn.Module):
    """
    SimSiam Projector: 3-layer MLP with BN.
    
    Maps backbone features to projection space.
    Structure: Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear -> BN
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 2048
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)  # No affine for output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """
    SimSiam Predictor: 2-layer MLP with BN.
    
    Only applied to one branch (asymmetric architecture).
    Structure: Linear -> BN -> ReLU -> Linear
    """
    
    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 512,
        out_dim: int = 2048
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# MIC-SimSiam Solver
# =============================================================================

@register_solver("mic_simsiam")
class MICSimSiamSolver(BaseSolver):
    """
    MIC-SimSiam Solver for Open Set Domain Adaptation.
    
    Training Framework:
    1. Warmup Phase: Train classifier on source domain only
    2. Adaptation Phase: 
       - MIC loss: masked image predicts full image features (SimSiam)
       - Classification loss with entropy reweighting
       
    The MIC loss encourages the model to learn global structural features
    that are robust to partial occlusion, improving rejection of unknown classes.
    """
    
    def build_model(self):
        """Build SE-ResNet50 backbone with SimSiam heads and classifier."""
        cfg = self.config.method
        
        # SE-ResNet50 backbone (Layer 3-4 with SE attention)
        self.backbone = build_se_resnet50(
            freeze_early=True,
            reduction=cfg.get("se_reduction", 16)
        ).to(self.device)
        
        # Feature dimension
        self.feat_dim = get_se_resnet50_feature_dim()
        
        # SimSiam projector and predictor
        proj_hidden = cfg.get("proj_hidden_dim", 2048)
        proj_out = cfg.get("proj_out_dim", 2048)
        pred_hidden = cfg.get("pred_hidden_dim", 512)
        
        self.projector = Projector(
            in_dim=self.feat_dim,
            hidden_dim=proj_hidden,
            out_dim=proj_out
        ).to(self.device)
        
        self.predictor = Predictor(
            in_dim=proj_out,
            hidden_dim=pred_hidden,
            out_dim=proj_out
        ).to(self.device)
        
        # Source classifier (for known classes only, not unknown)
        # num_classes already includes +1 for unknown in OSDA
        classifier_classes = self.num_classes - 1 if self.unknown_label is not None else self.num_classes
        self.classifier = nn.Linear(self.feat_dim, classifier_classes).to(self.device)
        
        # Random patch masker
        self.masker = RandomPatchMasker(
            mask_ratio_min=cfg.get("mask_ratio_min", 0.25),
            mask_ratio_max=cfg.get("mask_ratio_max", 0.40),
            patch_size=cfg.get("patch_size", 32)
        )
        
        # Store classifier class count for entropy computation
        self.classifier_classes = classifier_classes
        
        # ============== PCOSL: Learnable K+1 Prototype Space ==============
        # K known class prototypes + 1 learnable unknown prototype
        # Unlike fixed prototypes, these are learned during training
        self.prototypes = nn.Parameter(
            torch.randn(classifier_classes + 1, self.feat_dim) * 0.01
        ).to(self.device)
        
        # Track whether prototypes are initialized from source
        self.prototypes_initialized = False
        
        # For source negative mining
        self.source_entropy_threshold = 0.7  # High entropy = uncertain
        
        # Contrastive margin (learned from validation)
        self.contrastive_margin = cfg.get("contrastive_margin", 0.5)
        
        # Unknown ratio for percentile-based rejection
        self.unknown_ratio = cfg.get("unknown_ratio", 0.35)
        
        # For bi-modal threshold estimation
        self.rejection_threshold = None  # Will be computed adaptively
        
        logger.info(
            f"Built MIC-SimSiam + PCOSL model: backbone=SE-ResNet50, "
            f"feat_dim={self.feat_dim}, proj_out={proj_out}, "
            f"classifier_classes={classifier_classes}, "
            f"prototypes={classifier_classes}+1 (learnable)"
        )
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone (global average pooled)."""
        # Forward through backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def _compute_mic_loss(
        self,
        x_full: torch.Tensor,
        x_masked: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MIC (Masked Image Consistency) loss using SimSiam.
        
        The masked image branch tries to predict the full image features.
        Stop-gradient is applied to the full image branch (target).
        
        Args:
            x_full: Full (unmasked) images [B, C, H, W]
            x_masked: Masked images [B, C, H, W]
            
        Returns:
            MIC loss (negative cosine similarity)
        """
        # Extract features
        feat_full = self._extract_features(x_full)  # [B, D]
        feat_masked = self._extract_features(x_masked)  # [B, D]
        
        # Project features
        z_full = self.projector(feat_full)  # [B, proj_out]
        z_masked = self.projector(feat_masked)  # [B, proj_out]
        
        # Predictor only on masked branch
        p_masked = self.predictor(z_masked)  # [B, proj_out]
        
        # Negative cosine similarity (stop-gradient on full branch)
        z_full = z_full.detach()  # Stop gradient!
        
        # Normalize
        p_masked = F.normalize(p_masked, dim=1)
        z_full = F.normalize(z_full, dim=1)
        
        # Cosine similarity loss (maximize similarity = minimize negative)
        loss = -torch.mean(torch.sum(p_masked * z_full, dim=1))
        
        return loss
    
    def _compute_entropy_weight(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy-based sample weights for OSDA adaptation.
        
        Low entropy (confident/known) → high weight
        High entropy (uncertain/unknown) → low weight
        
        This allows known class samples to contribute more to learning
        while unknown samples are allowed to diverge.
        
        Args:
            logits: Classification logits [B, C]
            
        Returns:
            weights: Per-sample weights [B]
        """
        probs = F.softmax(logits, dim=1)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)  # [B]
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log(logits.size(1))
        normalized_entropy = entropy / max_entropy  # [0, 1]
        
        # Inverse: high entropy → low weight
        # Use exponential decay for smoother transition
        threshold = self.config.method.get("entropy_threshold", 0.5)
        
        # Samples with entropy < threshold get weight close to 1
        # Samples with entropy > threshold get exponentially lower weight
        weights = torch.exp(-5.0 * F.relu(normalized_entropy - threshold))
        
        return weights
    
    # ============== ADVANCED IMPROVEMENTS ==============
    
    def _compute_consistency_weight(
        self, 
        full_logits: torch.Tensor, 
        masked_logits: torch.Tensor,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        Compute sample reliability based on prediction consistency.
        
        Low KL divergence = high consistency = likely known class
        High KL divergence = low consistency = likely unknown/noise
        
        Args:
            full_logits: Logits from full (unmasked) view [B, C]
            masked_logits: Logits from masked view [B, C]
            temperature: Softmax temperature for smoothing
            
        Returns:
            weights: Per-sample consistency weights [B] in [0, 1]
        """
        full_probs = F.softmax(full_logits / temperature, dim=1)
        masked_probs = F.softmax(masked_logits / temperature, dim=1)
        
        # KL divergence: D_KL(full || masked)
        kl_div = F.kl_div(
            (masked_probs + 1e-8).log(), 
            full_probs, 
            reduction='none'
        ).sum(dim=1)
        
        # Convert to weight: low KL -> high weight
        # Use exponential decay: w = exp(-kl / tau)
        weights = torch.exp(-kl_div / temperature)
        
        return weights
    
    def _compute_dual_entropy_loss(
        self, 
        logits: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency-weighted entropy minimization.
        
        High weight (likely known): strong entropy minimization
        Low weight (likely unknown): weak/no entropy minimization (just ignore)
        
        Note: We don't maximize entropy for unknown samples as it caused
        collapse in experiments. Instead, we simply ignore them.
        
        Args:
            logits: Classification logits [B, C]
            weights: Sample weights [B] in [0, 1], high=likely known
            
        Returns:
            Weighted entropy loss (scalar)
        """
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Only entropy minimization, weighted by consistency
        # High weight samples: minimize their entropy
        # Low weight samples: effectively ignored
        weighted_entropy = (weights * entropy).sum() / (weights.sum() + 1e-8)
        
        return weighted_entropy
    
    def _compute_se_sparsity_loss(self) -> torch.Tensor:
        """
        Collect SE weights from all SE blocks and compute L1 penalty.
        
        This encourages the model to use fewer channels, making
        unknown samples have sparser (lower norm) features.
        
        Returns:
            L1 sparsity loss (scalar)
        """
        total_l1 = 0.0
        count = 0
        
        for module in self.backbone.modules():
            if hasattr(module, 'last_se_weights') and module.last_se_weights is not None:
                # last_se_weights: [B, C] or [C]
                se_weights = module.last_se_weights
                total_l1 = total_l1 + se_weights.abs().mean()
                count += 1
        
        return total_l1 / max(count, 1)
    
    # =============================================================================
    # PCOSL: Prototype-Contrastive Open-Set Learning Methods
    # =============================================================================
    
    def _initialize_prototypes_from_source(self):
        """Initialize K known prototypes from source domain class means."""
        self._set_eval_mode()
        
        # Accumulate features per class
        sums = torch.zeros(self.classifier_classes, self.feat_dim, device=self.device)
        counts = torch.zeros(self.classifier_classes, device=self.device)
        
        with torch.no_grad():
            for imgs, labels in self.source_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                feats = self._extract_features(imgs)
                for c in range(self.classifier_classes):
                    mask = labels == c
                    if mask.sum() > 0:
                        sums[c] += feats[mask].sum(dim=0)
                        counts[c] += mask.sum()
        
        # Set known prototypes (indices 0 to K-1)
        valid_classes = counts > 0
        with torch.no_grad():
            self.prototypes.data[:self.classifier_classes][valid_classes] = (
                sums[valid_classes] / counts[valid_classes].unsqueeze(1)
            )
            
            # Initialize unknown prototype as the mean + perturbation
            # This places it in a region far from known classes
            known_mean = self.prototypes.data[:self.classifier_classes].mean(dim=0)
            known_std = self.prototypes.data[:self.classifier_classes].std(dim=0)
            self.prototypes.data[-1] = known_mean + 2 * known_std  # Push away
        
        self.prototypes_initialized = True
        logger.info(f"Initialized {valid_classes.sum().item()}/{self.classifier_classes} known prototypes from source")
        self._set_train_mode()
    
    def _compute_contrastive_margin_loss(
        self, 
        features: torch.Tensor, 
        confidence: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        PCOSL Contrastive Margin Loss.
        
        - High confidence samples: Pull toward predicted class prototype
        - Low confidence samples: Push toward unknown prototype
        
        Args:
            features: Target domain features [B, D]
            confidence: Prediction confidence [B] in [0, 1]
            pseudo_labels: Predicted class labels [B]
        
        Returns:
            Contrastive margin loss (scalar)
        """
        B = features.size(0)
        if B == 0:
            return torch.tensor(0.0, device=self.device)
        
        # L2 normalize features and prototypes
        feats_norm = F.normalize(features, p=2, dim=1)  # [B, D]
        protos_norm = F.normalize(self.prototypes, p=2, dim=1)  # [K+1, D]
        
        # Distance to all prototypes
        dists = torch.cdist(feats_norm, protos_norm, p=2)  # [B, K+1]
        
        # Distance to predicted class prototype
        pos_dist = dists[torch.arange(B, device=self.device), pseudo_labels]  # [B]
        
        # Distance to unknown prototype (last one)
        unk_dist = dists[:, -1]  # [B]
        
        # Contrastive margin loss:
        # High confidence: minimize pos_dist (pull to known)
        # Low confidence: minimize unk_dist (push to unknown)
        
        # Threshold: confidence < 0.5 are treated as potential unknown
        is_likely_known = confidence > 0.5
        
        # Loss for likely known: pos_dist should be small
        loss_known = (confidence * pos_dist).mean()
        
        # Loss for likely unknown: unk_dist should be small
        loss_unknown = ((1 - confidence) * unk_dist).mean()
        
        # Margin constraint: pos_dist should be less than unk_dist for known samples
        margin_loss = F.relu(pos_dist - unk_dist + self.contrastive_margin)
        margin_loss = (is_likely_known.float() * margin_loss).mean()
        
        return loss_known + loss_unknown + margin_loss
    
    def _mine_source_negatives(
        self, 
        features: torch.Tensor, 
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Source Negative Mining: Find uncertain source samples.
        
        These samples are near class boundaries and share patterns with unknown.
        Use them to update the unknown prototype.
        
        Args:
            features: Source features [B, D]
            logits: Source logits [B, K]
        
        Returns:
            Features of uncertain samples [N, D] or empty tensor
        """
        probs = F.softmax(logits, dim=1)
        
        # Compute entropy (high entropy = uncertain)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = math.log(self.classifier_classes)
        norm_entropy = entropy / max_entropy
        
        # Also check for multi-class activation (top-2 are close)
        top2_probs, _ = probs.topk(2, dim=1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]
        
        # Uncertain if high entropy OR small margin
        uncertain_mask = (norm_entropy > 0.6) | (margin < 0.3)
        
        return features[uncertain_mask] if uncertain_mask.sum() > 0 else torch.empty(0, self.feat_dim, device=self.device)
    
    def _update_unknown_prototype(
        self, 
        uncertain_features: torch.Tensor, 
        momentum: float = 0.9
    ):
        """Update unknown prototype from uncertain source samples."""
        if len(uncertain_features) == 0:
            return
        
        with torch.no_grad():
            uncertain_mean = uncertain_features.mean(dim=0)
            self.prototypes.data[-1] = (
                momentum * self.prototypes.data[-1] + 
                (1 - momentum) * uncertain_mean
            )
    
    def _compute_otsu_threshold(self, distances: torch.Tensor) -> float:
        """
        Otsu's method: Find threshold that maximizes inter-class variance.
        
        Assumes bi-modal distribution: known (small distances) vs unknown (large).
        
        Args:
            distances: 1D tensor of distances to nearest prototype
        
        Returns:
            Optimal threshold (float)
        """
        if len(distances) < 10:
            return distances.median().item()
        
        distances_np = distances.cpu().numpy()
        
        # Create histogram
        hist, bin_edges = np.histogram(distances_np, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Otsu's algorithm
        total = hist.sum()
        sum_total = (hist * bin_centers).sum()
        
        sum_bg = 0.0
        weight_bg = 0
        max_variance = 0.0
        optimal_threshold = bin_centers[len(bin_centers) // 2]
        
        for i, (count, center) in enumerate(zip(hist, bin_centers)):
            weight_bg += count
            if weight_bg == 0:
                continue
            
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            
            sum_bg += count * center
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            
            # Inter-class variance
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = center
        
        return float(optimal_threshold)
    
    def _compute_rejection_threshold_otsu(self):
        """Compute rejection threshold using Otsu on target domain distances."""
        self._set_eval_mode()
        
        all_distances = []
        
        with torch.no_grad():
            for imgs, _ in self.target_loader:
                imgs = imgs.to(self.device)
                feats = self._extract_features(imgs)
                feats_norm = F.normalize(feats, p=2, dim=1)
                protos_norm = F.normalize(self.prototypes[:self.classifier_classes], p=2, dim=1)
                
                dists = torch.cdist(feats_norm, protos_norm, p=2)
                min_dists = dists.min(dim=1)[0]
                all_distances.append(min_dists)
        
        all_distances = torch.cat(all_distances)
        self.rejection_threshold = self._compute_otsu_threshold(all_distances)
        
        logger.info(f"Computed Otsu threshold: {self.rejection_threshold:.4f} "
                    f"(dist range: {all_distances.min():.4f} - {all_distances.max():.4f})")
        
        self._set_train_mode()
    
    def _set_train_mode(self):
        """Set all modules to training mode."""
        self.backbone.train()
        self.projector.train()
        self.predictor.train()
        self.classifier.train()
    
    def _set_eval_mode(self):
        """Set all modules to evaluation mode."""
        self.backbone.eval()
        self.projector.eval()
        self.predictor.eval()
        self.classifier.eval()
    
    # Note: Legacy _update_prototypes removed - PCOSL uses _initialize_prototypes_from_source
    
    def _compute_rejection_thresholds(self):
        """
        Compute rejection thresholds based on source domain statistics.
        Called after warmup to set adaptive thresholds.
        """
        self._set_eval_mode()
        
        all_entropies = []
        all_distances = []
        
        with torch.no_grad():
            for imgs, labels in self.source_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                feats = self._extract_features(imgs)
                logits = self.classifier(feats)
                probs = F.softmax(logits, dim=1)
                
                # Entropy (normalized)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                max_entropy = math.log(self.classifier_classes)
                norm_entropy = entropy / max_entropy
                all_entropies.append(norm_entropy)
                
                # Distance to nearest prototype
                feats_norm = F.normalize(feats, p=2, dim=1)
                protos_norm = F.normalize(self.prototypes, p=2, dim=1)
                dists = torch.cdist(feats_norm, protos_norm)  # [B, K]
                min_dist, _ = dists.min(dim=1)
                all_distances.append(min_dist)
        
        all_entropies = torch.cat(all_entropies)
        all_distances = torch.cat(all_distances)
        
        # Set thresholds at 99th percentile (softer to avoid over-rejection)
        computed_entropy = torch.quantile(all_entropies, 0.99).item()
        computed_distance = torch.quantile(all_distances, 0.99).item()
        
        # IMPORTANT: Set minimum floors to prevent over-rejection
        # As model becomes more confident, computed thresholds can become too strict
        min_entropy_threshold = 0.15  # At least 15% of max entropy
        min_distance_threshold = 0.8  # At least 0.8 cosine distance
        
        self.entropy_threshold = max(computed_entropy, min_entropy_threshold)
        self.distance_threshold = max(computed_distance, min_distance_threshold)
        
        logger.info(
            f"Computed rejection thresholds: entropy={self.entropy_threshold:.3f} "
            f"(min={min_entropy_threshold}), distance={self.distance_threshold:.3f} "
            f"(min={min_distance_threshold})"
        )
        
        self._set_train_mode()
    
    def _build_optimizer(self):
        """Build optimizer with separate learning rates."""
        cfg = self.config.method
        lr_backbone = cfg.get("lr_backbone", 1e-4)
        lr_head = cfg.get("lr_head", 1e-3)
        
        # Separate parameter groups
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.projector.parameters()) +
            list(self.predictor.parameters()) +
            list(self.classifier.parameters())
        )
        
        param_groups = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head}
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        
        return optimizer
    
    def _build_scheduler(self, optimizer, total_epochs: int):
        """Build learning rate scheduler (warmup + cosine decay)."""
        warmup_epochs = self.config.method.get("warmup_epochs", 10)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train(self):
        """Main training loop with warmup and adaptation phases."""
        cfg = self.config.method
        warmup_epochs = cfg.get("warmup_epochs", 10)
        adapt_epochs = cfg.get("adapt_epochs", 40)
        total_epochs = warmup_epochs + adapt_epochs
        
        # Build optimizer and scheduler
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, total_epochs)
        
        # Loss weights
        lambda_mic = cfg.get("lambda_mic", 1.0)
        lambda_cls = cfg.get("lambda_cls", 1.0)
        lambda_entropy = cfg.get("lambda_entropy", 0.3)  # Configurable entropy weight
        label_smoothing = cfg.get("label_smoothing", 0.1)  # Prevent source overfitting
        
        logger.info(
            f"Starting training: warmup={warmup_epochs}, adapt={adapt_epochs}, "
            f"lambda_mic={lambda_mic}, lambda_cls={lambda_cls}"
        )
        
        best_hscore = 0.0
        
        for epoch in range(total_epochs):
            self._set_train_mode()
            is_warmup = epoch < warmup_epochs
            
            # ============== PCOSL: Initialize prototypes at warmup end ==============
            if epoch == warmup_epochs:
                logger.info("Warmup complete. Initializing PCOSL prototypes from source...")
                self._initialize_prototypes_from_source()
                # Compute Otsu threshold ONCE after prototype initialization
                # (don't recalculate - it drifts as model learns, causing over-rejection)
                self._compute_rejection_threshold_otsu()
            
            # Metrics
            loss_meter = AverageMeter()
            cls_meter = AverageMeter()
            mic_meter = AverageMeter()
            
            # Create iterator
            target_iter = cycle(self.target_loader)
            
            pbar = tqdm(
                self.source_loader,
                desc=f"Epoch {epoch+1}/{total_epochs} ({'Warmup' if is_warmup else 'Adapt'})"
            )
            
            for src_imgs, src_labels in pbar:
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # ============== Source Classification ==============
                src_feats = self._extract_features(src_imgs)
                src_logits = self.classifier(src_feats)
                # Label smoothing to prevent source overfitting
                loss_cls_src = F.cross_entropy(
                    src_logits, src_labels, 
                    label_smoothing=label_smoothing
                )
                
                batch_size = src_imgs.size(0)
                total_loss = lambda_cls * loss_cls_src
                cls_meter.update(loss_cls_src.item(), batch_size)
                
                # Note: PCOSL uses _initialize_prototypes_from_source at warmup end
                # instead of running prototype updates
                
                # ============== MIC Loss (Adaptation Phase) ==============
                if not is_warmup:
                    # Get target batch
                    tgt_imgs, _ = next(target_iter)
                    tgt_imgs = tgt_imgs.to(self.device)
                    
                    # Combine source and target for MIC
                    all_imgs = torch.cat([src_imgs, tgt_imgs], dim=0)
                    
                    # Create masked version
                    masked_imgs, _ = self.masker(all_imgs)
                    
                    # MIC loss: masked predicts full (already mean-reduced)
                    loss_mic = self._compute_mic_loss(all_imgs, masked_imgs)
                    
                    total_loss = total_loss + lambda_mic * loss_mic
                    mic_meter.update(loss_mic.item(), all_imgs.size(0))
                    
                    # ============== PCOSL: Contrastive Margin Loss ==============
                    lambda_contrast = cfg.get("lambda_contrastive", 0.1)
                    if lambda_contrast > 0:
                        tgt_feats = self._extract_features(tgt_imgs)
                        tgt_logits = self.classifier(tgt_feats)
                        tgt_probs = F.softmax(tgt_logits, dim=1)
                        
                        confidence = tgt_probs.max(dim=1)[0]
                        pseudo_labels = tgt_probs.argmax(dim=1)
                        
                        loss_contrast = self._compute_contrastive_margin_loss(
                            tgt_feats, confidence, pseudo_labels
                        )
                        total_loss = total_loss + lambda_contrast * loss_contrast
                    
                    # ============== PCOSL: Source Negative Mining ==============
                    # Find uncertain source samples and use them to update unknown prototype
                    uncertain_src = self._mine_source_negatives(src_feats.detach(), src_logits.detach())
                    if len(uncertain_src) > 0:
                        self._update_unknown_prototype(uncertain_src, momentum=0.95)
                    
                    # ============== SE Channel Sparsity ==============
                    # This encourages sparse features, helping distinguish unknown
                    lambda_sparse = cfg.get("lambda_sparsity", 0.001)
                    if lambda_sparse > 0:
                        loss_sparsity = self._compute_se_sparsity_loss()
                        total_loss = total_loss + lambda_sparse * loss_sparsity
                
                # Backward
                total_loss.backward()
                optimizer.step()
                
                loss_meter.update(total_loss.item(), batch_size)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss_meter.avg:.4f}",
                    "cls": f"{cls_meter.avg:.4f}",
                    "mic": f"{mic_meter.avg:.4f}" if not is_warmup else "N/A"
                })
            
            # Step scheduler
            scheduler.step()
            
            # ============== Diagnostic Logging ==============
            if not is_warmup:
                self._set_eval_mode()
                with torch.no_grad():
                    # Sample target batch for diagnostics
                    diag_imgs, diag_labels = next(iter(self.target_test_loader))
                    diag_imgs = diag_imgs.to(self.device)
                    diag_labels = diag_labels.to(self.device)
                    
                    feats = self._extract_features(diag_imgs)
                    logits = self.classifier(feats)
                    probs = F.softmax(logits, dim=1)
                    
                    # Entropy distribution
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    max_entropy = math.log(logits.size(1))
                    norm_entropy = entropy / max_entropy
                    
                    # Confidence distribution
                    max_conf, preds = probs.max(dim=1)
                    
                    # Feature norm
                    feat_norm = feats.norm(dim=1)
                    
                    # Known vs Unknown separation
                    known_mask = diag_labels != self.unknown_label
                    unknown_mask = diag_labels == self.unknown_label
                    
                    known_conf = max_conf[known_mask].mean().item() if known_mask.sum() > 0 else 0
                    unknown_conf = max_conf[unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0
                    known_ent = norm_entropy[known_mask].mean().item() if known_mask.sum() > 0 else 0
                    unknown_ent = norm_entropy[unknown_mask].mean().item() if unknown_mask.sum() > 0 else 0
                    
                    logger.info(
                        f"  📊 Diag: feat_norm={feat_norm.mean():.1f}±{feat_norm.std():.1f}, "
                        f"known_conf={known_conf:.3f}, unknown_conf={unknown_conf:.3f}, "
                        f"known_ent={known_ent:.3f}, unknown_ent={unknown_ent:.3f}, "
                        f"conf_gap={known_conf - unknown_conf:.3f}"
                    )
                self._set_train_mode()
            
            # Evaluate
            hscore = self.evaluate()
            
            if hscore > best_hscore:
                best_hscore = hscore
            
            mic_val = mic_meter.avg if not is_warmup else 0.0
            logger.info(
                f"Epoch {epoch+1}/{total_epochs} - "
                f"Loss: {loss_meter.avg:.4f}, Cls: {cls_meter.avg:.4f}, "
                f"MIC: {mic_val:.4f}, "
                f"H-score: {hscore:.2f}% (best: {best_hscore:.2f}%)"
            )
        
        logger.info(f"Training complete. Best H-score: {best_hscore:.2f}%")
    
    def evaluate(self):
        """Evaluate with hybrid rejection (entropy + prototype distance)."""
        self._set_eval_mode()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        all_consistency = []
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                
                # Extract features and predictions for FULL images
                feats = self._extract_features(imgs)
                logits = self.classifier(feats)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                # Compute masked predictions for CONSISTENCY check
                masked_imgs, _ = self.masker(imgs)
                masked_feats = self._extract_features(masked_imgs)
                masked_logits = self.classifier(masked_feats)
                
                # Compute consistency weight (KL divergence based)
                consistency = self._compute_consistency_weight(
                    logits, masked_logits, temperature=2.0
                )
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
                all_probs.append(probs.cpu())
                all_features.append(feats.cpu())
                all_consistency.append(consistency.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        all_features = torch.cat(all_features)
        all_consistency = torch.cat(all_consistency)
        
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            # Apply hybrid rejection with consistency
            final_preds = self.predict_with_rejection(
                all_preds, all_probs, all_features.to(self.device),
                consistency=all_consistency.to(self.device)
            )
            # Ensure both tensors on CPU for comparison
            return self._compute_osda_metrics(final_preds.cpu(), all_labels.cpu())
        else:
            correct = (all_preds == all_labels).sum().item()
            return 100 * correct / len(all_labels)
    
    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass for evaluation (returns logits with unknown class)."""
        feats = self._extract_features(imgs)
        logits = self.classifier(feats)
        
        # Add a column for unknown class (score = 0)
        # The rejection will be handled by confidence thresholding
        if self.unknown_label is not None:
            unknown_col = torch.zeros(logits.size(0), 1, device=logits.device)
            logits = torch.cat([logits, unknown_col], dim=1)
        
        return logits
    
    def predict_with_rejection(
        self,
        preds: torch.Tensor,
        probs: torch.Tensor,
        features: torch.Tensor = None,
        consistency: torch.Tensor = None
    ) -> torch.Tensor:
        """
        PCOSL Rejection Strategy: Adaptive percentile + unknown prototype affinity.
        
        Reject if any criterion is met (OR logic for higher Unknown Acc):
        1. Distance in top percentile (far from known prototypes)
        2. Closer to unknown prototype than known
        """
        probs = probs.to(self.device)
        preds = preds.to(self.device)
        
        # Use base method if prototypes not initialized yet (during warmup)
        if features is None or not self.prototypes_initialized:
            return super().predict_with_rejection(preds, probs)
        
        features = features.to(self.device)
        
        # L2 normalize for distance computation
        feats_norm = F.normalize(features, p=2, dim=1)  # [B, D]
        protos_norm = F.normalize(self.prototypes, p=2, dim=1)  # [K+1, D]
        
        # Distance to all prototypes
        dists = torch.cdist(feats_norm, protos_norm, p=2)  # [B, K+1]
        
        # Distance to known prototypes (first K) and unknown prototype (last)
        known_dists = dists[:, :self.classifier_classes]  # [B, K]
        unknown_dist = dists[:, -1]  # [B]
        
        min_known_dist, _ = known_dists.min(dim=1)  # [B]
        
        # ============== Criterion 1: Percentile-based distance ==============
        # Reject samples in the top unknown_ratio% of known distances
        distance_percentile = torch.quantile(min_known_dist, 1.0 - self.unknown_ratio)
        dist_reject = min_known_dist > distance_percentile
        
        # ============== Criterion 2: Unknown prototype closer ==============
        # If closer to unknown prototype than nearest known, likely unknown
        unknown_closer = unknown_dist < min_known_dist
        
        # ============== Combined: OR logic (either triggers rejection) ==============
        rejected_mask = dist_reject | unknown_closer
        
        final_preds = preds.clone()
        final_preds[rejected_mask] = self.unknown_label
        
        return final_preds.cpu()
    
    def save_checkpoint(self, path):
        """Save all model components."""
        torch.save({
            "method": "mic_simsiam",
            "backbone": self.backbone.state_dict(),
            "projector": self.projector.state_dict(),
            "predictor": self.predictor.state_dict(),
            "classifier": self.classifier.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load all model components."""
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["backbone"])
        self.projector.load_state_dict(checkpoint["projector"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        logger.info(f"Checkpoint loaded from {path}")
