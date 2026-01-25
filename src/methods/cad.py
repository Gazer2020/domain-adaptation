"""
Channel Activation-based Domain Adaptation (CAD) Solver.

This method uses a Channel Gating Module to learn class-consistent channel
activation patterns. The key insight is:
1. Structure-Aware Loss: Forces known classes to have stable gate patterns
2. Anomaly-Aware Loss: Suppresses target channels that source doesn't use
3. Prototype-based rejection: Uses structural fingerprints (class prototypes)
   computed from gating vectors to identify unknown classes based on
   cosine similarity rather than classification confidence
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import ChannelGatingModule, SemanticHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


@register_solver("cad")
class CADSolver(BaseSolver):
    """
    Channel Activation-based Domain Adaptation solver.
    
    Three-stage training:
    1. Pretrain: Standard classification on source domain
    2. Adaptation: Fine-tune with Structure-Aware and Anomaly-Aware losses
    3. Prototype Computation: Calculate class prototypes from source gating vectors
    
    Key components:
    - Channel Gating Module: FC + Sigmoid that outputs gate values g ∈ (0,1)
    - Structure-Aware Loss: MSE between sample gates and class prototypes
    - Anomaly-Aware Loss: Penalizes target activations on source-unused channels
    - Prototype-based Rejection: Uses cosine similarity between sample gates
      and class prototypes to identify unknown classes (structural matching)
    """
    
    def build_model(self):
        """Build feature extractor, channel gating module, and classifier."""
        backbone_name = self.config.method.backbone
        
        # Feature extractor (backbone with fc replaced by Identity)
        backbone = get_backbone(backbone_name)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone.to(self.device)
        
        # Channel Gating Module
        gating_hidden = self.config.method.gating_hidden_dim
        self.gating_module = ChannelGatingModule(
            feature_dim=self.feature_dim,
            hidden_dim=gating_hidden
        ).to(self.device)
        
        # Classification head
        # Classification head
        semantic_hidden_dim = self.config.method.get("semantic_hidden_dim", 256)
        self.classifier = SemanticHead(
            in_features=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=semantic_hidden_dim
        ).to(self.device)
        
        # Number of source classes (for structure loss, excludes unknown class)
        # Default to OSDA behavior: num_classes includes unknown, num_src_classes doesn't
        self.num_src_classes = self.num_classes - 1
        
        # Prototype storage for rejection mechanism
        # Will be computed after training: [num_src_classes, feature_dim]
        self.class_prototypes = torch.zeros(
            self.num_src_classes, self.feature_dim, device=self.device
        )
        
        logger.info(f"Built CAD model: backbone={backbone_name}, "
                    f"feature_dim={self.feature_dim}, "
                    f"num_src_classes={self.num_src_classes}, "
                    f"num_classes={self.num_classes}")

        # Adaptive rejection threshold (to be computed)
        self.rejection_threshold = 0.5

    def _get_trainable_params(self):
        """Get all trainable parameters."""
        params = list(self.feature_extractor.parameters())
        params += list(self.gating_module.parameters())
        params += list(self.classifier.parameters())
        return params

    def _build_optimizer(self):
        """Build optimizer for all training stages."""
        self.optimizer = optim.SGD(
            self._get_trainable_params(),
            lr=self.config.method.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

    def _set_train_mode(self):
        """Set all components to training mode."""
        self.feature_extractor.train()
        self.gating_module.train()
        self.classifier.train()

    def _set_eval_mode(self):
        """Set all components to evaluation mode."""
        self.feature_extractor.eval()
        self.gating_module.eval()
        self.classifier.eval()

    def _forward_with_gating(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with channel gating.
        
        Args:
            imgs: Input images [B, C, H, W]
            
        Returns:
            logits: Classification logits [B, num_classes]
            features: Raw features from backbone [B, D]
            gate: Gate values [B, D] in (0, 1)
            gated_features: Features after gating [B, D]
        """
        # Get raw features from backbone (after GAP)
        features = self.feature_extractor(imgs)  # [B, D]
        
        # Get gate vector
        gate = self.gating_module(features)  # [B, D], values in (0, 1)
        
        # Apply gating (element-wise product)
        gated_features = features * gate  # [B, D]
        
        # Classification
        logits = self.classifier(gated_features)
        
        return logits, features, gate, gated_features

    def _compute_structure_loss(self, gates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Structure-Aware Loss (intra-class consistency).
        
        Forces samples of the same class to have similar gate patterns.
        "If it's a dog, channels 5, 12, 99 must be consistently ON."
        
        Args:
            gates: Gate values [B, D]
            labels: Class labels [B]
            
        Returns:
            structure_loss: MSE between sample gates and class prototypes
        """
        batch_size = gates.size(0)
        
        # Initialize class centers and counts
        class_centers = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        class_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        # Accumulate gates per class using index_add
        class_centers.index_add_(0, labels, gates)
        class_counts.index_add_(0, labels, torch.ones(batch_size, device=self.device))
        
        # Average to get prototypes (avoid division by zero)
        class_counts = class_counts.clamp(min=1)
        class_prototypes = class_centers / class_counts.unsqueeze(1)
        
        # Get prototype for each sample
        sample_prototypes = class_prototypes[labels]  # [B, D]
        
        # MSE loss between sample gate and its class prototype
        structure_loss = F.mse_loss(gates, sample_prototypes)
        
        return structure_loss

    def _compute_anomaly_loss(self, src_gates: torch.Tensor, tgt_gates: torch.Tensor) -> torch.Tensor:
        """
        Compute Anomaly-Aware Loss (suppress unknown channels).
        
        Penalizes target domain channels that source domain doesn't use.
        This helps suppress background/noise channels that might indicate unknown classes.
        
        Args:
            src_gates: Source domain gate values [B_s, D]
            tgt_gates: Target domain gate values [B_t, D]
            
        Returns:
            anomaly_loss: Penalty for activating source-unused channels in target
        """
        # Source domain channel importance (detached - no gradient!)
        # This represents the "global importance" of each channel for known classes
        src_importance = src_gates.mean(dim=0).detach()  # [D]
        
        # Source "non-importance" weights (channels that source doesn't use)
        src_non_importance = 1 - src_importance  # [D]
        
        # Penalize target gates that activate where source doesn't
        # This forces target to close "anomaly" channels
        anomaly_loss = (tgt_gates * src_non_importance).mean()
        
        return anomaly_loss

    def train(self):
        """Two-stage training: pretrain + adaptation."""
        self._build_optimizer()
        
        pretrain_epochs = self.config.method.pretrain_epochs
        adapt_epochs = self.config.method.adapt_epochs
        
        # Stage 1: Pretrain on source
        logger.info(f"Stage 1: Pretraining for {pretrain_epochs} epochs...")
        self._train_pretrain_stage(pretrain_epochs)
        
        # Stage 2: Adaptation with gating losses
        logger.info(f"Stage 2: Adaptation for {adapt_epochs} epochs...")
        self._train_adaptation_stage(adapt_epochs)
        
        # Stage 3: Compute class prototypes for rejection mechanism
        logger.info("Computing class prototypes from source domain...")
        self._compute_class_prototypes()
        
        # Evaluate the final model
        logger.info("Evaluating final model...")
        final_hos = self.evaluate()
        logger.info(f"Final evaluation - HOS: {final_hos:.2f}%")
        
        logger.info("Training finished.")

    def _train_pretrain_stage(self, max_epochs: int):
        """
        Stage 1: Standard classification on source domain.
        
        Train the feature extractor and classifier without gating losses.
        The gating module learns basic patterns but isn't explicitly supervised.
        """
        for epoch in range(max_epochs):
            self._set_train_mode()
            loss_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Pretrain {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward with gating
                logits, _, _, _ = self._forward_with_gating(src_imgs)
                
                # Cross-entropy loss only
                loss = self.criterion(logits, src_labels)
                
                loss.backward()
                self.optimizer.step()
                
                loss_meter.update(loss.item())
                pbar.set_postfix({"loss": loss_meter.avg})
            
            hos = self.evaluate()
            logger.info(f"Pretrain Epoch {epoch+1}: Loss={loss_meter.avg:.4f}, HOS={hos:.2f}%")

    def _train_adaptation_stage(self, max_epochs: int):
        """
        Stage 2: Fine-tune with Structure-Aware and Anomaly-Aware losses.
        
        Three losses are combined:
        1. Classification loss (CE) on source domain
        2. Structure loss: Enforce intra-class gate consistency
        3. Anomaly loss: Suppress target channels unused by source
        """
        for epoch in range(max_epochs):
            self._set_train_mode()
            
            tgt_iter = cycle(self.target_loader)
            cls_loss_meter = AverageMeter()
            struct_loss_meter = AverageMeter()
            anomaly_loss_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Adapt {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)
                
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                
                self.optimizer.zero_grad()
                
                loss, loss_dict = self._compute_total_loss_terms(src_imgs, src_labels, tgt_imgs)
                
                loss.backward()
                self.optimizer.step()
                
                cls_loss_meter.update(loss_dict["cls"])
                struct_loss_meter.update(loss_dict["struct"])
                anomaly_loss_meter.update(loss_dict["anom"])
                
                pbar.set_postfix({
                    "cls": cls_loss_meter.avg,
                    "struct": struct_loss_meter.avg,
                    "anom": anomaly_loss_meter.avg
                })
            
            hos = self.evaluate()
            logger.info(f"Adapt Epoch {epoch+1}: Cls={cls_loss_meter.avg:.4f}, "
                        f"Struct={struct_loss_meter.avg:.4f}, "
                        f"Anom={anomaly_loss_meter.avg:.4f}, HOS={hos:.2f}%")

    def _compute_total_loss_terms(self, src_imgs, src_labels, tgt_imgs):
        """
        Compute total loss and individual loss terms.
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dict of scalar values (for logging)
        """
        # Source forward with gating
        src_logits, _, src_gates, _ = self._forward_with_gating(src_imgs)
        
        # Target forward with gating
        _, _, tgt_gates, _ = self._forward_with_gating(tgt_imgs)
        
        # Loss 1: Classification loss on source
        cls_loss = self.criterion(src_logits, src_labels)
        
        # Loss 2: Structure-Aware loss (intra-class consistency)
        structure_loss = self._compute_structure_loss(src_gates, src_labels)
        
        # Loss 3: Anomaly-Aware loss (suppress unknown channels)
        anomaly_loss = self._compute_anomaly_loss(src_gates, tgt_gates)
        
        lambda_structure = self.config.method.lambda_structure
        lambda_anomaly = self.config.method.lambda_anomaly
        
        # Total loss
        loss = cls_loss + lambda_structure * structure_loss + lambda_anomaly * anomaly_loss
        
        loss_dict = {
            "cls": cls_loss.item(),
            "struct": structure_loss.item(),
            "anom": anomaly_loss.item()
        }
        
        return loss, loss_dict

    def _compute_class_prototypes(self):
        """
        Compute class prototypes (structural fingerprints) from source domain.
        
        For each known class, compute the average gating vector across all
        source samples of that class. These prototypes represent the characteristic
        channel activation patterns for each class.
        """
        self._set_eval_mode()
        
        # Accumulators for each class
        class_gate_sums = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        class_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.source_loader, desc="Computing prototypes"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Get gating vectors
                _, _, gates, _ = self._forward_with_gating(imgs)
                
                # Accumulate gates per class
                class_gate_sums.index_add_(0, labels, gates)
                class_counts.index_add_(0, labels, torch.ones(len(labels), device=self.device))
        
        # Compute prototypes (average gates per class)
        class_counts = class_counts.clamp(min=1)  # Avoid division by zero
        self.class_prototypes = class_gate_sums / class_counts.unsqueeze(1)
        
        
        # Compute adaptive rejection threshold
        # Based on distribution of source sample similarities to their class prototypes
        logger.info("Computing adaptive rejection threshold...")
        all_similarities = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.source_loader, desc="Computing threshold"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Get gating vectors
                _, _, gates, _ = self._forward_with_gating(imgs)
                
                # Normalize
                gates_norm = F.normalize(gates, p=2, dim=1)
                prototypes_norm = F.normalize(self.class_prototypes, p=2, dim=1)
                
                # Get prototype for each sample's ground truth class
                sample_prototypes = prototypes_norm[labels]
                
                # Compute cosine similarity
                # (B, D) * (B, D) -> (B,)
                sim = (gates_norm * sample_prototypes).sum(dim=1)
                all_similarities.append(sim)
                
        all_similarities = torch.cat(all_similarities)
        
        # Set threshold to 5th percentile (exclude outliers)
        # "If target sample is less similar to nearest prototype than 95% of source samples 
        # are to their own class prototype, reject it."
        q = 0.05
        self.rejection_threshold = torch.quantile(all_similarities, q).item()
        
        logger.info(f"Computed prototypes for {self.num_src_classes} classes, "
                    f"shape: {self.class_prototypes.shape}")
        logger.info(f"Adaptive rejection threshold (5th percentile): {self.rejection_threshold:.4f}")

    def predict_with_rejection(self, preds: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Custom rejection strategy for CAD.
        
        If prototypes are available, rejection is already handled in forward_for_eval
        via structural matching (cosine similarity).
        Otherwise, fall back to base class's confidence thresholding.
        """
        if self.class_prototypes.abs().sum() > 0:
            # Prototypes exist: predictions already reflect prototype-based rejection
            # via forward_for_eval boosting unknown class logits
            logger.info("Using prototype-based rejection (structural matching)")
            return preds
        else:
            logger.info(f"Using confidence-based rejection (threshold={self.unknown_threshold})")
            return super().predict_with_rejection(preds, probs)

    # Note: compute_loss is not implemented - CAD uses custom train()

    def forward_for_eval(self, imgs):
        """
        Forward pass for evaluation with prototype-based rejection.
        
        Instead of confidence-based rejection, we use structural matching:
        - Compute gating vector for each sample
        - Calculate cosine similarity with all class prototypes
        - If max similarity < threshold, boost unknown class logit
        
        Returns:
            logits: Modified logits with unknown class boosted for anomalous samples
        """
        logits, _, gates, _ = self._forward_with_gating(imgs)
        
        # Check if prototypes have been computed (non-zero)
        if self.class_prototypes.abs().sum() > 0:
            # Normalize gates and prototypes for cosine similarity
            gates_norm = F.normalize(gates, p=2, dim=1)  # [B, D]
            prototypes_norm = F.normalize(self.class_prototypes, p=2, dim=1)  # [C, D]
            
            # Compute cosine similarity: [B, C]
            similarities = torch.mm(gates_norm, prototypes_norm.t())
            
            # Get maximum similarity for each sample
            max_similarities, _ = similarities.max(dim=1)  # [B]
            
            # Get threshold (adaptive)
            threshold = self.rejection_threshold
            
            # Find anomalous samples (low structural match with all known classes)
            is_anomalous = max_similarities < threshold  # [B]
            
            # Boost unknown class logit for anomalous samples
            # Force them to be classified as unknown by setting a very high logit
            if is_anomalous.any():
                # Unknown class is the last class (index = num_classes - 1)
                logits[is_anomalous, -1] = 100.0  # Large value to force unknown prediction
        
        return logits

    def save_checkpoint(self, path):
        """Save all model components including prototypes."""
        torch.save({
            "method": "cad",
            "feature_extractor": self.feature_extractor.state_dict(),
            "gating_module": self.gating_module.state_dict(),
            "classifier": self.classifier.state_dict(),
            "class_prototypes": self.class_prototypes,
            "rejection_threshold": self.rejection_threshold,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """Load all model components including prototypes."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        
        if "feature_extractor" in checkpoint:
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            self.gating_module.load_state_dict(checkpoint["gating_module"])
            self.classifier.load_state_dict(checkpoint["classifier"])
            
            # Load prototypes if available
            if "class_prototypes" in checkpoint:
                self.class_prototypes = checkpoint["class_prototypes"].to(self.device)
                logger.info(f"Loaded class prototypes with shape {self.class_prototypes.shape}")
            else:
                logger.warning("No class prototypes found in checkpoint")
                
            if "rejection_threshold" in checkpoint:
                self.rejection_threshold = checkpoint["rejection_threshold"]
                logger.info(f"Loaded adaptive rejection threshold: {self.rejection_threshold:.4f}")
            else:
                logger.warning("No rejection threshold found, using default 0.5")
                self.rejection_threshold = 0.5
        else:
            logger.warning("Loading from old checkpoint format - may be incompatible")
            
        logger.info(f"Model loaded from {path}")
