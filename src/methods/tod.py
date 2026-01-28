"""
Tri-partition Orthogonal Decomposition (TOD) for Open Set Domain Adaptation.

This method implements:
1. Hard channel gating to split features into Invariant (Structure) and Specific (Texture/Noise).
2. Tri-Constraint Optimization:
   - Constraint A: Domain Decoupling (Inv -> Adversarial, Sp -> Cooperative)
   - Constraint B: Consistency (Inv should be robust to augmentation)
   - Constraint C: Information Maximization (Inv should reduce entropy on Target)
3. Structured Clustering for specific OSDA rejection.
"""

import logging
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import ChannelGatingModule, SemanticHead, DomainHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


class GradReverse(Function):
    """
    Gradient Reversal Layer for Adversarial Training.
    Forward: Identity
    Backward: Negate gradient * alpha
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


@register_solver("tod")
class TODSolver(BaseSolver):
    """
    Tri-partition Orthogonal Decomposition Solver.
    """

    def build_model(self):
        """Build all components: Backbone, Gate, Classifier, Discriminator."""
        backbone_name = self.config.method.backbone
        
        # 1. Feature Extractor (Backbone)
        backbone = get_backbone(backbone_name)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone.to(self.device)
        
        # 2. Channel Gating Module
        gating_hidden = self.config.method.gating_hidden_dim
        self.gating_module = ChannelGatingModule(
            feature_dim=self.feature_dim,
            hidden_dim=gating_hidden
        ).to(self.device)
        
        # 3. Task Classifier
        # Uses semantic_hidden_dim from config
        semantic_hidden = self.config.method.semantic_hidden_dim
        self.classifier = SemanticHead(
            in_features=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=semantic_hidden
        ).to(self.device)
        
        # 4. Domain Discriminator
        discriminator_hidden = self.config.method.discriminator_hidden_dim
        self.discriminator = DomainHead(
            in_features=self.feature_dim,
            hidden_dim=discriminator_hidden
        ).to(self.device)
        
        # Store Reference to known class prototypes (for structured clustering)
        # We only really care about source classes for structure
        self.num_src_classes = self.num_classes - 1 if self.unknown_label is not None else self.num_classes
        self.prototypes = None  # Will be initialized/updated during training

        logger.info(f"Built TOD model components on {self.device}")

    def _get_trainable_params(self):
        return (
            list(self.feature_extractor.parameters()) +
            list(self.gating_module.parameters()) +
            list(self.classifier.parameters()) +
            list(self.discriminator.parameters())
        )

    def _build_optimizer(self):
        self.optimizer = optim.SGD(
            self._get_trainable_params(),
            lr=self.config.method.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

    def _set_train_mode(self):
        self.feature_extractor.train()
        self.gating_module.train()
        self.classifier.train()
        self.discriminator.train()

    def _set_eval_mode(self):
        self.feature_extractor.eval()
        self.gating_module.eval()
        self.classifier.eval()
        self.discriminator.eval()

    def _compute_alpha(self, epoch: int, total_epochs: int) -> float:
        """Progressive alpha scheduling for GRL (DANN-style)."""
        p = epoch / max(total_epochs, 1)
        return float(2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

    def _forward_decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with decomposition.
        Returns:
            f: Raw features
            f_inv: Invariant features (Structure)
            f_sp: Specific features (Texture/Noise)
            gate: Gate values
        """
        f = self.feature_extractor(x)
        gate = self.gating_module(f)
        
        f_inv = f * gate
        f_sp = f * (1.0 - gate)
        
        return f, f_inv, f_sp, gate

    def train(self):
        """Full training procedure."""
        self._build_optimizer()
        
        pretrain_epochs = self.config.method.pretrain_epochs
        adapt_epochs = self.config.method.adapt_epochs
        
        # Stage 1: Source Initialization
        if pretrain_epochs > 0:
            logger.info(f"Starting Stage 1: Source Initialization ({pretrain_epochs} epochs)")
            self._train_source_init_stage(pretrain_epochs)
        
        # Stage 2: Tri-Constraint Adaptation
        if adapt_epochs > 0:
            # Reset prototypes to force recomputation with current model
            self.prototypes = None
            logger.info(f"Starting Stage 2: Tri-Constraint Adaptation ({adapt_epochs} epochs)")
            self._train_adaptation_stage(adapt_epochs)
            
        logger.info("Training finished.")

    def _train_source_init_stage(self, epochs: int):
        """
        Phase 1: Learn initial structure from Source.
        Minimize CrossEntropy on f_inv.
        """
        for epoch in range(epochs):
            self._set_train_mode()
            loss_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Init Epoch {epoch+1}/{epochs}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                _, f_inv, _, _ = self._forward_decompose(imgs)
                logits = self.classifier(f_inv)
                
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                loss_meter.update(loss.item())
                pbar.set_postfix({"loss": loss_meter.avg})
            
            # Optional: Eval
            acc = self.evaluate()
            logger.info(f"Init Epoch {epoch+1} - Loss: {loss_meter.avg:.4f} - Eval Acc: {acc:.2f}%")

    def _train_adaptation_stage(self, epochs: int):
        """
        Phase 2: Adversarial Adaptation with Tri-Constraints.
        """
        # Adaptation loop
        for epoch in range(epochs):
            self._set_train_mode()
            
            tgt_iter = cycle(self.target_loader)
            
            meter_cls = AverageMeter()
            meter_dom = AverageMeter()
            meter_con = AverageMeter()
            meter_ent = AverageMeter()
            meter_cluster = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Adapt Epoch {epoch+1}/{epochs}")
            
            # Update prototypes at start of epoch (or batch? Epoch is safer/faster)
            # For simplicity, we can do it on the fly or periodically.
            # Let's do it implicitly via current batch for now to save time, or accumulated.
            
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)
                # Strong augmentation simulated: 
                # Ideally we need a strong aug loader. For now, we apply some noise or random crop if not available.
                # Assuming loader output is already augmented or we rely on standard aug.
                # To simulate "strong augmentation" for Constraint B, let's just use the same batch 
                # but maybe with dropout or feature perturbation if we don't have image aug tensors.
                # Actually, let's assume tgt_imgs is one view. We need a second view for consistency.
                # Since we don't have easy access to strong aug inside the loop without changing loader,
                # we will use feature-level perturbation (e.g. Dropout) or just compare against itself (weak consistency).
                # BETTER: For this implementation, let's use the SAME image but enforce consistency 
                # implies invariance to small perturbations. 
                # OR pseudo-augment by flipping.
                tgt_imgs_aug = torch.flip(tgt_imgs, dims=[3]) # Simple horizontal flip
                
                src_imgs, src_labels = src_imgs.to(self.device), src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                tgt_imgs_aug = tgt_imgs_aug.to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- Step 1: Decomposition ---
                # Source
                _, f_inv_s, f_sp_s, gate_s = self._forward_decompose(src_imgs)
                # Target
                _, f_inv_t, f_sp_t, gate_t = self._forward_decompose(tgt_imgs)
                # Target Aug
                _, f_inv_t_aug, _, _ = self._forward_decompose(tgt_imgs_aug)
                
                # --- Task Loss (Source) ---
                cls_logits_s = self.classifier(f_inv_s)
                loss_cls = self.criterion(cls_logits_s, src_labels)
                
                # --- Constraint A: Domain Decoupling ---
                # A1: Invariant features should be domain-indistinguishable (Adversarial)
                # Progressive GRL alpha scheduling
                alpha = self._compute_alpha(epoch, epochs)
                f_inv_s_grl = grad_reverse(f_inv_s, alpha)
                f_inv_t_grl = grad_reverse(f_inv_t, alpha)
                
                dom_pred_inv_s = self.discriminator(f_inv_s_grl)
                dom_pred_inv_t = self.discriminator(f_inv_t_grl)
                
                # Domain Labels: Source=1, Target=0
                d_labels_s = torch.ones_like(dom_pred_inv_s)
                d_labels_t = torch.zeros_like(dom_pred_inv_t)
                
                loss_dom_inv = (
                    F.binary_cross_entropy_with_logits(dom_pred_inv_s, d_labels_s) +
                    F.binary_cross_entropy_with_logits(dom_pred_inv_t, d_labels_t)
                ) * 0.5
                
                # A2: Specific features should be domain-distinguishable (Cooperative)
                dom_pred_sp_s = self.discriminator(f_sp_s)
                dom_pred_sp_t = self.discriminator(f_sp_t)
                
                loss_dom_sp = (
                    F.binary_cross_entropy_with_logits(dom_pred_sp_s, d_labels_s) +
                    F.binary_cross_entropy_with_logits(dom_pred_sp_t, d_labels_t)
                ) * 0.5
                
                # --- Constraint B: Consistency ---
                # f_inv should be invariant to augmentation
                loss_con = F.mse_loss(f_inv_t, f_inv_t_aug)
                
                # --- Constraint C: Selective Entropy Minimization ---
                # Only minimize entropy on reliable (low-entropy) samples to preserve unknown detection
                logits_t = self.classifier(f_inv_t)
                probs_t = F.softmax(logits_t, dim=1)
                sample_entropy = -(probs_t * torch.log(probs_t + 1e-6)).sum(dim=1)
                
                with torch.no_grad():
                    reliable_mask = sample_entropy < self.config.method.entropy_threshold
                
                if reliable_mask.sum() > 0:
                    loss_ent = sample_entropy[reliable_mask].mean()
                else:
                    loss_ent = torch.tensor(0.0, device=self.device)
                
                # --- Structured Clustering (Simplified) ---
                # Ideally: maintain moving average prototypes. 
                # For this step, we can use the source batch centers as proxies or just omit if too complex.
                # Let's implement a simple "Center Loss" style on Source
                # and "Proto-alignment" on Target if confident.
                
                # Source Clustering: Pull f_inv_s to boolean class centers
                # Simple implementation: Same as CAD structure loss
                # Calculate prototypes from current source batch
                loss_cluster = torch.tensor(0.0, device=self.device)
                if self.config.method.lambda_cluster > 0:
                    # Intra-class compactness on Source
                    # We can use the logic from CAD or simple center distance
                    # For simplicity, let's reuse the logic:
                    # "Pull known classes" -> Minimize distance to class center
                    unique_labels = src_labels.unique()
                    for y in unique_labels:
                        mask = (src_labels == y)
                        center = f_inv_s[mask].mean(dim=0).detach()
                        loss_cluster += F.mse_loss(f_inv_s[mask], center.unsqueeze(0).expand_as(f_inv_s[mask]))
                    loss_cluster /= len(unique_labels)

                # Total Loss
                cfg = self.config.method
                total_loss = (
                    loss_cls + 
                    cfg.lambda_dom_inv * loss_dom_inv +
                    cfg.lambda_dom_sp * loss_dom_sp +
                    cfg.lambda_con * loss_con +
                    cfg.lambda_ent * loss_ent + 
                    cfg.lambda_cluster * loss_cluster
                )
                
                total_loss.backward()
                self.optimizer.step()
                
                # Logging
                meter_cls.update(loss_cls.item())
                meter_dom.update(loss_dom_inv.item()) # Log adversarial part mainly
                meter_con.update(loss_con.item())
                meter_ent.update(loss_ent.item())
                meter_cluster.update(loss_cluster.item())
                
                pbar.set_postfix({
                    "cls": f"{meter_cls.avg:.3f}",
                    "dom_inv": f"{meter_dom.avg:.3f}",
                    "ent": f"{meter_ent.avg:.3f}"
                })
            
            # Eval
            acc = self.evaluate()
            logger.info(
                f"Adapt Epoch {epoch+1} - "
                f"Cls: {meter_cls.avg:.4f}, DomInv: {meter_dom.avg:.4f}, "
                f"Con: {meter_con.avg:.4f}, Ent: {meter_ent.avg:.4f}, "
                f"Cluster: {meter_cluster.avg:.4f} - HOS: {acc:.2f}%"
            )

    def _compute_prototypes_and_threshold(self):
        """
        Compute prototypes and rejection threshold using Source Quantile Method.
        """
        self._set_eval_mode()
        
        # 1. Compute Prototypes (Mean of Gated Inv Features? Or Mean of Inv Features?)
        # User formula: d = || f_inv - (mu_k * w) ||^2.
        # This implies mu_k is the prototype of features f.
        # So we should compute mu_k as mean of f (or f_inv) for each class.
        # Let's assume mu_k is mean of f_inv.
        
        logger.info("Computing prototypes from source data...")
        class_sums = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        class_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        # Store all source distances for threshold calculation
        source_distances = []
        
        # First Pass: Compute Prototypes
        with torch.no_grad():
            for imgs, labels in tqdm(self.source_loader, desc="Computing Prototypes"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                _, f_inv, _, _ = self._forward_decompose(imgs)
                
                # Accumulate
                class_sums.index_add_(0, labels, f_inv)
                class_counts.index_add_(0, labels, torch.ones(len(labels), device=self.device))
        
        class_counts = class_counts.clamp(min=1)
        self.prototypes = class_sums / class_counts.unsqueeze(1)
        
        # Second Pass: Compute Threshold (Source Quantile)
        logger.info("Computing rejection threshold...")
        all_dists = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.source_loader, desc="Computing Threshold"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                f, f_inv, _, gate = self._forward_decompose(imgs)
                
                # Formula: d_k = || f_inv - (mu_k * w) ||^2
                # But here we know the label k. So we compute distance to correct prototype.
                
                # Get correct prototypes
                protos = self.prototypes[labels] # [B, D]
                
                # Weighted prototype: mu_k * w
                weighted_protos = protos * gate
                
                # Distance
                # d = || f_inv - weighted_protos ||^2
                # Sum over dimensions
                dists = (f_inv - weighted_protos).pow(2).sum(dim=1) # [B]
                all_dists.append(dists)
        
        all_dists = torch.cat(all_dists)
        
        # Compute threshold at rejection_quantile percentile
        q = self.config.method.rejection_quantile
        sorted_dists, _ = torch.sort(all_dists)
        threshold_idx = min(int(len(sorted_dists) * q), len(sorted_dists) - 1)
            
        self.rejection_threshold = sorted_dists[threshold_idx].item()
        
        logger.info(f"Computed Threshold (Q={q}): {self.rejection_threshold:.4f}")

    def evaluate(self):
        """
        Custom Evaluation Flow:
        1. Decompose
        2. Calc Structural Distance to all prototypes
        3. Rejection logic
        """
        if self.prototypes is None:
            self._compute_prototypes_and_threshold()
            
        self._set_eval_mode()
        
        all_preds = []
        all_labels = []
        all_dists = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.target_test_loader, desc="Evaluating"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # 1. Feature Decomposition
                f, f_inv, _, gate = self._forward_decompose(imgs)
                
                # 2. Structural Distance Calculation
                # Need distance to ALL prototypes.
                # d_k = || f_inv - (mu_k * w) ||^2
                # efficient broadcast?
                # f_inv: [B, D]
                # gate: [B, D]
                # prototypes: [K, D]
                
                # Expand to [B, K, D]
                B, D = f_inv.shape
                K = self.num_src_classes
                
                f_inv_exp = f_inv.unsqueeze(1) # [B, 1, D]
                gate_exp = gate.unsqueeze(1)   # [B, 1, D]
                protos_exp = self.prototypes.unsqueeze(0) # [1, K, D]
                
                weighted_protos = protos_exp * gate_exp # [B, K, D]
                
                dists = (f_inv_exp - weighted_protos).pow(2).sum(dim=2) # [B, K]
                
                # 3. Find Min Distance and Best Class
                min_dists, best_classes = dists.min(dim=1) # [B]
                
                # 4. Rejection
                # If min_dist >= tau, unknown.
                preds = best_classes.clone()
                unknown_mask = min_dists >= self.rejection_threshold
                
                if self.unknown_label is not None:
                    preds[unknown_mask] = self.unknown_label
                    
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # Use BaseSolver metrics
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            return self._compute_osda_metrics(all_preds, all_labels)
        else:
            correct = (all_preds == all_labels).sum().item()
            total = len(all_labels)
            return 100 * correct / total if total > 0 else 0.0

    def forward_for_eval(self, imgs):
        """Evaluation uses only the Invariant features."""
        _, f_inv, _, _ = self._forward_decompose(imgs)
        # Note: Standard forward_for_eval isn't used by custom evaluate(), 
        # but kept for compatibility if called otherwise.
        return self.classifier(f_inv)

    def save_checkpoint(self, path):
        torch.save({
            "method": "tod",
            "feature_extractor": self.feature_extractor.state_dict(),
            "gating_module": self.gating_module.state_dict(),
            "classifier": self.classifier.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.gating_module.load_state_dict(checkpoint["gating_module"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        logger.info(f"Model loaded from {path}")
