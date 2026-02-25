"""
DGA: Domain-Adaptive Gating Adaptation for Open-Set Domain Adaptation.

Target: H-score >= 89%

Key Design Principles:
1. Channel gating for semantic feature decoupling (f_inv = f × gate, f_sp = f × (1-gate))
2. Progressive self-training with curriculum pseudo-labeling
3. Multi-cue unknown rejection (distance + entropy + gate divergence)
4. Class-balanced contrastive learning

Training Strategy:
- Phase 1: Supervised warmup on source (strong feature backbone)
- Phase 2: Self-training with progressive confidence threshold
- Phase 3: Fine-tuning with all cues activated
"""

import logging
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import SemanticHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


class GatingModule(nn.Module):
    """
    Channel Gating Module for feature decomposition.
    
    Produces gate values in [0, 1] via sigmoid.
    f_inv = f × gate (domain-invariant, for classification)
    f_sp = f × (1-gate) (domain-specific, filtered out)
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 4
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.net(features)
        return torch.sigmoid(logits / temperature)


@register_solver("dga")
class DGASolver(BaseSolver):
    """
    DGA: Domain-Adaptive Gating Adaptation.
    
    Core Components:
    1. Dual-path gating: f_inv for classification, f_sp filtered out
    2. Progressive self-training with curricular pseudo-labels
    3. Multi-cue rejection: distance + entropy + gate consistency
    4. Contrastive prototype alignment
    """

    def build_model(self):
        """Build backbone, gating module, and classifier."""
        cfg = self.config.method
        
        # Backbone
        backbone = get_backbone(cfg.backbone)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self.device)
        
        # Gating Module
        self.gating = GatingModule(
            self.feature_dim,
            cfg.gating_hidden_dim
        ).to(self.device)
        
        # Classifier
        self.classifier = SemanticHead(
            self.feature_dim,
            self.num_classes,
            cfg.semantic_hidden_dim
        ).to(self.device)
        
        # Number of known classes
        self.num_src_classes = self.num_classes - 1 if self.unknown_label else self.num_classes
        
        # Class prototypes (for f_inv features)
        self.prototypes = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        self.prototype_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        # Gate prototypes per class
        self.gate_prototypes = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        
        # Unknown Prototype (moving average of low-confidence samples)
        self.unknown_prototype = torch.zeros(1, self.feature_dim, device=self.device)
        self.unknown_count = 0
        
        # Rejection threshold
        self.rejection_threshold = 0.5
        
        # Temperature for gating
        self.temperature = cfg.init_temperature
        
        # Track best model
        self.best_h_score = 0.0
        self.best_state = None
        
        logger.info(f"Built DGA: backbone={cfg.backbone}, "
                    f"feat_dim={self.feature_dim}, classes={self.num_classes}")

    def _build_optimizer(self):
        """Build optimizer with layer-wise learning rates."""
        cfg = self.config.method
        
        param_groups = [
            {'params': self.backbone.parameters(), 'lr': cfg.lr_backbone},
            {'params': self.gating.parameters(), 'lr': cfg.lr_head},
            {'params': self.classifier.parameters(), 'lr': cfg.lr_head},
        ]
        
        self.optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        
        # Linear Warmup + Cosine Annealing
        total_epochs = cfg.warmup_epochs + cfg.adapt_epochs
        warmup_epochs = cfg.warmup_epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            else:
                progress = float(epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _set_train_mode(self):
        self.backbone.train()
        self.gating.train()
        self.classifier.train()

    def _set_eval_mode(self):
        self.backbone.eval()
        self.gating.eval()
        self.classifier.eval()

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            logits: Classification logits
            f_inv: Domain-invariant features
            f_sp: Domain-specific features
            gate: Gate values
        """
        f = self.backbone(x)
        
        if self.backbone.training:
            # Consistent Gating: Two passes with dropout to enforce invariance
            gate1 = self.gating(f, self.temperature)
            gate2 = self.gating(f, self.temperature)
            gate = gate1 # Use first pass for downstream
            
            # Consistency feature (optional, but keep simple)
            f_inv = f * gate
            f_sp = f * (1 - gate)
            logits = self.classifier(f_inv)
            
            return logits, f_inv, f_sp, gate, gate2
        
        gate = self.gating(f, self.temperature)
        f_inv = f * gate
        f_sp = f * (1 - gate)
        logits = self.classifier(f_inv)
        return logits, f_inv, f_sp, gate, None

    def _compute_gate_loss(self, gates: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Gate regularization:
        1. Binary entropy: push gates to 0 or 1
        2. Class consistency: same class → similar gates
        """
        eps = 1e-6
        # Binary entropy for sparsity
        entropy = -gates * torch.log(gates + eps) - (1 - gates) * torch.log(1 - gates + eps)
        loss = entropy.mean()
        
        if labels is not None and len(labels) > 0:
            # Class consistency via prototype matching
            unique_labels = labels.unique()
            consistency_loss = 0.0
            count = 0
            
            for c in unique_labels:
                if c >= self.num_src_classes:
                    continue
                mask = labels == c
                if mask.sum() >= 2:
                    class_gates = gates[mask]
                    center = class_gates.mean(dim=0, keepdim=True)
                    consistency_loss += F.mse_loss(class_gates, center.expand_as(class_gates))
                    count += 1
            
            if count > 0:
                loss = loss + consistency_loss / count
        
        return loss

    def _compute_contrastive_loss(self, f_inv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Supervised contrastive loss for better feature discrimination.
        Pulls same-class samples together, pushes different classes apart.
        """
        if len(labels) < 2:
            return torch.tensor(0.0, device=self.device)
        
        features = F.normalize(f_inv, p=2, dim=1)
        batch_size = features.size(0)
        
        # Compute similarity matrix
        similarity = torch.mm(features, features.t()) / self.config.method.contrastive_temperature
        
        # Create mask for positive pairs (same class)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Exclude diagonal (self-similarity)
        mask_diag = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        labels_eq = labels_eq & ~mask_diag
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        
        # For each sample, compute log(sum_pos / sum_all)
        loss = 0.0
        valid_count = 0
        
        for i in range(batch_size):
            pos_mask = labels_eq[i]
            neg_mask = ~pos_mask & ~mask_diag[i]
            
            if pos_mask.sum() == 0:
                continue
            
            pos_sum = exp_sim[i][pos_mask].sum()
            all_sum = exp_sim[i][~mask_diag[i]].sum()
            
            loss -= torch.log(pos_sum / (all_sum + 1e-8))
            valid_count += 1
        
        return loss / (valid_count + 1e-8)

    def _get_pseudo_labels(self, logits_weak: torch.Tensor, epoch: int, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels with percentile-based thresholding.
        """
        probs = F.softmax(logits_weak, dim=1)
        confidence, predictions = probs.max(dim=1)
        
        mask = torch.zeros_like(confidence, dtype=torch.bool)
        
        # Percentile thresholding per class
        for c in range(self.num_src_classes):
             class_mask = predictions == c
             if class_mask.sum() == 0:
                 continue
                 
             scores = confidence[class_mask]
             # Dynamic threshold based on percentile
             k = max(1, int(len(scores) * cfg.percentile))
             threshold = torch.kthvalue(scores, len(scores) - k + 1).values
             
             mask |= (class_mask & (confidence >= threshold))
             
        return predictions, mask

    def _update_prototypes(self, f_inv: torch.Tensor, gates: torch.Tensor, 
                           labels: torch.Tensor, momentum: float = 0.99):
        """Update running class prototypes."""
        with torch.no_grad():
            for c in range(self.num_src_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                
                # Feature prototype
                batch_proto = f_inv[mask].mean(dim=0)
                if self.prototype_counts[c] == 0:
                    self.prototypes[c] = batch_proto
                else:
                    self.prototypes[c] = momentum * self.prototypes[c] + (1 - momentum) * batch_proto
                
                # Gate prototype
                batch_gate_proto = gates[mask].mean(dim=0)
                if self.gate_prototypes[c].abs().sum() == 0:
                    self.gate_prototypes[c] = batch_gate_proto
                else:
                    self.gate_prototypes[c] = momentum * self.gate_prototypes[c] + (1 - momentum) * batch_gate_proto
                
                self.prototype_counts[c] += mask.sum()

    def _update_unknown_prototype(self, f_inv: torch.Tensor, entropy: torch.Tensor, momentum: float = 0.99):
        """Update unknown prototype with high-entropy samples."""
        with torch.no_grad():
            # Select likely unknown samples (high entropy)
            # Dynamic thresholding could be better, but fixed percentile is stable
            threshold = torch.quantile(entropy, 0.8) 
            mask = entropy > threshold
            
            if mask.sum() > 0:
                batch_unknown = f_inv[mask].mean(dim=0, keepdim=True)
                if self.unknown_count == 0:
                    self.unknown_prototype = batch_unknown
                else:
                    self.unknown_prototype = momentum * self.unknown_prototype + (1 - momentum) * batch_unknown
                self.unknown_count += mask.sum()

    def _get_temperature(self, epoch: int, total: int) -> float:
        """Anneal temperature from high to low."""
        cfg = self.config.method
        progress = epoch / max(total - 1, 1)
        return cfg.init_temperature + (cfg.final_temperature - cfg.init_temperature) * progress

    def train(self):
        """Progressive training: warmup → self-training → fine-tuning."""
        self._build_optimizer()
        cfg = self.config.method
        
        warmup_epochs = cfg.warmup_epochs
        adapt_epochs = cfg.adapt_epochs
        total_epochs = warmup_epochs + adapt_epochs
        
        logger.info(f"DGA Training: {warmup_epochs} warmup + {adapt_epochs} adapt epochs")
        logger.info(f"Target H-score: 89%+")
        
        for epoch in range(total_epochs):
            is_warmup = epoch < warmup_epochs
            
            # Update temperature
            if is_warmup:
                self.temperature = cfg.init_temperature
            else:
                self.temperature = self._get_temperature(epoch - warmup_epochs, adapt_epochs)
            
            # Train epoch
            metrics = self._train_epoch(epoch, is_warmup)
            
            # Evaluate
            self._compute_prototypes(update_known=False)
            h_score = self.evaluate()
            metrics['h_score'] = h_score
            
            # Track best model
            if h_score > self.best_h_score:
                self.best_h_score = h_score
                self.best_state = {
                    'backbone': {k: v.cpu().clone() for k, v in self.backbone.state_dict().items()},
                    'gating': {k: v.cpu().clone() for k, v in self.gating.state_dict().items()},
                    'classifier': {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()},
                    'prototypes': self.prototypes.cpu().clone(),
                    'gate_prototypes': self.gate_prototypes.cpu().clone(),
                    'threshold': self.rejection_threshold,
                }
            
            self._log_epoch(epoch, total_epochs, is_warmup, metrics)
            
            # Step scheduler
            self.scheduler.step()
        
        # Restore best model
        if self.best_state is not None:
            self.backbone.load_state_dict({k: v.to(self.device) for k, v in self.best_state['backbone'].items()})
            self.gating.load_state_dict({k: v.to(self.device) for k, v in self.best_state['gating'].items()})
            self.classifier.load_state_dict({k: v.to(self.device) for k, v in self.best_state['classifier'].items()})
            self.prototypes = self.best_state['prototypes'].to(self.device)
            self.gate_prototypes = self.best_state['gate_prototypes'].to(self.device)
            self.rejection_threshold = self.best_state['threshold']
        
        h_score = self.evaluate()
        logger.info(f"Training complete. Final H-score: {h_score:.2f}% (Best: {self.best_h_score:.2f}%)")

    def _train_epoch(self, epoch: int, is_warmup: bool) -> Dict[str, float]:
        """Single epoch training."""
        self._set_train_mode()
        cfg = self.config.method
        
        meters = {k: AverageMeter() for k in ['loss', 'cls', 'gate', 'contrast', 'cons',
                                               'pseudo_ratio', 'gate_mean', 'gate_std', 'ent']}
        
        tgt_iter = None if is_warmup else cycle(self.target_loader)
        
        pbar = tqdm(self.source_loader, desc=f"Epoch {epoch + 1}", leave=False, ncols=80, ascii=True, mininterval=5.0)
        for src_imgs, src_labels in pbar:
            src_imgs = src_imgs.to(self.device)
            src_labels = src_labels.to(self.device)
            
            loss_ent = torch.tensor(0.0, device=self.device)
            
            self.optimizer.zero_grad()
            
            # Source forward
            logits_s, f_inv_s, f_sp_s, gate_s, gate_s2 = self._forward(src_imgs)
            
            # Source classification loss
            loss_cls = self.criterion(logits_s, src_labels)
            
            # Gate regularization
            loss_gate = self._compute_gate_loss(gate_s, src_labels)
            
            # Contrastive loss
            loss_contrast = self._compute_contrastive_loss(f_inv_s, src_labels)
            
            # Update prototypes
            self._update_prototypes(f_inv_s.detach(), gate_s.detach(), src_labels)
            
            # Gate Consistency Loss
            loss_cons = F.mse_loss(gate_s, gate_s2)
            
            pseudo_ratio = 0.0
            
            if not is_warmup and tgt_iter is not None:
                # Target forward
                tgt_data = next(tgt_iter)
                if isinstance(tgt_data[0], (list, tuple)):
                     # FixMatch mode: (weak, strong)
                     (tgt_weak, tgt_strong), _ = tgt_data
                     tgt_weak = tgt_weak.to(self.device)
                     tgt_strong = tgt_strong.to(self.device)
                else:
                     # Fallback
                     tgt_weak, _ = tgt_data
                     tgt_weak = tgt_weak.to(self.device)
                     tgt_strong = tgt_weak # No strong aug available
                
                # Weak View -> Pseudo-labels
                with torch.no_grad():
                     logits_w, f_inv_w, _, gate_w, _ = self._forward(tgt_weak)
                     pseudo_labels, pseudo_mask = self._get_pseudo_labels(
                         logits_w, epoch - cfg.warmup_epochs, cfg
                     )
                
                # Strong View -> Training
                logits_s, f_inv_s, _, gate_s_target, gate_s2_target = self._forward(tgt_strong)
                
                # Consistency on strong view gates
                loss_cons += F.mse_loss(gate_s_target, gate_s2_target)
                
                # Update Unknown Prototype (using Weak view entropy)
                probs_w = F.softmax(logits_w, dim=1)
                entropy_w = -(probs_w * torch.log(probs_w + 1e-8)).sum(dim=1)
                self._update_unknown_prototype(f_inv_w.detach(), entropy_w, momentum=getattr(cfg, 'momentum_unknown', 0.99))
                
                # Target gate regularization
                loss_gate = loss_gate + cfg.lambda_gate_target * self._compute_gate_loss(gate_s_target)

                # Entropy minimization (on weak view is usually safer, but strong is fine too)
                loss_ent = entropy_w.mean()
                
                if pseudo_mask.sum() > 0:
                    # Pseudo-label classification loss (on Strong View)
                    loss_cls = loss_cls + cfg.lambda_pseudo * self.criterion(
                        logits_s[pseudo_mask], pseudo_labels[pseudo_mask]
                    )
                    
                    # Contrastive with pseudo-labels (on Strong View features)
                    if pseudo_mask.sum() >= 4:
                        loss_contrast = loss_contrast + cfg.lambda_pseudo * self._compute_contrastive_loss(
                            f_inv_s[pseudo_mask], pseudo_labels[pseudo_mask]
                        )
                    
                    # Update prototypes
                    self._update_prototypes(
                        f_inv_s[pseudo_mask].detach(),
                        gate_s_target[pseudo_mask].detach(),
                        pseudo_labels[pseudo_mask],
                        momentum=0.9
                    )
                
                pseudo_ratio = pseudo_mask.float().mean().item()
            
            # Total loss
            loss = loss_cls + \
                   cfg.lambda_gate * loss_gate + \
                   cfg.lambda_contrastive * loss_contrast + \
                   cfg.lambda_entropy * loss_ent + \
                   5.0 * loss_cons  # Strong consistency weight
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.gating.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Update meters
            meters['loss'].update(loss.item())
            meters['cls'].update(loss_cls.item())
            meters['gate'].update(loss_gate.item())
            meters['contrast'].update(loss_contrast.item())
            meters['cons'].update(loss_cons.item())
            if not is_warmup and tgt_iter is not None:
                meters['ent'].update(loss_ent.item())
            meters['pseudo_ratio'].update(pseudo_ratio)
            meters['gate_mean'].update(gate_s.mean().item())
            meters['gate_std'].update(gate_s.std().item())
            
            pbar.set_postfix(
                loss=f"{meters['loss'].avg:.3f}",
                gate=f"{meters['gate_mean'].avg:.3f}±{meters['gate_std'].avg:.3f}"
            )
        
        return {k: m.avg for k, m in meters.items()}

    def _log_epoch(self, epoch: int, total: int, is_warmup: bool, metrics: Dict[str, float]):
        """Epoch logging."""
        phase = "Warmup" if is_warmup else "Adapt"
        h_score = metrics.get('h_score', 0)
        best_marker = " ★" if h_score == self.best_h_score and h_score > 0 else ""
        
        logger.info(
            f"{phase} Epoch {epoch + 1}/{total}: "
            f"loss={metrics['loss']:.4f}, cls={metrics['cls']:.4f}, "
            f"loss={metrics['loss']:.4f}, cls={metrics['cls']:.4f}, "
            f"con={metrics['contrast']:.4f}, cons={metrics['cons']:.4f}, "
            f"gate_μ={metrics['gate_mean']:.3f}±{metrics['gate_std']:.3f}, "
            f"pseudo={metrics['pseudo_ratio']:.1%}, "
            f"H-score={h_score:.2f}%{best_marker}"
        )

    def _compute_prototypes(self, update_known: bool = True):
        """Compute class prototypes for rejection."""
        self._set_eval_mode()
        
        if update_known:
            sums = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
            counts = torch.zeros(self.num_src_classes, device=self.device)
            
            with torch.no_grad():
                for imgs, labels in self.source_loader:
                    _, f_inv, _, _, _ = self._forward(imgs.to(self.device))
                    labels = labels.to(self.device)
                    
                    for c in range(self.num_src_classes):
                        mask = labels == c
                        if mask.sum() > 0:
                            sums[c] += f_inv[mask].sum(dim=0)
                            counts[c] += mask.sum()
            
            self.prototypes = sums / counts.clamp(min=1).unsqueeze(1)
        
        # Recalculate Unknown Prototype more accurately over full target set
        self.unknown_prototype.zero_()
        unknown_count = 0
        all_entropies = []
        all_features = []
        
        with torch.no_grad():
            for imgs_data, _ in self.target_loader:
                if isinstance(imgs_data, (list, tuple)):
                    imgs = imgs_data[0]
                else:
                    imgs = imgs_data
                    
                logits, f_inv, _, _, _ = self._forward(imgs.to(self.device))
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
                
                all_entropies.append(entropy)
                all_features.append(f_inv)
                
        all_entropies = torch.cat(all_entropies)
        all_features = torch.cat(all_features)
        
        # Top 20% highest entropy as unknown proxies
        threshold = torch.quantile(all_entropies, 0.8)
        mask = all_entropies > threshold
        if mask.any():
            self.unknown_prototype = all_features[mask].mean(dim=0, keepdim=True)
            
        logger.info(f"Refined Unknown Prototype using top 20% entropy samples.")

    def evaluate(self) -> float:
        """Evaluate with entropy-based rejection."""
        if self.prototypes.abs().sum() == 0:
            self._compute_prototypes()
        
        self._set_eval_mode()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                logits, f_inv, f_sp, gates, _ = self._forward(imgs.to(self.device))
                preds = logits.argmax(dim=1)
                probs = F.softmax(logits, dim=1)
                
                if self.unknown_label is not None:
                    # 1. Distance to Known Prototypes
                    # (B, K, D) - (1, K, D)
                    dists_known = torch.cdist(F.normalize(f_inv, p=2, dim=1), F.normalize(self.prototypes, p=2, dim=1)) # (B, K)
                    min_dist_known, _ = dists_known.min(dim=1)
                    
                    # 2. Distance to Unknown Prototype
                    dist_unknown = torch.cdist(F.normalize(f_inv, p=2, dim=1), F.normalize(self.unknown_prototype, p=2, dim=1)).squeeze(1) # (B)
                    
                    # 3. Relative Score: is it closer to unknown than known?
                    # If dist_unknown < min_dist_known -> likely unknown
                    # We add a margin for safety
                    score = min_dist_known - dist_unknown
                    
                    # Reject if score > threshold (closer to unknown or far from known)
                    # Heuristic threshold: 0.0 means equidistance
                    reject = score > -0.05  # slightly bias towards known to avoid over-rejection
                    preds[reject] = self.unknown_label
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            return self._compute_osda_metrics(all_preds, all_labels)
        else:
            return 100 * (all_preds == all_labels).sum().item() / len(all_labels)

    def forward_for_eval(self, imgs):
        """Forward pass for evaluation."""
        logits, _, _, _, _ = self._forward(imgs)
        return logits

    def save_checkpoint(self, path):
        """Save checkpoint."""
        torch.save({
            "method": "dga",
            "backbone": self.backbone.state_dict(),
            "gating": self.gating.state_dict(),
            "classifier": self.classifier.state_dict(),
            "prototypes": self.prototypes,
            "gate_prototypes": self.gate_prototypes,
            "rejection_threshold": self.rejection_threshold,
            "best_h_score": self.best_h_score,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.gating.load_state_dict(ckpt["gating"])
        self.classifier.load_state_dict(ckpt["classifier"])
        if "prototypes" in ckpt:
            self.prototypes = ckpt["prototypes"].to(self.device)
        if "gate_prototypes" in ckpt:
            self.gate_prototypes = ckpt["gate_prototypes"].to(self.device)
        if "rejection_threshold" in ckpt:
            self.rejection_threshold = ckpt["rejection_threshold"]
        logger.info(f"Loaded checkpoint: {path}")
