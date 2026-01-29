"""
Tri-partition Orthogonal Decomposition (TOD) for Open Set Domain Adaptation.

Core Innovation: Channel Gating for Feature Decomposition
- f_inv = f × gate → Invariant/Structure features (domain-shared)
- f_sp = f × (1-gate) → Specific/Texture features (domain-specific)

Key Improvements:
1. Progressive Gate Warmup: temperature scheduling for stable training
2. Sparsity Regularization: encourage binary-like gates
3. Orthogonality Constraint: ensure f_inv ⊥ f_sp
4. MMD Alignment: stable domain adaptation (replaces adversarial GRL)
5. Feature-based rejection: uses f_inv distance to class prototypes for unknown detection
"""

import logging
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import SemanticHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


class TemperatureGatingModule(nn.Module):
    """
    Channel Gating Module with Temperature Scheduling.
    
    Uses temperature to control gate sharpness:
    - High temperature (e.g., 10) → gates ≈ 0.5 → soft gating
    - Low temperature (e.g., 1) → gates → 0 or 1 → hard gating
    
    With proper initialization, gates start diverse and become specialized.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = None):
        super().__init__()
        self.feature_dim = feature_dim
        
        if hidden_dim is None:
            hidden_dim = feature_dim // 4
        
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        
        # Initialize with small weights to start with balanced gates (~0.5)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute gate values with temperature scaling.
        
        Args:
            features: [B, D] feature vectors
            temperature: temperature for sigmoid (higher = softer)
            
        Returns:
            gate: [B, D] gate values in (0, 1)
        """
        x = F.relu(self.bn1(self.fc1(features)))
        logits = self.fc2(x)
        gate = torch.sigmoid(logits / temperature)
        return gate


def compute_mmd(source: torch.Tensor, target: torch.Tensor, kernel: str = 'rbf') -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy between source and target features.
    
    MMD is a stable alternative to adversarial domain alignment.
    """
    n_s, n_t = source.size(0), target.size(0)
    
    if n_s == 0 or n_t == 0:
        return torch.tensor(0.0, device=source.device)
    
    # Compute pairwise distances
    def rbf_kernel(x, y, sigma=1.0):
        dist = torch.cdist(x, y, p=2)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    # Auto bandwidth selection (median heuristic)
    with torch.no_grad():
        all_data = torch.cat([source, target], dim=0)
        pairwise_dist = torch.cdist(all_data, all_data, p=2)
        sigma = torch.median(pairwise_dist[pairwise_dist > 0]).item()
        sigma = max(sigma, 0.1)  # Prevent too small sigma
    
    k_ss = rbf_kernel(source, source, sigma)
    k_tt = rbf_kernel(target, target, sigma)
    k_st = rbf_kernel(source, target, sigma)
    
    mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
    return torch.clamp(mmd, min=0.0)


@register_solver("tod")
class TODSolver(BaseSolver):
    """
    Tri-partition Orthogonal Decomposition Solver.
    
    Core mechanism: Channel gating decomposes features into:
    - f_inv (invariant): domain-shared structural features
    - f_sp (specific): domain-specific texture/noise features
    
    Classifier operates on f_inv to learn domain-invariant representations.
    """

    def build_model(self):
        """Build all components: Backbone, Gate, Classifier."""
        backbone_name = self.config.method.backbone
        
        # 1. Feature Extractor (Backbone)
        backbone = get_backbone(backbone_name)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone.to(self.device)
        
        # 2. Temperature-based Gating Module
        gating_hidden = self.config.method.gating_hidden_dim
        self.gating_module = TemperatureGatingModule(
            feature_dim=self.feature_dim,
            hidden_dim=gating_hidden
        ).to(self.device)
        
        # 3. Task Classifier (operates on f_inv, the gated invariant features)
        semantic_hidden = self.config.method.semantic_hidden_dim
        self.classifier = SemanticHead(
            in_features=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=semantic_hidden
        ).to(self.device)
        
        # Number of source classes (excludes unknown)
        self.num_src_classes = self.num_classes - 1 if self.unknown_label is not None else self.num_classes
        
        # Prototype storage for rejection mechanism
        self.class_prototypes = torch.zeros(
            self.num_src_classes, self.feature_dim, device=self.device
        )
        self.rejection_threshold = 0.5
        
        # Temperature scheduling
        self.current_temperature = self.config.method.init_temperature

        logger.info(f"Built TOD model with gating core on {self.device}")

    def _get_trainable_params(self):
        return (
            list(self.feature_extractor.parameters()) +
            list(self.gating_module.parameters()) +
            list(self.classifier.parameters())
        )

    def _build_optimizer(self):
        self.optimizer = optim.SGD(
            self._get_trainable_params(),
            lr=self.config.method.lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )

    def _set_train_mode(self):
        self.feature_extractor.train()
        self.gating_module.train()
        self.classifier.train()

    def _set_eval_mode(self):
        self.feature_extractor.eval()
        self.gating_module.eval()
        self.classifier.eval()

    def _forward_decompose(self, x: torch.Tensor, temperature: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with feature decomposition via gating.
        
        This is the CORE of TOD:
        - f_inv = f × gate (invariant features for classification)
        - f_sp = f × (1-gate) (specific features, domain-dependent)
        
        Returns:
            f: Raw features from backbone
            f_inv: Invariant features (gated)
            f_sp: Specific features (complement)
            gate: Gate values [B, D]
        """
        if temperature is None:
            temperature = self.current_temperature
            
        f = self.feature_extractor(x)
        gate = self.gating_module(f, temperature)
        
        # Core decomposition
        f_inv = f * gate           # Invariant (structure)
        f_sp = f * (1.0 - gate)    # Specific (texture/noise)
        
        return f, f_inv, f_sp, gate

    def _compute_gate_sparsity_loss(self, gate: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss to encourage binary-like gates.
        
        Gate entropy: H(g) = -g*log(g) - (1-g)*log(1-g)
        We want to minimize this (push gates to 0 or 1).
        """
        eps = 1e-6
        entropy = -gate * torch.log(gate + eps) - (1 - gate) * torch.log(1 - gate + eps)
        return entropy.mean()

    def _compute_orthogonality_loss(self, f_inv: torch.Tensor, f_sp: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonality constraint between f_inv and f_sp.
        
        For gated decomposition f_inv = f * g and f_sp = f * (1-g):
        - f_inv · f_sp = f² * g * (1-g)
        - This is naturally 0 when g→0 or g→1
        
        We use the actual inner product divided by feature magnitude to measure orthogonality.
        """
        # Inner product: sum(f_inv * f_sp) = sum(f² * g * (1-g))
        inner_prod = (f_inv * f_sp).sum(dim=1)
        
        # Normalize by feature magnitude for scale invariance
        f_norm = torch.norm(f_inv + f_sp, dim=1).clamp(min=1e-6)
        orth_loss = (inner_prod / f_norm).abs().mean()
        
        return orth_loss

    def _compute_gate_diversity_loss(self, gates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Encourage different classes to have different gate patterns.
        
        Inter-class diversity: gates of different classes should be different.
        """
        # Compute class centroids for gates
        class_gates = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        class_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        class_gates.index_add_(0, labels, gates)
        class_counts.index_add_(0, labels, torch.ones(labels.size(0), device=self.device))
        class_counts = class_counts.clamp(min=1)
        
        class_centroids = class_gates / class_counts.unsqueeze(1)
        
        # Only compute for classes that appear in this batch
        active_classes = (class_counts > 0).nonzero().squeeze(-1)
        if len(active_classes) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute pairwise similarity between class centroids (want it to be low)
        active_centroids = class_centroids[active_classes]
        active_norm = F.normalize(active_centroids, p=2, dim=1)
        
        sim_matrix = torch.mm(active_norm, active_norm.t())
        
        # Exclude diagonal (self-similarity)
        n = len(active_classes)
        mask = ~torch.eye(n, dtype=torch.bool, device=self.device)
        off_diag_sim = sim_matrix[mask]
        
        # Maximize dissimilarity = minimize similarity
        diversity_loss = off_diag_sim.mean()
        
        return diversity_loss

    def _compute_structure_loss(self, gates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Structure-Aware Loss (intra-class gate consistency).
        
        Forces samples of the same class to have similar gate patterns.
        """
        batch_size = gates.size(0)
        
        class_centers = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        class_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        class_centers.index_add_(0, labels, gates)
        class_counts.index_add_(0, labels, torch.ones(batch_size, device=self.device))
        
        class_counts = class_counts.clamp(min=1)
        class_prototypes = class_centers / class_counts.unsqueeze(1)
        
        sample_prototypes = class_prototypes[labels]
        structure_loss = F.mse_loss(gates, sample_prototypes.detach())
        
        return structure_loss

    def _get_temperature(self, epoch: int, total_epochs: int, stage: str) -> float:
        """
        Get temperature for current epoch based on schedule.
        
        Stage 1 (warmup): High temperature (gates ≈ 1, f_inv ≈ f)
        Stage 2 (adapt): Temperature decreases linearly
        """
        init_temp = self.config.method.init_temperature
        final_temp = self.config.method.final_temperature
        
        if stage == 'warmup':
            # Keep high temperature during warmup
            return init_temp
        else:
            # Linear decay during adaptation
            progress = epoch / max(total_epochs - 1, 1)
            return init_temp + (final_temp - init_temp) * progress

    def train(self):
        """Three-stage training with progressive gating."""
        self._build_optimizer()
        
        pretrain_epochs = self.config.method.pretrain_epochs
        adapt_epochs = self.config.method.adapt_epochs
        
        # Stage 1: Source Warmup (high temperature)
        logger.info(f"Stage 1: Source Warmup ({pretrain_epochs} epochs, temp={self.config.method.init_temperature})")
        self._train_warmup_stage(pretrain_epochs)
        
        # Stage 2: Adaptation with progressive gating
        if adapt_epochs > 0:
            logger.info(f"Stage 2: Adaptation ({adapt_epochs} epochs, temp: {self.config.method.init_temperature} → {self.config.method.final_temperature})")
            self._train_adaptation_stage(adapt_epochs)
        
        # Compute final prototypes
        logger.info("Computing final class prototypes...")
        self._compute_class_prototypes()
        
        # Final evaluation
        final_hos = self.evaluate()
        logger.info(f"Training finished. Final HOS: {final_hos:.2f}%")

    def _train_warmup_stage(self, epochs: int):
        """
        Stage 1: Warmup on source with high temperature.
        
        High temperature → gates ≈ 1 → f_inv ≈ f → normal classification
        This ensures classifier learns good features before gating kicks in.
        """
        temperature = self.config.method.init_temperature
        lambda_struct = self.config.method.lambda_struct
        
        for epoch in range(epochs):
            self._set_train_mode()
            self.current_temperature = temperature
            
            loss_meter = AverageMeter()
            cls_meter = AverageMeter()
            struct_meter = AverageMeter()
            gate_mean_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Warmup {epoch+1}/{epochs}")
            for imgs, labels in pbar:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward with gating (high temp → gates ≈ 1)
                f, f_inv, f_sp, gate = self._forward_decompose(imgs, temperature)
                
                # Classification on f_inv (≈ f during warmup)
                logits = self.classifier(f_inv)
                loss_cls = self.criterion(logits, labels)
                
                # Structure loss for gate consistency
                loss_struct = self._compute_structure_loss(gate, labels)
                
                loss = loss_cls + lambda_struct * loss_struct
                
                loss.backward()
                self.optimizer.step()
                
                loss_meter.update(loss.item())
                cls_meter.update(loss_cls.item())
                struct_meter.update(loss_struct.item())
                gate_mean_meter.update(gate.mean().item())
                
                pbar.set_postfix({
                    "cls": f"{cls_meter.avg:.3f}",
                    "gate_μ": f"{gate_mean_meter.avg:.3f}"
                })
            
            known_acc = self._evaluate_known_accuracy()
            logger.info(f"Warmup Epoch {epoch+1}: Cls={cls_meter.avg:.4f}, "
                       f"Struct={struct_meter.avg:.4f}, Gate_μ={gate_mean_meter.avg:.3f}, "
                       f"Known Acc={known_acc:.2f}%")

    def _train_adaptation_stage(self, epochs: int):
        """
        Stage 2: Adaptation with progressive gating and domain alignment.
        
        Temperature decreases → gates become sharper (0 or 1)
        MMD aligns f_inv across domains
        Sparsity and orthogonality regularize the decomposition
        """
        lambda_struct = self.config.method.lambda_struct
        lambda_sparse = self.config.method.lambda_sparse
        lambda_orth = self.config.method.lambda_orth
        lambda_mmd = self.config.method.lambda_mmd
        lambda_pseudo = self.config.method.lambda_pseudo
        pseudo_threshold = self.config.method.pseudo_threshold
        
        for epoch in range(epochs):
            self._set_train_mode()
            
            # Update temperature (decreasing schedule)
            temperature = self._get_temperature(epoch, epochs, 'adapt')
            self.current_temperature = temperature
            
            tgt_iter = cycle(self.target_loader)
            
            cls_meter = AverageMeter()
            sparse_meter = AverageMeter()
            orth_meter = AverageMeter()
            mmd_meter = AverageMeter()
            pseudo_meter = AverageMeter()
            gate_mean_meter = AverageMeter()
            n_pseudo = 0
            
            pbar = tqdm(self.source_loader, desc=f"Adapt {epoch+1}/{epochs} (T={temperature:.2f})")
            
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)
                
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                
                self.optimizer.zero_grad()
                
                # ===== Source Forward =====
                f_s, f_inv_s, f_sp_s, gate_s = self._forward_decompose(src_imgs, temperature)
                
                # ===== Target Forward =====
                f_t, f_inv_t, f_sp_t, gate_t = self._forward_decompose(tgt_imgs, temperature)
                
                # ===== Source Losses =====
                # Classification on f_inv
                src_logits = self.classifier(f_inv_s)
                loss_cls = self.criterion(src_logits, src_labels)
                
                # Sparsity: encourage binary gates
                loss_sparse_s = self._compute_gate_sparsity_loss(gate_s)
                loss_sparse_t = self._compute_gate_sparsity_loss(gate_t)
                loss_sparse = (loss_sparse_s + loss_sparse_t) / 2
                
                # Orthogonality: f_inv ⊥ f_sp
                loss_orth_s = self._compute_orthogonality_loss(f_inv_s, f_sp_s, gate_s)
                loss_orth_t = self._compute_orthogonality_loss(f_inv_t, f_sp_t, gate_t)
                loss_orth = (loss_orth_s + loss_orth_t) / 2
                
                # Gate diversity: different classes should have different gates
                loss_div = self._compute_gate_diversity_loss(gate_s, src_labels)
                
                # Structure loss
                loss_struct = self._compute_structure_loss(gate_s, src_labels)
                
                # ===== Domain Alignment via MMD =====
                loss_mmd = compute_mmd(f_inv_s, f_inv_t)
                
                # ===== Pseudo-Labeling =====
                tgt_logits = self.classifier(f_inv_t)
                
                with torch.no_grad():
                    tgt_probs = F.softmax(tgt_logits, dim=1)
                    tgt_conf, tgt_pseudo = tgt_probs.max(dim=1)
                    confident_mask = (tgt_conf >= pseudo_threshold) & (tgt_pseudo < self.num_src_classes)
                
                loss_pseudo = torch.tensor(0.0, device=self.device)
                if confident_mask.sum() > 0:
                    n_pseudo += confident_mask.sum().item()
                    loss_pseudo = self.criterion(tgt_logits[confident_mask], tgt_pseudo[confident_mask])
                    
                    # Also add structure loss on pseudo-labeled samples
                    loss_struct = loss_struct + self._compute_structure_loss(
                        gate_t[confident_mask], tgt_pseudo[confident_mask]
                    )
                
                # ===== Total Loss =====
                loss = (loss_cls + 
                       lambda_struct * loss_struct +
                       lambda_sparse * loss_sparse +
                       lambda_orth * loss_orth +
                       0.1 * loss_div +  # Gate diversity
                       lambda_mmd * loss_mmd +
                       lambda_pseudo * loss_pseudo)
                
                loss.backward()
                self.optimizer.step()
                
                # Logging
                cls_meter.update(loss_cls.item())
                sparse_meter.update(loss_sparse.item())
                orth_meter.update(loss_orth.item())
                mmd_meter.update(loss_mmd.item())
                pseudo_meter.update(loss_pseudo.item())
                gate_mean_meter.update((gate_s.mean().item() + gate_t.mean().item()) / 2)
                
                pbar.set_postfix({
                    "cls": f"{cls_meter.avg:.3f}",
                    "orth": f"{orth_meter.avg:.3f}",
                    "mmd": f"{mmd_meter.avg:.3f}",
                    "gate_μ": f"{gate_mean_meter.avg:.3f}"
                })
            
            # Evaluate
            self._compute_class_prototypes()
            hos = self.evaluate()
            logger.info(f"Adapt Epoch {epoch+1}: T={temperature:.2f}, Cls={cls_meter.avg:.4f}, "
                       f"Sparse={sparse_meter.avg:.4f}, Orth={orth_meter.avg:.4f}, "
                       f"MMD={mmd_meter.avg:.4f}, Gate_μ={gate_mean_meter.avg:.3f}, "
                       f"N_Pseudo={n_pseudo}, HOS={hos:.2f}%")

    def _evaluate_known_accuracy(self):
        """Evaluate accuracy only on known classes."""
        self._set_eval_mode()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                if self.unknown_label is not None:
                    mask = labels != self.unknown_label
                    if mask.sum() == 0:
                        continue
                    imgs = imgs[mask]
                    labels = labels[mask]
                
                _, f_inv, _, _ = self._forward_decompose(imgs)
                logits = self.classifier(f_inv)
                _, predicted = torch.max(logits, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
        return 100.0 * correct / total if total > 0 else 0.0

    def _compute_class_prototypes(self):
        """Compute class prototypes using f_inv features from source domain.
        
        Uses f_inv (invariant features) for prototypes as they are more discriminative
        than gate patterns for distinguishing known vs unknown samples.
        """
        self._set_eval_mode()
        
        class_feat_sums = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        class_counts = torch.zeros(self.num_src_classes, device=self.device)
        
        with torch.no_grad():
            for imgs, labels in self.source_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Use f_inv features for prototypes
                _, f_inv, _, _ = self._forward_decompose(imgs)
                
                class_feat_sums.index_add_(0, labels, f_inv)
                class_counts.index_add_(0, labels, torch.ones(len(labels), device=self.device))
        
        class_counts = class_counts.clamp(min=1)
        self.class_prototypes = class_feat_sums / class_counts.unsqueeze(1)
        
        # Compute adaptive threshold using GMM on min distances
        target_distances = []
        
        with torch.no_grad():
            for imgs, _ in self.target_loader:
                imgs = imgs.to(self.device)
                _, f_inv, _, _ = self._forward_decompose(imgs)
                
                # Compute min Euclidean distance to class prototypes
                # dist[i, j] = ||f_inv[i] - proto[j]||_2
                dists = torch.cdist(f_inv, self.class_prototypes, p=2)
                min_dists, _ = dists.min(dim=1)
                target_distances.append(min_dists.cpu())
        
        target_distances = torch.cat(target_distances).numpy().reshape(-1, 1)
        
        if len(target_distances) >= 10:
            try:
                gmm = GaussianMixture(n_components=2, covariance_type='spherical',
                                      reg_covar=1e-3, random_state=42)
                gmm.fit(target_distances)
                
                means = gmm.means_.flatten()
                variances = gmm.covariances_.flatten()
                
                # Unknown samples have HIGHER distances, known have LOWER
                if means[0] > means[1]:
                    unk_idx, knw_idx = 0, 1
                else:
                    unk_idx, knw_idx = 1, 0
                
                m_unk, v_unk = means[unk_idx], variances[unk_idx]
                m_knw, v_knw = means[knw_idx], variances[knw_idx]
                
                # Decision boundary: weighted mean of the two cluster means
                self.rejection_threshold = float((m_unk * v_knw + m_knw * v_unk) / (v_unk + v_knw))
                logger.info(f"GMM (dist): Known μ={m_knw:.4f}, Unknown μ={m_unk:.4f}, threshold={self.rejection_threshold:.4f}")
            except Exception as e:
                logger.warning(f"GMM failed: {e}, using default threshold")
                # Use median as fallback threshold
                self.rejection_threshold = float(np.median(target_distances))
        else:
            self.rejection_threshold = float(np.median(target_distances)) if len(target_distances) > 0 else 10.0

    def evaluate(self):
        """Evaluation with prototype-based rejection using f_inv feature distances."""
        if self.class_prototypes.abs().sum() == 0:
            self._compute_class_prototypes()
            
        self._set_eval_mode()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Decompose features
                _, f_inv, _, _ = self._forward_decompose(imgs)
                
                # Classification on f_inv
                logits = self.classifier(f_inv)
                _, preds = logits.max(dim=1)
                
                # Distance-based rejection: unknown samples are far from all prototypes
                dists = torch.cdist(f_inv, self.class_prototypes, p=2)
                min_dists, _ = dists.min(dim=1)
                
                # Samples with min distance > threshold are rejected as unknown
                unknown_mask = min_dists > self.rejection_threshold
                
                if self.unknown_label is not None:
                    preds[unknown_mask] = self.unknown_label
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            return self._compute_osda_metrics(all_preds, all_labels)
        else:
            correct = (all_preds == all_labels).sum().item()
            total = len(all_labels)
            return 100 * correct / total if total > 0 else 0.0

    def forward_for_eval(self, imgs):
        """Forward pass for evaluation - uses f_inv (gated features)."""
        _, f_inv, _, _ = self._forward_decompose(imgs)
        return self.classifier(f_inv)

    def save_checkpoint(self, path):
        torch.save({
            "method": "tod",
            "feature_extractor": self.feature_extractor.state_dict(),
            "gating_module": self.gating_module.state_dict(),
            "classifier": self.classifier.state_dict(),
            "class_prototypes": self.class_prototypes,
            "rejection_threshold": self.rejection_threshold,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.gating_module.load_state_dict(checkpoint["gating_module"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        
        if "class_prototypes" in checkpoint:
            self.class_prototypes = checkpoint["class_prototypes"].to(self.device)
        if "rejection_threshold" in checkpoint:
            self.rejection_threshold = checkpoint["rejection_threshold"]
            
        logger.info(f"Model loaded from {path}")
