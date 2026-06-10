"""
Domain-Conditioned Feature Modulation (DCFM) solver.

Core idea: Instead of forcing domain-invariant features (like DANN), DCFM
*embraces* domain information. A domain classifier explicitly identifies which
domain the input belongs to, and its internal representation conditions a FiLM
modulation layer that adapts feature extraction per-domain.

Architecture:
    1. Shared backbone → base features h
    2. Domain classifier (NO gradient reversal) → domain logits + z_domain
    3. FiLM modulation: h' = (1 + γ(z_domain)) * BN(h) + β(z_domain)
    4. Task classifier → class predictions from h'
    5. Cross-domain Feature Hallucination: apply target style to source content.

Training strategy:
    Stage 1 (Warmup): Source task loss + domain classification loss
    Stage 2 (Joint):  + Information Maximization + Cross-Domain Feature Hallucination
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import GpuLossAccumulator, cycle

logger = logging.getLogger(__name__)


class DomainClassifier(nn.Module):
    """
    Domain classifier that directly (non-adversarially) predicts source vs target.
    Returns both the domain logits and an intermediate representation z_domain
    used by the FiLM modulation module.
    """

    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        z_domain = F.relu(self.ln1(self.fc1(x)))
        domain_logits = self.fc2(z_domain)
        return domain_logits, z_domain


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Generates per-sample scale (γ) and shift (β) from a domain representation,
    then applies: h' = (1 + γ) * h + β.

    Initialized to identity (γ=0, β=0) so modulation has no effect at the start.
    """

    def __init__(self, feat_dim: int, domain_dim: int = 256, scale_factor: float = 4.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.gamma_net = nn.Sequential(
            nn.Linear(domain_dim, feat_dim),
            nn.Sigmoid(),  # bounded γ ∈ [0, scale_factor]
        )
        self.beta_net = nn.Linear(domain_dim, feat_dim)

        # Initialize to identity modulation:
        # Sigmoid(0) = 0.5, so scale = 0.5 * scale_factor.
        # We want initial scale ≈ 1.0, so bias = sigmoid_inv(1/scale_factor).
        # For scale_factor=4, we want Sigmoid output = 0.25 → bias ≈ -1.1
        # Simpler: just zero-init weights and set bias so Sigmoid → 1/scale_factor
        nn.init.zeros_(self.gamma_net[0].weight)
        init_bias = -math.log(scale_factor - 1.0)  # sigmoid(init_bias) = 1/scale_factor
        nn.init.constant_(self.gamma_net[0].bias, init_bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: torch.Tensor, z_domain: torch.Tensor):
        gamma = self.gamma_net(z_domain) * self.scale_factor  # γ ∈ [0, scale_factor]
        beta = self.beta_net(z_domain)
        return gamma * h + beta


class DCFMNetwork(nn.Module):
    """
    Full DCFM network combining backbone, domain classifier, FiLM modulation,
    and task classifier.
    """

    def __init__(self, backbone_name: str, num_classes: int,
                 domain_hidden_dim: int = 256, bottleneck_dim: int = 0,
                 film_scale: float = 4.0):
        super().__init__()
        self.backbone = get_backbone(backbone_name)

        if hasattr(self.backbone, 'fc'):
            self.feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError("Backbone feature dimension not found.")

        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.domain_classifier = DomainClassifier(self.feat_dim, domain_hidden_dim)
        self.modulator = FiLMModulation(self.feat_dim, domain_hidden_dim, film_scale)

        # Configurable classifier head
        if bottleneck_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.feat_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(bottleneck_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(self.feat_dim, num_classes)

    def extract_features(self, x: torch.Tensor):
        """Extract base features from the backbone."""
        h = self.backbone(x)
        return h

    def get_domain_logits(self, h: torch.Tensor):
        """Get domain logits from DETACHED features (for domain loss only)."""
        domain_logits, _ = self.domain_classifier(h.detach())
        return domain_logits

    def get_domain_z(self, h: torch.Tensor):
        """Get domain representation z_domain WITH gradients (for FiLM modulation)."""
        _, z_domain = self.domain_classifier(h)
        return z_domain

    def forward_modulated(self, h: torch.Tensor, z_domain: torch.Tensor):
        """Apply FiLM modulation and task classification."""
        h_normed = self.feat_bn(h)
        h_mod = self.modulator(h_normed, z_domain)
        task_logits = self.classifier(h_mod)
        return task_logits

    def forward(self, x: torch.Tensor):
        h = self.extract_features(x)
        domain_logits = self.get_domain_logits(h)
        z_domain = self.get_domain_z(h)
        task_logits = self.forward_modulated(h, z_domain)
        return task_logits, domain_logits


def information_maximization_loss(logits: torch.Tensor,
                                  diversity_weight: float = 2.0) -> torch.Tensor:
    """
    Information Maximization (IM) loss for unsupervised target adaptation.

    Combines two complementary objectives:
    1. Entropy minimization: Encourage confident individual predictions
    2. Diversity maximization: Prevent collapse to a single class by
       maximizing entropy of the mean prediction across the batch

    L_IM = mean_i[H(p_i)] - diversity_weight * H(mean_i[p_i])

    Minimizing this encourages each sample to be confident while
    ensuring diversity across the batch.
    """
    probs = F.softmax(logits, dim=1)

    # Individual entropy: encourage each prediction to be confident
    ent_individual = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

    # Diversity: maximize entropy of mean prediction (prevent collapse)
    mean_probs = probs.mean(dim=0)
    ent_diversity = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()

    # Minimize individual entropy, maximize diversity
    return ent_individual - diversity_weight * ent_diversity


@register_solver("dcfm")
class DCFMSolver(BaseSolver):
    """
    Domain-Conditioned Feature Modulation (DCFM) solver.

    Two-stage training:
        Stage 1 (Warmup): Train on source with task + domain losses.
        Stage 2 (Joint):  Add Information Maximization on target predictions.
    """

    def build_model(self):
        backbone_name = self.config.method.get("backbone", "resnet50")
        domain_hidden_dim = self.config.method.get("domain_hidden_dim", 256)
        bottleneck_dim = self.config.method.get("bottleneck_dim", 0)
        film_scale = self.config.method.get("film_scale", 4.0)

        self.net = DCFMNetwork(
            backbone_name, self.num_classes, domain_hidden_dim,
            bottleneck_dim, film_scale,
        ).to(self.device)

        # Hyperparameters
        self.lambda_domain = self.config.method.get("lambda_domain", 1.0)
        self.lambda_im = self.config.method.get("lambda_im", 0.3)
        self.lambda_cf = self.config.method.get("lambda_cf", 1.0)
        self.lambda_div = self.config.method.get("lambda_div", 2.0)
        self.label_smoothing = self.config.method.get("label_smoothing", 0.1)

        # Loss functions
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.criterion_domain = nn.CrossEntropyLoss()

    def forward_for_eval(self, imgs):
        task_logits, _ = self.net(imgs)
        return task_logits

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(self):
        warmup_epochs = self.config.method.get("warmup_epochs", 5)
        max_epochs = self.config.method.epochs
        lr = self.config.method.lr

        # Optimizer with differential LR
        optimizer = optim.SGD([
            {'params': self.net.backbone.parameters(), 'lr': lr * 0.1},
            {'params': self.net.feat_bn.parameters(), 'lr': lr},
            {'params': self.net.domain_classifier.parameters(), 'lr': lr},
            {'params': self.net.modulator.parameters(), 'lr': lr},
            {'params': self.net.classifier.parameters(), 'lr': lr},
        ], momentum=0.9, weight_decay=5e-4, nesterov=True)

        total_iters = (warmup_epochs + max_epochs) * len(self.source_loader)

        # Linear warmup + cosine annealing
        warmup_iters = warmup_epochs * len(self.source_loader)

        def lr_lambda(step):
            if step < warmup_iters:
                return float(step) / max(1, warmup_iters)
            progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self.register_training_state(optimizer=optimizer, scheduler=scheduler)

        # Beta distribution for mixup (PyTorch-based for reproducibility)
        beta_dist = torch.distributions.Beta(
            torch.tensor(2.0), torch.tensor(2.0),
        )

        # Best model tracking
        best_acc = self._best_metric

        logger.info(f"DCFM Training: {warmup_epochs} warmup + {max_epochs} joint epochs")

        # ===================== Stage 1: Warmup ===================== #
        logger.info("=== Stage 1: Source Warmup ===")
        for epoch in self._epoch_range(warmup_epochs):
            self.net.train()
            acc_meter = GpuLossAccumulator(device=self.device)
            tgt_iter = cycle(self.target_loader)

            for src_imgs, src_labels in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs, src_labels = self._to_device(src_imgs), self._to_device(src_labels)
                tgt_imgs = self._to_device(tgt_imgs)
                bs_src, bs_tgt = src_imgs.size(0), tgt_imgs.size(0)

                self._zero_grad(optimizer)

                with self._auto_cast():
                    # --- Decoupled Forward (consistent with Stage 2) ---
                    h_src = self.net.extract_features(src_imgs)
                    h_tgt = self.net.extract_features(tgt_imgs)

                    # Domain logits (detached) for domain loss
                    domain_logits_src = self.net.get_domain_logits(h_src)
                    domain_logits_tgt = self.net.get_domain_logits(h_tgt)

                    # z_domain (with gradients) for FiLM modulation
                    z_src = self.net.get_domain_z(h_src)

                    # Task loss (source only)
                    task_logits_src = self.net.forward_modulated(h_src, z_src)
                    loss_task = self.criterion_task(task_logits_src, src_labels)

                    # Domain loss (source=0, target=1)
                    domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(bs_src, dtype=torch.long, device=self.device),
                        torch.ones(bs_tgt, dtype=torch.long, device=self.device),
                    ])
                    loss_domain = self.criterion_domain(domain_logits, domain_labels)

                    loss = loss_task + self.lambda_domain * loss_domain
                self._optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    clip_params=self.net.parameters(),
                    clip_max_norm=5.0,
                )
                scheduler.step()

                acc_meter.update("task", loss_task)
                acc_meter.update("domain", loss_domain)
                acc_meter.update("total", loss)
                acc_meter.step()

            acc_val = self.evaluate()
            if acc_val > best_acc:
                best_acc = acc_val
            self._maybe_save_best(acc_val, epoch + 1)
            self._log_epoch_summary(
                epoch + 1,
                warmup_epochs,
                metrics=acc_meter.compute(),
                score=acc_val,
                best_score=best_acc,
                score_name="Acc",
                prefix="DCFM Warmup",
            )

        # ===================== Stage 2: Joint ===================== #
        logger.info("=== Stage 2: Joint Training with Feature Hallucination & IM ===")
        for epoch in self._epoch_range(max_epochs, offset=warmup_epochs):
            self.net.train()
            acc_meter = GpuLossAccumulator(device=self.device)
            tgt_iter = cycle(self.target_loader)

            # Gradually ramp up IM and CF loss to prevent early disruption
            ramp = min(1.0, (epoch + 1) / max(1, max_epochs * 0.3))

            for src_imgs, src_labels in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs, src_labels = self._to_device(src_imgs), self._to_device(src_labels)
                tgt_imgs = self._to_device(tgt_imgs)
                bs_src, bs_tgt = src_imgs.size(0), tgt_imgs.size(0)

                self._zero_grad(optimizer)

                with self._auto_cast():
                    # --- Decoupled Forward Pass ---
                    h_src = self.net.extract_features(src_imgs)
                    h_tgt = self.net.extract_features(tgt_imgs)

                    # Domain logits (detached) for domain loss
                    domain_logits_src = self.net.get_domain_logits(h_src)
                    domain_logits_tgt = self.net.get_domain_logits(h_tgt)

                    # z_domain (with gradients) for FiLM modulation
                    z_src = self.net.get_domain_z(h_src)
                    z_tgt = self.net.get_domain_z(h_tgt)

                    task_logits_src = self.net.forward_modulated(h_src, z_src)
                    task_logits_tgt = self.net.forward_modulated(h_tgt, z_tgt)

                    # 1) Source task loss
                    loss_task = self.criterion_task(task_logits_src, src_labels)

                    # 2) Domain classification loss
                    domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(bs_src, dtype=torch.long, device=self.device),
                        torch.ones(bs_tgt, dtype=torch.long, device=self.device),
                    ])
                    loss_domain = self.criterion_domain(domain_logits, domain_labels)

                    # 3) Information Maximization on target
                    loss_im = information_maximization_loss(
                        task_logits_tgt, diversity_weight=self.lambda_div,
                    )

                    # 4) Cross-Domain Joint Manifold Mixup
                    min_bs = min(bs_src, bs_tgt)

                    # Shuffle target to pair randomly with source
                    shuffle_idx = torch.randperm(min_bs, device=self.device)
                    h_tgt_shuffled = h_tgt[:min_bs][shuffle_idx]
                    z_tgt_shuffled = z_tgt[:min_bs][shuffle_idx]

                    # Sample lambda from Beta(2,2), ensure source dominance
                    lam = beta_dist.sample().item()
                    lam = max(lam, 1.0 - lam)  # source-dominant for label reliability

                    # Joint Manifold Interpolation
                    h_cross = lam * h_src[:min_bs] + (1 - lam) * h_tgt_shuffled
                    z_cross = lam * z_src[:min_bs].detach() + (1 - lam) * z_tgt_shuffled.detach()

                    # Modulate and Classify
                    task_logits_cross = self.net.classifier(
                        self.net.modulator(self.net.feat_bn(h_cross), z_cross)
                    )

                    # Mixed Labels Loss
                    prob_tgt = F.softmax(task_logits_tgt[:min_bs][shuffle_idx].detach(), dim=1)
                    loss_src_part = lam * F.cross_entropy(task_logits_cross, src_labels[:min_bs])
                    log_prob_cross = F.log_softmax(task_logits_cross, dim=1)
                    loss_tgt_part = (1 - lam) * torch.sum(-prob_tgt * log_prob_cross, dim=1).mean()

                    loss_cf = loss_src_part + loss_tgt_part

                    # Total Loss with ramp-ups
                    loss = (loss_task
                            + self.lambda_domain * loss_domain
                            + self.lambda_im * ramp * loss_im
                            + self.lambda_cf * ramp * loss_cf)

                self._optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    clip_params=self.net.parameters(),
                    clip_max_norm=5.0,
                )
                scheduler.step()

                acc_meter.update("task", loss_task)
                acc_meter.update("domain", loss_domain)
                acc_meter.update("im", loss_im)
                acc_meter.update("cf", loss_cf)
                acc_meter.update("total", loss)
                acc_meter.step()

            acc_val = self.evaluate()
            if acc_val > best_acc:
                best_acc = acc_val
            self._maybe_save_best(acc_val, warmup_epochs + epoch + 1)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics=acc_meter.compute(),
                extras={"rmp": (ramp, ".2f")},
                score=acc_val,
                best_score=best_acc,
                score_name="Acc",
                prefix="DCFM Joint",
            )

        # Load best model weights at the end of training
        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")

    def save_checkpoint(self, path):
        self._save_named_modules_checkpoint(path, modules={"model": self.net})

    def load_checkpoint(self, path):
        self._load_named_modules_checkpoint(
            path,
            modules={"model": self.net},
            fallback_key="model",
        )
