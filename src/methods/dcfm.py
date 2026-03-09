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

Training strategy:
    Stage 1 (Warmup): Source task loss + domain classification loss
    Stage 2 (Joint):  + Information Maximization on target predictions
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import AverageMeter, cycle

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
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        z_domain = F.relu(self.bn1(self.fc1(x)))
        domain_logits = self.fc2(z_domain)
        return domain_logits, z_domain


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Generates per-sample scale (γ) and shift (β) from a domain representation,
    then applies: h' = (1 + γ) * h + β.

    Initialized to identity (γ=0, β=0) so modulation has no effect at the start.
    """

    def __init__(self, feat_dim: int, domain_dim: int = 256):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(domain_dim, feat_dim),
            nn.Tanh(),  # bound γ to [-1, 1] for stability
        )
        self.beta_net = nn.Linear(domain_dim, feat_dim)

        # Initialize to identity modulation
        nn.init.zeros_(self.gamma_net[0].weight)
        nn.init.zeros_(self.gamma_net[0].bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: torch.Tensor, z_domain: torch.Tensor):
        gamma = self.gamma_net(z_domain)
        beta = self.beta_net(z_domain)
        return (1.0 + gamma) * h + beta


class DCFMNetwork(nn.Module):
    """
    Full DCFM network combining backbone, domain classifier, FiLM modulation,
    and task classifier.
    """

    def __init__(self, backbone_name: str, num_classes: int, domain_hidden_dim: int = 256):
        super().__init__()
        self.backbone = get_backbone(backbone_name)

        if hasattr(self.backbone, 'fc'):
            self.feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError("Backbone feature dimension not found.")

        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.domain_classifier = DomainClassifier(self.feat_dim, domain_hidden_dim)
        self.modulator = FiLMModulation(self.feat_dim, domain_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)

        # Domain classifier on detached features —
        # domain loss shapes domain_classifier only, not the backbone.
        # Backbone gradient comes from task loss through modulated features.
        domain_logits, z_domain = self.domain_classifier(h.detach())

        # FiLM modulation conditioned on domain representation
        h_normed = self.feat_bn(h)
        h_mod = self.modulator(h_normed, z_domain)

        task_logits = self.classifier(h_mod)
        return task_logits, domain_logits


def information_maximization_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Information Maximization (IM) loss for unsupervised target adaptation.

    Combines two complementary objectives:
    1. Entropy minimization: Encourage confident individual predictions
    2. Diversity maximization: Prevent collapse to a single class by
       maximizing entropy of the mean prediction across the batch

    L_IM = mean_i[H(p_i)] - H(mean_i[p_i])
         = (individual entropy) - (batch diversity)

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
    return ent_individual - ent_diversity


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

        self.net = DCFMNetwork(backbone_name, self.num_classes, domain_hidden_dim).to(self.device)

        # Hyperparameters
        self.lambda_domain = self.config.method.get("lambda_domain", 1.0)
        self.lambda_im = self.config.method.get("lambda_im", 0.3)
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

        logger.info(f"DCFM Training: {warmup_epochs} warmup + {max_epochs} joint epochs")
        best_acc = 0.0

        # ===================== Stage 1: Warmup ===================== #
        logger.info("=== Stage 1: Source Warmup ===")
        for epoch in range(warmup_epochs):
            self.net.train()
            meters = {k: AverageMeter() for k in ['task', 'domain', 'total']}
            tgt_iter = cycle(self.target_loader)

            for src_imgs, src_labels in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs, src_labels = src_imgs.to(self.device), src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                bs_src, bs_tgt = src_imgs.size(0), tgt_imgs.size(0)

                all_imgs = torch.cat([src_imgs, tgt_imgs], dim=0)

                optimizer.zero_grad()
                task_logits, domain_logits = self.net(all_imgs)

                # Task loss (source only)
                loss_task = self.criterion_task(task_logits[:bs_src], src_labels)

                # Domain loss (source=0, target=1)
                domain_labels = torch.cat([
                    torch.zeros(bs_src, dtype=torch.long, device=self.device),
                    torch.ones(bs_tgt, dtype=torch.long, device=self.device),
                ])
                loss_domain = self.criterion_domain(domain_logits, domain_labels)

                loss = loss_task + self.lambda_domain * loss_domain
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters['task'].update(loss_task.item())
                meters['domain'].update(loss_domain.item())
                meters['total'].update(loss.item())

            acc = self.evaluate()
            best_acc = max(best_acc, acc)
            logger.info(
                f"Warmup {epoch+1} | task={meters['task'].avg:.4f} "
                f"dom={meters['domain'].avg:.4f} | Acc={acc:.2f}% (best={best_acc:.2f}%)"
            )

        # ===================== Stage 2: Joint ===================== #
        logger.info("=== Stage 2: Joint Training with Information Maximization ===")
        for epoch in range(max_epochs):
            self.net.train()
            meters = {k: AverageMeter() for k in ['task', 'domain', 'im', 'total']}
            tgt_iter = cycle(self.target_loader)

            # Gradually ramp up IM loss to prevent early disruption
            ramp = min(1.0, (epoch + 1) / max(1, max_epochs * 0.3))

            for src_imgs, src_labels in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs, src_labels = src_imgs.to(self.device), src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                bs_src, bs_tgt = src_imgs.size(0), tgt_imgs.size(0)

                all_imgs = torch.cat([src_imgs, tgt_imgs], dim=0)

                optimizer.zero_grad()
                task_logits, domain_logits = self.net(all_imgs)

                task_logits_src = task_logits[:bs_src]
                task_logits_tgt = task_logits[bs_src:]

                # 1) Source task loss
                loss_task = self.criterion_task(task_logits_src, src_labels)

                # 2) Domain classification loss
                domain_labels = torch.cat([
                    torch.zeros(bs_src, dtype=torch.long, device=self.device),
                    torch.ones(bs_tgt, dtype=torch.long, device=self.device),
                ])
                loss_domain = self.criterion_domain(domain_logits, domain_labels)

                # 3) Information Maximization on target
                loss_im = information_maximization_loss(task_logits_tgt)

                loss = (loss_task
                        + self.lambda_domain * loss_domain
                        + self.lambda_im * ramp * loss_im)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters['task'].update(loss_task.item())
                meters['domain'].update(loss_domain.item())
                meters['im'].update(loss_im.item())
                meters['total'].update(loss.item())

            acc = self.evaluate()
            best_acc = max(best_acc, acc)
            logger.info(
                f"Joint {epoch+1} | task={meters['task'].avg:.4f} "
                f"dom={meters['domain'].avg:.4f} im={meters['im'].avg:.4f} "
                f"ramp={ramp:.2f} | Acc={acc:.2f}% (best={best_acc:.2f}%)"
            )

        logger.info(f"Training finished. Best Acc: {best_acc:.2f}%")

    def save_checkpoint(self, path):
        torch.save({
            "method": "dcfm",
            "model": self.net.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if "model" in checkpoint:
            self.net.load_state_dict(checkpoint["model"])
        else:
            self.net.load_state_dict(checkpoint)
        logger.info(f"Model loaded from {path}")
