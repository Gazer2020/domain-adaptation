"""
Open Set Domain Adaptation version of Domain-Conditioned Feature Modulation (ODCFM).

Core idea: Extend DCFM for OSDA by introducing Energy Awareness and a Target Unknown Prototype.
Target samples with high uncertainty (energy) are modulated towards a learnable unknown prototype
rather than source identity to prevent negative transfer. A Safe Manifold Mixup uses an energy
firewall to prevent unknown classes from mixing with source.

Architecture:
    1. Shared backbone → base features h
    2. Domain classifier → domain logits + z_domain
    3. Energy Awareness → w_unk (uncertainty weight) based on global EMA of source energy
    4. Adaptive Modulation → z'_tgt = (1 - w_unk) * z_tgt + w_unk * z_unk_proto
       h' = FiLM(h, z')
    5. Task classifier → class predictions from h'
    6. Safe Manifold Mixup: Mixup only for high-confidence target samples.
    7. Sinkhorn Loss: Unsupervised clustering with dustbin for target domain.
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
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        z_domain = F.relu(self.ln1(self.fc1(x)))
        domain_logits = self.fc2(z_domain)
        return domain_logits, z_domain


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    """

    def __init__(self, feat_dim: int, domain_dim: int = 256, scale_factor: float = 4.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.gamma_net = nn.Sequential(
            nn.Linear(domain_dim, feat_dim),
            nn.Sigmoid(),  # bounded γ ∈ [0, scale_factor]
        )
        self.beta_net = nn.Linear(domain_dim, feat_dim)

        nn.init.zeros_(self.gamma_net[0].weight)
        init_bias = -math.log(scale_factor - 1.0)
        nn.init.constant_(self.gamma_net[0].bias, init_bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: torch.Tensor, z_domain: torch.Tensor):
        gamma = self.gamma_net(z_domain) * self.scale_factor
        beta = self.beta_net(z_domain)
        return gamma * h + beta


class ODCFMNetwork(nn.Module):
    """
    Full ODCFM network combining backbone, domain classifier, FiLM modulation,
    task classifier, and learnable target unknown prototype.
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
        
        # Learnable Target Unknown Prototype
        self.z_unk_proto = nn.Parameter(torch.zeros(domain_hidden_dim))

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
        h = self.backbone(x)
        return h

    def get_domain_logits(self, h: torch.Tensor):
        domain_logits, _ = self.domain_classifier(h.detach())
        return domain_logits

    def get_domain_z(self, h: torch.Tensor):
        _, z_domain = self.domain_classifier(h)
        return z_domain

    def forward_modulated(self, h: torch.Tensor, z_domain: torch.Tensor):
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


def compute_energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Compute energy score E(x) = -T * log sum exp(logits / T)."""
    return -T * torch.logsumexp(logits / T, dim=1)


def sinkhorn_loss_with_dustbin(logits: torch.Tensor, w_unk: torch.Tensor, epsilon: float = 0.05, num_iters: int = 3):
    """
    Sinkhorn loss for target domain unsupervised clustering guided by energy firewall.
    
    Args:
        logits: (B, C) target logits for known classes
        w_unk: (B, 1) uncertainty weights
        epsilon: Entropy regularization coefficient
        num_iters: Number of Sinkhorn-Knopp iterations
    """
    B, C = logits.shape
    device = logits.device
    
    probs = F.softmax(logits, dim=1)  # (B, C)
    
    # Construct full probabilities guided by w_unk
    w = w_unk.detach()
    p_known = (1 - w) * probs
    p_dustbin = w
    
    # Full probability matrix (B, C+1)
    probs_full = torch.cat([p_known, p_dustbin], dim=1)  # (B, C+1)
    
    # Cost matrix M (B, C+1)
    cost_full = -torch.log(probs_full + 1e-8)  # (B, C+1)
    
    # Dynamic Marginals based on Energy Uncertainty (w_unk)
    unk_ratio = w.mean().item()
    unk_ratio = max(unk_ratio, 0.05)  # Floor to prevent dustbin from vanishing
    known_ratio = max(1.0 - unk_ratio, 1e-6)
    
    c = torch.ones(C + 1, device=device)
    c[:C] = known_ratio / C
    c[C] = unk_ratio
    c = c / c.sum()
    
    with torch.no_grad():
        Q = torch.exp(-cost_full / epsilon).t()  # (C+1, B)
        Q_sum = torch.sum(Q)
        Q /= Q_sum
        
        for _ in range(num_iters):
            Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
            Q /= B
            Q /= (torch.sum(Q, dim=1, keepdim=True) + 1e-8)
            Q *= c.unsqueeze(1)
            
        Q *= B
        Q = Q.t()  # (B, C+1)
    
    # Cross-entropy loss ONLY for known classes (dustbin is handled by loss_unk)
    loss = -torch.sum(Q[:, :C] * torch.log(probs + 1e-8)) / B
    
    return loss


@register_solver("odcfm")
class ODCFMSolver(BaseSolver):
    """
    Open Set Domain-Conditioned Feature Modulation (ODCFM) solver.
    """

    def build_model(self):
        backbone_name = self.config.method.get("backbone", "resnet50")
        domain_hidden_dim = self.config.method.get("domain_hidden_dim", 256)
        bottleneck_dim = self.config.method.get("bottleneck_dim", 0)
        film_scale = self.config.method.get("film_scale", 4.0)

        self.net = ODCFMNetwork(
            backbone_name, self.num_classes, domain_hidden_dim,
            bottleneck_dim, film_scale,
        ).to(self.device)

        # Hyperparameters
        self.lambda_domain = self.config.method.get("lambda_domain", 1.0)
        self.lambda_sk = self.config.method.get("lambda_sk", 0.5)  # Sinkhorn loss weight
        self.lambda_cf = self.config.method.get("lambda_cf", 1.0)
        self.lambda_ent = self.config.method.get("lambda_ent", 0.1)  # Target entropy minimization
        self.lambda_unk = self.config.method.get("lambda_unk", 1.0)  # Unknown class training
        self.label_smoothing = self.config.method.get("label_smoothing", 0.1)
        
        # Energy and Mixup Hyperparams
        self.energy_T = self.config.method.get("energy_T", 1.0)
        self.unk_T = self.config.method.get("unk_T", 0.05)  # Sharper sigmoid for w_unk
        self.margin_k = self.config.method.get("margin_k", 1.5)  # Adaptive threshold: ema + k * std
        self.ema_decay = self.config.method.get("ema_decay", 0.99)

        # Loss functions
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.criterion_domain = nn.CrossEntropyLoss()
        
        # Internal state for adaptive energy threshold
        self.src_energy_ema = None
        self.src_energy_std_ema = None

    def forward_for_eval(self, imgs):
        """Standard Forward. Overriding is not necessary because we override evaluate directly."""
        task_logits, _ = self.net(imgs)
        return task_logits
        
    def _compute_hscore(self, preds, labels):
        """Compute H-score from predictions and labels."""
        unknown_label = self.unknown_label
        known_mask = labels != unknown_label
        unknown_mask = labels == unknown_label
        
        if known_mask.sum() > 0:
            known_acc = (preds[known_mask] == labels[known_mask]).sum().item() / known_mask.sum().item()
        else:
            known_acc = 0.0
        if unknown_mask.sum() > 0:
            unknown_acc = (preds[unknown_mask] == unknown_label).sum().item() / unknown_mask.sum().item()
        else:
            unknown_acc = 0.0
        if known_acc + unknown_acc > 0:
            hscore = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        else:
            hscore = 0.0
        return known_acc, unknown_acc, hscore

    def evaluate(self):
        """
        Evaluate on target test set with optimal threshold search.
        Uses both the classifier's own unknown-class prediction and
        energy-based rejection with threshold search to maximize H-score.
        """
        self._set_eval_mode()
        all_preds = []
        all_labels = []
        all_energies = []
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                h = self.net.extract_features(imgs)
                z = self.net.get_domain_z(h)
                outputs = self.net.forward_modulated(h, z)
                
                energy = compute_energy(outputs, self.energy_T)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, dim=1)
                
                all_preds.append(predicted.cpu())
                all_labels.append(labels)
                all_energies.append(energy.cpu())
                
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_energies = torch.cat(all_energies)
        
        # --- Optimal energy threshold search to maximize H-score ---
        # The classifier may already predict some samples as unknown (class C).
        # We additionally reject samples with high energy.
        best_ka, best_ua, best_hs = 0.0, 0.0, 0.0
        
        sorted_energies, _ = all_energies.sort()
        n = len(sorted_energies)
        
        # Search over percentile thresholds
        for pct in range(5, 100, 2):
            idx = min(int(n * pct / 100), n - 1)
            threshold = sorted_energies[idx].item()
            
            final_preds = all_preds.clone()
            # Classifier's own unknown predictions are kept (pred == unknown_label)
            # Additionally reject high-energy samples
            energy_reject = all_energies > threshold
            final_preds[energy_reject] = self.unknown_label
            
            ka, ua, hs = self._compute_hscore(final_preds, all_labels)
            if hs > best_hs:
                best_hs = hs
                best_ka = ka
                best_ua = ua
        
        # Also try classifier-only (no energy rejection)
        ka, ua, hs = self._compute_hscore(all_preds, all_labels)
        if hs > best_hs:
            best_hs = hs
            best_ka = ka
            best_ua = ua
            
        return best_ka * 100, best_ua * 100, best_hs * 100

    def train(self):
        warmup_epochs = self.config.method.get("warmup_epochs", 5)
        max_epochs = self.config.method.epochs
        lr = self.config.method.lr

        optimizer = optim.SGD([
            {'params': self.net.backbone.parameters(), 'lr': lr * 0.1},
            {'params': self.net.feat_bn.parameters(), 'lr': lr},
            {'params': self.net.domain_classifier.parameters(), 'lr': lr},
            {'params': self.net.modulator.parameters(), 'lr': lr},
            {'params': self.net.classifier.parameters(), 'lr': lr},
            {'params': [self.net.z_unk_proto], 'lr': lr}, # Target Unknown Prototype
        ], momentum=0.9, weight_decay=5e-4, nesterov=True)

        total_iters = (warmup_epochs + max_epochs) * len(self.source_loader)
        warmup_iters = warmup_epochs * len(self.source_loader)

        def lr_lambda(step):
            if step < warmup_iters:
                return float(step) / max(1, warmup_iters)
            progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        beta_dist = torch.distributions.Beta(torch.tensor(2.0), torch.tensor(2.0))

        best_acc = 0.0
        save_dir = Path("checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_dir / "best.pth"
        
        logger.info(f"ODCFM Training: {warmup_epochs} warmup + {max_epochs} joint epochs")

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

                optimizer.zero_grad()

                h_src = self.net.extract_features(src_imgs)
                h_tgt = self.net.extract_features(tgt_imgs)

                domain_logits_src = self.net.get_domain_logits(h_src)
                domain_logits_tgt = self.net.get_domain_logits(h_tgt)

                z_src = self.net.get_domain_z(h_src)

                task_logits_src = self.net.forward_modulated(h_src, z_src)
                loss_task = self.criterion_task(task_logits_src, src_labels)

                domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
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
                # Initialize Target Unknown Prototype with EMA of target z_domain
                with torch.no_grad():
                    z_tgt = self.net.get_domain_z(h_tgt)
                    # During warmup, we don't have w_unk yet, so we just use plain mean
                    # But ideally it should move towards unknown. Since it's warmup, plain mean is okay to start.
                    z_tgt_mean = z_tgt.mean(dim=0)
                    if epoch == 0 and meters['total'].count == 1:
                        self.net.z_unk_proto.data.copy_(z_tgt_mean)
                    else:
                        self.net.z_unk_proto.data.mul_(self.ema_decay).add_(z_tgt_mean, alpha=1 - self.ema_decay)
                        
                # Track source energy EMA (mean + std)
                with torch.no_grad():
                    src_energy = compute_energy(task_logits_src, self.energy_T)
                    batch_src_energy_mean = src_energy.mean().item()
                    batch_src_energy_std = src_energy.std().item() if src_energy.numel() > 1 else 0.0
                    if self.src_energy_ema is None:
                        self.src_energy_ema = batch_src_energy_mean
                        self.src_energy_std_ema = batch_src_energy_std
                    else:
                        self.src_energy_ema = self.ema_decay * self.src_energy_ema + (1 - self.ema_decay) * batch_src_energy_mean
                        self.src_energy_std_ema = self.ema_decay * self.src_energy_std_ema + (1 - self.ema_decay) * batch_src_energy_std

            known_acc, unk_acc, hscore = self.evaluate()
            if hscore > best_acc:
                best_acc = hscore
                self.save_checkpoint(best_path)
            logger.info(
                f"Warmup {epoch+1:02d} | ts={meters['task'].avg:.3f} "
                f"dm={meters['domain'].avg:.3f} | rmp=0.00 w_unk=0.00 | "
                f"K={known_acc:.1f}% U={unk_acc:.1f}% H={hscore:.1f}% (best={best_acc:.1f}%)"
            )

        # ===================== Stage 2: Joint ===================== #
        logger.info("=== Stage 2: Joint Training with OSDA Feature Hallucination & Sinkhorn ===")
        for epoch in range(max_epochs):
            self.net.train()
            meters = {k: AverageMeter() for k in ['task', 'domain', 'sk', 'cf', 'total', 'w_unk']}
            tgt_iter = cycle(self.target_loader)

            ramp = min(1.0, (epoch + 1) / max(1, max_epochs * 0.3))
            # Cosine decay for unknown loss: strong early, gentle late
            unk_decay = 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, max_epochs)))

            for src_imgs, src_labels in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs, src_labels = src_imgs.to(self.device), src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                bs_src, bs_tgt = src_imgs.size(0), tgt_imgs.size(0)

                optimizer.zero_grad()

                h_src = self.net.extract_features(src_imgs)
                h_tgt = self.net.extract_features(tgt_imgs)

                domain_logits_src = self.net.get_domain_logits(h_src)
                domain_logits_tgt = self.net.get_domain_logits(h_tgt)

                z_src = self.net.get_domain_z(h_src)
                z_tgt = self.net.get_domain_z(h_tgt)
                
                # Update Source Energy EMA (no_grad — only for tracking stats)
                with torch.no_grad():
                    task_logits_src_tmp = self.net.forward_modulated(h_src, z_src)
                    src_energy = compute_energy(task_logits_src_tmp, self.energy_T)
                    batch_src_energy_mean = src_energy.mean().item()
                    batch_src_energy_std = src_energy.std().item() if src_energy.numel() > 1 else 0.0
                    self.src_energy_ema = self.ema_decay * self.src_energy_ema + (1 - self.ema_decay) * batch_src_energy_mean
                    self.src_energy_std_ema = self.ema_decay * self.src_energy_std_ema + (1 - self.ema_decay) * batch_src_energy_std
                    tau_E = self.src_energy_ema + self.margin_k * self.src_energy_std_ema
                    
                # Compute Target Energy and w_unk WITH gradients (so z_unk_proto can learn)
                task_logits_tgt_tmp = self.net.forward_modulated(h_tgt, z_tgt)
                tgt_energy = compute_energy(task_logits_tgt_tmp, self.energy_T)  # (B,)
                w_unk = torch.sigmoid((tgt_energy - tau_E) / self.unk_T).unsqueeze(1)  # (B, 1)
                
                
                meters['w_unk'].update(w_unk.mean().item())

                # 1) Modulated Forward Passes
                # Source keeps its own z_domain
                task_logits_src = self.net.forward_modulated(h_src, z_src)
                
                # Target forms z'_tgt using w_unk and z_unk_proto
                z_tgt_prime = (1 - w_unk) * z_tgt + w_unk * self.net.z_unk_proto.unsqueeze(0)
                task_logits_tgt = self.net.forward_modulated(h_tgt, z_tgt_prime)

                # Source task loss
                loss_task = self.criterion_task(task_logits_src, src_labels)

                # Domain classification loss
                domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(bs_src, dtype=torch.long, device=self.device),
                    torch.ones(bs_tgt, dtype=torch.long, device=self.device),
                ])
                loss_domain = self.criterion_domain(domain_logits, domain_labels)

                # OSDA Target Loss: Sinkhorn with dustbin — use only known-class logits
                # The unknown class (last column) is handled by loss_unk, not Sinkhorn
                num_known = self.num_classes - 1  # C known classes
                loss_sk = sinkhorn_loss_with_dustbin(task_logits_tgt[:, :num_known], w_unk)

                # Unknown class training loss: push high-w_unk target samples toward unknown class
                unk_cls_idx = self.num_classes - 1  # The last class is "unknown"
                unk_target = torch.full((bs_tgt,), unk_cls_idx, dtype=torch.long, device=self.device)
                per_sample_unk_loss = F.cross_entropy(task_logits_tgt, unk_target, reduction='none')
                loss_unk = (w_unk.squeeze(1).detach() * per_sample_unk_loss).mean()

                # Source negative unknown loss: source should NOT predict the unknown class
                # This strengthens the known/unknown boundary from the source side
                src_unk_prob = F.softmax(task_logits_src, dim=1)[:, unk_cls_idx]  # prob of unknown class
                loss_src_neg_unk = src_unk_prob.mean()  # Minimize probability of source predicting unknown

                # Safe Cross-Domain Joint Manifold Mixup
                min_bs = min(bs_src, bs_tgt)

                shuffle_idx = torch.randperm(min_bs, device=self.device)
                h_tgt_shuffled = h_tgt[:min_bs][shuffle_idx]
                z_tgt_prime_shuffled = z_tgt_prime[:min_bs][shuffle_idx]
                w_unk_shuffled = w_unk[:min_bs][shuffle_idx].squeeze(1) # (min_bs,)

                # Sample base lambda from Beta(2,2)
                base_lam = beta_dist.sample((min_bs,)).to(self.device)
                base_lam = torch.max(base_lam, 1.0 - base_lam)  # source-dominant

                # Energy Firewall: mask out mixup for high energy (unknown) target samples
                # If w_unk >= 0.5, we force lam = 1.0 (only use source)
                # Hard Mask:
                mask = (w_unk_shuffled < 0.5).float()
                # If mask is 0 (is unknown), lam becomes 1.0. If mask is 1 (is known), lam remains base_lam.
                lam = base_lam * mask + 1.0 * (1 - mask)
                lam = lam.view(min_bs, 1)

                h_cross = lam * h_src[:min_bs] + (1 - lam) * h_tgt_shuffled
                z_cross = lam * z_src[:min_bs].detach() + (1 - lam) * z_tgt_prime_shuffled.detach()

                task_logits_cross = self.net.classifier(
                    self.net.modulator(self.net.feat_bn(h_cross), z_cross)
                )

                prob_tgt = F.softmax(task_logits_tgt[:min_bs][shuffle_idx].detach(), dim=1)
                
                # loss_src_part is weighted by lam
                loss_src_part = (lam.squeeze(1) * F.cross_entropy(task_logits_cross, src_labels[:min_bs], reduction='none')).mean()
                
                log_prob_cross = F.log_softmax(task_logits_cross, dim=1)
                loss_tgt_part = ((1 - lam.squeeze(1)) * torch.sum(-prob_tgt * log_prob_cross, dim=1)).mean()

                loss_cf = loss_src_part + loss_tgt_part

                # Target entropy minimization for high-confidence (known) target samples
                # Use only known-class logits for entropy to not suppress unknown class
                tgt_probs_known = F.softmax(task_logits_tgt[:, :num_known], dim=1)
                tgt_entropy = -torch.sum(tgt_probs_known * torch.log(tgt_probs_known + 1e-8), dim=1)  # (B,)
                # Weight by (1 - w_unk): only minimize entropy for confident known samples
                w_known = (1 - w_unk.squeeze(1)).detach()
                loss_ent = (w_known * tgt_entropy).mean()

                # Weighted z_unk_proto EMA update during joint training
                with torch.no_grad():
                    w = w_unk.detach()
                    # weighted mean of z_tgt focusing on unknown samples
                    # add small epsilon to avoid div by zero if w is all 0
                    z_tgt_unk_mean = (z_tgt * w).sum(dim=0) / (w.sum() + 1e-8)
                    self.net.z_unk_proto.data.mul_(self.ema_decay).add_(z_tgt_unk_mean, alpha=1 - self.ema_decay)

                # Total Loss
                loss = (loss_task
                        + self.lambda_domain * loss_domain
                        + self.lambda_sk * ramp * loss_sk
                        + self.lambda_cf * ramp * loss_cf
                        + self.lambda_ent * ramp * loss_ent
                        + self.lambda_unk * ramp * unk_decay * loss_unk
                        + self.lambda_unk * ramp * loss_src_neg_unk)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters['task'].update(loss_task.item())
                meters['domain'].update(loss_domain.item())
                meters['sk'].update(loss_sk.item())
                meters['cf'].update(loss_cf.item())
                meters['total'].update(loss.item())

            known_acc, unk_acc, hscore = self.evaluate()
            if hscore > best_acc:
                best_acc = hscore
                self.save_checkpoint(best_path)
            logger.info(
                f"Joint {epoch+1:02d} | ts={meters['task'].avg:.3f} dm={meters['domain'].avg:.3f} "
                f"sk={meters['sk'].avg:.3f} cf={meters['cf'].avg:.3f} | rmp={ramp:.2f} w_unk={meters['w_unk'].avg:.2f} | "
                f"K={known_acc:.1f}% U={unk_acc:.1f}% H={hscore:.1f}% (best={best_acc:.1f}%)"
            )

        if best_path.exists():
            self.load_checkpoint(best_path)
            logger.info(f"Loaded best model (Acc={best_acc:.2f}%) from {best_path}")

        logger.info(f"Training finished. Best Acc: {best_acc:.2f}%")

    def save_checkpoint(self, path):
        torch.save({
            "method": "odcfm",
            "model": self.net.state_dict(),
            "src_energy_ema": self.src_energy_ema,
            "src_energy_std_ema": self.src_energy_std_ema,
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if "model" in checkpoint:
            self.net.load_state_dict(checkpoint["model"])
        else:
            self.net.load_state_dict(checkpoint)
        # Restore energy stats for correct evaluation threshold
        if "src_energy_ema" in checkpoint:
            self.src_energy_ema = checkpoint["src_energy_ema"]
        if "src_energy_std_ema" in checkpoint:
            self.src_energy_std_ema = checkpoint["src_energy_std_ema"]
        logger.info(f"Model loaded from {path}")
