"""
Multi-source Domain-Conditioned Feature Modulation (MDCFM).

This is a multi-source extension of DCFM for closed-set MSDA:
- Multiple source domains (S) + one target domain.
- Domain classifier predicts S+1 domain classes (S sources + target).
- FiLM modulation is conditioned on the domain representation z_domain.

Data contract (only for setting=msda):
- source_loader yields (src_imgs, src_labels, src_domain_id) where src_domain_id ∈ [0..S-1]
- target_loader yields (tgt_imgs, _); solver creates tgt_domain_id = S
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
    Multi-class domain classifier that predicts S source domains + 1 target domain.
    Returns (domain_logits, z_domain).
    """

    def __init__(self, in_features: int, num_domains: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_domains)

    def forward(self, x: torch.Tensor):
        z_domain = F.relu(self.ln1(self.fc1(x)))
        domain_logits = self.fc2(z_domain)
        return domain_logits, z_domain


class FiLMModulation(nn.Module):
    def __init__(self, feat_dim: int, domain_dim: int = 256, scale_factor: float = 4.0):
        super().__init__()
        self.scale_factor = float(scale_factor)
        self.gamma_net = nn.Sequential(
            nn.Linear(domain_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.beta_net = nn.Linear(domain_dim, feat_dim)

        nn.init.zeros_(self.gamma_net[0].weight)
        init_bias = -math.log(self.scale_factor - 1.0)
        nn.init.constant_(self.gamma_net[0].bias, init_bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: torch.Tensor, z_domain: torch.Tensor):
        gamma = self.gamma_net(z_domain) * self.scale_factor
        beta = self.beta_net(z_domain)
        return gamma * h + beta


class MDCFMNetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_domains: int,
        domain_hidden_dim: int = 256,
        bottleneck_dim: int = 0,
        film_scale: float = 4.0,
    ):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        if hasattr(self.backbone, "fc"):
            self.feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError("Backbone feature dimension not found.")

        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.domain_classifier = DomainClassifier(self.feat_dim, num_domains=num_domains, hidden_dim=domain_hidden_dim)
        self.modulator = FiLMModulation(self.feat_dim, domain_hidden_dim, film_scale)

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
        return self.backbone(x)

    def get_domain_logits(self, h: torch.Tensor):
        dom_logits, _ = self.domain_classifier(h.detach())
        return dom_logits

    def get_domain_z(self, h: torch.Tensor):
        _, z_domain = self.domain_classifier(h)
        return z_domain

    def forward_modulated(self, h: torch.Tensor, z_domain: torch.Tensor):
        h_normed = self.feat_bn(h)
        h_mod = self.modulator(h_normed, z_domain)
        return self.classifier(h_mod)

    def forward(self, x: torch.Tensor):
        h = self.extract_features(x)
        domain_logits = self.get_domain_logits(h)
        z_domain = self.get_domain_z(h)
        task_logits = self.forward_modulated(h, z_domain)
        return task_logits, domain_logits


def information_maximization_loss(logits: torch.Tensor, diversity_weight: float = 2.0) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    ent_individual = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
    mean_probs = probs.mean(dim=0)
    ent_diversity = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
    return ent_individual - diversity_weight * ent_diversity


@register_solver("mdcfm")
class MDCFMSolver(BaseSolver):
    """
    Multi-source extension of DCFM under setting=msda.
    """

    def build_model(self):
        backbone_name = self.config.method.get("backbone", "resnet50")
        domain_hidden_dim = int(self.config.method.get("domain_hidden_dim", 256))
        bottleneck_dim = int(self.config.method.get("bottleneck_dim", 0))
        film_scale = float(self.config.method.get("film_scale", 4.0))

        sources = getattr(self.config.dataset, "sources", None)
        if sources is None or len(list(sources)) == 0:
            raise ValueError("mdcfm requires config.dataset.sources to be a non-empty list")
        self.num_source_domains = len(list(sources))
        self.target_domain_id = self.num_source_domains
        self.num_domains = self.num_source_domains + 1

        self.net = MDCFMNetwork(
            backbone_name=backbone_name,
            num_classes=self.num_classes,
            num_domains=self.num_domains,
            domain_hidden_dim=domain_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            film_scale=film_scale,
        ).to(self.device)

        self.lambda_domain = float(self.config.method.get("lambda_domain", 1.0))
        self.lambda_im = float(self.config.method.get("lambda_im", 0.3))
        self.lambda_cf = float(self.config.method.get("lambda_cf", 1.0))
        self.lambda_div = float(self.config.method.get("lambda_div", 2.0))
        self.label_smoothing = float(self.config.method.get("label_smoothing", 0.1))

        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.criterion_domain = nn.CrossEntropyLoss()

    def forward_for_eval(self, imgs):
        logits, _ = self.net(imgs)
        return logits

    def train(self):
        warmup_epochs = int(self.config.method.get("warmup_epochs", 5))
        max_epochs = int(self.config.method.epochs)
        lr = float(self.config.method.lr)

        optimizer = optim.SGD(
            [
                {"params": self.net.backbone.parameters(), "lr": lr * 0.1},
                {"params": self.net.feat_bn.parameters(), "lr": lr},
                {"params": self.net.domain_classifier.parameters(), "lr": lr},
                {"params": self.net.modulator.parameters(), "lr": lr},
                {"params": self.net.classifier.parameters(), "lr": lr},
            ],
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

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

        logger.info(f"MDCFM Training (MSDA): {warmup_epochs} warmup + {max_epochs} joint epochs | S={self.num_source_domains}")

        # ===================== Stage 1: Warmup ===================== #
        logger.info("=== Stage 1: Source Warmup ===")
        for epoch in range(warmup_epochs):
            self.net.train()
            meters = {k: AverageMeter() for k in ["task", "domain", "total"]}
            tgt_iter = cycle(self.target_loader)

            for src_imgs, src_labels, src_dom in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                src_dom = src_dom.to(self.device)
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

                tgt_dom = torch.full((bs_tgt,), self.target_domain_id, dtype=torch.long, device=self.device)
                domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
                domain_labels = torch.cat([src_dom.long(), tgt_dom], dim=0)
                loss_domain = self.criterion_domain(domain_logits, domain_labels)

                loss = loss_task + self.lambda_domain * loss_domain
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters["task"].update(loss_task.item())
                meters["domain"].update(loss_domain.item())
                meters["total"].update(loss.item())

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
                self.save_checkpoint(best_path)
            logger.info(
                f"Warmup {epoch+1} | task={meters['task'].avg:.4f} dom={meters['domain'].avg:.4f} "
                f"| Acc={acc:.2f}% (best={best_acc:.2f}%)"
            )

        # ===================== Stage 2: Joint ===================== #
        logger.info("=== Stage 2: Joint Training with Feature Hallucination & IM ===")
        for epoch in range(max_epochs):
            self.net.train()
            meters = {k: AverageMeter() for k in ["task", "domain", "im", "cf", "total"]}
            tgt_iter = cycle(self.target_loader)
            ramp = min(1.0, (epoch + 1) / max(1, max_epochs * 0.3))

            for src_imgs, src_labels, src_dom in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)
                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                src_dom = src_dom.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                bs_src, bs_tgt = src_imgs.size(0), tgt_imgs.size(0)

                optimizer.zero_grad()

                h_src = self.net.extract_features(src_imgs)
                h_tgt = self.net.extract_features(tgt_imgs)

                domain_logits_src = self.net.get_domain_logits(h_src)
                domain_logits_tgt = self.net.get_domain_logits(h_tgt)

                z_src = self.net.get_domain_z(h_src)
                z_tgt = self.net.get_domain_z(h_tgt)

                task_logits_src = self.net.forward_modulated(h_src, z_src)
                task_logits_tgt = self.net.forward_modulated(h_tgt, z_tgt)

                loss_task = self.criterion_task(task_logits_src, src_labels)

                tgt_dom = torch.full((bs_tgt,), self.target_domain_id, dtype=torch.long, device=self.device)
                domain_logits = torch.cat([domain_logits_src, domain_logits_tgt], dim=0)
                domain_labels = torch.cat([src_dom.long(), tgt_dom], dim=0)
                loss_domain = self.criterion_domain(domain_logits, domain_labels)

                loss_im = information_maximization_loss(task_logits_tgt, diversity_weight=self.lambda_div)

                min_bs = min(bs_src, bs_tgt)
                shuffle_idx = torch.randperm(min_bs, device=self.device)
                h_tgt_shuffled = h_tgt[:min_bs][shuffle_idx]
                z_tgt_shuffled = z_tgt[:min_bs][shuffle_idx]

                lam = beta_dist.sample().item()
                lam = max(lam, 1.0 - lam)

                h_cross = lam * h_src[:min_bs] + (1 - lam) * h_tgt_shuffled
                z_cross = lam * z_src[:min_bs].detach() + (1 - lam) * z_tgt_shuffled.detach()

                task_logits_cross = self.net.classifier(self.net.modulator(self.net.feat_bn(h_cross), z_cross))

                prob_tgt = F.softmax(task_logits_tgt[:min_bs][shuffle_idx].detach(), dim=1)
                loss_src_part = lam * F.cross_entropy(task_logits_cross, src_labels[:min_bs])
                log_prob_cross = F.log_softmax(task_logits_cross, dim=1)
                loss_tgt_part = (1 - lam) * torch.sum(-prob_tgt * log_prob_cross, dim=1).mean()
                loss_cf = loss_src_part + loss_tgt_part

                loss = (
                    loss_task
                    + self.lambda_domain * loss_domain
                    + self.lambda_im * ramp * loss_im
                    + self.lambda_cf * ramp * loss_cf
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters["task"].update(loss_task.item())
                meters["domain"].update(loss_domain.item())
                meters["im"].update(loss_im.item())
                meters["cf"].update(loss_cf.item())
                meters["total"].update(loss.item())

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
                self.save_checkpoint(best_path)
            logger.info(
                f"Joint {epoch+1:02d} | ts={meters['task'].avg:.3f} dm={meters['domain'].avg:.3f} "
                f"im={meters['im'].avg:.3f} cf={meters['cf'].avg:.3f} | rmp={ramp:.2f} | "
                f"Acc={acc:.2f}% (best={best_acc:.2f}%)"
            )

        if best_path.exists():
            self.load_checkpoint(best_path)
            logger.info(f"Loaded best model (Acc={best_acc:.2f}%) from {best_path}")

        logger.info(f"Training finished. Best Acc: {best_acc:.2f}%")

    def save_checkpoint(self, path):
        torch.save(
            {
                "method": "mdcfm",
                "model": self.net.state_dict(),
                "num_domains": self.num_domains,
                "num_source_domains": self.num_source_domains,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if "model" in checkpoint:
            self.net.load_state_dict(checkpoint["model"])
        else:
            self.net.load_state_dict(checkpoint)
        logger.info(f"Model loaded from {path}")

