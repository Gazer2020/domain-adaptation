"""
DARE: relation-conditioned class-aware FiLM + EMA for MSDA.

Design:
- Use one shared feature space for source prototypes, relation inference,
  FiLM modulation, and final classification.
- For each class, compare a sample to every source-domain prototype of that
  class, producing class-specific cross-domain relation weights.
- Use those relations to build a class-aware FiLM-conditioned feature bank.
- Use an EMA teacher on weak target views to supervise strong target views.
"""

import copy
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import AverageMeter, cycle

logger = logging.getLogger(__name__)


class RelationConditionedClassFiLM(nn.Module):
    """Generate per-class FiLM parameters in the shared feature space."""

    def __init__(self, feat_dim: int, num_source_domains: int, hidden_dim: int = 256, scale: float = 0.5):
        super().__init__()
        self.scale = float(scale)
        in_dim = feat_dim * 3 + num_source_domains
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h_shared: torch.Tensor,
        proto_context: torch.Tensor,
        relation_weights: torch.Tensor,
    ) -> torch.Tensor:
        h_expand = h_shared.unsqueeze(1).expand(-1, proto_context.size(1), -1)
        cond = torch.cat(
            [h_expand, proto_context, (h_expand - proto_context).abs(), relation_weights],
            dim=-1,
        )
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1.0 + self.scale * torch.tanh(gamma)) * h_expand + self.scale * beta


class DARENetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_source_domains: int,
        *,
        bottleneck_dim: int = 0,
        relation_hidden_dim: int = 256,
        relation_temperature: float = 0.10,
        film_scale: float = 0.5,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.num_source_domains = int(num_source_domains)
        self.relation_temperature = max(1e-6, float(relation_temperature))

        self.backbone = get_backbone(backbone_name)
        if not hasattr(self.backbone, "fc"):
            raise NotImplementedError("Backbone feature dimension not found (missing `fc`).")

        feat_dim_raw = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if bottleneck_dim and int(bottleneck_dim) > 0:
            bottleneck_dim = int(bottleneck_dim)
            self.bottleneck = nn.Sequential(
                nn.Linear(feat_dim_raw, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.feat_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            self.feat_dim = feat_dim_raw

        self.feature_norm = nn.LayerNorm(self.feat_dim)
        self.relation_film = RelationConditionedClassFiLM(
            feat_dim=self.feat_dim,
            num_source_domains=self.num_source_domains,
            hidden_dim=int(relation_hidden_dim),
            scale=float(film_scale),
        )
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)

        self.register_buffer(
            "src_prototypes",
            torch.zeros(self.num_source_domains, self.num_classes, self.feat_dim),
            persistent=True,
        )
        self.register_buffer(
            "src_proto_inited",
            torch.zeros(self.num_source_domains, self.num_classes, dtype=torch.bool),
            persistent=True,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.backbone(x))

    def normalize_features(self, h: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(h)

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.src_prototypes.zero_()
        self.src_proto_inited.zero_()

    def _build_relation(
        self,
        h_shared: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proto_mask = self.src_proto_inited.transpose(0, 1)  # [C, D]
        proto_norm = F.normalize(self.src_prototypes, dim=-1)  # [D, C, F]
        h_rel = F.normalize(h_shared, dim=-1)  # [B, F]

        relation_logits = torch.einsum("bf,dcf->bcd", h_rel, proto_norm)
        relation_logits = relation_logits / self.relation_temperature
        relation_logits = relation_logits.masked_fill(~proto_mask.unsqueeze(0), -1e4)

        relation_weights = torch.softmax(relation_logits, dim=-1)
        valid_classes = proto_mask.any(dim=-1)
        if (~valid_classes).any():
            relation_weights = relation_weights.clone()
            relation_weights[:, ~valid_classes, :] = 1.0 / float(self.num_source_domains)

        proto_context = torch.einsum("bcd,dcf->bcf", relation_weights, self.src_prototypes)
        if (~valid_classes).any():
            proto_context = proto_context.clone()
            proto_context[:, ~valid_classes, :] = 0.0

        return relation_logits, relation_weights, proto_context

    def forward_relation_logits_from_shared(
        self,
        h_shared: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        relation_logits, relation_weights, proto_context = self._build_relation(h_shared)
        h_class = self.relation_film(h_shared, proto_context, relation_weights)

        base_logits = self.classifier(h_shared)
        weight = self.classifier.weight.unsqueeze(0)
        delta_logits = ((h_class - h_shared.unsqueeze(1)) * weight).sum(dim=-1)
        logits = base_logits + delta_logits

        aux = {
            "base_logits": base_logits,
            "delta_logits": delta_logits,
            "relation_logits": relation_logits,
            "relation_weights": relation_weights,
            "proto_context": proto_context,
            "h_shared": h_shared,
        }
        return logits, aux

    def forward_relation_logits(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if h_shared is None:
            if x is None:
                raise ValueError("Either x or h_shared must be provided.")
            h = self.extract_features(x)
            h_shared = self.normalize_features(h)
        return self.forward_relation_logits_from_shared(h_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_relation_logits(x=x)
        return logits


def soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    losses = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if weights is not None:
        weights = weights.detach()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()


def _unwrap_weak_strong_from_maybe_tuple(tgt_imgs):
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) >= 2:
        return tgt_imgs[0], tgt_imgs[1]
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) == 1:
        return tgt_imgs[0], tgt_imgs[0]
    return tgt_imgs, tgt_imgs


@register_solver("dare")
class DARESolver(BaseSolver):
    def _resolve_epoch_steps(self) -> int:
        source_steps = max(1, len(self.source_loader))
        target_steps = max(1, len(self.target_loader))
        mode = str(self.config.method.get("epoch_steps_mode", "max")).strip().lower()

        if mode == "source":
            return source_steps
        if mode == "target":
            return target_steps
        if mode == "mean":
            return max(1, int(round((source_steps + target_steps) / 2.0)))
        if mode == "max":
            return max(source_steps, target_steps)

        raise ValueError(f"Unsupported epoch_steps_mode: {mode}")

    def build_model(self):
        m = self.config.method
        backbone_name = m.get("backbone", "resnet50")
        sources = getattr(self.config.dataset, "sources", None)
        if sources is None or len(list(sources)) == 0:
            raise ValueError("dare requires config.dataset.sources to be a non-empty list")

        self.num_source_domains = len(list(sources))

        self.bottleneck_dim = int(m.get("bottleneck_dim", 256))
        self.relation_hidden_dim = int(m.get("relation_hidden_dim", 256))
        self.relation_temperature = float(m.get("relation_temperature", 0.10))
        self.film_scale = float(m.get("film_scale", 0.5))
        self.prototype_momentum = float(m.get("prototype_momentum", 0.9))

        self.lambda_pseudo = float(m.get("lambda_pseudo", 0.08))
        self.lambda_source_relation = float(m.get("lambda_source_relation", 0.25))
        self.lambda_proto_align = float(m.get("lambda_proto_align", 0.05))
        self.relation_label_smoothing = float(m.get("relation_label_smoothing", 0.10))
        self.pseudo_conf_power = float(m.get("pseudo_conf_power", 2.0))
        self.pseudo_start_epoch = int(m.get("pseudo_start_epoch", 6))
        self.enable_confidence_weighting = self._is_truthy(m.get("enable_confidence_weighting", True))
        self.refresh_source_prototypes_each_epoch = bool(
            m.get("refresh_source_prototypes_each_epoch", True)
        )
        self.enable_relation_forward = self._is_truthy(m.get("enable_relation_forward", True))
        self.enable_source_relation_loss = self._is_truthy(m.get("enable_source_relation_loss", True))
        self.enable_ema_pseudo = self._is_truthy(m.get("enable_ema_pseudo", True))
        self.enable_proto_loss = self._is_truthy(m.get("enable_proto_loss", True))

        self.T_sem = float(m.get("T_sem", 0.8))
        self.total_epochs = int(m.get("epochs", 20))
        self.ramp_denom = float(m.get("ramp_denom", max(1.0, self.total_epochs * 0.3)))
        self.grad_clip = float(m.get("grad_clip", 5.0))
        self.save_ckpt_after_epoch = int(m.get("save_ckpt_after_epoch", 0))
        self.epoch_steps_mode = str(m.get("epoch_steps_mode", "max")).strip().lower()
        self.ema_decay_start = float(m.get("ema_decay_start", 0.996))
        self.ema_decay_end = float(m.get("ema_decay_end", 0.9995))

        self.label_smoothing = float(m.get("label_smoothing", 0.05))
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.net = DARENetwork(
            backbone_name=backbone_name,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            bottleneck_dim=self.bottleneck_dim,
            relation_hidden_dim=self.relation_hidden_dim,
            relation_temperature=self.relation_temperature,
            film_scale=self.film_scale,
        ).to(self.device)

        self.ema_net = copy.deepcopy(self.net)
        for param in self.ema_net.parameters():
            param.requires_grad_(False)

        logger.info(
            "DARE relation-FiLM: bottleneck=%d rel_hidden=%d rel_temp=%.3f "
            "film_scale=%.3f lambda_rel=%.3f lambda_proto=%.3f lambda_pseudo=%.3f "
            "pseudo_start=%d proto_refresh_each_epoch=%s relation_forward=%s rel_loss=%s "
            "ema_pseudo=%s proto_loss=%s",
            self.bottleneck_dim,
            self.relation_hidden_dim,
            self.relation_temperature,
            self.film_scale,
            self.lambda_source_relation,
            self.lambda_proto_align,
            self.lambda_pseudo,
            self.pseudo_start_epoch,
            self.refresh_source_prototypes_each_epoch,
            self.enable_relation_forward,
            self.enable_source_relation_loss,
            self.enable_ema_pseudo,
            self.enable_proto_loss,
        )

    def _forward_logits(
        self,
        model: DARENetwork,
        *,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if h_shared is None:
            if x is None:
                raise ValueError("Either x or h_shared must be provided.")
            h = model.extract_features(x)
            h_shared = model.normalize_features(h)

        if self.enable_relation_forward:
            return model.forward_relation_logits(h_shared=h_shared)

        logits = model.classifier(h_shared)
        zeros_delta = torch.zeros_like(logits)
        proto_context = self._plain_proto_context(model, h_shared)
        relation_logits = torch.zeros(
            h_shared.size(0),
            self.num_classes,
            self.num_source_domains,
            device=h_shared.device,
            dtype=h_shared.dtype,
        )
        relation_weights = torch.full_like(
            relation_logits,
            1.0 / float(max(1, self.num_source_domains)),
        )
        aux = {
            "base_logits": logits,
            "delta_logits": zeros_delta,
            "relation_logits": relation_logits,
            "relation_weights": relation_weights,
            "proto_context": proto_context,
            "h_shared": h_shared,
        }
        return logits, aux

    def _plain_proto_context(self, model: DARENetwork, h_shared: torch.Tensor) -> torch.Tensor:
        proto_mask = model.src_proto_inited.transpose(0, 1)  # [C, D]
        valid_counts = proto_mask.sum(dim=1).clamp_min(1).to(model.src_prototypes.dtype)
        proto_sum = (
            model.src_prototypes
            * model.src_proto_inited.unsqueeze(-1).to(model.src_prototypes.dtype)
        ).sum(dim=0)
        class_proto = proto_sum / valid_counts.unsqueeze(-1)
        valid_classes = proto_mask.any(dim=1)
        class_proto = torch.where(
            valid_classes.unsqueeze(-1),
            class_proto,
            torch.zeros_like(class_proto),
        )
        return class_proto.to(dtype=h_shared.dtype).unsqueeze(0).expand(h_shared.size(0), -1, -1)

    def _set_eval_mode(self):
        self.net.eval()
        self.ema_net.eval()

    @torch.no_grad()
    def _update_ema(self, decay: float):
        for p_ema, p_student in zip(self.ema_net.parameters(), self.net.parameters()):
            p_ema.data.mul_(decay).add_(p_student.data, alpha=1.0 - decay)
        for b_ema, b_student in zip(self.ema_net.buffers(), self.net.buffers()):
            b_ema.data.copy_(b_student.data)

    def _ema_decay_at(self, step: int, total_steps: int) -> float:
        progress = min(1.0, step / max(1, total_steps))
        return self.ema_decay_start + (self.ema_decay_end - self.ema_decay_start) * progress

    def _source_relation_loss(
        self,
        relation_logits: torch.Tensor,
        src_labels: torch.Tensor,
        src_dom: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(src_labels.size(0), device=src_labels.device)
        true_class_relation = relation_logits[batch_idx, src_labels]
        num_domains = true_class_relation.size(1)
        if num_domains <= 1:
            return torch.zeros((), device=true_class_relation.device, dtype=true_class_relation.dtype)

        off_value = self.relation_label_smoothing / float(num_domains - 1)
        target = torch.full_like(true_class_relation, off_value)
        target.scatter_(1, src_dom.unsqueeze(1), 1.0 - self.relation_label_smoothing)
        return soft_target_cross_entropy(true_class_relation, target)

    def _target_proto_alignment_loss(
        self,
        h_shared: torch.Tensor,
        proto_context: torch.Tensor,
        class_probs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        proto_sel = torch.einsum("bc,bcf->bf", class_probs.detach(), proto_context)
        h_n = F.normalize(h_shared, dim=1)
        p_n = F.normalize(proto_sel, dim=1)
        losses = 1.0 - (h_n * p_n).sum(dim=1)
        weights = weights.detach()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    @torch.no_grad()
    def _teacher_guidance(self, tgt_weak: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.ema_net.eval()
        logits_pre, _ = self._forward_logits(self.ema_net, x=tgt_weak)
        q_tgt = F.softmax(logits_pre / self.T_sem, dim=1)
        conf = q_tgt.max(dim=1).values.detach()
        return q_tgt, conf

    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        self.ema_net.eval()
        if isinstance(imgs, (tuple, list)):
            imgs = imgs[0]
        with torch.no_grad():
            logits, _ = self._forward_logits(self.ema_net, x=imgs)
            return logits

    @torch.no_grad()
    def _recompute_source_prototypes(self, model: DARENetwork):
        was_training = model.training
        model.eval()

        feat_sums = torch.zeros(
            self.num_source_domains,
            self.num_classes,
            model.feat_dim,
            device=self.device,
        )
        counts = torch.zeros(
            self.num_source_domains,
            self.num_classes,
            device=self.device,
        )
        feat_sums_flat = feat_sums.view(-1, model.feat_dim)
        counts_flat = counts.view(-1)

        for src_imgs, src_labels, src_dom in self.source_loader:
            src_imgs = self._to_device(src_imgs)
            src_labels = self._to_device(src_labels)
            src_dom = self._to_device(src_dom)

            with self._auto_cast():
                h = model.extract_features(src_imgs)
                h_shared = model.normalize_features(h)

            flat_index = src_dom.long() * self.num_classes + src_labels.long()
            feat_sums_flat.index_add_(0, flat_index, h_shared)
            counts_flat.index_add_(
                0,
                flat_index,
                torch.ones_like(flat_index, dtype=counts_flat.dtype),
            )

        model.reset_source_prototypes()
        valid = counts > 0
        safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
        prototypes = feat_sums / safe_counts
        prototypes = torch.where(valid.unsqueeze(-1), prototypes, torch.zeros_like(prototypes))
        model.src_proto_inited.copy_(valid)
        model.src_prototypes.copy_(prototypes)

        model.train(was_training)

    def save_checkpoint(self, path):
        self._save_named_modules_checkpoint(
            path,
            modules={
                "student": self.net,
                "ema": self.ema_net,
            },
        )

    def load_checkpoint(self, path):
        checkpoint = self._load_checkpoint_file(path)
        student_state = checkpoint["student"] if isinstance(checkpoint, dict) and "student" in checkpoint else checkpoint
        ema_state = checkpoint["ema"] if isinstance(checkpoint, dict) and "ema" in checkpoint else student_state

        self.net.load_state_dict(student_state, strict=False)
        self.ema_net.load_state_dict(ema_state, strict=False)
        logger.info("%s checkpoint loaded from %s", self._solver_display_name(), path)

    def train(self):
        base_lr = float(self.config.method.lr)
        param_groups = [
            {"params": list(self.net.backbone.parameters()), "lr": base_lr * 0.1},
            {"params": list(self.net.bottleneck.parameters()), "lr": base_lr},
            {"params": list(self.net.feature_norm.parameters()), "lr": base_lr},
            {"params": list(self.net.relation_film.parameters()), "lr": base_lr},
            {"params": list(self.net.classifier.parameters()), "lr": base_lr},
        ]
        param_groups = [group for group in param_groups if len(group["params"]) > 0]

        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        epoch_steps = self._resolve_epoch_steps()
        total_iters = self.total_epochs * epoch_steps

        def lr_lambda(step):
            progress = step / max(1, total_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self.register_training_state(optimizer=optimizer, scheduler=scheduler)
        best_acc = self._best_metric

        global_step = self._training_global_step
        logger.info(
            "DARE Training: relation-conditioned class-aware FiLM | epoch_steps_mode=%s "
            "source_steps=%d target_steps=%d epoch_steps=%d",
            self.epoch_steps_mode,
            len(self.source_loader),
            len(self.target_loader),
            epoch_steps,
        )

        needs_source_prototypes = self.enable_relation_forward or (
            self.enable_ema_pseudo and self.enable_proto_loss
        )

        for epoch in self._epoch_range(self.total_epochs):
            if needs_source_prototypes and (self.refresh_source_prototypes_each_epoch or epoch == 0):
                self._recompute_source_prototypes(self.net)
                self.ema_net.src_prototypes.copy_(self.net.src_prototypes)
                self.ema_net.src_proto_inited.copy_(self.net.src_proto_inited)

            self.net.train()
            meters = {
                key: AverageMeter()
                for key in ["task", "rel", "proto", "pseudo", "conf", "pwt", "rmax", "dlogit", "total"]
            }
            src_iter = cycle(self.source_loader)
            tgt_iter = cycle(self.target_loader)
            ramp = min(1.0, (epoch + 1) / max(1.0, self.ramp_denom))
            pseudo_ramp = 1.0 if (epoch + 1) >= self.pseudo_start_epoch else 0.0

            for _ in range(epoch_steps):
                src_imgs, src_labels, src_dom = next(src_iter)
                tgt_batch = next(tgt_iter)
                tgt_imgs = tgt_batch[0] if isinstance(tgt_batch, (tuple, list)) else tgt_batch
                tgt_weak, tgt_strong = _unwrap_weak_strong_from_maybe_tuple(tgt_imgs)

                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)
                src_dom = self._to_device(src_dom)
                tgt_weak = self._to_device(tgt_weak)
                tgt_strong = self._to_device(tgt_strong)

                self._zero_grad(optimizer)

                with self._auto_cast():
                    src_h = self.net.extract_features(src_imgs)
                    src_h_shared = self.net.normalize_features(src_h)
                    logits_src, src_aux = self._forward_logits(self.net, h_shared=src_h_shared)
                    loss_task = self.criterion_task(logits_src, src_labels)
                    if self.enable_relation_forward and self.enable_source_relation_loss:
                        loss_rel = self._source_relation_loss(
                            src_aux["relation_logits"],
                            src_labels,
                            src_dom,
                        )
                    else:
                        loss_rel = torch.zeros((), device=self.device, dtype=loss_task.dtype)

                    tgt_h = self.net.extract_features(tgt_strong)
                    tgt_h_shared = self.net.normalize_features(tgt_h)
                    logits_tgt, tgt_aux = self._forward_logits(self.net, h_shared=tgt_h_shared)

                    if self.enable_ema_pseudo:
                        with torch.no_grad():
                            with self._auto_cast():
                                q_tgt, conf_tgt = self._teacher_guidance(tgt_weak)
                            if self.enable_confidence_weighting:
                                pseudo_weights = conf_tgt.pow(self.pseudo_conf_power)
                            else:
                                pseudo_weights = torch.ones_like(conf_tgt)
                        pseudo_loss_weights = pseudo_weights * pseudo_ramp
                        loss_pseudo = soft_target_cross_entropy(
                            logits_tgt,
                            q_tgt.detach(),
                            weights=pseudo_loss_weights,
                        )
                        if self.enable_proto_loss:
                            loss_proto = self._target_proto_alignment_loss(
                                tgt_h_shared,
                                tgt_aux["proto_context"],
                                q_tgt.detach(),
                                pseudo_loss_weights,
                            )
                        else:
                            loss_proto = torch.zeros((), device=self.device, dtype=loss_task.dtype)
                    else:
                        conf_tgt = torch.zeros(tgt_h_shared.size(0), device=self.device, dtype=loss_task.dtype)
                        pseudo_loss_weights = torch.zeros_like(conf_tgt)
                        loss_pseudo = torch.zeros((), device=self.device, dtype=loss_task.dtype)
                        loss_proto = torch.zeros((), device=self.device, dtype=loss_task.dtype)

                    loss = (
                        loss_task
                        + self.lambda_source_relation * loss_rel
                        + self.lambda_proto_align * ramp * loss_proto
                        + self.lambda_pseudo * ramp * loss_pseudo
                    )
                self._optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    clip_params=self.net.parameters(),
                    clip_max_norm=self.grad_clip,
                )
                scheduler.step()

                self._update_ema(self._ema_decay_at(global_step, total_iters))
                global_step += 1

                if self.enable_relation_forward:
                    src_true_rel = src_aux["relation_weights"][
                        torch.arange(src_labels.size(0), device=self.device),
                        src_labels,
                    ]
                    rmax_value = src_true_rel.max(dim=1).values.mean().item()
                else:
                    rmax_value = 0.0
                meters["task"].update(loss_task.item())
                meters["rel"].update(loss_rel.item())
                meters["proto"].update(loss_proto.item())
                meters["pseudo"].update(loss_pseudo.item())
                meters["conf"].update(conf_tgt.mean().item())
                meters["pwt"].update(pseudo_loss_weights.mean().item())
                meters["rmax"].update(rmax_value)
                meters["dlogit"].update(tgt_aux["delta_logits"].abs().mean().item())
                meters["total"].update(loss.item())

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            if epoch + 1 > self.save_ckpt_after_epoch:
                self._maybe_save_best(acc, epoch + 1)

            self._log_epoch_summary(
                epoch + 1,
                self.total_epochs,
                metrics={
                    "task": meters["task"].avg,
                    "rel": meters["rel"].avg,
                    "proto": meters["proto"].avg,
                    "pseudo": meters["pseudo"].avg,
                    "qconf": (meters["conf"].avg, ".3f"),
                    "pwt": (meters["pwt"].avg, ".3f"),
                    "rmax": (meters["rmax"].avg, ".3f"),
                    "dlogit": meters["dlogit"].avg,
                    "total": meters["total"].avg,
                },
                extras={"rmp": (ramp, ".2f")},
                score=acc,
                best_score=best_acc,
                score_name="Acc",
            )

        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")
