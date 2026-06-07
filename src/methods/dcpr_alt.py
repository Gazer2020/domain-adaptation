"""
DCPR-ALT: prototype-relation target adaptation.

This method is copied from DCPR and keeps the source-domain prototype graph,
routing, EMA teacher, and prototype refresh mechanics. Its objective has two
parts:
- L_src: source supervised classification through the prototype classifier.
- L_rel: target EMA consistency over the domain-class prototype relation.
"""

import copy
import logging
import math
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from methods.base_solver import BaseSolver
from methods.components import TargetViewBuilder, linear_ema_decay, update_ema_model
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import CudaBatchPrefetcher, cycle

logger = logging.getLogger(__name__)


def soft_prob_cross_entropy(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    student_probs = student_probs.clamp_min(1e-8)
    teacher_probs = teacher_probs.detach()
    losses = -(teacher_probs * student_probs.log()).sum(dim=1)
    if weights is not None:
        weights = weights.detach()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)
    return losses.mean()


class PrototypeRelationRouter(nn.Module):
    """Prototype classifier plus class-conditioned source-domain routing."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_source_domains: int,
        relation_temperature: float = 0.10,
        relation_space_mode: str = "standard",
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.num_source_domains = int(num_source_domains)
        self.relation_space_mode = str(relation_space_mode).strip().lower()
        if self.relation_space_mode not in {"standard", "domain_relative"}:
            raise ValueError("relation_space_mode must be 'standard' or 'domain_relative'")
        self.register_buffer(
            "relation_temperature",
            torch.tensor(max(1e-6, float(relation_temperature)), dtype=torch.float32),
            persistent=True,
        )

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
        self.register_buffer(
            "src_domain_centers",
            torch.zeros(self.num_source_domains, self.feat_dim),
            persistent=True,
        )
        self.register_buffer(
            "src_domain_center_inited",
            torch.zeros(self.num_source_domains, dtype=torch.bool),
            persistent=True,
        )

    @torch.no_grad()
    def set_relation_temperature(self, value: float):
        self.relation_temperature.fill_(max(1e-6, float(value)))

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.src_prototypes.zero_()
        self.src_proto_inited.zero_()
        self.src_domain_centers.zero_()
        self.src_domain_center_inited.zero_()

    def source_relation_prototypes(self) -> torch.Tensor:
        proto = self.src_prototypes
        if self.relation_space_mode != "domain_relative":
            return proto
        centers = self.src_domain_centers.unsqueeze(1)
        valid_centers = self.src_domain_center_inited.view(-1, 1, 1)
        return torch.where(valid_centers, proto - centers, proto)

    def center_features(
        self,
        h_relation: torch.Tensor,
        *,
        domain_ids: Optional[torch.Tensor] = None,
        sample_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.relation_space_mode != "domain_relative":
            return h_relation
        if sample_center is not None:
            center = sample_center.to(device=h_relation.device, dtype=h_relation.dtype)
            if center.ndim == 1:
                center = center.unsqueeze(0)
            return h_relation - center
        if domain_ids is not None:
            domain_ids = domain_ids.to(device=h_relation.device).long().clamp(0, self.num_source_domains - 1)
            centers = self.src_domain_centers.to(device=h_relation.device, dtype=h_relation.dtype)[domain_ids]
            valid = self.src_domain_center_inited.to(device=h_relation.device)[domain_ids].unsqueeze(1)
            return h_relation - torch.where(valid, centers, torch.zeros_like(centers))
        valid = self.src_domain_center_inited.to(device=h_relation.device)
        if bool(valid.any()):
            center = self.src_domain_centers.to(device=h_relation.device, dtype=h_relation.dtype)[valid].mean(dim=0)
            return h_relation - center.unsqueeze(0)
        return h_relation

    def parse(
        self,
        h_relation: torch.Tensor,
        *,
        domain_ids: Optional[torch.Tensor] = None,
        sample_center: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return class logits plus sparse source-domain routing statistics."""
        proto = self.source_relation_prototypes()
        mask = self.src_proto_inited
        h_relation = self.center_features(
            h_relation,
            domain_ids=domain_ids,
            sample_center=sample_center,
        )

        proto_n = F.normalize(proto, dim=-1)
        h_n = F.normalize(h_relation, dim=-1)

        node_logits_bdc = torch.einsum("bf,dcf->bdc", h_n, proto_n)
        node_logits_bdc = node_logits_bdc / self.relation_temperature.to(dtype=node_logits_bdc.dtype)
        node_logits_bdc = node_logits_bdc.masked_fill(~mask.unsqueeze(0), -1e4)

        # Class evidence is the log-sum-exp over source-domain prototypes that
        # share the same class label.
        class_logits_rel = torch.logsumexp(node_logits_bdc, dim=1)
        valid_classes = mask.any(dim=0)
        class_logits_rel = class_logits_rel.masked_fill(~valid_classes.unsqueeze(0), -1e4)

        domain_logits = node_logits_bdc.permute(0, 2, 1).contiguous()
        domain_mask = mask.transpose(0, 1).unsqueeze(0)
        domain_logits = domain_logits.masked_fill(~domain_mask, -1e4)
        domain_weights = torch.softmax(domain_logits, dim=-1) * domain_mask.float()
        domain_weights = domain_weights / domain_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        domain_weights = torch.where(
            valid_classes.unsqueeze(0).unsqueeze(-1),
            domain_weights,
            torch.full_like(domain_weights, 1.0 / float(max(1, self.num_source_domains))),
        )

        return {
            "class_logits_rel": class_logits_rel,
            "domain_weights": domain_weights,
            "valid_classes": valid_classes,
            "h_relation": h_relation,
        }


class DCPRNetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_source_domains: int,
        *,
        bottleneck_dim: int = 256,
        relation_temperature: float = 0.10,
        adaptive_head_scale: float = 10.0,
        adaptive_logit_weight: float = 0.0,
        target_logit_weight: float = 0.0,
        relation_space_mode: str = "standard",
        class_logits_mode: str = "single",
        classifier_anchor_blend: float = 0.0,
        logit_fusion: str = "linear",
        proto_logit_weight: Optional[float] = None,
        proto_logit_temperature: float = 1.0,
        adaptive_logit_temperature: float = 1.0,
        target_logit_temperature: float = 1.0,
        feature_norm_mode: str = "layernorm",
        classification_structure: str = "legacy",
        target_classifier_mode: str = "head",
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.num_source_domains = int(num_source_domains)

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

        self.feature_norm_mode = str(feature_norm_mode).strip().lower()
        if self.feature_norm_mode == "layernorm":
            self.feature_norm = nn.LayerNorm(self.feat_dim)
        elif self.feature_norm_mode in {"l2", "identity"}:
            self.feature_norm = nn.Identity()
        else:
            raise ValueError("feature_norm_mode must be 'layernorm', 'l2', or 'identity'")
        self.relation_feat_dim = self.feat_dim
        self.adaptive_logit_weight = float(adaptive_logit_weight)
        self.target_logit_weight = float(target_logit_weight)
        self.adaptive_head_scale = float(adaptive_head_scale)
        self.relation_space_mode = str(relation_space_mode).strip().lower()
        self.class_logits_mode = str(class_logits_mode).strip().lower()
        if self.class_logits_mode not in {"single", "fusion"}:
            raise ValueError("class_logits_mode must be 'single' or 'fusion'")
        self.classifier_anchor_blend = min(1.0, max(0.0, float(classifier_anchor_blend)))
        self.logit_fusion = str(logit_fusion).strip().lower()
        if self.logit_fusion not in {"linear", "logprob"}:
            raise ValueError("logit_fusion must be 'linear' or 'logprob'")
        self.proto_logit_weight = (
            None if proto_logit_weight is None else min(1.0, max(0.0, float(proto_logit_weight)))
        )
        self.proto_logit_temperature = max(1e-6, float(proto_logit_temperature))
        self.adaptive_logit_temperature = max(1e-6, float(adaptive_logit_temperature))
        self.target_logit_temperature = max(1e-6, float(target_logit_temperature))
        self.classification_structure = str(classification_structure).strip().lower()
        if self.classification_structure not in {"legacy", "decoupled"}:
            raise ValueError("classification_structure must be 'legacy' or 'decoupled'")
        self.target_classifier_mode = str(target_classifier_mode).strip().lower()
        if self.target_classifier_mode not in {"head", "prototype"}:
            raise ValueError("target_classifier_mode must be 'head' or 'prototype'")

        self.relation_router = PrototypeRelationRouter(
            feat_dim=self.relation_feat_dim,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            relation_temperature=relation_temperature,
            relation_space_mode=self.relation_space_mode,
        )
        self.adaptive_classifier = nn.Linear(self.relation_feat_dim, self.num_classes, bias=False)
        self.target_classifier = nn.Linear(self.relation_feat_dim, self.num_classes, bias=False)
        self.register_buffer(
            "target_prototypes",
            torch.zeros(self.num_classes, self.relation_feat_dim),
            persistent=True,
        )
        self.register_buffer(
            "target_proto_inited",
            torch.zeros(self.num_classes, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "target_center",
            torch.zeros(self.relation_feat_dim),
            persistent=True,
        )
        self.register_buffer(
            "target_center_inited",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "class_relation_anchors",
            torch.zeros(self.num_classes, self.relation_feat_dim),
            persistent=True,
        )
        self.register_buffer(
            "class_anchor_inited",
            torch.zeros(self.num_classes, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "classifier_anchor_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "target_classifier_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    def extract_relation_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.backbone(x))

    def normalize_relation_features(self, h: torch.Tensor) -> torch.Tensor:
        h = self.feature_norm(h)
        if self.feature_norm_mode == "l2":
            h = F.normalize(h, dim=-1)
        return h

    def _encode_shared(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h_shared is not None:
            return h_shared
        if x is None:
            raise ValueError("Either x or h_shared must be provided.")
        h = self.extract_relation_features(x)
        return self.normalize_relation_features(h)

    @torch.no_grad()
    def set_relation_temperature(self, value: float):
        self.relation_router.set_relation_temperature(value)

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.relation_router.reset_source_prototypes()

    @torch.no_grad()
    def set_class_relation_anchors(self, valid: torch.Tensor, anchors: torch.Tensor):
        self.class_anchor_inited.copy_(valid.to(device=self.class_anchor_inited.device))
        self.class_relation_anchors.copy_(anchors.to(device=self.class_relation_anchors.device))

    def classifier_anchor_loss(self) -> torch.Tensor:
        valid = self.class_anchor_inited
        if not bool(valid.any()):
            return self.adaptive_classifier.weight.sum() * 0.0
        weights = F.normalize(self.adaptive_classifier.weight[valid], dim=-1)
        anchors = F.normalize(self.class_relation_anchors[valid], dim=-1)
        return (1.0 - (weights * anchors).sum(dim=1)).mean()

    @torch.no_grad()
    def maybe_init_classifier_from_anchors(self, *, force: bool = False):
        if bool(self.classifier_anchor_initialized) and not force:
            return
        valid = self.class_anchor_inited
        if not bool(valid.any()):
            return
        anchors = F.normalize(self.class_relation_anchors[valid], dim=-1)
        self.adaptive_classifier.weight.data[valid].copy_(anchors.to(dtype=self.adaptive_classifier.weight.dtype))
        self.classifier_anchor_initialized.fill_(True)

    @torch.no_grad()
    def maybe_init_target_classifier_from_source(self, *, force: bool = False):
        if bool(self.target_classifier_initialized) and not force:
            return
        self.target_classifier.weight.copy_(self.adaptive_classifier.weight)
        self.target_classifier_initialized.fill_(True)

    @torch.no_grad()
    def update_target_center(self, h_shared: torch.Tensor, momentum: float):
        if self.relation_space_mode != "domain_relative" or h_shared.numel() == 0:
            return
        batch_center = h_shared.detach().mean(dim=0).to(dtype=self.target_center.dtype)
        momentum = min(0.9999, max(0.0, float(momentum)))
        if bool(self.target_center_inited):
            self.target_center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)
        else:
            self.target_center.copy_(batch_center)
            self.target_center_inited.fill_(True)

    def _target_sample_center(self, h_shared: torch.Tensor) -> Optional[torch.Tensor]:
        if self.relation_space_mode != "domain_relative" or not bool(self.target_center_inited):
            return None
        return self.target_center.to(device=h_shared.device, dtype=h_shared.dtype)

    @torch.no_grad()
    def update_target_prototypes(
        self,
        h_shared: torch.Tensor,
        probs: torch.Tensor,
        conf: torch.Tensor,
        *,
        threshold: float,
        momentum: float,
    ) -> torch.Tensor:
        pseudo = probs.argmax(dim=1)
        selected = conf >= float(threshold)
        selected_rate = selected.float().mean()
        if not bool(selected.any()):
            return selected_rate

        h_selected = h_shared[selected].detach()
        y_selected = pseudo[selected].detach()
        momentum = min(0.9999, max(0.0, float(momentum)))
        for cls in y_selected.unique(sorted=False):
            cls_idx = int(cls.item())
            cls_mask = y_selected == cls
            cls_feat = h_selected[cls_mask].mean(dim=0)
            if bool(self.target_proto_inited[cls_idx]):
                self.target_prototypes[cls_idx].mul_(momentum).add_(cls_feat, alpha=1.0 - momentum)
            else:
                self.target_prototypes[cls_idx].copy_(cls_feat)
                self.target_proto_inited[cls_idx] = True
        return selected_rate

    @torch.no_grad()
    def update_target_prototypes_soft(
        self,
        h_shared: torch.Tensor,
        responsibilities: torch.Tensor,
        *,
        momentum: float,
    ) -> torch.Tensor:
        """Update target anchors with soft class responsibility, no hard sample gate."""
        if responsibilities.numel() == 0:
            return torch.zeros((), device=h_shared.device)

        resp = responsibilities.detach().clamp_min(0.0)
        resp = resp / resp.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mass = resp.sum(dim=0)
        valid = mass > 1e-6
        if not bool(valid.any()):
            return torch.zeros((), device=h_shared.device)

        proto = resp.t() @ h_shared.detach()
        proto = proto / mass.clamp_min(1e-6).unsqueeze(1)
        momentum = min(0.9999, max(0.0, float(momentum)))
        for cls in valid.nonzero(as_tuple=False).flatten():
            cls_idx = int(cls.item())
            if bool(self.target_proto_inited[cls_idx]):
                self.target_prototypes[cls_idx].mul_(momentum).add_(proto[cls_idx], alpha=1.0 - momentum)
            else:
                self.target_prototypes[cls_idx].copy_(proto[cls_idx])
                self.target_proto_inited[cls_idx] = True
        return mass[valid].sum() / float(max(1, h_shared.size(0)))

    def _adaptive_logits(self, h_shared: torch.Tensor) -> torch.Tensor:
        h_n = F.normalize(h_shared, dim=-1)
        weight = self.adaptive_classifier.weight
        if self.class_logits_mode == "single" and self.classifier_anchor_blend > 0.0:
            valid = self.class_anchor_inited
            if bool(valid.any()):
                blended = weight.clone()
                anchors = self.class_relation_anchors.to(device=weight.device, dtype=weight.dtype)
                blend = self.classifier_anchor_blend
                blended[valid] = (1.0 - blend) * weight[valid] + blend * anchors[valid]
                weight = blended
        w_n = F.normalize(weight, dim=-1)
        return F.linear(h_n, w_n) * self.adaptive_head_scale

    def _target_head_logits(self, h_shared: torch.Tensor) -> torch.Tensor:
        h_n = F.normalize(h_shared, dim=-1)
        w_n = F.normalize(self.target_classifier.weight, dim=-1)
        return F.linear(h_n, w_n) * self.adaptive_head_scale

    def _target_prototype_logits(self, h_shared: torch.Tensor) -> torch.Tensor:
        target_n = F.normalize(self.target_prototypes, dim=-1)
        h_n = F.normalize(h_shared, dim=-1)
        logits = F.linear(h_n, target_n) * self.adaptive_head_scale
        return logits

    @staticmethod
    def _branch_log_probs(
        logits: torch.Tensor,
        valid_classes: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        masked_logits = logits.masked_fill(~valid_classes.unsqueeze(0), -1e4)
        return F.log_softmax(masked_logits / max(1e-6, float(temperature)), dim=1)

    def _linear_fuse_class_logits(
        self,
        proto_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        target_logits: torch.Tensor,
        *,
        use_target_logits: bool,
    ) -> torch.Tensor:
        class_logits = proto_logits
        if self.adaptive_logit_weight > 0.0:
            adaptive_weight = min(1.0, max(0.0, self.adaptive_logit_weight))
            class_logits = (1.0 - adaptive_weight) * class_logits + adaptive_weight * adaptive_logits
        if use_target_logits and self.target_logit_weight > 0.0:
            target_weight = min(1.0, max(0.0, self.target_logit_weight))
            valid_weight = self.target_proto_inited.to(dtype=class_logits.dtype).unsqueeze(0) * target_weight
            class_logits = (1.0 - valid_weight) * class_logits + valid_weight * target_logits
        return class_logits

    def _logprob_fuse_class_logits(
        self,
        proto_logits: torch.Tensor,
        adaptive_logits: torch.Tensor,
        target_logits: torch.Tensor,
        valid_classes: torch.Tensor,
        *,
        use_target_logits: bool,
    ) -> torch.Tensor:
        adaptive_weight = min(1.0, max(0.0, self.adaptive_logit_weight))
        target_weight = min(1.0, max(0.0, self.target_logit_weight)) if use_target_logits else 0.0
        if self.proto_logit_weight is None:
            proto_weight = max(0.0, 1.0 - adaptive_weight - target_weight)
        else:
            proto_weight = self.proto_logit_weight

        if proto_weight <= 0.0 and adaptive_weight <= 0.0 and target_weight <= 0.0:
            proto_weight = 1.0

        proto_lp = self._branch_log_probs(proto_logits, valid_classes, self.proto_logit_temperature)
        fused = proto_lp.new_zeros(proto_lp.shape)
        effective_weight = proto_lp.new_full((1, proto_lp.size(1)), 1e-8)

        if proto_weight > 0.0:
            fused = fused + proto_weight * proto_lp
            effective_weight = effective_weight + proto_weight * valid_classes.to(dtype=proto_lp.dtype).unsqueeze(0)

        if adaptive_weight > 0.0:
            adaptive_lp = self._branch_log_probs(
                adaptive_logits,
                valid_classes,
                self.adaptive_logit_temperature,
            )
            fused = fused + adaptive_weight * adaptive_lp
            effective_weight = effective_weight + adaptive_weight * valid_classes.to(dtype=proto_lp.dtype).unsqueeze(0)

        target_valid = valid_classes & self.target_proto_inited if target_weight > 0.0 else valid_classes.new_zeros(valid_classes.shape)
        if target_weight > 0.0 and bool(target_valid.any()):
            target_lp = self._branch_log_probs(
                target_logits,
                target_valid,
                self.target_logit_temperature,
            )
            target_mask = target_valid.to(dtype=proto_lp.dtype).unsqueeze(0)
            fused = fused + target_weight * target_mask * target_lp
            effective_weight = effective_weight + target_weight * target_mask

        fused = fused / effective_weight.clamp_min(1e-6)
        return fused.masked_fill(~valid_classes.unsqueeze(0), -1e4)

    def forward_relation_logits(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        use_target_center: bool = False,
        use_target_logits: bool = True,
        classification_role: str = "target",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_shared = self._encode_shared(x=x, h_shared=h_shared)
        target_center = self._target_sample_center(h_shared) if use_target_center else None
        relations = self.relation_router.parse(
            h_shared,
            domain_ids=domain_ids,
            sample_center=target_center,
        )
        h_relation = relations["h_relation"]
        proto_class_logits = relations["class_logits_rel"].masked_fill(
            ~relations["valid_classes"].unsqueeze(0), -1e4
        )
        adaptive_logits = self._adaptive_logits(h_relation)
        target_head_logits = self._target_head_logits(h_relation)
        if self.classification_structure == "decoupled":
            role = str(classification_role).strip().lower()
            if role == "source":
                class_logits = adaptive_logits
            elif role == "target":
                class_logits = (
                    target_head_logits
                    if self.target_classifier_mode == "head"
                    else proto_class_logits
                )
            else:
                raise ValueError("classification_role must be 'source' or 'target'")
            target_logits = target_head_logits
        elif self.class_logits_mode == "single":
            target_logits = torch.zeros_like(adaptive_logits)
            class_logits = adaptive_logits
        else:
            target_logits = self._target_prototype_logits(h_relation)
            if self.logit_fusion == "logprob":
                class_logits = self._logprob_fuse_class_logits(
                    proto_class_logits,
                    adaptive_logits,
                    target_logits,
                    relations["valid_classes"],
                    use_target_logits=use_target_logits,
                )
            else:
                class_logits = self._linear_fuse_class_logits(
                    proto_class_logits,
                    adaptive_logits,
                    target_logits,
                    use_target_logits=use_target_logits,
                )

        class_logits = class_logits.masked_fill(~relations["valid_classes"].unsqueeze(0), -1e4)
        class_probs = torch.softmax(class_logits, dim=1)
        domain_weights = relations["domain_weights"]
        node_mass = class_probs.unsqueeze(-1) * domain_weights

        aux = {
            "h_shared": h_shared,
            "h_relation": h_relation,
            "class_logits": class_logits,
            "proto_class_logits": proto_class_logits,
            "adaptive_class_logits": adaptive_logits,
            "source_class_logits": adaptive_logits,
            "target_head_logits": target_head_logits,
            "target_class_logits": target_logits,
            "class_probs": class_probs,
            "domain_weights": domain_weights,
            "node_mass": node_mass,
            "valid_classes": relations["valid_classes"],
            "target_proto_inited": self.target_proto_inited,
        }
        return class_logits, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_relation_logits(x=x)
        return logits


@register_solver("dcpr_alt")
class DCPRAltSolver(BaseSolver):
    def _resolve_epoch_steps(self) -> int:
        return max(1, len(self.source_loader), len(self.target_loader))

    def build_model(self):
        m = self.config.method
        backbone_name = m.get("backbone", "resnet50")
        sources = list(getattr(self.config.dataset, "sources", []) or [])
        if len(sources) == 0:
            raise ValueError("dcpr_alt requires config.dataset.sources to be a non-empty list")

        self.num_source_domains = len(sources)

        for key, cast, default in [
            ("bottleneck_dim", int, 256),
            ("consistency_conf_power", float, 2.0),
            ("consistency_start_epoch", int, 5),
            ("grad_clip", float, 5.0),
            ("save_ckpt_after_epoch", int, 15),
            ("ema_decay_start", float, 0.996),
            ("ema_decay_end", float, 0.9995),
            ("label_smoothing", float, 0.05),
        ]:
            setattr(self, key, cast(m.get(key, default)))

        self.lambda_relation_consistency = float(m.get("lambda_relation_consistency", 0.40))
        self.lambda_target_pseudo_ce = float(m.get("lambda_target_pseudo_ce", 0.0))
        self.lambda_target_im = float(m.get("lambda_target_im", 0.0))
        self.lambda_source_proto_ce = float(m.get("lambda_source_proto_ce", 0.0))
        self.lambda_source_adaptive_ce = float(m.get("lambda_source_adaptive_ce", 0.0))
        self.lambda_classifier_anchor = float(m.get("lambda_classifier_anchor", 0.0))
        self.lambda_ambiguity_margin = float(m.get("lambda_ambiguity_margin", 0.0))
        self.pseudo_threshold = float(m.get("pseudo_threshold", 0.80))
        self.pseudo_start_epoch = int(m.get("pseudo_start_epoch", self.consistency_start_epoch))
        self.pseudo_conf_power = float(m.get("pseudo_conf_power", 1.0))
        self.target_im_start_epoch = int(m.get("target_im_start_epoch", self.consistency_start_epoch))
        self.ambiguity_start_epoch = int(m.get("ambiguity_start_epoch", self.consistency_start_epoch))
        self.ambiguity_margin = float(m.get("ambiguity_margin", 1.0))
        self.ambiguity_min_conf = float(m.get("ambiguity_min_conf", 0.45))
        self.ambiguity_threshold = float(m.get("ambiguity_threshold", 0.30))
        self.ambiguity_relation_boost = float(m.get("ambiguity_relation_boost", 0.0))
        self.ambiguity_power = float(m.get("ambiguity_power", 1.0))
        self.relation_space_mode = str(m.get("relation_space_mode", "standard")).strip().lower()
        if self.relation_space_mode not in {"standard", "domain_relative"}:
            raise ValueError("method.relation_space_mode must be 'standard' or 'domain_relative'")
        self.target_center_momentum = float(m.get("target_center_momentum", 0.98))
        self.target_prototype_momentum = float(m.get("target_prototype_momentum", 0.98))
        self.target_prototype_threshold = float(m.get("target_prototype_threshold", 0.70))
        self.target_prototype_update = str(m.get("target_prototype_update", "hard")).strip().lower()
        if self.target_prototype_update not in {"hard", "soft_relation"}:
            raise ValueError("method.target_prototype_update must be 'hard' or 'soft_relation'")
        self.target_soft_temperature = float(m.get("target_soft_temperature", 0.70))
        self.target_soft_entropy_power = float(m.get("target_soft_entropy_power", 1.0))
        self.target_soft_balance = self._is_truthy(m.get("target_soft_balance", True))
        self.target_soft_ambiguity_smooth = float(m.get("target_soft_ambiguity_smooth", 0.30))
        self.target_prototype_start_epoch = int(
            m.get("target_prototype_start_epoch", self.pseudo_start_epoch)
        )
        decision_mode = str(m.get("decision_mode", "prototype")).strip().lower()
        if decision_mode == "prototype":
            adaptive_weight_default = 0.0
            target_weight_default = 0.0
        elif decision_mode == "adaptive":
            adaptive_weight_default = 1.0
            target_weight_default = 0.0
        elif decision_mode == "target":
            adaptive_weight_default = 0.0
            target_weight_default = 1.0
        elif decision_mode in {"hybrid", "adaptive_target", "target_adaptive"}:
            adaptive_weight_default = 0.35
            target_weight_default = 0.35
        else:
            raise ValueError(
                "dcpr_alt method.decision_mode must be one of "
                "prototype, adaptive, target, hybrid/adaptive_target"
            )
        self.decision_mode = decision_mode
        self.adaptive_logit_weight = float(m.get("adaptive_logit_weight", adaptive_weight_default))
        self.target_logit_weight = float(m.get("target_logit_weight", target_weight_default))
        self.adaptive_head_scale = float(m.get("adaptive_head_scale", 10.0))
        self.class_logits_mode = str(m.get("class_logits_mode", "single")).strip().lower()
        if self.class_logits_mode not in {"single", "fusion"}:
            raise ValueError("method.class_logits_mode must be 'single' or 'fusion'")
        self.classifier_init_from_anchors = self._is_truthy(m.get("classifier_init_from_anchors", True))
        self.classifier_anchor_blend = float(m.get("classifier_anchor_blend", 0.0))
        self.feature_norm_mode = str(m.get("feature_norm_mode", "layernorm")).strip().lower()
        if self.feature_norm_mode not in {"layernorm", "l2", "identity"}:
            raise ValueError("method.feature_norm_mode must be 'layernorm', 'l2', or 'identity'")
        self.classification_structure = str(m.get("classification_structure", "legacy")).strip().lower()
        if self.classification_structure not in {"legacy", "decoupled"}:
            raise ValueError("method.classification_structure must be 'legacy' or 'decoupled'")
        self.target_classifier_mode = str(m.get("target_classifier_mode", "head")).strip().lower()
        if self.target_classifier_mode not in {"head", "prototype"}:
            raise ValueError("method.target_classifier_mode must be 'head' or 'prototype'")
        self.target_head_init_from_source = self._is_truthy(m.get("target_head_init_from_source", True))
        self.target_head_warmup_sync = self._is_truthy(m.get("target_head_warmup_sync", True))
        self.logit_fusion = str(m.get("logit_fusion", "linear")).strip().lower()
        if self.logit_fusion not in {"linear", "logprob"}:
            raise ValueError("method.logit_fusion must be 'linear' or 'logprob'")
        proto_logit_weight = m.get("proto_logit_weight", None)
        self.proto_logit_weight = None if proto_logit_weight is None else float(proto_logit_weight)
        self.proto_logit_temperature = float(m.get("proto_logit_temperature", 1.0))
        self.adaptive_logit_temperature = float(m.get("adaptive_logit_temperature", 1.0))
        self.target_logit_temperature = float(m.get("target_logit_temperature", 1.0))
        self.update_target_prototypes = self.target_logit_weight > 0.0

        self.total_epochs = int(m.get("epochs", 20))
        self.ramp_denom = float(m.get("ramp_denom", 16.0))
        prototype_batch_size = m.get("prototype_batch_size", None)
        self.prototype_batch_size = (
            int(prototype_batch_size)
            if prototype_batch_size is not None
            else int(self.config.batch_size)
        )
        self.prototype_batch_size = max(1, self.prototype_batch_size)
        self.prototype_prefetch_factor = max(1, int(m.get("prototype_prefetch_factor", 4)))
        self.prototype_persistent_workers = self._is_truthy(
            m.get("prototype_persistent_workers", True)
        )
        self.prototype_cuda_prefetch = self._resolve_auto_bool(
            m.get("prototype_cuda_prefetch", "auto"),
            auto_value=(self.device.type == "cuda"),
        )

        self.temperature_start = float(m.get("temperature_start", 0.15))
        self.temperature_end = float(m.get("temperature_end", 0.10))
        self.relation_temperature = self.temperature_start
        self.refresh_source_prototypes_each_epoch = True

        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        perf_cfg = self._cfg_get(self.config, "performance", {})
        prefetch_cfg = self._cfg_get(perf_cfg, "cuda_batch_prefetch", "auto")
        if isinstance(prefetch_cfg, str):
            lowered = prefetch_cfg.strip().lower()
            if lowered == "auto":
                self.cuda_batch_prefetch = self.device.type == "cuda"
            else:
                self.cuda_batch_prefetch = lowered in {"1", "true", "yes", "on"}
        else:
            self.cuda_batch_prefetch = bool(prefetch_cfg)

        self.net = DCPRNetwork(
            backbone_name=backbone_name,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            bottleneck_dim=self.bottleneck_dim,
            relation_temperature=self.relation_temperature,
            adaptive_head_scale=self.adaptive_head_scale,
            adaptive_logit_weight=self.adaptive_logit_weight,
            target_logit_weight=self.target_logit_weight,
            relation_space_mode=self.relation_space_mode,
            class_logits_mode=self.class_logits_mode,
            classifier_anchor_blend=self.classifier_anchor_blend,
            logit_fusion=self.logit_fusion,
            proto_logit_weight=self.proto_logit_weight,
            proto_logit_temperature=self.proto_logit_temperature,
            adaptive_logit_temperature=self.adaptive_logit_temperature,
            target_logit_temperature=self.target_logit_temperature,
            feature_norm_mode=self.feature_norm_mode,
            classification_structure=self.classification_structure,
            target_classifier_mode=self.target_classifier_mode,
        ).to(self.device)

        self.ema_net = copy.deepcopy(self.net)
        for param in self.ema_net.parameters():
            param.requires_grad_(False)

        self._forward_logits_student = self.net.forward_relation_logits
        self._student_forward_compiled = False
        self.class_ambiguity_weights = torch.zeros(self.num_classes, device=self.device)
        self._src_prefetch_stream = None
        self._tgt_prefetch_stream = None
        if self.cuda_batch_prefetch and self.device.type == "cuda":
            self._src_prefetch_stream = torch.cuda.Stream()
            self._tgt_prefetch_stream = torch.cuda.Stream()
        self._target_view_builder = TargetViewBuilder(
            config=self.config,
            device=self.device,
            to_device=self._to_device,
            logger=logger,
            display_name="DCPR-ALT",
        )

        logger.info(
            "DCPR-ALT: bottleneck=%d rel_space=%s temp=%.2f->%.2f "
            "rel_space_dim=%d norm=%s structure=%s target_cls=%s "
            "logits=%s decision=%s fusion=%s proto_w=%s adaptive_w=%.2f target_w=%.2f "
            "lambda_rel=%.2f lambda_pseudo=%.2f "
            "pseudo_thr=%.2f pseudo_start=%d lambda_im=%.2f im_start=%d "
            "target_proto=%s init_anchor=%s anchor_blend=%.2f "
            "lambda_src_proto=%.2f lambda_src_adapt=%.2f lambda_anchor=%.2f "
            "lambda_amb=%.2f amb_boost=%.2f "
            "proto=full_eval proto_bs=%d proto_prefetch=%s "
            "ramp_start=%d ramp_denom=%.1f prefetch=%s",
            self.bottleneck_dim,
            self.relation_space_mode,
            self.temperature_start,
            self.temperature_end,
            self.net.relation_feat_dim,
            self.feature_norm_mode,
            self.classification_structure,
            self.target_classifier_mode,
            self.class_logits_mode,
            self.decision_mode,
            self.logit_fusion,
            "auto" if self.proto_logit_weight is None else f"{self.proto_logit_weight:.2f}",
            self.adaptive_logit_weight,
            self.target_logit_weight,
            self.lambda_relation_consistency,
            self.lambda_target_pseudo_ce,
            self.pseudo_threshold,
            self.pseudo_start_epoch,
            self.lambda_target_im,
            self.target_im_start_epoch,
            self.target_prototype_update,
            str(self.classifier_init_from_anchors),
            self.classifier_anchor_blend,
            self.lambda_source_proto_ce,
            self.lambda_source_adaptive_ce,
            self.lambda_classifier_anchor,
            self.lambda_ambiguity_margin,
            self.ambiguity_relation_boost,
            self.prototype_batch_size,
            str(self.prototype_cuda_prefetch),
            self.consistency_start_epoch,
            self.ramp_denom,
            str(self.cuda_batch_prefetch),
        )

    def _uses_target_loader_in_training(self) -> bool:
        return (
            self.lambda_relation_consistency > 0.0
            or self.lambda_target_pseudo_ce > 0.0
            or self.lambda_target_im > 0.0
            or self.update_target_prototypes
            or self.lambda_ambiguity_margin > 0.0
            or self.ambiguity_relation_boost > 0.0
        )

    def _temperature_at_epoch(self, epoch_number: int) -> float:
        if self.temperature_start == self.temperature_end:
            return self.temperature_end
        if epoch_number <= self.consistency_start_epoch:
            return self.temperature_start
        anneal_span = max(1.0, self.ramp_denom - float(self.consistency_start_epoch))
        progress = min(1.0, (float(epoch_number) - float(self.consistency_start_epoch)) / anneal_span)
        return self.temperature_start + (self.temperature_end - self.temperature_start) * progress

    def _set_relation_temperature(self, value: float):
        self.relation_temperature = float(value)
        self.net.set_relation_temperature(value)
        self.ema_net.set_relation_temperature(value)

    def _forward_logits(
        self,
        model: DCPRNetwork,
        *,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        use_target_center: bool = False,
        use_target_logits: bool = True,
        classification_role: str = "target",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if model is self.net:
            return self._forward_logits_student(
                x,
                h_shared,
                domain_ids,
                use_target_center,
                use_target_logits,
                classification_role,
            )
        return model.forward_relation_logits(
            x=x,
            h_shared=h_shared,
            domain_ids=domain_ids,
            use_target_center=use_target_center,
            use_target_logits=use_target_logits,
            classification_role=classification_role,
        )

    def _setup_compiled_student_forward(self):
        if self._student_forward_compiled:
            return
        self._forward_logits_student = self._compile_callable(
            self.net.forward_relation_logits,
            "dcpr_alt_student.forward_relation_logits",
        )
        self._student_forward_compiled = True

    def _prepare_target_views(self, tgt_imgs):
        return self._target_view_builder.prepare(tgt_imgs)

    def _load_source_batch_to_device(self, src_batch):
        src_imgs, src_labels, src_dom = src_batch
        return self._to_device(src_imgs), self._to_device(src_labels), self._to_device(src_dom)

    def _load_target_batch_to_views(self, tgt_batch):
        tgt_imgs = tgt_batch[0] if isinstance(tgt_batch, (tuple, list)) else tgt_batch
        return self._prepare_target_views(tgt_imgs)

    # NOTE: Under AMP, LayerNorm and log_softmax/logsumexp internally promote to
    # float32 for numerical stability even when autocast wraps the region. The
    # backbone runs in BF16/FP16 (tensor cores), but the relation head operates in
    # FP32. This is expected PyTorch behaviour and is not a correctness issue.

    @torch.no_grad()
    def _update_ema(self, decay: float):
        update_ema_model(self.ema_net, self.net, decay)

    def _ema_decay_at(self, step: int, total_steps: int) -> float:
        return linear_ema_decay(
            step,
            total_steps,
            self.ema_decay_start,
            self.ema_decay_end,
        )

    @staticmethod
    def _normalize_distribution(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        return x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _domain_class_relation(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Target relation over class and source-domain prototype nodes."""
        return self._normalize_distribution(aux["node_mass"].flatten(1))

    def _class_responsibility_from_relation(
        self,
        aux: Dict[str, torch.Tensor],
        *,
        detach: bool = False,
    ) -> torch.Tensor:
        """Collapse domain-class relation to class responsibility."""
        node_mass = aux["node_mass"]
        if detach:
            node_mass = node_mass.detach()
        return self._normalize_distribution(node_mass.sum(dim=2))

    def _relation_consistency_loss(
        self,
        student_aux: Dict[str, torch.Tensor],
        teacher_aux: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Match teacher/student target domain-class prototype relations."""
        student_relation = self._domain_class_relation(student_aux)
        teacher_relation = self._domain_class_relation(teacher_aux)
        return soft_prob_cross_entropy(student_relation, teacher_relation, weights=weights)

    def _soft_target_responsibilities(
        self,
        teacher_aux: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        resp = self._class_responsibility_from_relation(teacher_aux, detach=True)
        temperature = max(1e-6, self.target_soft_temperature)
        if temperature != 1.0:
            resp = resp.clamp_min(1e-8).pow(1.0 / temperature)
            resp = self._normalize_distribution(resp)

        if self.target_soft_ambiguity_smooth > 0.0:
            amb = self.class_ambiguity_weights.to(device=resp.device, dtype=resp.dtype).unsqueeze(0)
            uniform = torch.full_like(resp, 1.0 / float(max(1, resp.size(1))))
            smooth = (self.target_soft_ambiguity_smooth * amb).clamp(0.0, 0.95)
            resp = (1.0 - smooth) * resp + smooth * uniform
            resp = self._normalize_distribution(resp)

        if self.target_soft_entropy_power > 0.0:
            entropy = -(resp.clamp_min(1e-8) * resp.clamp_min(1e-8).log()).sum(dim=1)
            confidence = 1.0 - entropy / math.log(float(max(2, resp.size(1))))
            resp = resp * confidence.clamp_min(0.0).pow(self.target_soft_entropy_power).unsqueeze(1)

        if self.target_soft_balance:
            class_mass = resp.mean(dim=0, keepdim=True).clamp_min(1e-3)
            resp = resp / class_mass

        return self._normalize_distribution(resp)

    def _pseudo_label_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pseudo_labels = teacher_probs.argmax(dim=1)
        selected = teacher_conf >= self.pseudo_threshold
        selected_rate = selected.float().mean()
        if not bool(selected.any()):
            return torch.zeros((), device=student_logits.device, dtype=torch.float32), selected_rate

        losses = F.cross_entropy(
            student_logits[selected],
            pseudo_labels[selected],
            reduction="none",
        )
        weights = teacher_conf[selected].detach().pow(self.pseudo_conf_power)
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        return loss, selected_rate

    @staticmethod
    def _information_maximization_loss(class_probs: torch.Tensor) -> torch.Tensor:
        probs = class_probs.clamp_min(1e-8)
        sample_entropy = -(probs * probs.log()).sum(dim=1).mean()
        mean_probs = probs.mean(dim=0)
        balance = (mean_probs * mean_probs.clamp_min(1e-8).log()).sum() + math.log(float(probs.size(1)))
        return sample_entropy + balance

    @torch.no_grad()
    def _update_source_class_ambiguity(
        self,
        valid: torch.Tensor,
        prototypes: torch.Tensor,
        domain_centers: Optional[torch.Tensor] = None,
    ):
        valid_class = valid.any(dim=0)
        if self.relation_space_mode == "domain_relative" and domain_centers is not None:
            prototypes = prototypes - domain_centers.unsqueeze(1)
        proto = F.normalize(prototypes, dim=-1)
        valid_f = valid.to(dtype=proto.dtype).unsqueeze(-1)
        class_sum = (proto * valid_f).sum(dim=0)
        class_count = valid_f.sum(dim=0).clamp_min(1.0)
        class_proto = F.normalize(class_sum / class_count, dim=-1)
        sim = class_proto @ class_proto.t()
        sim.fill_diagonal_(-1.0)
        top_sim = sim.max(dim=1).values
        denom = max(1e-6, 1.0 - self.ambiguity_threshold)
        weights = ((top_sim - self.ambiguity_threshold) / denom).clamp(0.0, 1.0)
        weights = torch.where(valid_class, weights, torch.zeros_like(weights))
        self.class_ambiguity_weights.copy_(weights.detach())

    def _compute_class_relation_anchors(
        self,
        valid: torch.Tensor,
        prototypes: torch.Tensor,
        domain_centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_class = valid.any(dim=0)
        relation_proto = prototypes
        if self.relation_space_mode == "domain_relative":
            relation_proto = relation_proto - domain_centers.unsqueeze(1)
        proto = F.normalize(relation_proto, dim=-1)
        valid_f = valid.to(dtype=proto.dtype).unsqueeze(-1)
        class_sum = (proto * valid_f).sum(dim=0)
        class_count = valid_f.sum(dim=0).clamp_min(1.0)
        anchors = F.normalize(class_sum / class_count, dim=-1)
        anchors = torch.where(valid_class.unsqueeze(1), anchors, torch.zeros_like(anchors))
        return valid_class, anchors

    def _ambiguity_relation_weights(
        self,
        teacher_probs: torch.Tensor,
        teacher_conf: torch.Tensor,
    ) -> torch.Tensor:
        if self.ambiguity_relation_boost <= 0.0:
            return torch.ones_like(teacher_conf)
        top2 = teacher_probs.topk(k=min(2, teacher_probs.size(1)), dim=1)
        top1 = top2.indices[:, 0]
        if top2.values.size(1) > 1:
            margin = top2.values[:, 0] - top2.values[:, 1]
        else:
            margin = torch.ones_like(teacher_conf)
        class_weight = self.class_ambiguity_weights[top1].to(device=teacher_conf.device)
        sample_weight = (1.0 - margin.clamp(0.0, 1.0)).pow(self.ambiguity_power)
        sample_weight = sample_weight * class_weight
        return 1.0 + self.ambiguity_relation_boost * sample_weight

    def _ambiguity_margin_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        teacher_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if student_logits.size(1) < 2:
            return torch.zeros((), device=student_logits.device), torch.zeros((), device=student_logits.device)
        top2 = teacher_probs.topk(k=2, dim=1)
        top1 = top2.indices[:, 0]
        runner_up = top2.indices[:, 1]
        class_weight = self.class_ambiguity_weights[top1].to(device=student_logits.device)
        selected = (teacher_conf >= self.ambiguity_min_conf) & (class_weight > 0.0)
        selected_rate = selected.float().mean()
        if not bool(selected.any()):
            return torch.zeros((), device=student_logits.device), selected_rate

        rows = torch.arange(student_logits.size(0), device=student_logits.device)[selected]
        pos = top1[selected]
        neg = runner_up[selected]
        margin = student_logits[rows, pos] - student_logits[rows, neg]
        losses = F.relu(self.ambiguity_margin - margin)
        weights = class_weight[selected].detach()
        loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        return loss, selected_rate

    @torch.no_grad()
    def _teacher_guidance(self, tgt_weak: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.ema_net.eval()
        h_shared = self.ema_net._encode_shared(x=tgt_weak)
        self.net.update_target_center(h_shared, momentum=self.target_center_momentum)
        self.ema_net.update_target_center(h_shared, momentum=self.target_center_momentum)
        _, aux = self._forward_logits(self.ema_net, h_shared=h_shared, use_target_center=True)
        conf = aux["class_probs"].max(dim=1).values.detach()
        guide = {
            "class_probs": aux["class_probs"].detach(),
            "node_mass": aux["node_mass"].detach(),
            "h_relation": aux["h_relation"].detach(),
        }
        return conf, guide

    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        self.ema_net.eval()
        if isinstance(imgs, (tuple, list)):
            imgs = imgs[0]
        with torch.no_grad():
            logits, _ = self._forward_logits(self.ema_net, x=imgs, use_target_center=True)
            return logits

    @torch.no_grad()
    def _sync_relation_buffers_to_ema(self):
        for name in ["src_prototypes", "src_proto_inited", "src_domain_centers", "src_domain_center_inited"]:
            getattr(self.ema_net.relation_router, name).copy_(getattr(self.net.relation_router, name))
        for name in [
            "target_prototypes",
            "target_proto_inited",
            "target_center",
            "target_center_inited",
            "class_relation_anchors",
            "class_anchor_inited",
            "classifier_anchor_initialized",
            "target_classifier_initialized",
        ]:
            getattr(self.ema_net, name).copy_(getattr(self.net, name))

    @torch.no_grad()
    def _sync_classifier_initialization_to_ema(self):
        self.ema_net.adaptive_classifier.weight.copy_(self.net.adaptive_classifier.weight)
        self.ema_net.target_classifier.weight.copy_(self.net.target_classifier.weight)

    @torch.no_grad()
    def _sync_target_head_from_source(self):
        self.net.target_classifier.weight.copy_(self.net.adaptive_classifier.weight)
        self.ema_net.target_classifier.weight.copy_(self.net.adaptive_classifier.weight)
        self.net.target_classifier_initialized.fill_(True)
        self.ema_net.target_classifier_initialized.fill_(True)

    def _build_optimizer(self):
        base_lr = float(self.config.method.lr)
        param_groups = [
            {"params": list(self.net.backbone.parameters()), "lr": base_lr * 0.1},
            {"params": list(self.net.bottleneck.parameters()), "lr": base_lr},
            {"params": list(self.net.feature_norm.parameters()), "lr": base_lr},
            {"params": list(self.net.adaptive_classifier.parameters()), "lr": base_lr},
            {"params": list(self.net.target_classifier.parameters()), "lr": base_lr},
            {"params": list(self.net.relation_router.parameters()), "lr": base_lr},
        ]
        param_groups = [group for group in param_groups if len(group["params"]) > 0]
        return optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

    def _build_scheduler(self, optimizer, total_iters: int):
        scheduler_t_max_epochs = getattr(self.config.method, "scheduler_t_max_epochs", None)
        if scheduler_t_max_epochs is not None:
            epoch_steps = max(1, self._resolve_epoch_steps())
            total_iters = max(1, int(round(float(scheduler_t_max_epochs) * epoch_steps)))
            logger.info("DCPR scheduler horizon override: t_max_epochs=%.2f", float(scheduler_t_max_epochs))

        def lr_lambda(step):
            progress = step / max(1, total_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _create_prefetchers(self, uses_target_loader: bool):
        src_iter = cycle(self.source_loader)
        use_cuda_prefetch = bool(self.cuda_batch_prefetch and self.device.type == "cuda")
        src_prefetcher = CudaBatchPrefetcher(
            src_iter,
            self._load_source_batch_to_device,
            enabled=use_cuda_prefetch,
            stream=self._src_prefetch_stream,
        )
        tgt_prefetcher = None
        if uses_target_loader:
            tgt_iter = cycle(self.target_loader)
            tgt_prefetcher = CudaBatchPrefetcher(
                tgt_iter,
                self._load_target_batch_to_views,
                enabled=use_cuda_prefetch,
                stream=self._tgt_prefetch_stream,
            )
        return src_prefetcher, tgt_prefetcher

    def _source_eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _source_domain_datasets(self):
        dataset = self.source_loader.dataset
        if hasattr(dataset, "datasets"):
            return list(dataset.datasets)
        return [dataset]

    @contextmanager
    def _temporary_source_transform(self, transform):
        datasets = self._source_domain_datasets()
        originals = [getattr(dataset, "transform", None) for dataset in datasets]
        for dataset in datasets:
            if hasattr(dataset, "transform"):
                dataset.transform = transform
        try:
            yield
        finally:
            for dataset, original in zip(datasets, originals):
                if hasattr(dataset, "transform"):
                    dataset.transform = original

    def _prototype_source_loader(self):
        kwargs = {
            "batch_size": self.prototype_batch_size,
            "shuffle": False,
            "drop_last": False,
            "num_workers": int(getattr(self.source_loader, "num_workers", 0)),
            "pin_memory": bool(getattr(self.source_loader, "pin_memory", False)),
        }
        if kwargs["num_workers"] > 0:
            kwargs["persistent_workers"] = self.prototype_persistent_workers
            kwargs["prefetch_factor"] = self.prototype_prefetch_factor
        return DataLoader(self.source_loader.dataset, **kwargs)

    @contextmanager
    def _prototype_source_iter(self):
        with self._temporary_source_transform(self._source_eval_transform()):
            yield self._prototype_source_loader()

    def _iter_prototype_source_batches(self, loader):
        if not (self.prototype_cuda_prefetch and self.device.type == "cuda"):
            yield from loader
            return

        prefetcher = CudaBatchPrefetcher(
            iter(loader),
            self._load_source_batch_to_device,
            enabled=True,
            stream=self._src_prefetch_stream,
        )
        try:
            while True:
                yield prefetcher.pop()
        except StopIteration:
            return
        finally:
            prefetcher.close()

    def _compute_source_prototypes(self, model: DCPRNetwork, prototype_loader):
        """Single-pass source prototype refresh."""
        feat_sums = torch.zeros(
            self.num_source_domains,
            self.num_classes,
            model.relation_feat_dim,
            device=self.device,
        )
        counts = torch.zeros(
            self.num_source_domains,
            self.num_classes,
            device=self.device,
        )
        domain_sums = torch.zeros(
            self.num_source_domains,
            model.relation_feat_dim,
            device=self.device,
        )
        domain_counts = torch.zeros(
            self.num_source_domains,
            device=self.device,
        )
        feat_sums_flat = feat_sums.view(-1, model.relation_feat_dim)
        counts_flat = counts.view(-1)

        for src_imgs, src_labels, src_dom in prototype_loader:
            src_imgs = self._to_device(src_imgs)
            src_labels = self._to_device(src_labels)
            src_dom = self._to_device(src_dom)

            with self._auto_cast():
                h = model.extract_relation_features(src_imgs)
                h_shared = model.normalize_relation_features(h)

            flat_index = src_dom.long() * self.num_classes + src_labels.long()
            # Sort by flat_index so index_add_ writes are coalesced on CUDA.
            order = flat_index.argsort()
            flat_index_sorted = flat_index[order]
            feat_sums_flat.index_add_(0, flat_index_sorted, h_shared[order])
            ones = torch.ones(flat_index_sorted.size(0), dtype=counts_flat.dtype, device=self.device)
            counts_flat.index_add_(0, flat_index_sorted, ones)
            domain_order = src_dom.long().argsort()
            domain_sorted = src_dom.long()[domain_order]
            domain_sums.index_add_(0, domain_sorted, h_shared[domain_order])
            domain_counts.index_add_(
                0,
                domain_sorted,
                torch.ones(domain_sorted.size(0), dtype=domain_counts.dtype, device=self.device),
            )

        valid = counts > 0
        safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
        prototypes = feat_sums / safe_counts
        prototypes = torch.where(valid.unsqueeze(-1), prototypes, torch.zeros_like(prototypes))
        domain_valid = domain_counts > 0
        domain_centers = domain_sums / domain_counts.clamp_min(1.0).unsqueeze(1)
        domain_centers = torch.where(
            domain_valid.unsqueeze(1),
            domain_centers,
            torch.zeros_like(domain_centers),
        )

        return valid, prototypes, domain_valid, domain_centers

    def _recompute_source_prototypes(self, model: DCPRNetwork):
        with torch.inference_mode():
            was_training = model.training
            model.eval()

            with self._prototype_source_iter() as prototype_loader:
                batch_iter = self._iter_prototype_source_batches(prototype_loader)
                started_at = time.time()
                valid, prototypes, domain_valid, domain_centers = self._compute_source_prototypes(model, batch_iter)
                elapsed_minutes = (time.time() - started_at) / 60.0

                model.reset_source_prototypes()
                model.relation_router.src_proto_inited.copy_(valid)
                model.relation_router.src_prototypes.copy_(prototypes)
                model.relation_router.src_domain_center_inited.copy_(domain_valid)
                model.relation_router.src_domain_centers.copy_(domain_centers)
                anchor_valid, class_anchors = self._compute_class_relation_anchors(
                    valid,
                    prototypes,
                    domain_centers,
                )
                model.set_class_relation_anchors(anchor_valid, class_anchors)
                if self.classifier_init_from_anchors:
                    model.maybe_init_classifier_from_anchors()
                if self.target_head_init_from_source:
                    model.maybe_init_target_classifier_from_source()
                if model is self.net:
                    self._update_source_class_ambiguity(valid, prototypes, domain_centers)
                logger.info(
                    "DCPR source prototype refresh | proto_bs=%d single_pass "
                    "elapsed_min=%.2f amb_mean=%.3f amb_max=%.3f",
                    self.prototype_batch_size,
                    elapsed_minutes,
                    float(self.class_ambiguity_weights.mean().item()),
                    float(self.class_ambiguity_weights.max().item()),
                )

            model.train(was_training)

    def _train_step(
        self,
        optimizer,
        scheduler,
        src_batch,
        tgt_batch,
        relation_ramp: float,
        pseudo_ramp: float,
        im_ramp: float,
        target_proto_ramp: float,
        ambiguity_ramp: float,
        ema_decay: float,
    ):
        src_imgs, src_labels, src_dom = src_batch
        tgt_weak, tgt_strong = tgt_batch if tgt_batch is not None else (None, None)

        self._zero_grad(optimizer)

        loss_rel = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_pseudo = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_im = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_src_proto = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_src_adaptive = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_anchor = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_ambiguity = torch.zeros((), device=self.device, dtype=torch.float32)
        conf_tgt = torch.zeros((), device=self.device, dtype=torch.float32)
        pseudo_selected = torch.zeros((), device=self.device, dtype=torch.float32)
        target_proto_selected = torch.zeros((), device=self.device, dtype=torch.float32)
        ambiguity_selected = torch.zeros((), device=self.device, dtype=torch.float32)

        with self._auto_cast():
            logits_src, src_aux = self._forward_logits(
                self.net,
                x=src_imgs,
                domain_ids=src_dom,
                use_target_logits=False,
                classification_role="source",
            )
            self._probe_amp_tensor(logits_src, "dcpr_alt/logits_src", warn_on_float32=False)
            loss_src = self.criterion_task(logits_src, src_labels)
            loss = loss_src
            if self.lambda_source_proto_ce > 0.0:
                loss_src_proto = self.criterion_task(src_aux["proto_class_logits"], src_labels)
                loss = loss + self.lambda_source_proto_ce * loss_src_proto
            if self.lambda_source_adaptive_ce > 0.0:
                loss_src_adaptive = self.criterion_task(src_aux["adaptive_class_logits"], src_labels)
                loss = loss + self.lambda_source_adaptive_ce * loss_src_adaptive
            if self.lambda_classifier_anchor > 0.0:
                loss_anchor = self.net.classifier_anchor_loss()
                loss = loss + self.lambda_classifier_anchor * loss_anchor

            if tgt_weak is not None and tgt_strong is not None:
                with torch.no_grad():
                    with self._auto_cast():
                        conf_tgt, teacher_aux = self._teacher_guidance(tgt_weak)
                    if self.update_target_prototypes and target_proto_ramp > 0.0:
                        if self.target_prototype_update == "soft_relation":
                            soft_resp = self._soft_target_responsibilities(teacher_aux)
                            target_proto_selected = self.net.update_target_prototypes_soft(
                                teacher_aux["h_relation"],
                                soft_resp,
                                momentum=self.target_prototype_momentum,
                            )
                        else:
                            target_proto_selected = self.net.update_target_prototypes(
                                teacher_aux["h_relation"],
                                teacher_aux["class_probs"],
                                conf_tgt,
                                threshold=self.target_prototype_threshold,
                                momentum=self.target_prototype_momentum,
                            )

                logits_tgt, tgt_aux = self._forward_logits(self.net, x=tgt_strong, use_target_center=True)

                if self.lambda_relation_consistency > 0.0:
                    rel_weights = conf_tgt.pow(self.consistency_conf_power)
                    if ambiguity_ramp > 0.0 and self.ambiguity_relation_boost > 0.0:
                        amb_weights = self._ambiguity_relation_weights(
                            teacher_aux["class_probs"],
                            conf_tgt,
                        )
                        rel_weights = rel_weights * (1.0 + ambiguity_ramp * (amb_weights - 1.0))
                    loss_rel = self._relation_consistency_loss(
                        tgt_aux,
                        teacher_aux,
                        rel_weights,
                    )
                    loss = loss + relation_ramp * self.lambda_relation_consistency * loss_rel

                if self.lambda_target_pseudo_ce > 0.0 and pseudo_ramp > 0.0:
                    loss_pseudo, pseudo_selected = self._pseudo_label_loss(
                        logits_tgt,
                        teacher_aux["class_probs"],
                        conf_tgt,
                    )
                    loss = loss + pseudo_ramp * self.lambda_target_pseudo_ce * loss_pseudo

                if self.lambda_target_im > 0.0 and im_ramp > 0.0:
                    loss_im = self._information_maximization_loss(tgt_aux["class_probs"])
                    loss = loss + im_ramp * self.lambda_target_im * loss_im

                if self.lambda_ambiguity_margin > 0.0 and ambiguity_ramp > 0.0:
                    loss_ambiguity, ambiguity_selected = self._ambiguity_margin_loss(
                        logits_tgt,
                        teacher_aux["class_probs"],
                        conf_tgt,
                    )
                    loss = loss + ambiguity_ramp * self.lambda_ambiguity_margin * loss_ambiguity

        self._optimizer_step_with_optional_clip(
            loss,
            optimizer,
            clip_params=self.net.parameters(),
            clip_max_norm=self.grad_clip,
        )
        scheduler.step()
        self._update_ema(ema_decay)

        metrics = {
            "src": loss_src.detach().float(),
            "srcp": loss_src_proto.detach().float(),
            "srca": loss_src_adaptive.detach().float(),
            "anch": loss_anchor.detach().float(),
            "rel": loss_rel.detach().float(),
            "pseudo": loss_pseudo.detach().float(),
            "psel": pseudo_selected.detach().float(),
            "im": loss_im.detach().float(),
            "amb": loss_ambiguity.detach().float(),
            "asel": ambiguity_selected.detach().float(),
            "tpsel": target_proto_selected.detach().float(),
            "conf": conf_tgt.detach().mean().float() if conf_tgt.ndim > 0 else conf_tgt.detach().float(),
            "total": loss.detach().float(),
        }
        return metrics

    def _run_train_epoch(
        self,
        optimizer,
        scheduler,
        epoch_steps: int,
        uses_target_loader: bool,
        relation_ramp: float,
        pseudo_ramp: float,
        im_ramp: float,
        target_proto_ramp: float,
        ambiguity_ramp: float,
        global_step: int,
    ):
        metric_keys = ("src", "srcp", "srca", "anch", "rel", "pseudo", "psel", "im", "amb", "asel", "tpsel", "conf", "total")
        metric_sums = {
            key: torch.zeros((), device=self.device, dtype=torch.float32)
            for key in metric_keys
        }

        src_prefetcher, tgt_prefetcher = self._create_prefetchers(uses_target_loader)

        for _ in range(epoch_steps):
            src_batch = src_prefetcher.pop()
            tgt_batch = tgt_prefetcher.pop() if tgt_prefetcher is not None else None
            step_metrics = self._train_step(
                optimizer=optimizer,
                scheduler=scheduler,
                src_batch=src_batch,
                tgt_batch=tgt_batch,
                relation_ramp=relation_ramp,
                pseudo_ramp=pseudo_ramp,
                im_ramp=im_ramp,
                target_proto_ramp=target_proto_ramp,
                ambiguity_ramp=ambiguity_ramp,
                ema_decay=self._ema_decay_at(global_step, self._total_iters),
            )
            global_step += 1
            for key, value in step_metrics.items():
                metric_sums[key].add_(value)

        src_prefetcher.close()
        if tgt_prefetcher is not None:
            tgt_prefetcher.close()

        scale = 1.0 / float(max(1, epoch_steps))
        metrics = {key: (value * scale).item() for key, value in metric_sums.items()}
        return metrics, global_step

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

        student_load = self.net.load_state_dict(student_state, strict=False)
        ema_load = self.ema_net.load_state_dict(ema_state, strict=False)
        missing = set(student_load.missing_keys) | set(ema_load.missing_keys)
        if "relation_router.relation_temperature" in missing:
            self._set_relation_temperature(self.temperature_end)
        else:
            self.relation_temperature = float(self.ema_net.relation_router.relation_temperature.item())
        logger.info("%s checkpoint loaded from %s", self._solver_display_name(), path)

    def train(self):
        self._save_start_epoch = int(self.save_ckpt_after_epoch)
        optimizer = self._build_optimizer()
        epoch_steps = self._resolve_epoch_steps()
        total_iters = self.total_epochs * epoch_steps
        self._total_iters = total_iters
        scheduler = self._build_scheduler(optimizer, total_iters)
        self.register_training_state(optimizer=optimizer, scheduler=scheduler)
        self._setup_compiled_student_forward()
        best_acc = self._best_metric
        uses_target_loader = self._uses_target_loader_in_training()

        global_step = self._training_global_step
        logger.info(
            "DCPR-ALT Training: epoch_steps=max source_steps=%d target_steps=%d "
            "epoch_steps=%d use_target=%s rel=domain_class lambda_rel=%.2f",
            len(self.source_loader),
            len(self.target_loader),
            epoch_steps,
            str(uses_target_loader),
            self.lambda_relation_consistency,
        )

        for epoch in self._epoch_range(self.total_epochs):
            current_temperature = self._temperature_at_epoch(epoch + 1)
            self._set_relation_temperature(current_temperature)

            if self.refresh_source_prototypes_each_epoch or epoch == 0:
                self._recompute_source_prototypes(self.net)
                self._sync_relation_buffers_to_ema()
                if epoch == 0:
                    self._sync_classifier_initialization_to_ema()
            if (
                self.classification_structure == "decoupled"
                and self.target_classifier_mode == "head"
                and self.target_head_warmup_sync
                and (epoch + 1) <= self.consistency_start_epoch
            ):
                self._sync_target_head_from_source()

            self.net.train()
            ramp = min(1.0, (epoch + 1) / max(1.0, self.ramp_denom))
            consistency_ramp = 1.0 if (epoch + 1) >= self.consistency_start_epoch else 0.0
            pseudo_ramp = ramp if (epoch + 1) >= self.pseudo_start_epoch else 0.0
            im_ramp = ramp if (epoch + 1) >= self.target_im_start_epoch else 0.0
            target_proto_ramp = ramp if (epoch + 1) >= self.target_prototype_start_epoch else 0.0
            ambiguity_ramp = ramp if (epoch + 1) >= self.ambiguity_start_epoch else 0.0
            metrics, global_step = self._run_train_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_steps=epoch_steps,
                uses_target_loader=uses_target_loader,
                relation_ramp=ramp * consistency_ramp,
                pseudo_ramp=pseudo_ramp,
                im_ramp=im_ramp,
                target_proto_ramp=target_proto_ramp,
                ambiguity_ramp=ambiguity_ramp,
                global_step=global_step,
            )

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            self._maybe_save_best(acc, epoch + 1)

            self._log_epoch_summary(
                epoch + 1,
                self.total_epochs,
                metrics={
                    "src": metrics["src"],
                    "srcp": metrics["srcp"],
                    "srca": metrics["srca"],
                    "anch": metrics["anch"],
                    "rel": metrics["rel"],
                    "pseudo": metrics["pseudo"],
                    "psel": (metrics["psel"], ".2f"),
                    "im": metrics["im"],
                    "amb": metrics["amb"],
                    "asel": (metrics["asel"], ".2f"),
                    "tpsel": (metrics["tpsel"], ".2f"),
                    "conf": (metrics["conf"], ".3f"),
                    "total": metrics["total"],
                },
                extras={
                    "rmp": (ramp, ".2f"),
                    "crmp": (consistency_ramp, ".2f"),
                    "prmp": (pseudo_ramp, ".2f"),
                    "irm": (im_ramp, ".2f"),
                    "trmp": (target_proto_ramp, ".2f"),
                    "armp": (ambiguity_ramp, ".2f"),
                    "tmp": (current_temperature, ".3f"),
                },
                score=acc,
                best_score=best_acc,
                score_name="Acc",
            )

        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")
