"""
RGR: Relational Graph Representation for MSDA.

Core pipeline:
Feature Extractor -> Relation Graph Builder -> Relation Parser
-> Relation-conditioned Representation Generator -> Classifier.

Key idea:
- Domain information participates in representation construction.
- Cross-domain relations participate in target recognition.
"""

import copy
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import AverageMeter, cycle

logger = logging.getLogger(__name__)


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


def _unwrap_weak_strong_from_maybe_tuple(tgt_imgs):
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) >= 2:
        return tgt_imgs[0], tgt_imgs[1]
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) == 1:
        return tgt_imgs[0], tgt_imgs[0]
    return tgt_imgs, tgt_imgs


class RelationGraphBuilder(nn.Module):
    """Build multi-source class relation graph from source prototypes."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_source_domains: int,
        relation_temperature: float = 0.10,
        boundary_temperature: float = 0.15,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.num_source_domains = int(num_source_domains)
        self.relation_temperature = max(1e-6, float(relation_temperature))
        self.boundary_temperature = max(1e-6, float(boundary_temperature))

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

        pairs = list(combinations(range(self.num_source_domains), 2))
        if pairs:
            pair_tensor = torch.tensor(pairs, dtype=torch.long)
        else:
            pair_tensor = torch.empty(0, 2, dtype=torch.long)
        self.register_buffer("domain_pairs", pair_tensor, persistent=False)

    @property
    def num_domain_pairs(self) -> int:
        return int(self.domain_pairs.size(0))

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.src_prototypes.zero_()
        self.src_proto_inited.zero_()

    def _boundary_node_messages(self) -> torch.Tensor:
        """Domain-internal inter-class boundary messages for each node."""
        proto = self.src_prototypes
        mask = self.src_proto_inited
        d, c, _ = proto.shape

        if c <= 1:
            return torch.zeros_like(proto)

        p_n = F.normalize(proto, dim=-1)
        sim = torch.einsum("dcf,dkf->dck", p_n, p_n) / self.boundary_temperature

        valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(c, device=proto.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = valid_pair & (~eye)

        safe_logits = sim.masked_fill(~valid_pair, -1e4)
        weights = torch.softmax(safe_logits, dim=-1)
        valid_row = valid_pair.any(dim=-1)
        weights = torch.where(valid_row.unsqueeze(-1), weights, torch.zeros_like(weights))

        diffs = proto.unsqueeze(2) - proto.unsqueeze(1)
        return torch.einsum("dck,dckf->dcf", weights, diffs)

    def _transition_context(
        self,
        domain_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build same-class cross-domain transition context.

        Returns:
            transition_per_class: [B, C, F]
            transition_mass: [B, C, E] where E = num domain pairs
        """
        bsz = domain_weights.size(0)
        c = self.num_classes
        e = self.num_domain_pairs
        device = domain_weights.device
        dtype = domain_weights.dtype

        if e == 0:
            transition_per_class = torch.zeros(
                bsz, c, self.feat_dim, device=device, dtype=dtype
            )
            transition_mass = torch.zeros(bsz, c, 0, device=device, dtype=dtype)
            return transition_per_class, transition_mass

        pair_d1 = self.domain_pairs[:, 0]
        pair_d2 = self.domain_pairs[:, 1]

        proto = self.src_prototypes
        mask = self.src_proto_inited

        edge_feat = 0.5 * (proto[pair_d1] + proto[pair_d2])  # [E, C, F]
        edge_feat = edge_feat.permute(1, 0, 2).contiguous()  # [C, E, F]

        edge_valid = (mask[pair_d1] & mask[pair_d2]).transpose(0, 1).contiguous()  # [C, E]

        wd1 = domain_weights[:, :, pair_d1].clamp_min(1e-8)
        wd2 = domain_weights[:, :, pair_d2].clamp_min(1e-8)
        pair_logits = torch.log(wd1) + torch.log(wd2)  # [B, C, E]
        pair_logits = pair_logits.masked_fill(~edge_valid.unsqueeze(0), -1e4)

        pair_weights = torch.softmax(pair_logits, dim=-1)
        valid_row = edge_valid.any(dim=-1)
        pair_weights = torch.where(
            valid_row.unsqueeze(0).unsqueeze(-1),
            pair_weights,
            torch.zeros_like(pair_weights),
        )

        transition_per_class = torch.einsum("bce,cef->bcf", pair_weights, edge_feat)
        transition_mass = pair_weights
        return transition_per_class, transition_mass

    def parse(self, h_shared: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse sample-to-graph relations."""
        proto = self.src_prototypes
        mask = self.src_proto_inited

        proto_n = F.normalize(proto, dim=-1)
        h_n = F.normalize(h_shared, dim=-1)

        node_logits_bdc = torch.einsum("bf,dcf->bdc", h_n, proto_n)
        node_logits_bdc = node_logits_bdc / self.relation_temperature
        node_logits_bdc = node_logits_bdc.masked_fill(~mask.unsqueeze(0), -1e4)

        # Class-level evidence from cross-domain nodes.
        class_logits_rel = torch.logsumexp(node_logits_bdc, dim=1)  # [B, C]
        valid_classes = mask.any(dim=0)  # [C]
        class_logits_rel = class_logits_rel.masked_fill(~valid_classes.unsqueeze(0), -1e4)

        # For each class, parse which source-domain version is closer.
        domain_logits = node_logits_bdc.permute(0, 2, 1).contiguous()  # [B, C, D]
        domain_mask = mask.transpose(0, 1).unsqueeze(0)  # [1, C, D]
        domain_logits = domain_logits.masked_fill(~domain_mask, -1e4)
        domain_weights = torch.softmax(domain_logits, dim=-1)
        domain_weights = torch.where(
            valid_classes.unsqueeze(0).unsqueeze(-1),
            domain_weights,
            torch.full_like(domain_weights, 1.0 / float(max(1, self.num_source_domains))),
        )

        node_context = torch.einsum("bcd,dcf->bcf", domain_weights, proto)
        node_context = torch.where(
            valid_classes.unsqueeze(0).unsqueeze(-1),
            node_context,
            torch.zeros_like(node_context),
        )

        boundary_node = self._boundary_node_messages()  # [D, C, F]
        transition_per_class, transition_mass = self._transition_context(domain_weights)

        return {
            "class_logits_rel": class_logits_rel,
            "domain_logits": domain_logits,
            "domain_weights": domain_weights,
            "node_context": node_context,
            "boundary_node": boundary_node,
            "transition_per_class": transition_per_class,
            "transition_mass": transition_mass,
            "valid_classes": valid_classes,
        }


class RelationParser(nn.Module):
    """Fuse class prior and relation evidence into sample class relation state."""

    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.class_prior = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        h_shared: torch.Tensor,
        class_logits_rel: torch.Tensor,
        valid_classes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prior_logits = self.class_prior(h_shared)
        alpha = torch.sigmoid(self.mix_logit)
        class_logits = alpha * prior_logits + (1.0 - alpha) * class_logits_rel
        class_logits = class_logits.masked_fill(~valid_classes.unsqueeze(0), -1e4)
        class_probs = torch.softmax(class_logits, dim=1)
        return {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "class_prior_logits": prior_logits,
            "class_rel_logits": class_logits_rel,
            "mix_alpha": alpha,
        }


class RelationConditionedRepresentationGenerator(nn.Module):
    """Generate transferable semantic representation conditioned on relations."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.class_embed = nn.Linear(num_classes, feat_dim, bias=False)
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.out_norm = nn.LayerNorm(feat_dim)

    def forward(
        self,
        h_shared: torch.Tensor,
        class_context: torch.Tensor,
        transition_context: torch.Tensor,
        boundary_context: torch.Tensor,
        class_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        class_hint = self.class_embed(class_probs)
        cond = torch.cat(
            [h_shared, class_context, transition_context, boundary_context, class_hint],
            dim=-1,
        )
        delta = self.net(cond)
        z = self.out_norm(h_shared + delta)
        return z, delta


class RGRNetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_source_domains: int,
        *,
        bottleneck_dim: int = 0,
        relation_hidden_dim: int = 256,
        generator_hidden_dim: int = 512,
        relation_temperature: float = 0.10,
        boundary_temperature: float = 0.15,
        generator_dropout: float = 0.2,
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

        self.feature_norm = nn.LayerNorm(self.feat_dim)
        self.graph_builder = RelationGraphBuilder(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            relation_temperature=relation_temperature,
            boundary_temperature=boundary_temperature,
        )
        self.relation_parser = RelationParser(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes,
            hidden_dim=relation_hidden_dim,
        )
        self.representation_generator = RelationConditionedRepresentationGenerator(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes,
            hidden_dim=generator_hidden_dim,
            dropout=generator_dropout,
        )
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.backbone(x))

    def normalize_features(self, h: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(h)

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.graph_builder.reset_source_prototypes()

    def forward_relation_logits_from_shared(
        self,
        h_shared: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        graph = self.graph_builder.parse(h_shared)
        parsed = self.relation_parser(
            h_shared=h_shared,
            class_logits_rel=graph["class_logits_rel"],
            valid_classes=graph["valid_classes"],
        )
        class_probs = parsed["class_probs"]
        domain_weights = graph["domain_weights"]
        node_mass = class_probs.unsqueeze(-1) * domain_weights  # [B, C, D]

        class_context = torch.einsum("bc,bcf->bf", class_probs, graph["node_context"])
        transition_context = torch.einsum("bc,bcf->bf", class_probs, graph["transition_per_class"])
        boundary_context = torch.einsum("bcd,dcf->bf", node_mass, graph["boundary_node"])

        z, delta = self.representation_generator(
            h_shared=h_shared,
            class_context=class_context,
            transition_context=transition_context,
            boundary_context=boundary_context,
            class_probs=class_probs,
        )
        logits = self.classifier(z)
        cls_probs_from_logits = torch.softmax(logits, dim=1)

        transition_mass = graph["transition_mass"] * class_probs.unsqueeze(-1)
        if transition_mass.numel() > 0:
            transition_global = transition_mass.flatten(1)
            transition_global = transition_global / transition_global.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        else:
            transition_global = transition_mass.new_zeros(transition_mass.size(0), 0)

        aux = {
            "h_shared": h_shared,
            "z": z,
            "delta": delta,
            "class_logits": parsed["class_logits"],
            "class_probs": class_probs,
            "class_prior_logits": parsed["class_prior_logits"],
            "class_rel_logits": parsed["class_rel_logits"],
            "mix_alpha": parsed["mix_alpha"],
            "domain_logits": graph["domain_logits"],
            "domain_weights": domain_weights,
            "node_context": graph["node_context"],
            "node_mass": node_mass,
            "class_context": class_context,
            "transition_per_class": graph["transition_per_class"],
            "transition_mass": graph["transition_mass"],
            "transition_global": transition_global,
            "boundary_node": graph["boundary_node"],
            "boundary_context": boundary_context,
            "valid_classes": graph["valid_classes"],
            "cls_probs_from_logits": cls_probs_from_logits,
        }
        return logits, aux

    def forward_relation_logits(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h_shared is None:
            if x is None:
                raise ValueError("Either x or h_shared must be provided.")
            h = self.extract_features(x)
            h_shared = self.normalize_features(h)
        return self.forward_relation_logits_from_shared(h_shared)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_relation_logits(x=x)
        return logits


@register_solver("rgr")
class RGRSolver(BaseSolver):
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
            raise ValueError("rgr requires config.dataset.sources to be a non-empty list")

        self.num_source_domains = len(list(sources))

        self.bottleneck_dim = int(m.get("bottleneck_dim", 256))
        self.relation_hidden_dim = int(m.get("relation_hidden_dim", 256))
        self.generator_hidden_dim = int(m.get("generator_hidden_dim", 512))
        self.relation_temperature = float(m.get("relation_temperature", 0.10))
        self.boundary_temperature = float(m.get("boundary_temperature", 0.15))
        self.generator_dropout = float(m.get("generator_dropout", 0.2))

        self.lambda_source_relation = float(m.get("lambda_source_relation", 0.20))
        self.lambda_relation_consistency = float(m.get("lambda_relation_consistency", 0.40))
        self.lambda_local_consistency = float(m.get("lambda_local_consistency", 0.15))
        self.lambda_explain_consistency = float(m.get("lambda_explain_consistency", 0.10))

        self.consistency_conf_power = float(m.get("consistency_conf_power", 2.0))
        self.consistency_start_epoch = int(m.get("consistency_start_epoch", 4))
        self.local_knn = int(m.get("local_knn", 5))
        self.refresh_source_prototypes_each_epoch = bool(
            m.get("refresh_source_prototypes_each_epoch", True)
        )

        self.total_epochs = int(m.get("epochs", 20))
        self.ramp_denom = float(m.get("ramp_denom", max(1.0, self.total_epochs * 0.3)))
        self.grad_clip = float(m.get("grad_clip", 5.0))
        self.save_ckpt_after_epoch = int(m.get("save_ckpt_after_epoch", 0))
        self.epoch_steps_mode = str(m.get("epoch_steps_mode", "max")).strip().lower()
        self.ema_decay_start = float(m.get("ema_decay_start", 0.996))
        self.ema_decay_end = float(m.get("ema_decay_end", 0.9995))
        self.relation_label_smoothing = float(m.get("relation_label_smoothing", 0.10))

        self.label_smoothing = float(m.get("label_smoothing", 0.05))
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.net = RGRNetwork(
            backbone_name=backbone_name,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            bottleneck_dim=self.bottleneck_dim,
            relation_hidden_dim=self.relation_hidden_dim,
            generator_hidden_dim=self.generator_hidden_dim,
            relation_temperature=self.relation_temperature,
            boundary_temperature=self.boundary_temperature,
            generator_dropout=self.generator_dropout,
        ).to(self.device)

        self.ema_net = copy.deepcopy(self.net)
        for param in self.ema_net.parameters():
            param.requires_grad_(False)

        logger.info(
            "RGR: bottleneck=%d rel_hidden=%d gen_hidden=%d rel_temp=%.3f "
            "boundary_temp=%.3f lambda_rel=%.3f lambda_rel_cons=%.3f "
            "lambda_local=%.3f lambda_explain=%.3f consistency_start=%d knn=%d",
            self.bottleneck_dim,
            self.relation_hidden_dim,
            self.generator_hidden_dim,
            self.relation_temperature,
            self.boundary_temperature,
            self.lambda_source_relation,
            self.lambda_relation_consistency,
            self.lambda_local_consistency,
            self.lambda_explain_consistency,
            self.consistency_start_epoch,
            self.local_knn,
        )

    def _forward_logits(
        self,
        model: RGRNetwork,
        *,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h_shared is None:
            if x is None:
                raise ValueError("Either x or h_shared must be provided.")
            h = model.extract_features(x)
            h_shared = model.normalize_features(h)
        return model.forward_relation_logits_from_shared(h_shared=h_shared)

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
        domain_logits: torch.Tensor,
        src_labels: torch.Tensor,
        src_dom: torch.Tensor,
    ) -> torch.Tensor:
        # domain_logits: [B, C, D], supervise selected class on source domain id.
        batch_idx = torch.arange(src_labels.size(0), device=src_labels.device)
        true_class_domain_logits = domain_logits[batch_idx, src_labels]
        num_domains = true_class_domain_logits.size(1)
        if num_domains <= 1:
            return torch.zeros((), device=true_class_domain_logits.device, dtype=true_class_domain_logits.dtype)

        off_value = self.relation_label_smoothing / float(num_domains - 1)
        target = torch.full_like(true_class_domain_logits, off_value)
        target.scatter_(1, src_dom.unsqueeze(1), 1.0 - self.relation_label_smoothing)
        return soft_target_cross_entropy(true_class_domain_logits, target)

    @staticmethod
    def _normalize_distribution(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        return x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _position_distribution(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        class_part = aux["class_probs"]  # [B, C]
        node_part = aux["node_mass"].flatten(1)  # [B, C*D], sums to ~1
        node_part = self._normalize_distribution(node_part)

        trans_part = aux["transition_global"]  # [B, C*E] or [B, 0]
        if trans_part.size(1) > 0:
            trans_part = self._normalize_distribution(trans_part)
            return torch.cat([class_part, node_part, trans_part], dim=1)
        return torch.cat([class_part, node_part], dim=1)

    def _relation_consistency_loss(
        self,
        student_aux: Dict[str, torch.Tensor],
        teacher_aux: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = soft_prob_cross_entropy(
            student_aux["class_probs"],
            teacher_aux["class_probs"],
            weights=weights,
        )

        node_s = self._normalize_distribution(student_aux["node_mass"].flatten(1))
        node_t = self._normalize_distribution(teacher_aux["node_mass"].flatten(1))
        node_loss = soft_prob_cross_entropy(node_s, node_t, weights=weights)

        if student_aux["transition_global"].size(1) > 0:
            trans_s = self._normalize_distribution(student_aux["transition_global"])
            trans_t = self._normalize_distribution(teacher_aux["transition_global"])
            trans_loss = soft_prob_cross_entropy(trans_s, trans_t, weights=weights)
        else:
            trans_loss = torch.zeros((), device=self.device, dtype=cls_loss.dtype)

        return cls_loss, node_loss, trans_loss

    def _local_structure_loss(
        self,
        student_aux: Dict[str, torch.Tensor],
        teacher_aux: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        pos_s = self._position_distribution(student_aux)  # [B, P]
        pos_t = self._position_distribution(teacher_aux).detach()  # [B, P]
        z_t = teacher_aux["z"].detach()

        bsz = pos_s.size(0)
        if bsz <= 1:
            return torch.zeros((), device=self.device, dtype=pos_s.dtype)

        k = min(max(1, self.local_knn), bsz - 1)
        z_t_n = F.normalize(z_t, dim=1)
        sim = z_t_n @ z_t_n.t()
        sim.fill_diagonal_(-1e4)
        topv, topi = sim.topk(k, dim=1)
        neigh_w = torch.softmax(topv, dim=1)

        neigh_pos = pos_t[topi]  # [B, k, P]
        target_pos = torch.einsum("bk,bkp->bp", neigh_w, neigh_pos).detach()

        losses = ((pos_s - target_pos) ** 2).sum(dim=1)
        weights = weights.detach()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-6)

    @torch.no_grad()
    def _teacher_guidance(self, tgt_weak: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.ema_net.eval()
        _, aux = self._forward_logits(self.ema_net, x=tgt_weak)
        conf = aux["class_probs"].max(dim=1).values.detach()
        guide = {
            "class_probs": aux["class_probs"].detach(),
            "node_mass": aux["node_mass"].detach(),
            "transition_global": aux["transition_global"].detach(),
            "z": aux["z"].detach(),
        }
        return conf, guide

    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        self.ema_net.eval()
        if isinstance(imgs, (tuple, list)):
            imgs = imgs[0]
        with torch.no_grad():
            logits, _ = self._forward_logits(self.ema_net, x=imgs)
            return logits

    @torch.no_grad()
    def _recompute_source_prototypes(self, model: RGRNetwork):
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

        for src_imgs, src_labels, src_dom in self.source_loader:
            src_imgs = src_imgs.to(self.device)
            src_labels = src_labels.to(self.device)
            src_dom = src_dom.to(self.device)

            h = model.extract_features(src_imgs)
            h_shared = model.normalize_features(h)

            for dom_id in src_dom.unique().tolist():
                dom_mask = src_dom == dom_id
                feats_dom = h_shared[dom_mask]
                labels_dom = src_labels[dom_mask]
                for cls_id in labels_dom.unique().tolist():
                    cls_mask = labels_dom == cls_id
                    feat_sums[dom_id, cls_id] += feats_dom[cls_mask].sum(dim=0)
                    counts[dom_id, cls_id] += float(cls_mask.sum().item())

        model.reset_source_prototypes()
        valid = counts > 0
        model.graph_builder.src_proto_inited.copy_(valid)
        for dom_id in range(self.num_source_domains):
            for cls_id in range(self.num_classes):
                if bool(valid[dom_id, cls_id].item()):
                    model.graph_builder.src_prototypes[dom_id, cls_id].copy_(
                        feat_sums[dom_id, cls_id] / counts[dom_id, cls_id]
                    )

        model.train(was_training)

    def save_checkpoint(self, path):
        torch.save(
            {
                "method": "rgr",
                "student": self.net.state_dict(),
                "ema": self.ema_net.state_dict(),
            },
            path,
        )
        logger.info(f"RGR checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        student_state = checkpoint["student"] if isinstance(checkpoint, dict) and "student" in checkpoint else checkpoint
        ema_state = checkpoint["ema"] if isinstance(checkpoint, dict) and "ema" in checkpoint else student_state

        self.net.load_state_dict(student_state, strict=False)
        self.ema_net.load_state_dict(ema_state, strict=False)
        logger.info(f"RGR checkpoint loaded from {path}")

    def train(self):
        base_lr = float(self.config.method.lr)
        param_groups = [
            {"params": list(self.net.backbone.parameters()), "lr": base_lr * 0.1},
            {"params": list(self.net.bottleneck.parameters()), "lr": base_lr},
            {"params": list(self.net.feature_norm.parameters()), "lr": base_lr},
            {"params": list(self.net.graph_builder.parameters()), "lr": base_lr},
            {"params": list(self.net.relation_parser.parameters()), "lr": base_lr},
            {"params": list(self.net.representation_generator.parameters()), "lr": base_lr},
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
        best_acc = 0.0
        best_save_acc = -1e18
        best_path = Path("checkpoints") / "best_rgr.pth"
        best_path.parent.mkdir(parents=True, exist_ok=True)

        global_step = 0
        logger.info(
            "RGR Training: extractor->graph->parser->generator->classifier | "
            "epoch_steps_mode=%s source_steps=%d target_steps=%d epoch_steps=%d",
            self.epoch_steps_mode,
            len(self.source_loader),
            len(self.target_loader),
            epoch_steps,
        )

        for epoch in range(self.total_epochs):
            if self.refresh_source_prototypes_each_epoch or epoch == 0:
                self._recompute_source_prototypes(self.net)
                self.ema_net.graph_builder.src_prototypes.copy_(
                    self.net.graph_builder.src_prototypes
                )
                self.ema_net.graph_builder.src_proto_inited.copy_(
                    self.net.graph_builder.src_proto_inited
                )

            self.net.train()
            meters = {
                key: AverageMeter()
                for key in [
                    "src",
                    "srel",
                    "rcls",
                    "rnode",
                    "rtrans",
                    "local",
                    "expl",
                    "conf",
                    "mix",
                    "gdelta",
                    "total",
                ]
            }

            src_iter = cycle(self.source_loader)
            tgt_iter = cycle(self.target_loader)
            ramp = min(1.0, (epoch + 1) / max(1.0, self.ramp_denom))
            consistency_ramp = 1.0 if (epoch + 1) >= self.consistency_start_epoch else 0.0

            for _ in range(epoch_steps):
                src_imgs, src_labels, src_dom = next(src_iter)
                tgt_batch = next(tgt_iter)
                tgt_imgs = tgt_batch[0] if isinstance(tgt_batch, (tuple, list)) else tgt_batch
                tgt_weak, tgt_strong = _unwrap_weak_strong_from_maybe_tuple(tgt_imgs)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                src_dom = src_dom.to(self.device)
                tgt_weak = tgt_weak.to(self.device)
                tgt_strong = tgt_strong.to(self.device)

                optimizer.zero_grad()

                logits_src, src_aux = self._forward_logits(self.net, x=src_imgs)
                loss_src = self.criterion_task(logits_src, src_labels)
                loss_src_rel = self._source_relation_loss(
                    src_aux["domain_logits"],
                    src_labels,
                    src_dom,
                )

                logits_tgt, tgt_aux = self._forward_logits(self.net, x=tgt_strong)
                with torch.no_grad():
                    conf_tgt, teacher_aux = self._teacher_guidance(tgt_weak)
                    rel_weights = conf_tgt.pow(self.consistency_conf_power)

                loss_rcls, loss_rnode, loss_rtrans = self._relation_consistency_loss(
                    tgt_aux,
                    teacher_aux,
                    rel_weights,
                )
                loss_local = self._local_structure_loss(
                    tgt_aux,
                    teacher_aux,
                    rel_weights,
                )
                loss_explain = soft_target_cross_entropy(
                    logits_tgt,
                    tgt_aux["class_probs"].detach(),
                    weights=rel_weights,
                )

                loss = (
                    loss_src
                    + self.lambda_source_relation * loss_src_rel
                    + ramp
                    * consistency_ramp
                    * (
                        self.lambda_relation_consistency * (loss_rcls + loss_rnode + loss_rtrans)
                        + self.lambda_local_consistency * loss_local
                        + self.lambda_explain_consistency * loss_explain
                    )
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                scheduler.step()

                self._update_ema(self._ema_decay_at(global_step, total_iters))
                global_step += 1

                meters["src"].update(loss_src.item())
                meters["srel"].update(loss_src_rel.item())
                meters["rcls"].update(loss_rcls.item())
                meters["rnode"].update(loss_rnode.item())
                meters["rtrans"].update(loss_rtrans.item())
                meters["local"].update(loss_local.item())
                meters["expl"].update(loss_explain.item())
                meters["conf"].update(conf_tgt.mean().item())
                meters["mix"].update(float(tgt_aux["mix_alpha"].item()))
                meters["gdelta"].update(tgt_aux["delta"].abs().mean().item())
                meters["total"].update(loss.item())

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            if epoch + 1 > self.save_ckpt_after_epoch and acc > best_save_acc:
                best_save_acc = acc
                self.save_checkpoint(best_path)

            logger.info(
                f"RGR {epoch+1}/{self.total_epochs} | "
                f"src={meters['src'].avg:.4f} "
                f"srel={meters['srel'].avg:.4f} "
                f"rcls={meters['rcls'].avg:.4f} "
                f"rnode={meters['rnode'].avg:.4f} "
                f"rtrans={meters['rtrans'].avg:.4f} "
                f"local={meters['local'].avg:.4f} "
                f"expl={meters['expl'].avg:.4f} "
                f"conf={meters['conf'].avg:.3f} "
                f"mix={meters['mix'].avg:.3f} "
                f"gdelta={meters['gdelta'].avg:.4f} "
                f"total={meters['total'].avg:.4f} | "
                f"rmp={ramp:.2f} crmp={consistency_ramp:.2f} | "
                f"Acc={acc:.2f}% (best={best_acc:.2f}%)"
            )

        if best_path.exists():
            self.load_checkpoint(best_path)
            logger.info(f"Loaded best RGR checkpoint from {best_path} with Acc={best_save_acc:.2f}%")
