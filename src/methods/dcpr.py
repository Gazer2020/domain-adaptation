"""
DCPR: Domain-Class Prototype Relation for Multi-Source Domain Adaptation.

Mainline from the implementation:
- Refresh source-domain class prototypes from the full source set each epoch.
- Classify by similarity to those prototypes, aggregating same-class evidence
  across source domains.
- Use softmax to route each predicted class through the closest source-domain
  prototypes.
- Build a per-domain, off-diagonal class-confusion profile from source
  classifier probabilities.
- Adapt target data with EMA teacher consistency on strong/weak views, matching
  both prototype-node mass and routed confusion profiles.
- Anneal the relation temperature after consistency starts so early routing is
  smoother and late logits are sharper.
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


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax over `dim` with the same shape as softmax outputs."""
    if logits.numel() == 0:
        return logits

    z = logits.transpose(dim, -1)
    orig_shape = z.shape
    z = z.reshape(-1, orig_shape[-1])

    z_sorted, _ = torch.sort(z, dim=-1, descending=True)
    z_cumsum = z_sorted.cumsum(dim=-1)
    ks = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype).unsqueeze(0)
    support = 1 + ks * z_sorted > z_cumsum
    support_size = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(-1, support_size - 1) - 1.0) / support_size.to(z.dtype)
    out = (z - tau).clamp_min(0.0)
    out = out.reshape(orig_shape).transpose(dim, -1)
    return out


def _resolve_ambiguity_reciprocal_mode(value) -> str:
    if isinstance(value, bool):
        return "geometric" if value else "none"
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on", "sqrt", "geometric", "geom"}:
        return "geometric"
    if lowered in {"0", "false", "no", "off", "none", ""}:
        return "none"
    if lowered in {"mean", "avg", "average", "arithmetic", "sym_mean"}:
        return "mean"
    if lowered in {"harmonic", "hmean", "sym_harmonic"}:
        return "harmonic"
    raise ValueError(
        f"Unsupported DCPR ambiguity_reciprocal={value}. "
        "Expected none, geometric, mean, or harmonic."
    )


class PrototypeRelationRouter(nn.Module):
    """Parse prototype-routed class, domain, and confusion relations."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_source_domains: int,
        relation_temperature: float = 0.10,
        routing_mode: str = "softmax",
        routing_scope: str = "class",
        ambiguity_source: str = "domain_class",
        ambiguity_reciprocal="harmonic",
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.num_source_domains = int(num_source_domains)
        self.confusion_topk = 4
        self.routing_mode = str(routing_mode).lower()
        self.routing_scope = str(routing_scope).lower()
        self.ambiguity_source = str(ambiguity_source).lower()
        self.ambiguity_reciprocal = _resolve_ambiguity_reciprocal_mode(ambiguity_reciprocal)
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
            "src_classifier_confusion",
            torch.zeros(self.num_source_domains, self.num_classes, self.num_classes),
            persistent=True,
        )

    @torch.no_grad()
    def set_relation_temperature(self, value: float):
        self.relation_temperature.fill_(max(1e-6, float(value)))

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.src_prototypes.zero_()
        self.src_proto_inited.zero_()
        self.src_classifier_confusion.zero_()

    def _apply_routing(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_logits = logits.masked_fill(~valid_mask, -1e4)
        if self.routing_mode == "softmax":
            weights = torch.softmax(safe_logits, dim=-1)
        elif self.routing_mode == "uniform":
            weights = valid_mask.float()
        elif self.routing_mode in {"hard", "hardmax", "argmax"}:
            best = safe_logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(-1, best, 1.0)
        elif self.routing_mode == "sparsemax":
            weights = sparsemax(safe_logits, dim=-1)
        else:
            raise ValueError(
                f"Unsupported DCPR routing_mode={self.routing_mode}. "
                "Expected sparsemax, softmax, uniform, or hardmax."
            )
        weights = weights * valid_mask.float()
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _classifier_confusion_weights(self) -> torch.Tensor:
        scores = self.src_classifier_confusion
        mask = self.src_proto_inited
        d, c, _ = scores.shape

        if c <= 1:
            return torch.zeros(d, c, c, device=scores.device, dtype=scores.dtype)

        valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(c, device=scores.device, dtype=torch.bool).unsqueeze(0)
        keep_mask = valid_pair & (~eye)

        if self.ambiguity_source == "uniform":
            weights = torch.where(keep_mask, torch.ones_like(scores), torch.zeros_like(scores))
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.ambiguity_source == "global":
            valid_rows = mask.float().sum(dim=0).clamp_min(1.0).view(c, 1)
            global_scores = (scores * mask.unsqueeze(-1).float()).sum(dim=0) / valid_rows
            scores = global_scores.unsqueeze(0).expand(d, c, c)
        elif self.ambiguity_source in {"domain_class", "random"}:
            pass
        else:
            raise ValueError(
                f"Unsupported DCPR ambiguity_source={self.ambiguity_source}. "
                "Expected domain_class, global, uniform, or random."
            )

        if self.ambiguity_reciprocal != "none":
            reverse_scores = scores.transpose(1, 2)
            scores = scores.clamp_min(0.0)
            reverse_scores = reverse_scores.clamp_min(0.0)
            if self.ambiguity_reciprocal == "geometric":
                scores = (scores * reverse_scores).sqrt()
            elif self.ambiguity_reciprocal == "mean":
                scores = 0.5 * (scores + reverse_scores)
            elif self.ambiguity_reciprocal == "harmonic":
                scores = (2.0 * scores * reverse_scores) / (scores + reverse_scores).clamp_min(1e-8)
            else:
                raise ValueError(
                    f"Unsupported DCPR ambiguity_reciprocal={self.ambiguity_reciprocal}. "
                    "Expected none, geometric, mean, or harmonic."
                )

        if self.confusion_topk > 0 and self.confusion_topk < (c - 1):
            topk = min(self.confusion_topk, c - 1)
            safe_scores = scores.masked_fill(~keep_mask, -1e4)
            topi = safe_scores.topk(topk, dim=-1).indices
            topk_mask = torch.zeros_like(keep_mask)
            topk_mask.scatter_(-1, topi, True)
            keep_mask = keep_mask & topk_mask

        weights = torch.where(keep_mask, scores, torch.zeros_like(scores))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        valid_row = keep_mask.any(dim=-1)
        return torch.where(valid_row.unsqueeze(-1), weights, torch.zeros_like(weights))

    def parse(self, h_relation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return class logits plus sparse source-domain routing statistics."""
        proto = self.src_prototypes
        mask = self.src_proto_inited

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

        # For each candidate class, route through the source domains with
        # active prototype support.
        domain_logits = node_logits_bdc.permute(0, 2, 1).contiguous()
        domain_mask = mask.transpose(0, 1).unsqueeze(0)
        if self.routing_scope == "instance":
            domain_valid = mask.any(dim=1).unsqueeze(0)
            instance_domain_logits = torch.logsumexp(node_logits_bdc, dim=2)
            instance_weights = self._apply_routing(instance_domain_logits, domain_valid)
            domain_weights = instance_weights.unsqueeze(1).expand(-1, self.num_classes, -1)
            domain_weights = domain_weights * domain_mask.float()
            domain_weights = domain_weights / domain_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.routing_scope == "class":
            domain_logits = domain_logits.masked_fill(~domain_mask, -1e4)
            domain_weights = self._apply_routing(domain_logits, domain_mask)
        elif self.routing_scope in {"none", "off", "shared"}:
            domain_weights = domain_mask.float().expand(domain_logits.size(0), -1, -1)
            domain_weights = domain_weights / domain_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            raise ValueError(
                f"Unsupported DCPR routing_scope={self.routing_scope}. "
                "Expected class, instance, or none."
            )
        domain_weights = torch.where(
            valid_classes.unsqueeze(0).unsqueeze(-1),
            domain_weights,
            torch.full_like(domain_weights, 1.0 / float(max(1, self.num_source_domains))),
        )

        # The routed source-domain mixture selects a class-confusion profile
        # conditioned on where this sample is closest in prototype space.
        confusion_dck = self._classifier_confusion_weights()
        confusion_per_class = torch.einsum("bcd,dck->bck", domain_weights, confusion_dck)
        confusion_per_class = torch.where(
            valid_classes.unsqueeze(0).unsqueeze(-1),
            confusion_per_class,
            torch.zeros_like(confusion_per_class),
        )

        return {
            "class_logits_rel": class_logits_rel,
            "domain_logits": domain_logits,
            "domain_weights": domain_weights,
            "confusion_per_class": confusion_per_class,
            "valid_classes": valid_classes,
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
        routing_mode: str = "softmax",
        routing_scope: str = "class",
        ambiguity_source: str = "domain_class",
        ambiguity_reciprocal="harmonic",
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
        self.relation_feat_dim = self.feat_dim

        self.relation_router = PrototypeRelationRouter(
            feat_dim=self.relation_feat_dim,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            relation_temperature=relation_temperature,
            routing_mode=routing_mode,
            routing_scope=routing_scope,
            ambiguity_source=ambiguity_source,
            ambiguity_reciprocal=ambiguity_reciprocal,
        )

    def extract_relation_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.backbone(x))

    def normalize_relation_features(self, h: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(h)

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

    def forward_relation_logits(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_shared = self._encode_shared(x=x, h_shared=h_shared)
        relations = self.relation_router.parse(h_shared)
        class_logits = relations["class_logits_rel"].masked_fill(
            ~relations["valid_classes"].unsqueeze(0), -1e4
        )
        class_probs = torch.softmax(class_logits, dim=1)
        domain_weights = relations["domain_weights"]
        node_mass = class_probs.unsqueeze(-1) * domain_weights

        confusion_per_class = relations["confusion_per_class"]
        confusion_profile = torch.einsum("bc,bck->bk", class_probs, confusion_per_class)
        confusion_profile = confusion_profile / confusion_profile.sum(dim=1, keepdim=True).clamp_min(1e-8)

        aux = {
            "h_relation": h_shared,
            "class_logits": class_logits,
            "class_probs": class_probs,
            "domain_logits": relations["domain_logits"],
            "domain_weights": domain_weights,
            "node_mass": node_mass,
            "confusion_per_class": confusion_per_class,
            "confusion_profile": confusion_profile,
            "valid_classes": relations["valid_classes"],
        }
        return class_logits, aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_relation_logits(x=x)
        return logits


@register_solver("dcpr")
class DCPRSolver(BaseSolver):
    def _resolve_epoch_steps(self) -> int:
        return max(1, len(self.source_loader), len(self.target_loader))

    def build_model(self):
        m = self.config.method
        backbone_name = m.get("backbone", "resnet50")
        sources = list(getattr(self.config.dataset, "sources", []) or [])
        if len(sources) == 0:
            raise ValueError("dcpr requires config.dataset.sources to be a non-empty list")

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

        self.prototype_granularity = str(m.get("prototype_granularity", "domain_class")).lower()
        self.routing_mode = str(m.get("routing_mode", "softmax")).lower()
        self.routing_scope = str(m.get("routing_scope", "class")).lower()
        self.support_consistency_target = str(
            m.get("support_consistency_target", "node_mass")
        ).lower()
        self.ambiguity_source = str(m.get("ambiguity_source", "domain_class")).lower()
        self.ambiguity_reciprocal = _resolve_ambiguity_reciprocal_mode(
            m.get("ambiguity_reciprocal", "harmonic")
        )
        self.use_ambiguity_consistency = self._is_truthy(m.get("ambiguity_consistency", True))
        self.lambda_relation_consistency = float(m.get("lambda_relation_consistency", 0.40))
        self.lambda_rel_consistency_node = float(m.get("lambda_rel_consistency_node", 0.25))
        self.lambda_rel_consistency_conf = float(m.get("lambda_rel_consistency_conf", 1.0))

        if self.prototype_granularity in {"class", "shared", "shared_class"}:
            if self.routing_scope == "class":
                logger.info(
                    "DCPR shared-class prototype graph uses routing_scope=none "
                    "instead of class-wise source routing."
                )
                self.routing_scope = "none"
            if self.ambiguity_source == "domain_class":
                logger.info(
                    "DCPR shared-class prototype graph uses ambiguity_source=global "
                    "because there are no domain-class nodes."
                )
                self.ambiguity_source = "global"

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
        self.confusion_topk = 4
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
            routing_mode=self.routing_mode,
            routing_scope=self.routing_scope,
            ambiguity_source=self.ambiguity_source,
            ambiguity_reciprocal=self.ambiguity_reciprocal,
        ).to(self.device)

        self.ema_net = copy.deepcopy(self.net)
        for param in self.ema_net.parameters():
            param.requires_grad_(False)

        self._forward_logits_student = self.net.forward_relation_logits
        self._student_forward_compiled = False
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
            display_name="DCPR",
        )

        logger.info(
            "DCPR mainline: bottleneck=%d temp=%.2f->%.2f conf_topk=%d "
            "routing=%s/%s proto=%s amb_src=%s amb_recip=%s support=%s amb_cons=%s "
            "rel_space_dim=%d lambda_rel_cons=%.2f "
            "rel_w(node/conf)=%.2f/%.2f proto=full_eval proto_bs=%d proto_prefetch=%s "
            "ramp_start=%d ramp_denom=%.1f prefetch=%s",
            self.bottleneck_dim,
            self.temperature_start,
            self.temperature_end,
            self.confusion_topk,
            self.routing_mode,
            self.routing_scope,
            self.prototype_granularity,
            self.ambiguity_source,
            str(self.ambiguity_reciprocal),
            self.support_consistency_target,
            str(self.use_ambiguity_consistency),
            self.net.relation_feat_dim,
            self.lambda_relation_consistency,
            self.lambda_rel_consistency_node,
            self.lambda_rel_consistency_conf,
            self.prototype_batch_size,
            str(self.prototype_cuda_prefetch),
            self.consistency_start_epoch,
            self.ramp_denom,
            str(self.cuda_batch_prefetch),
        )

    def _uses_target_loader_in_training(self) -> bool:
        uses_support = self.support_consistency_target not in {"", "none", "off", "false"}
        uses_ambiguity = bool(self.use_ambiguity_consistency)
        return self.lambda_relation_consistency > 0.0 and (uses_support or uses_ambiguity)

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if model is self.net:
            return self._forward_logits_student(x, h_shared)
        return model.forward_relation_logits(x=x, h_shared=h_shared)

    def _setup_compiled_student_forward(self):
        if self._student_forward_compiled:
            return
        self._forward_logits_student = self._compile_callable(
            self.net.forward_relation_logits,
            "dcpr_student.forward_relation_logits",
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

    def _relation_consistency_loss(
        self,
        student_aux: Dict[str, torch.Tensor],
        teacher_aux: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match teacher/student target relations, not just class marginals."""
        support_losses = []
        if self.support_consistency_target in {"node", "node_mass", "m", "prototype", "support"}:
            student_node = self._normalize_distribution(student_aux["node_mass"].flatten(1))
            teacher_node = self._normalize_distribution(teacher_aux["node_mass"].flatten(1))
            support_losses.append(soft_prob_cross_entropy(student_node, teacher_node, weights=weights))
        elif self.support_consistency_target in {"class", "class_probs", "prob", "probs", "pi"}:
            support_losses.append(
                soft_prob_cross_entropy(
                    self._normalize_distribution(student_aux["class_probs"]),
                    self._normalize_distribution(teacher_aux["class_probs"]),
                    weights=weights,
                )
            )
        elif self.support_consistency_target in {"both", "node_class", "m_pi"}:
            student_node = self._normalize_distribution(student_aux["node_mass"].flatten(1))
            teacher_node = self._normalize_distribution(teacher_aux["node_mass"].flatten(1))
            support_losses.append(soft_prob_cross_entropy(student_node, teacher_node, weights=weights))
            support_losses.append(
                soft_prob_cross_entropy(
                    self._normalize_distribution(student_aux["class_probs"]),
                    self._normalize_distribution(teacher_aux["class_probs"]),
                    weights=weights,
                )
            )
        elif self.support_consistency_target in {"", "none", "off", "false"}:
            pass
        else:
            raise ValueError(
                f"Unsupported DCPR support_consistency_target={self.support_consistency_target}. "
                "Expected node_mass, class_probs, both, or none."
            )

        if support_losses:
            node_loss = torch.stack(support_losses).mean()
        else:
            node_loss = torch.zeros((), device=weights.device, dtype=torch.float32)

        if self.use_ambiguity_consistency:
            conf_loss = soft_prob_cross_entropy(
                self._normalize_distribution(student_aux["confusion_profile"]),
                self._normalize_distribution(teacher_aux["confusion_profile"]),
                weights=weights,
            )
        else:
            conf_loss = torch.zeros((), device=weights.device, dtype=torch.float32)
        return node_loss, conf_loss

    @torch.no_grad()
    def _teacher_guidance(self, tgt_weak: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.ema_net.eval()
        _, aux = self._forward_logits(self.ema_net, x=tgt_weak)
        conf = aux["class_probs"].max(dim=1).values.detach()
        guide = {
            "class_probs": aux["class_probs"].detach(),
            "node_mass": aux["node_mass"].detach(),
            "confusion_profile": aux["confusion_profile"].detach(),
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
    def _sync_relation_buffers_to_ema(self):
        for name in ["src_prototypes", "src_proto_inited", "src_classifier_confusion"]:
            getattr(self.ema_net.relation_router, name).copy_(getattr(self.net.relation_router, name))

    def _build_optimizer(self):
        base_lr = float(self.config.method.lr)
        param_groups = [
            {"params": list(self.net.backbone.parameters()), "lr": base_lr * 0.1},
            {"params": list(self.net.bottleneck.parameters()), "lr": base_lr},
            {"params": list(self.net.feature_norm.parameters()), "lr": base_lr},
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
        """Single-pass: compute per-class prototypes and classifier confusion."""
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
        confusion = torch.zeros(
            self.num_source_domains,
            self.num_classes,
            self.num_classes,
            device=self.device,
        )
        feat_sums_flat = feat_sums.view(-1, model.relation_feat_dim)
        counts_flat = counts.view(-1)
        confusion_flat = confusion.view(-1, self.num_classes)

        for src_imgs, src_labels, src_dom in prototype_loader:
            src_imgs = self._to_device(src_imgs)
            src_labels = self._to_device(src_labels)
            src_dom = self._to_device(src_dom)

            with self._auto_cast():
                h = model.extract_relation_features(src_imgs)
                h_shared = model.normalize_relation_features(h)
                logits, _ = self._forward_logits(model, h_shared=h_shared)
                probs = torch.softmax(logits, dim=1)

            flat_index = src_dom.long() * self.num_classes + src_labels.long()
            # Sort by flat_index so index_add_ writes are coalesced on CUDA.
            order = flat_index.argsort()
            flat_index_sorted = flat_index[order]
            feat_sums_flat.index_add_(0, flat_index_sorted, h_shared[order])
            ones = torch.ones(flat_index_sorted.size(0), dtype=counts_flat.dtype, device=self.device)
            counts_flat.index_add_(0, flat_index_sorted, ones)
            confusion_flat.index_add_(0, flat_index_sorted, probs[order])

        valid = counts > 0
        safe_counts = counts.clamp_min(1.0).unsqueeze(-1)
        prototypes = feat_sums / safe_counts
        prototypes = torch.where(valid.unsqueeze(-1), prototypes, torch.zeros_like(prototypes))

        eye = torch.eye(self.num_classes, device=self.device).unsqueeze(0)
        confusion_scores = confusion * (1.0 - eye)
        confusion = confusion_scores / confusion_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        confusion = torch.where(valid.unsqueeze(-1), confusion, torch.zeros_like(confusion))

        if self.prototype_granularity in {"class", "shared", "shared_class"}:
            class_counts = counts.sum(dim=0)
            class_feat_sums = feat_sums.sum(dim=0)
            class_valid = class_counts > 0
            class_prototypes = class_feat_sums / class_counts.clamp_min(1.0).unsqueeze(-1)
            class_prototypes = torch.where(
                class_valid.unsqueeze(-1),
                class_prototypes,
                torch.zeros_like(class_prototypes),
            )
            shared_prototypes = torch.zeros_like(prototypes)
            shared_valid = torch.zeros_like(valid)
            shared_prototypes[0] = class_prototypes
            shared_valid[0] = class_valid

            class_confusion_scores = confusion_scores.sum(dim=0)
            class_confusion = (
                class_confusion_scores
                / class_confusion_scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            )
            class_confusion = torch.where(
                class_valid.unsqueeze(-1),
                class_confusion,
                torch.zeros_like(class_confusion),
            )
            shared_confusion = torch.zeros_like(confusion)
            shared_confusion[0] = class_confusion

            prototypes = shared_prototypes
            valid = shared_valid
            confusion = shared_confusion
        elif self.prototype_granularity == "domain_class":
            pass
        else:
            raise ValueError(
                f"Unsupported DCPR prototype_granularity={self.prototype_granularity}. "
                "Expected domain_class or shared_class."
            )

        if self.ambiguity_source == "random":
            random_scores = torch.rand_like(confusion)
            random_scores = random_scores * (1.0 - eye)
            confusion = torch.where(valid.unsqueeze(-1), random_scores, torch.zeros_like(random_scores))
            confusion = confusion / confusion.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.ambiguity_source in {"domain_class", "global", "uniform"}:
            pass
        else:
            raise ValueError(
                f"Unsupported DCPR ambiguity_source={self.ambiguity_source}. "
                "Expected domain_class, global, uniform, or random."
            )

        return valid, prototypes, confusion

    def _recompute_source_prototypes(self, model: DCPRNetwork):
        with torch.inference_mode():
            was_training = model.training
            model.eval()

            with self._prototype_source_iter() as prototype_loader:
                batch_iter = self._iter_prototype_source_batches(prototype_loader)
                started_at = time.time()
                valid, prototypes, confusion = self._compute_source_prototypes(model, batch_iter)
                elapsed_minutes = (time.time() - started_at) / 60.0

                model.reset_source_prototypes()
                model.relation_router.src_proto_inited.copy_(valid)
                model.relation_router.src_prototypes.copy_(prototypes)
                model.relation_router.src_classifier_confusion.copy_(confusion)
                logger.info(
                    "DCPR source prototype refresh | proto_bs=%d single_pass "
                    "elapsed_min=%.2f",
                    self.prototype_batch_size,
                    elapsed_minutes,
                )

            model.train(was_training)

    def _train_step(
        self,
        optimizer,
        scheduler,
        src_batch,
        tgt_batch,
        ramp: float,
        consistency_ramp: float,
        global_step: int,
        total_iters: int,
    ):
        src_imgs, src_labels, _ = src_batch
        tgt_weak, tgt_strong = tgt_batch if tgt_batch is not None else (None, None)

        self._zero_grad(optimizer)

        loss_rnode = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_rconf = torch.zeros((), device=self.device, dtype=torch.float32)
        conf_tgt = torch.zeros((), device=self.device, dtype=torch.float32)

        with self._auto_cast():
            logits_src, _ = self._forward_logits(self.net, x=src_imgs)
            self._probe_amp_tensor(logits_src, "dcpr/logits_src", warn_on_float32=False)
            loss_src = self.criterion_task(logits_src, src_labels)
            loss = loss_src

            if tgt_weak is not None and tgt_strong is not None:
                _, tgt_aux = self._forward_logits(self.net, x=tgt_strong)
                with torch.no_grad():
                    with self._auto_cast():
                        conf_tgt, teacher_aux = self._teacher_guidance(tgt_weak)

                rel_weights = conf_tgt.pow(self.consistency_conf_power)
                loss_rnode, loss_rconf = self._relation_consistency_loss(
                    tgt_aux,
                    teacher_aux,
                    rel_weights,
                )
                consistency_loss = self.lambda_relation_consistency * (
                    self.lambda_rel_consistency_node * loss_rnode
                    + self.lambda_rel_consistency_conf * loss_rconf
                )
                loss = loss + ramp * consistency_ramp * consistency_loss

        self._optimizer_step_with_optional_clip(
            loss,
            optimizer,
            clip_params=self.net.parameters(),
            clip_max_norm=self.grad_clip,
        )
        scheduler.step()
        self._update_ema(self._ema_decay_at(global_step, total_iters))

        metrics = {
            "src": loss_src.detach().float(),
            "rnode": loss_rnode.detach().float(),
            "rconf": loss_rconf.detach().float(),
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
        ramp: float,
        consistency_ramp: float,
        global_step: int,
        total_iters: int,
    ):
        metric_keys = ("src", "rnode", "rconf", "conf", "total")
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
                ramp=ramp,
                consistency_ramp=consistency_ramp,
                global_step=global_step,
                total_iters=total_iters,
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
        scheduler = self._build_scheduler(optimizer, total_iters)
        self.register_training_state(optimizer=optimizer, scheduler=scheduler)
        self._setup_compiled_student_forward()
        best_acc = self._best_metric
        uses_target_loader = self._uses_target_loader_in_training()

        global_step = self._training_global_step
        logger.info(
            "DCPR Training(mainline): epoch_steps=max source_steps=%d target_steps=%d "
            "epoch_steps=%d use_target=%s",
            len(self.source_loader),
            len(self.target_loader),
            epoch_steps,
            str(uses_target_loader),
        )

        for epoch in self._epoch_range(self.total_epochs):
            current_temperature = self._temperature_at_epoch(epoch + 1)
            self._set_relation_temperature(current_temperature)

            if self.refresh_source_prototypes_each_epoch or epoch == 0:
                self._recompute_source_prototypes(self.net)
                self._sync_relation_buffers_to_ema()

            self.net.train()
            ramp = min(1.0, (epoch + 1) / max(1.0, self.ramp_denom))
            consistency_ramp = 1.0 if (epoch + 1) >= self.consistency_start_epoch else 0.0
            metrics, global_step = self._run_train_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_steps=epoch_steps,
                uses_target_loader=uses_target_loader,
                ramp=ramp,
                consistency_ramp=consistency_ramp,
                global_step=global_step,
                total_iters=total_iters,
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
                    "rnode": metrics["rnode"],
                    "rconf": metrics["rconf"],
                    "conf": (metrics["conf"], ".3f"),
                    "total": metrics["total"],
                },
                extras={
                    "rmp": (ramp, ".2f"),
                    "crmp": (consistency_ramp, ".2f"),
                    "tmp": (current_temperature, ".3f"),
                },
                score=acc,
                best_score=best_acc,
                score_name="Acc",
            )

        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")
