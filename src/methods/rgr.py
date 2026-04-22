"""
RGR: Relational Graph Representation for MSDA.

Final winner variant (single-path):
- Shared feature encoder + relation graph parsing.
- Cross-domain same-class reference relations.
- Local inter-class confusion structure relations.
- Target adaptation via node/confusion relation consistency on weak/strong views.
- Relation-only classifier head (no prior-fusion branch).

Core statement:
Multi-source adaptation is better framed as relation-assisted discrimination
on shared semantics, rather than relation-driven feature generation.
"""

import copy
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import cycle

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


def _unwrap_weak_strong_from_maybe_tuple(tgt_imgs):
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) >= 2:
        return tgt_imgs[0], tgt_imgs[1]
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) == 1:
        return tgt_imgs[0], tgt_imgs[0]
    return tgt_imgs, tgt_imgs


def _record_stream_recursive(batch: Any, stream: torch.cuda.Stream):
    if torch.is_tensor(batch):
        if batch.is_cuda:
            batch.record_stream(stream)
        return
    if isinstance(batch, (list, tuple)):
        for v in batch:
            _record_stream_recursive(v, stream)
        return
    if isinstance(batch, dict):
        for v in batch.values():
            _record_stream_recursive(v, stream)


class _CudaBatchPrefetcher:
    """Prefetch next batch to device stream while current batch is computing."""

    def __init__(
        self,
        iterator,
        load_fn: Callable[[Any], Any],
        enabled: bool,
    ):
        self.iterator = iterator
        self.load_fn = load_fn
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.stream = torch.cuda.Stream() if self.enabled else None
        self._next_batch = None
        self._preload()

    def _preload(self):
        try:
            raw = next(self.iterator)
        except StopIteration:
            self._next_batch = None
            return

        if not self.enabled or self.stream is None:
            self._next_batch = self.load_fn(raw)
            return

        with torch.cuda.stream(self.stream):
            self._next_batch = self.load_fn(raw)

    def pop(self):
        if self._next_batch is None:
            raise StopIteration

        if self.enabled and self.stream is not None:
            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.stream)
            _record_stream_recursive(self._next_batch, current_stream)

        batch = self._next_batch
        self._preload()
        return batch


class RelationGraphBuilder(nn.Module):
    """Build relation primitives from source domain class prototypes."""

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_source_domains: int,
        relation_temperature: float = 0.10,
        confusion_temperature: float = 0.15,
        confusion_topk: int = 4,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.num_source_domains = int(num_source_domains)
        self.relation_temperature = max(1e-6, float(relation_temperature))
        self.confusion_temperature = max(1e-6, float(confusion_temperature))
        self.confusion_topk = max(0, int(confusion_topk))

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

    @torch.no_grad()
    def reset_source_prototypes(self):
        self.src_prototypes.zero_()
        self.src_proto_inited.zero_()

    def _class_confusion_weights(self) -> torch.Tensor:
        """Domain-wise class confusion structure: [D, C, C]."""
        proto = self.src_prototypes
        mask = self.src_proto_inited
        d, c, _ = proto.shape

        if c <= 1:
            return torch.zeros(d, c, c, device=proto.device, dtype=proto.dtype)

        p_n = F.normalize(proto, dim=-1)
        sim = torch.einsum("dcf,dkf->dck", p_n, p_n) / self.confusion_temperature

        valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(c, device=proto.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = valid_pair & (~eye)

        if self.confusion_topk > 0 and self.confusion_topk < (c - 1):
            topk = min(self.confusion_topk, c - 1)
            safe_logits = sim.masked_fill(~valid_pair, -1e4)
            topi = safe_logits.topk(topk, dim=-1).indices
            keep = torch.zeros_like(valid_pair)
            keep.scatter_(-1, topi, True)
            valid_pair = valid_pair & keep

        safe_logits = sim.masked_fill(~valid_pair, -1e4)
        weights = torch.softmax(safe_logits, dim=-1)
        valid_row = valid_pair.any(dim=-1)
        return torch.where(valid_row.unsqueeze(-1), weights, torch.zeros_like(weights))

    def parse(self, h_relation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse sample-to-graph relations from relation-space features."""
        proto = self.src_prototypes
        mask = self.src_proto_inited

        proto_n = F.normalize(proto, dim=-1)
        h_n = F.normalize(h_relation, dim=-1)

        # Sample-to-domain-class relation logits: [B, D, C]
        node_logits_bdc = torch.einsum("bf,dcf->bdc", h_n, proto_n)
        node_logits_bdc = node_logits_bdc / self.relation_temperature
        node_logits_bdc = node_logits_bdc.masked_fill(~mask.unsqueeze(0), -1e4)

        # Same-class cross-domain evidence.
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

        # Same-class cross-domain reference features [B, C, F].
        node_context = torch.einsum("bcd,dcf->bcf", domain_weights, proto)
        node_context = torch.where(
            valid_classes.unsqueeze(0).unsqueeze(-1),
            node_context,
            torch.zeros_like(node_context),
        )

        # Local inter-class confusion structure conditioned on domain reference.
        # confusion_dck: [D, C(anchor), C(confusing)]
        confusion_dck = self._class_confusion_weights()
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
            "node_context": node_context,
            "confusion_per_class": confusion_per_class,
            "valid_classes": valid_classes,
        }


class RelationParser(nn.Module):
    """Final parser used by the winning setup: relation-only classification."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        class_logits_rel: torch.Tensor,
        valid_classes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        class_logits = class_logits_rel
        class_logits = class_logits.masked_fill(~valid_classes.unsqueeze(0), -1e4)
        class_probs = torch.softmax(class_logits, dim=1)
        return {
            "class_logits": class_logits,
            "class_probs": class_probs,
        }


class RGRNetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_source_domains: int,
        *,
        bottleneck_dim: int = 0,
        relation_temperature: float = 0.10,
        confusion_temperature: float = 0.10,
        confusion_topk: int = 8,
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

        self.graph_builder = RelationGraphBuilder(
            feat_dim=self.relation_feat_dim,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            relation_temperature=relation_temperature,
            confusion_temperature=confusion_temperature,
            confusion_topk=confusion_topk,
        )
        self.relation_parser = RelationParser()

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
    def reset_source_prototypes(self):
        self.graph_builder.reset_source_prototypes()

    def forward_relation_logits(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_shared = self._encode_shared(x=x, h_shared=h_shared)
        graph = self.graph_builder.parse(h_shared)
        parsed = self.relation_parser(
            class_logits_rel=graph["class_logits_rel"],
            valid_classes=graph["valid_classes"],
        )

        class_probs = parsed["class_probs"]
        domain_weights = graph["domain_weights"]
        node_mass = class_probs.unsqueeze(-1) * domain_weights  # [B, C, D]

        confusion_per_class = graph["confusion_per_class"]  # [B, C, C]
        confusion_profile = torch.einsum("bc,bck->bk", class_probs, confusion_per_class)
        confusion_profile = confusion_profile / confusion_profile.sum(dim=1, keepdim=True).clamp_min(1e-8)

        logits = parsed["class_logits"]

        aux = {
            "h_relation": h_shared,
            "class_logits": parsed["class_logits"],
            "class_probs": class_probs,
            "domain_logits": graph["domain_logits"],
            "domain_weights": domain_weights,
            "node_mass": node_mass,
            # Kept for analysis/visualization even though final training loss
            # only uses node_mass and confusion_profile.
            "node_context": graph["node_context"],
            "confusion_per_class": confusion_per_class,
            "confusion_profile": confusion_profile,
            "valid_classes": graph["valid_classes"],
        }
        return logits, aux

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
        sources = list(getattr(self.config.dataset, "sources", []) or [])
        if len(sources) == 0:
            raise ValueError("rgr requires config.dataset.sources to be a non-empty list")

        self.num_source_domains = len(sources)

        for key, cast, default in [
            ("bottleneck_dim", int, 256),
            ("relation_temperature", float, 0.10),
            ("lambda_relation_consistency", float, 0.40),
            ("consistency_conf_power", float, 2.0),
            ("consistency_start_epoch", int, 4),
            ("refresh_source_prototypes_each_epoch", bool, True),
            ("grad_clip", float, 5.0),
            ("save_ckpt_after_epoch", int, 0),
            ("ema_decay_start", float, 0.996),
            ("ema_decay_end", float, 0.9995),
            ("label_smoothing", float, 0.05),
        ]:
            setattr(self, key, cast(m.get(key, default)))

        self.total_epochs = int(m.get("epochs", 20))
        self.ramp_denom = float(m.get("ramp_denom", max(1.0, self.total_epochs * 0.3)))
        self.epoch_steps_mode = str(m.get("epoch_steps_mode", "max")).strip().lower()
        # Fixed by the final winning setup.
        self.confusion_temperature = 0.10
        self.confusion_topk = 8
        self.lambda_rel_consistency_node = 0.25
        self.lambda_rel_consistency_conf = 1.0

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

        self.net = RGRNetwork(
            backbone_name=backbone_name,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            bottleneck_dim=self.bottleneck_dim,
            relation_temperature=self.relation_temperature,
            confusion_temperature=self.confusion_temperature,
            confusion_topk=self.confusion_topk,
        ).to(self.device)

        self.ema_net = copy.deepcopy(self.net)
        for param in self.ema_net.parameters():
            param.requires_grad_(False)
        self._forward_logits_student = self.net.forward_relation_logits
        self._student_forward_compiled = False
        self._target_tensor_aug_enabled = False
        self._target_weak_aug = None
        self._target_strong_aug = None
        self._setup_target_tensor_augment()

        logger.info(
            "RGR(final-winner): bottleneck=%d rel_temp=%.3f "
            "conf_temp=%.3f conf_topk=%d rel_space_dim=%d "
            "lambda_rel_cons=%.3f prefetch=%s "
            "rel_w(node/conf)=%.2f/%.2f consistency_start=%d",
            self.bottleneck_dim,
            self.relation_temperature,
            self.confusion_temperature,
            self.confusion_topk,
            self.net.relation_feat_dim,
            self.lambda_relation_consistency,
            str(self.cuda_batch_prefetch),
            self.lambda_rel_consistency_node,
            self.lambda_rel_consistency_conf,
            self.consistency_start_epoch,
        )

    def _forward_logits(
        self,
        model: RGRNetwork,
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

        def _student_forward(x: Optional[torch.Tensor], h_shared: Optional[torch.Tensor]):
            return self.net.forward_relation_logits(x=x, h_shared=h_shared)

        self._forward_logits_student = self._compile_callable(
            _student_forward,
            "rgr_student.forward_relation_logits",
        )
        self._student_forward_compiled = True

    def _setup_target_tensor_augment(self):
        perf_cfg = getattr(self.config, "performance", None)
        aug_cfg = getattr(perf_cfg, "augmentation", None) if perf_cfg is not None else None
        enabled = bool(getattr(aug_cfg, "target_tensor_v2", False)) if aug_cfg is not None else False
        if not enabled:
            return
        if not bool(getattr(self.config.method, "strong_aug", False)):
            logger.warning(
                "target_tensor_v2 requested, but method.strong_aug=False. "
                "Falling back to dataloader transforms."
            )
            return
        color_space_cfg = getattr(self.config.method, "color_space", None)
        if color_space_cfg is not None and bool(getattr(color_space_cfg, "enabled", False)):
            logger.warning(
                "target_tensor_v2 requested with color_space.enabled=True, which is unsupported. "
                "Falling back to dataloader transforms."
            )
            return

        target_aug_cfg = getattr(self.config.method, "target_aug", None)
        randaugment_num_ops = (
            int(getattr(target_aug_cfg, "randaugment_num_ops", 2))
            if target_aug_cfg is not None
            else 2
        )
        randaugment_magnitude = (
            int(getattr(target_aug_cfg, "randaugment_magnitude", 10))
            if target_aug_cfg is not None
            else 10
        )

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self._target_weak_aug = transforms_v2.Compose(
            [
                transforms_v2.RandomCrop(224),
                transforms_v2.RandomHorizontalFlip(),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean, std),
            ]
        )
        self._target_strong_aug = transforms_v2.Compose(
            [
                transforms_v2.RandomCrop(224),
                transforms_v2.RandomHorizontalFlip(),
                transforms_v2.RandAugment(
                    num_ops=randaugment_num_ops,
                    magnitude=randaugment_magnitude,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean, std),
            ]
        )
        self._target_tensor_aug_enabled = True
        logger.info(
            "RGR target tensor augmentation enabled (v2, GPU-capable): "
            "weak=RandomCrop+HFlip, strong=RandomCrop+HFlip+RandAugment(%d,%d)",
            randaugment_num_ops,
            randaugment_magnitude,
        )

    def _to_uint8_image_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            return x
        if torch.is_floating_point(x):
            if x.max() <= 1.0 and x.min() >= 0.0:
                return (x * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
            return x.round().clamp(0.0, 255.0).to(torch.uint8)
        return x.clamp(0, 255).to(torch.uint8)

    def _prepare_target_views(self, tgt_imgs):
        if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) >= 2:
            tgt_weak, tgt_strong = tgt_imgs[0], tgt_imgs[1]
            return self._to_device(tgt_weak), self._to_device(tgt_strong)

        if not self._target_tensor_aug_enabled:
            tgt_weak, tgt_strong = _unwrap_weak_strong_from_maybe_tuple(tgt_imgs)
            return self._to_device(tgt_weak), self._to_device(tgt_strong)

        base = self._to_device(tgt_imgs)
        base = self._to_uint8_image_tensor(base)
        tgt_weak = self._target_weak_aug(base)
        tgt_strong = self._target_strong_aug(base)
        return tgt_weak, tgt_strong

    def _load_source_batch_to_device(self, src_batch):
        src_imgs, src_labels, src_dom = src_batch
        return self._to_device(src_imgs), self._to_device(src_labels), self._to_device(src_dom)

    def _load_target_batch_to_views(self, tgt_batch):
        tgt_imgs = tgt_batch[0] if isinstance(tgt_batch, (tuple, list)) else tgt_batch
        return self._prepare_target_views(tgt_imgs)

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
        node_loss = soft_prob_cross_entropy(
            self._normalize_distribution(student_aux["node_mass"].flatten(1)),
            self._normalize_distribution(teacher_aux["node_mass"].flatten(1)),
            weights=weights,
        )

        conf_loss = soft_prob_cross_entropy(
            self._normalize_distribution(student_aux["confusion_profile"]),
            self._normalize_distribution(teacher_aux["confusion_profile"]),
            weights=weights,
        )

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

    def _recompute_source_prototypes(self, model: RGRNetwork):
        with torch.inference_mode():
            was_training = model.training
            model.eval()

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
            feat_sums_flat = feat_sums.view(-1, model.relation_feat_dim)
            counts_flat = counts.view(-1)

            for src_imgs, src_labels, src_dom in self.source_loader:
                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)
                src_dom = self._to_device(src_dom)

                with self._auto_cast():
                    h = model.extract_relation_features(src_imgs)
                    h_shared = model.normalize_relation_features(h)

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

            model.graph_builder.src_proto_inited.copy_(valid)
            model.graph_builder.src_prototypes.copy_(prototypes)

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
        self._setup_compiled_student_forward()
        best_acc = 0.0
        best_save_acc = -1e18
        best_path = Path("checkpoints") / "best_rgr.pth"
        best_path.parent.mkdir(parents=True, exist_ok=True)

        global_step = 0
        logger.info(
            "RGR Training(final-winner): relation consistency | "
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
            metric_keys = ("src", "rnode", "rconf", "conf", "total")
            metric_sums = {
                key: torch.zeros((), device=self.device, dtype=torch.float32)
                for key in metric_keys
            }

            src_iter = cycle(self.source_loader)
            tgt_iter = cycle(self.target_loader)
            use_cuda_prefetch = bool(self.cuda_batch_prefetch and self.device.type == "cuda")
            src_prefetcher = _CudaBatchPrefetcher(
                src_iter,
                self._load_source_batch_to_device,
                enabled=use_cuda_prefetch,
            )
            tgt_prefetcher = _CudaBatchPrefetcher(
                tgt_iter,
                self._load_target_batch_to_views,
                enabled=use_cuda_prefetch,
            )
            ramp = min(1.0, (epoch + 1) / max(1.0, self.ramp_denom))
            consistency_ramp = 1.0 if (epoch + 1) >= self.consistency_start_epoch else 0.0

            for _ in range(epoch_steps):
                src_imgs, src_labels, _ = src_prefetcher.pop()
                tgt_weak, tgt_strong = tgt_prefetcher.pop()

                self._zero_grad(optimizer)

                with self._auto_cast():
                    logits_src, _ = self._forward_logits(self.net, x=src_imgs)
                    self._probe_amp_tensor(logits_src, "rgr/logits_src")
                    loss_src = self.criterion_task(logits_src, src_labels)

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
                    loss = loss_src + ramp * consistency_ramp * consistency_loss

                self._optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    clip_params=self.net.parameters(),
                    clip_max_norm=self.grad_clip,
                )
                scheduler.step()

                self._update_ema(self._ema_decay_at(global_step, total_iters))
                global_step += 1

                metric_sums["src"].add_(loss_src.detach().float())
                metric_sums["rnode"].add_(loss_rnode.detach().float())
                metric_sums["rconf"].add_(loss_rconf.detach().float())
                metric_sums["conf"].add_(conf_tgt.detach().mean().float())
                metric_sums["total"].add_(loss.detach().float())

            scale = 1.0 / float(max(1, epoch_steps))
            metrics = {key: (value * scale).item() for key, value in metric_sums.items()}

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            if epoch + 1 > self.save_ckpt_after_epoch and acc > best_save_acc:
                best_save_acc = acc
                self.save_checkpoint(best_path)

            logger.info(
                f"RGR {epoch+1}/{self.total_epochs} | "
                f"src={metrics['src']:.4f} "
                f"rnode={metrics['rnode']:.4f} "
                f"rconf={metrics['rconf']:.4f} "
                f"conf={metrics['conf']:.3f} "
                f"total={metrics['total']:.4f} | "
                f"rmp={ramp:.2f} crmp={consistency_ramp:.2f} | "
                f"Acc={acc:.2f}% (best={best_acc:.2f}%)"
            )

        if best_path.exists():
            self.load_checkpoint(best_path)
            logger.info(f"Loaded best RGR checkpoint from {best_path} with Acc={best_save_acc:.2f}%")
