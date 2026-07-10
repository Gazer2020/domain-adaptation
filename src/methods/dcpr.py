"""
DCPR: domain-class prototype relation adaptation.

Source supervision uses a cosine classifier head (adaptive_logits) with CE loss.
Target adaptation uses EMA consistency over the domain-class prototype relation
graph, with the teacher producing domain-class node mass distributions that the
student is trained to match on strongly-augmented views.

Architecture:
- Backbone features -> LayerNorm -> domain-relative prototype relation
- One cosine classifier is shared by source and target
- Prototypes affect only the domain-class relation, never class logits
- Target-centre EMA places target features in their own relative coordinates
- Classifier-margin ranks emphasize ambiguous classes inside the relation loss
"""

import copy
import csv
import json
import logging
import math
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
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


def _select_routing_indices(
    class_probs: torch.Tensor,
    routing_weights: torch.Tensor,
    num_samples: int,
    candidate_classes: int,
) -> list[int]:
    """Select samples whose top candidate classes route most differently."""
    if class_probs.ndim != 2 or routing_weights.ndim != 3:
        raise ValueError("Expected class_probs [N,C] and routing_weights [N,C,K].")
    if class_probs.size(0) != routing_weights.size(0):
        raise ValueError("Probability and routing batches must have the same size.")

    topk = min(max(2, int(candidate_classes)), class_probs.size(1))
    top_classes = class_probs.topk(topk, dim=1).indices
    scores = []
    for index in range(class_probs.size(0)):
        routes = routing_weights[index, top_classes[index]]
        diversity = torch.cdist(routes, routes, p=1).max().item() * 0.5
        scores.append((diversity, index, int(top_classes[index, 0].item())))

    selected = []
    used_predictions = set()
    for _score, index, prediction in sorted(scores, reverse=True):
        if prediction in used_predictions:
            continue
        selected.append(index)
        used_predictions.add(prediction)
        if len(selected) >= num_samples:
            return selected
    for _score, index, _prediction in sorted(scores, reverse=True):
        if index not in selected:
            selected.append(index)
        if len(selected) >= num_samples:
            break
    return selected


def _select_ambiguous_cases(
    source_probs: torch.Tensor,
    dcpr_probs: torch.Tensor,
    labels: torch.Tensor,
    ambiguity_weights: torch.Tensor,
    num_pairs: int,
    samples_per_pair: int,
) -> list[dict]:
    """Find Source Only errors corrected by DCPR, grouped by class pair."""
    source_pred = source_probs.argmax(dim=1)
    dcpr_pred = dcpr_probs.argmax(dim=1)
    corrected = (source_pred != labels) & (dcpr_pred == labels)
    grouped: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)

    for index in corrected.nonzero(as_tuple=False).flatten().tolist():
        true_class = int(labels[index].item())
        confused_class = int(source_pred[index].item())
        sample_score = (
            float(source_probs[index, confused_class].item())
            + float(dcpr_probs[index, true_class].item())
            - float(source_probs[index, true_class].item())
        )
        grouped[(true_class, confused_class)].append((sample_score, index))

    ranked_pairs = []
    for pair, candidates in grouped.items():
        true_class, confused_class = pair
        ambiguity = 0.5 * (
            float(ambiguity_weights[true_class].item())
            + float(ambiguity_weights[confused_class].item())
        )
        ranked_pairs.append((len(candidates), ambiguity, pair, candidates))

    cases = []
    for count, ambiguity, pair, candidates in sorted(ranked_pairs, reverse=True)[:num_pairs]:
        for rank, (_score, index) in enumerate(
            sorted(candidates, reverse=True)[:samples_per_pair]
        ):
            cases.append(
                {
                    "index": index,
                    "true_class": pair[0],
                    "confused_class": pair[1],
                    "pair_count": count,
                    "pair_ambiguity": ambiguity,
                    "sample_rank": rank,
                }
            )
    return cases


def _safe_analysis_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _copy_analysis_image(source: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _export_routing_materials(
    output_dir: Path,
    *,
    samples: Sequence[tuple[str, int]],
    class_names: Sequence[str],
    source_domains: Sequence[str],
    labels: torch.Tensor,
    class_probs: torch.Tensor,
    prototype_class_probs: torch.Tensor,
    routing_weights: torch.Tensor,
    num_samples: int,
    candidate_classes: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    selected = _select_routing_indices(
        class_probs,
        routing_weights,
        num_samples=min(num_samples, len(samples)),
        candidate_classes=candidate_classes,
    )
    topk = min(max(2, candidate_classes), class_probs.size(1))
    manifest = []

    with (output_dir / "routing_weights.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "dataset_index",
                "candidate_class",
                "candidate_name",
                "source_domain",
                "rho",
                "is_top_candidate",
            ],
        )
        writer.writeheader()
        for sample_id, index in enumerate(selected):
            path, _label = samples[index]
            top_classes = class_probs[index].topk(topk).indices.tolist()
            image_path = _copy_analysis_image(
                path,
                image_dir / f"{sample_id:02d}_{_safe_analysis_name(Path(path).name)}",
            )
            routes = routing_weights[index, top_classes]
            diversity = float(torch.cdist(routes, routes, p=1).max().item() * 0.5)
            manifest.append(
                {
                    "sample_id": sample_id,
                    "dataset_index": index,
                    "image": image_path,
                    "original_image": path,
                    "true_class": int(labels[index].item()),
                    "true_name": class_names[int(labels[index].item())],
                    "predicted_class": int(class_probs[index].argmax().item()),
                    "predicted_name": class_names[int(class_probs[index].argmax().item())],
                    "top_candidate_classes": top_classes,
                    "top_candidate_names": [class_names[value] for value in top_classes],
                    "routing_diversity": diversity,
                }
            )
            top_set = set(top_classes)
            for class_id in range(routing_weights.size(1)):
                for domain_id, domain_name in enumerate(source_domains):
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "dataset_index": index,
                            "candidate_class": class_id,
                            "candidate_name": class_names[class_id],
                            "source_domain": domain_name,
                            "rho": float(routing_weights[index, class_id, domain_id].item()),
                            "is_top_candidate": class_id in top_set,
                        }
                    )

    selected_tensor = torch.tensor(selected, dtype=torch.long)
    np.savez_compressed(
        output_dir / "routing_materials.npz",
        dataset_indices=selected_tensor.numpy(),
        labels=labels[selected_tensor].numpy(),
        class_probs=class_probs[selected_tensor].numpy(),
        prototype_class_probs=prototype_class_probs[selected_tensor].numpy(),
        routing_weights=routing_weights[selected_tensor].numpy(),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "class_names": list(class_names),
                "source_domains": list(source_domains),
                "samples": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir / "manifest.json"


def _export_ambiguous_materials(
    output_dir: Path,
    *,
    samples: Sequence[tuple[str, int]],
    class_names: Sequence[str],
    source_domains: Sequence[str],
    labels: torch.Tensor,
    source_probs: torch.Tensor,
    dcpr_probs: torch.Tensor,
    routing_weights: torch.Tensor,
    ambiguity_weights: torch.Tensor,
    num_pairs: int,
    samples_per_pair: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    cases = _select_ambiguous_cases(
        source_probs,
        dcpr_probs,
        labels,
        ambiguity_weights,
        num_pairs=num_pairs,
        samples_per_pair=samples_per_pair,
    )
    first_by_class = {}
    for index, (_path, label) in enumerate(samples):
        first_by_class.setdefault(int(label), index)

    probability_rows = []
    routing_rows = []
    manifest = []
    for case_id, case in enumerate(cases):
        index = case["index"]
        true_class = case["true_class"]
        confused_class = case["confused_class"]
        pair_name = (
            f"{_safe_analysis_name(class_names[true_class])}__"
            f"{_safe_analysis_name(class_names[confused_class])}"
        )
        case_dir = image_dir / f"{case_id:02d}_{pair_name}"
        corrected_path = _copy_analysis_image(
            samples[index][0],
            case_dir / f"corrected_{Path(samples[index][0]).name}",
        )
        true_ref_index = first_by_class[true_class]
        confused_ref_index = first_by_class[confused_class]
        true_ref_path = _copy_analysis_image(
            samples[true_ref_index][0],
            case_dir / f"true_reference_{Path(samples[true_ref_index][0]).name}",
        )
        confused_ref_path = _copy_analysis_image(
            samples[confused_ref_index][0],
            case_dir / f"confused_reference_{Path(samples[confused_ref_index][0]).name}",
        )

        candidate_classes = [true_class, confused_class]
        for model_name, probs in [("source_only", source_probs), ("dcpr", dcpr_probs)]:
            for class_id in candidate_classes:
                probability_rows.append(
                    {
                        "case_id": case_id,
                        "model": model_name,
                        "candidate_class": class_id,
                        "candidate_name": class_names[class_id],
                        "probability": float(probs[index, class_id].item()),
                    }
                )
        for class_id in candidate_classes:
            for domain_id, domain_name in enumerate(source_domains):
                routing_rows.append(
                    {
                        "case_id": case_id,
                        "candidate_class": class_id,
                        "candidate_name": class_names[class_id],
                        "source_domain": domain_name,
                        "rho": float(routing_weights[index, class_id, domain_id].item()),
                    }
                )
        manifest.append(
            {
                **case,
                "case_id": case_id,
                "true_name": class_names[true_class],
                "confused_name": class_names[confused_class],
                "corrected_image": corrected_path,
                "true_reference_image": true_ref_path,
                "confused_reference_image": confused_ref_path,
                "original_corrected_image": samples[index][0],
                "source_only_prediction": int(source_probs[index].argmax().item()),
                "dcpr_prediction": int(dcpr_probs[index].argmax().item()),
            }
        )

    for filename, rows, fields in [
        (
            "candidate_probabilities.csv",
            probability_rows,
            ["case_id", "model", "candidate_class", "candidate_name", "probability"],
        ),
        (
            "candidate_routing_weights.csv",
            routing_rows,
            ["case_id", "candidate_class", "candidate_name", "source_domain", "rho"],
        ),
    ]:
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    selected = torch.tensor([case["index"] for case in cases], dtype=torch.long)
    np.savez_compressed(
        output_dir / "ambiguous_materials.npz",
        dataset_indices=selected.numpy(),
        labels=labels[selected].numpy(),
        source_probs=source_probs[selected].numpy(),
        dcpr_probs=dcpr_probs[selected].numpy(),
        routing_weights=routing_weights[selected].numpy(),
        ambiguity_weights=ambiguity_weights.numpy(),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "class_names": list(class_names),
                "source_domains": list(source_domains),
                "cases": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# DCPR Ambiguous-Class Materials",
        "",
        f"- corrected cases exported: {len(cases)}",
        f"- requested class pairs: {num_pairs}",
        "",
        "| case | true class | Source Only confusion | pair count | ambiguity |",
        "|---:|---|---|---:|---:|",
    ]
    summary_lines.extend(
        f"| {case['case_id']} | {case['true_name']} | {case['confused_name']} | "
        f"{case['pair_count']} | {case['pair_ambiguity']:.4f} |"
        for case in manifest
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_dir / "manifest.json"


class PrototypeRelationRouter(nn.Module):
    """Class-conditioned routing over source domain-class prototypes."""

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
        """Return source-domain routing statistics for each class."""
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

        valid_classes = mask.any(dim=0)
        class_logits_rel = torch.logsumexp(node_logits_bdc, dim=1)
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
        bottleneck_dim: int = 0,
        relation_temperature: float = 0.10,
        adaptive_head_scale: float = 10.0,
        relation_space_mode: str = "domain_relative",
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
        self.adaptive_head_scale = float(adaptive_head_scale)
        self.relation_space_mode = str(relation_space_mode).strip().lower()

        self.relation_router = PrototypeRelationRouter(
            feat_dim=self.relation_feat_dim,
            num_classes=self.num_classes,
            num_source_domains=self.num_source_domains,
            relation_temperature=relation_temperature,
            relation_space_mode=self.relation_space_mode,
        )
        self.adaptive_classifier = nn.Linear(self.relation_feat_dim, self.num_classes, bias=False)
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

    def _adaptive_logits(self, h_shared: torch.Tensor) -> torch.Tensor:
        h_n = F.normalize(h_shared, dim=-1)
        w_n = F.normalize(self.adaptive_classifier.weight, dim=-1)
        return F.linear(h_n, w_n) * self.adaptive_head_scale

    def forward_relation_logits(
        self,
        x: Optional[torch.Tensor] = None,
        h_shared: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        use_target_center: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_shared = self._encode_shared(x=x, h_shared=h_shared)
        sample_center = self._target_sample_center(h_shared) if use_target_center else None
        relations = self.relation_router.parse(
            h_shared,
            domain_ids=domain_ids,
            sample_center=sample_center,
        )
        class_logits = self._adaptive_logits(h_shared)
        class_logits = class_logits.masked_fill(~relations["valid_classes"].unsqueeze(0), -1e4)
        class_probs = torch.softmax(class_logits, dim=1)
        prototype_class_probs = torch.softmax(relations["class_logits_rel"], dim=1)
        domain_weights = relations["domain_weights"]
        node_mass = class_probs.unsqueeze(-1) * domain_weights
        aux = {
            "h_shared": h_shared,
            "h_relation": relations["h_relation"],
            "class_logits": class_logits,
            "class_probs": class_probs,
            "prototype_class_probs": prototype_class_probs,
            "domain_weights": domain_weights,
            "node_mass": node_mass,
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
            ("bottleneck_dim", int, 0),
            ("consistency_conf_power", float, 2.0),
            ("consistency_start_epoch", int, 5),
            ("grad_clip", float, 5.0),
            ("save_ckpt_after_epoch", int, 15),
            ("ema_decay_start", float, 0.996),
            ("ema_decay_end", float, 0.9995),
            ("label_smoothing", float, 0.05),
            ("target_center_momentum", float, 0.98),
        ]:
            setattr(self, key, cast(m.get(key, default)))

        self.lambda_relation_consistency = float(m.get("lambda_relation_consistency", 0.40))
        self.consistency_target = str(m.get("consistency_target", "relation")).strip().lower()
        if self.consistency_target not in {"relation", "classification", "class_only"}:
            raise ValueError(
                "method.consistency_target must be 'relation', 'classification', "
                "or 'class_only'"
            )
        self.relation_space_mode = str(m.get("relation_space_mode", "domain_relative")).strip().lower()
        if self.relation_space_mode not in {"standard", "domain_relative"}:
            raise ValueError("method.relation_space_mode must be 'standard' or 'domain_relative'")
        self.adaptive_head_scale = float(m.get("adaptive_head_scale", 10.0))
        self.ambiguity_relation_boost = float(m.get("ambiguity_relation_boost", 0.5))
        analysis_cfg = m.get("analysis_export", {})
        self.analysis_mode = str(analysis_cfg.get("mode", "none")).strip().lower()
        if self.analysis_mode not in {"none", "routing", "ambiguous"}:
            raise ValueError("method.analysis_export.mode must be none, routing, or ambiguous")
        self.analysis_output_dir = Path(str(analysis_cfg.get("output_dir", "analysis_materials")))
        self.analysis_num_samples = max(1, int(analysis_cfg.get("num_samples", 12)))
        self.analysis_candidate_classes = max(2, int(analysis_cfg.get("candidate_classes", 5)))
        self.analysis_source_checkpoint = analysis_cfg.get("source_checkpoint", None)
        self.analysis_num_pairs = max(1, int(analysis_cfg.get("num_pairs", 6)))
        self.analysis_samples_per_pair = max(1, int(analysis_cfg.get("samples_per_pair", 1)))

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
            relation_space_mode=self.relation_space_mode,
        ).to(self.device)
        self.class_ambiguity_weights = torch.zeros(self.num_classes, device=self.device)

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
            "DCPR: bottleneck=%d rel_space=%s rel_dim=%d temp=%.2f->%.2f "
            "lambda_rel=%.2f target=%s head_scale=%.1f amb_boost=%.2f proto_bs=%d "
            "ramp_start=%d ramp_denom=%.1f prefetch=%s",
            self.bottleneck_dim,
            self.relation_space_mode,
            self.net.relation_feat_dim,
            self.temperature_start,
            self.temperature_end,
            self.lambda_relation_consistency,
            self.consistency_target,
            self.adaptive_head_scale,
            self.ambiguity_relation_boost,
            self.prototype_batch_size,
            self.consistency_start_epoch,
            self.ramp_denom,
            str(self.cuda_batch_prefetch),
        )
        if self.analysis_mode != "none":
            logger.info(
                "DCPR analysis export: mode=%s output=%s",
                self.analysis_mode,
                self.analysis_output_dir,
            )

    def _uses_target_loader_in_training(self) -> bool:
        return self.lambda_relation_consistency > 0.0

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if model is self.net:
            return self._forward_logits_student(
                x, h_shared, domain_ids, use_target_center,
            )
        return model.forward_relation_logits(
            x=x,
            h_shared=h_shared,
            domain_ids=domain_ids,
            use_target_center=use_target_center,
        )

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

    def _domain_class_relation(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Target relation over class and source-domain prototype nodes."""
        return self._normalize_distribution(aux["node_mass"].flatten(1))

    def _consistency_distribution(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.consistency_target == "relation":
            return self._domain_class_relation(aux)
        if self.consistency_target == "classification":
            return aux["class_probs"]
        if self.consistency_target == "class_only":
            return aux["prototype_class_probs"]
        raise ValueError(f"Unsupported consistency target: {self.consistency_target}")

    def _relation_consistency_loss(
        self,
        student_aux: Dict[str, torch.Tensor],
        teacher_aux: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Match the configured teacher/student target distribution."""
        student_distribution = self._consistency_distribution(student_aux)
        teacher_distribution = self._consistency_distribution(teacher_aux)
        return soft_prob_cross_entropy(
            student_distribution,
            teacher_distribution,
            weights=weights,
        )

    def _ambiguity_sample_weights(self, class_probs: torch.Tensor) -> torch.Tensor:
        if self.ambiguity_relation_boost <= 0.0:
            return torch.ones(class_probs.size(0), device=class_probs.device, dtype=class_probs.dtype)
        ambiguity = self.class_ambiguity_weights.to(
            device=class_probs.device,
            dtype=class_probs.dtype,
        )
        expected_ambiguity = (class_probs * ambiguity.unsqueeze(0)).sum(dim=1)
        centered_ambiguity = 2.0 * expected_ambiguity - 1.0
        return (1.0 + self.ambiguity_relation_boost * centered_ambiguity).clamp_min(1e-3)

    @torch.no_grad()
    def _teacher_guidance(self, tgt_weak: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.ema_net.eval()
        h_shared = self.ema_net._encode_shared(x=tgt_weak)
        self.net.update_target_center(h_shared, momentum=self.target_center_momentum)
        self.ema_net.update_target_center(h_shared, momentum=self.target_center_momentum)
        _, aux = self._forward_logits(
            self.ema_net,
            h_shared=h_shared,
            use_target_center=True,
        )
        classifier_probs = aux["class_probs"]
        node_mass = classifier_probs.unsqueeze(-1) * aux["domain_weights"]
        weight_class_probs = (
            aux["prototype_class_probs"]
            if self.consistency_target == "class_only"
            else classifier_probs
        )
        conf = weight_class_probs.max(dim=1).values.detach()
        guide = {
            "class_probs": weight_class_probs.detach(),
            "prototype_class_probs": aux["prototype_class_probs"].detach(),
            "domain_weights": aux["domain_weights"].detach(),
            "node_mass": node_mass.detach(),
        }
        return conf, guide

    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        model = self.ema_net
        model.eval()
        if isinstance(imgs, (tuple, list)):
            imgs = imgs[0]
        with torch.no_grad():
            logits, _ = self._forward_logits(
                model,
                x=imgs,
                use_target_center=True,
            )
            return logits

    @torch.no_grad()
    def _sync_relation_buffers_to_ema(self):
        for name in ["src_prototypes", "src_proto_inited", "src_domain_centers", "src_domain_center_inited"]:
            getattr(self.ema_net.relation_router, name).copy_(getattr(self.net.relation_router, name))

    @torch.no_grad()
    def _sync_classifier_initialization_to_ema(self):
        self.ema_net.adaptive_classifier.weight.copy_(self.net.adaptive_classifier.weight)

    def _build_optimizer(self):
        base_lr = float(self.config.method.lr)
        param_groups = [
            {"params": list(self.net.backbone.parameters()), "lr": base_lr * 0.1},
            {"params": list(self.net.bottleneck.parameters()), "lr": base_lr},
            {"params": list(self.net.feature_norm.parameters()), "lr": base_lr},
            {"params": list(self.net.adaptive_classifier.parameters()), "lr": base_lr},
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
            # This loader is recreated for each prototype refresh, so persistent
            # workers would outlive the temporary loader and delay process exit.
            kwargs["persistent_workers"] = False
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
                self._update_class_ambiguity(model, valid, prototypes)
                logger.info(
                    "DCPR source prototype refresh | proto_bs=%d single_pass "
                    "elapsed_min=%.2f amb_mean=%.3f amb_max=%.3f",
                    self.prototype_batch_size,
                    elapsed_minutes,
                    float(self.class_ambiguity_weights.mean().item()),
                    float(self.class_ambiguity_weights.max().item()),
                )

            model.train(was_training)

    @torch.no_grad()
    def _update_class_ambiguity(
        self,
        model: DCPRNetwork,
        valid: torch.Tensor,
        prototypes: torch.Tensor,
    ):
        weights = valid.float().unsqueeze(-1)
        class_counts = weights.sum(dim=0).clamp_min(1.0)
        class_proto = (prototypes * weights).sum(dim=0) / class_counts
        class_valid = valid.any(dim=0)
        valid_indices = class_valid.nonzero(as_tuple=False).flatten()
        ambiguity = torch.zeros(self.num_classes, device=prototypes.device)

        if valid_indices.numel() > 1:
            logits = model._adaptive_logits(class_proto[valid_indices])
            logits = logits[:, valid_indices]
            row = torch.arange(valid_indices.numel(), device=prototypes.device)
            correct = logits[row, row]
            wrong_logits = logits.clone()
            wrong_logits[row, row] = -torch.inf
            margins = correct - wrong_logits.max(dim=1).values

            # Lowest classifier margin is most ambiguous. Rank normalization
            # keeps the weighting scale stable as the class count changes.
            order = margins.argsort(descending=False, stable=True)
            ranked = torch.linspace(
                1.0,
                0.0,
                steps=valid_indices.numel(),
                device=prototypes.device,
                dtype=prototypes.dtype,
            )
            valid_ambiguity = torch.empty_like(ranked)
            valid_ambiguity[order] = ranked
            ambiguity[valid_indices] = valid_ambiguity

        self.class_ambiguity_weights.copy_(
            torch.where(class_valid, ambiguity, torch.zeros_like(ambiguity))
        )

    def _train_step(
        self,
        optimizer,
        scheduler,
        src_batch,
        tgt_batch,
        relation_ramp: float,
        ema_decay: float,
    ):
        src_imgs, src_labels, src_dom = src_batch
        tgt_weak, tgt_strong = tgt_batch if tgt_batch is not None else (None, None)

        self._zero_grad(optimizer)

        loss_rel = torch.zeros((), device=self.device, dtype=torch.float32)
        conf_tgt = torch.zeros((), device=self.device, dtype=torch.float32)

        with self._auto_cast():
            logits_src, _ = self._forward_logits(
                self.net, x=src_imgs, domain_ids=src_dom,
            )
            self._probe_amp_tensor(logits_src, "dcpr/logits_src", warn_on_float32=False)
            loss_src = self.criterion_task(logits_src, src_labels)
            loss = loss_src

            if tgt_weak is not None and tgt_strong is not None:
                with torch.no_grad():
                    with self._auto_cast():
                        conf_tgt, teacher_aux = self._teacher_guidance(tgt_weak)

                _, tgt_aux = self._forward_logits(
                    self.net,
                    x=tgt_strong,
                    use_target_center=True,
                )

                if self.lambda_relation_consistency > 0.0:
                    rel_weights = conf_tgt.pow(self.consistency_conf_power)
                    rel_weights = rel_weights * self._ambiguity_sample_weights(
                        teacher_aux["class_probs"]
                    )
                    loss_rel = self._relation_consistency_loss(
                        tgt_aux,
                        teacher_aux,
                        rel_weights,
                    )
                    loss = loss + relation_ramp * self.lambda_relation_consistency * loss_rel

        self._optimizer_step_with_optional_clip(
            loss, optimizer, clip_params=self.net.parameters(), clip_max_norm=self.grad_clip,
        )
        scheduler.step()
        self._update_ema(ema_decay)

        metrics = {
            "src": loss_src.detach().float(),
            "rel": loss_rel.detach().float(),
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
        global_step: int,
    ):
        metric_keys = ("src", "rel", "conf", "total")
        metric_sums = {key: torch.zeros((), device=self.device, dtype=torch.float32) for key in metric_keys}

        src_prefetcher, tgt_prefetcher = self._create_prefetchers(uses_target_loader)

        for _ in range(epoch_steps):
            src_batch = src_prefetcher.pop()
            tgt_batch = tgt_prefetcher.pop() if tgt_prefetcher is not None else None
            step_metrics = self._train_step(
                optimizer=optimizer, scheduler=scheduler,
                src_batch=src_batch, tgt_batch=tgt_batch,
                relation_ramp=relation_ramp,
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
            path, modules={"student": self.net, "ema": self.ema_net},
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

    def extra_training_state_dict(self) -> dict[str, Any]:
        return {
            "relation_temperature": self.relation_temperature,
        }

    def load_extra_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if "relation_temperature" in state:
            self._set_relation_temperature(float(state["relation_temperature"]))

    def _analysis_dataset_metadata(self):
        dataset = self.target_test_loader.dataset
        samples = list(getattr(dataset, "samples", []))
        class_names = list(getattr(dataset, "class_names", []))
        if not samples or not isinstance(samples[0][0], str):
            raise ValueError(
                "DCPR analysis export requires the files storage backend with image paths."
            )
        if len(class_names) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} target class names, got {len(class_names)}."
            )
        return samples, class_names

    @torch.no_grad()
    def _collect_analysis_outputs(self, model: DCPRNetwork):
        model.eval()
        labels_all = []
        class_probs_all = []
        prototype_probs_all = []
        routing_all = []
        for imgs, labels in self.target_test_loader:
            imgs = self._to_device(imgs)
            with self._auto_cast():
                _, aux = self._forward_logits(
                    model,
                    x=imgs,
                    use_target_center=True,
                )
            labels_all.append(labels.cpu())
            class_probs_all.append(aux["class_probs"].float().cpu())
            prototype_probs_all.append(aux["prototype_class_probs"].float().cpu())
            routing_all.append(aux["domain_weights"].float().cpu())
        return {
            "labels": torch.cat(labels_all),
            "class_probs": torch.cat(class_probs_all),
            "prototype_class_probs": torch.cat(prototype_probs_all),
            "routing_weights": torch.cat(routing_all),
        }

    def _load_analysis_source_model(self) -> DCPRNetwork:
        if self.analysis_source_checkpoint is None:
            raise ValueError(
                "method.analysis_export.source_checkpoint is required for ambiguous mode."
            )
        checkpoint_path = Path(str(self.analysis_source_checkpoint))
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint = self._load_checkpoint_file(checkpoint_path)
        state = checkpoint.get("ema", checkpoint.get("student", checkpoint))
        model = copy.deepcopy(self.ema_net)
        model.load_state_dict(state, strict=False)
        model.eval()
        logger.info("Loaded Source Only analysis checkpoint from %s", checkpoint_path)
        return model

    def _export_analysis(self):
        if self.analysis_mode == "none":
            return
        samples, class_names = self._analysis_dataset_metadata()
        source_domains = list(self.config.dataset.sources)
        outputs = self._collect_analysis_outputs(self.ema_net)
        if self.analysis_mode == "routing":
            manifest = _export_routing_materials(
                self.analysis_output_dir,
                samples=samples,
                class_names=class_names,
                source_domains=source_domains,
                labels=outputs["labels"],
                class_probs=outputs["class_probs"],
                prototype_class_probs=outputs["prototype_class_probs"],
                routing_weights=outputs["routing_weights"],
                num_samples=self.analysis_num_samples,
                candidate_classes=self.analysis_candidate_classes,
            )
        else:
            source_model = self._load_analysis_source_model()
            source_outputs = self._collect_analysis_outputs(source_model)
            manifest = _export_ambiguous_materials(
                self.analysis_output_dir,
                samples=samples,
                class_names=class_names,
                source_domains=source_domains,
                labels=outputs["labels"],
                source_probs=source_outputs["class_probs"],
                dcpr_probs=outputs["class_probs"],
                routing_weights=outputs["routing_weights"],
                ambiguity_weights=self.class_ambiguity_weights.detach().float().cpu(),
                num_pairs=self.analysis_num_pairs,
                samples_per_pair=self.analysis_samples_per_pair,
            )
        logger.info("DCPR analysis materials exported to %s", manifest)

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
            "DCPR Training: source_steps=%d target_steps=%d epoch_steps=%d "
            "use_target=%s lambda_rel=%.2f rel_space=%s",
            len(self.source_loader), len(self.target_loader),
            epoch_steps, str(uses_target_loader),
            self.lambda_relation_consistency, self.relation_space_mode,
        )

        for epoch in self._epoch_range(self.total_epochs):
            current_temperature = self._temperature_at_epoch(epoch + 1)
            self._set_relation_temperature(current_temperature)

            if self.refresh_source_prototypes_each_epoch or epoch == 0:
                self._recompute_source_prototypes(self.net)
                self._sync_relation_buffers_to_ema()
                if epoch == 0:
                    self._sync_classifier_initialization_to_ema()

            self.net.train()
            ramp = min(1.0, (epoch + 1) / max(1.0, self.ramp_denom))
            consistency_ramp = 1.0 if (epoch + 1) >= self.consistency_start_epoch else 0.0
            metrics, global_step = self._run_train_epoch(
                optimizer=optimizer, scheduler=scheduler,
                epoch_steps=epoch_steps, uses_target_loader=uses_target_loader,
                relation_ramp=ramp * consistency_ramp,
                global_step=global_step,
            )

            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            self._maybe_save_best(acc, epoch + 1)

            self._log_epoch_summary(
                epoch + 1, self.total_epochs,
                metrics={
                    "src": metrics["src"],
                    "rel": metrics["rel"],
                    "conf": (metrics["conf"], ".3f"),
                    "total": metrics["total"],
                },
                extras={
                    "rmp": (ramp, ".2f"),
                    "crmp": (consistency_ramp, ".2f"),
                    "tmp": (current_temperature, ".3f"),
                },
                score=acc, best_score=best_acc, score_name="Acc",
            )

        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._export_analysis()
        self._log_training_complete(best_score=best_acc, score_name="Acc")
