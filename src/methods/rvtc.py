"""
RVTC V2: orthodox ResNet-50 logits_task + KinematicHeadV2 (layer3→layer4) with asymmetric gradients.
Source: CE trains backbone; kin loss trains head only (detached backbone in kinematic branch).
Target: kin loss pulls backbone toward head; optional entropy on logits_task.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import ResNet50_Weights, resnet50

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.rvtc_head import KinematicHeadV2
from utils import GpuLossAccumulator, cycle

logger = logging.getLogger(__name__)

# ResNet-50: layer3 channels=1024, layer4 channels=2048
RESNET50_L3_DIM = 1024
RESNET50_L4_DIM = 2048


def _entropy_minimization(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()


class RVTCNetV2(nn.Module):
    """
    Standard ResNet-50 forward; kinematic stream uses GAP(x3), GAP(x4) with asymmetric detach (see forward).
    """

    def __init__(
        self,
        num_classes: int,
        kin_hidden: int = 512,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        body = resnet50(weights=weights)
        in_f = body.fc.in_features
        body.fc = nn.Linear(in_f, num_classes)
        self.backbone = body
        self.kinematic_head = KinematicHeadV2(
            in_dim=RESNET50_L3_DIM,
            out_dim=RESNET50_L4_DIM,
            hidden=kin_hidden,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.backbone.fc.parameters():
                p.requires_grad_(True)

    def forward(
        self,
        x: torch.Tensor,
        is_source: bool = True,
        return_kinematic: bool = True,
    ):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        pooled = self.backbone.avgpool(x4)
        feat = torch.flatten(pooled, 1)
        logits = self.backbone.fc(feat)

        if not return_kinematic:
            return logits

        v3 = self.backbone.avgpool(x3).flatten(1)
        v4 = self.backbone.avgpool(x4).flatten(1)

        v3_dir = F.normalize(v3, p=2, dim=1, eps=1e-6)
        v4_dir = F.normalize(v4, p=2, dim=1, eps=1e-6)

        if is_source:
            v4_pred = self.kinematic_head(v3_dir.detach())
            kin_loss = 1.0 - (v4_pred * v4_dir.detach()).sum(dim=1).mean()
        else:
            v4_pred = self.kinematic_head(v3_dir)
            kin_loss = 1.0 - (v4_pred * v4_dir).sum(dim=1).mean()

        return logits, kin_loss


@register_solver("rvtc")
class RVTCSolver(BaseSolver):
    """RVTC V2: CE on source logits; kinematic + optional ent; asymmetric kin via RVTCNetV2."""

    def build_model(self):
        kin_hidden = int(self.config.method.get("kin_hidden", 512))
        pretrained = self._is_truthy(self.config.method.get("pretrained", True))
        freeze_backbone = self._is_truthy(self.config.method.get("freeze_backbone", False))
        self.net = RVTCNetV2(
            num_classes=self.num_classes,
            kin_hidden=kin_hidden,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        ).to(self.device)

        self.lambda_kin = float(self.config.method.get("lambda_kin", 0.5))
        self.lambda_ent = float(self.config.method.get("lambda_ent", 0.1))
        self.kin_on_source = self._is_truthy(self.config.method.get("kin_on_source", True))
        rd_k = self.config.method.get("kin_ramp_denom", None)
        rd_e = self.config.method.get("ent_ramp_denom", None)
        max_ep = float(self.config.method.epochs)
        self.kin_ramp_denom = max(1e-8, float(rd_k) if rd_k is not None else max(0.3 * max_ep, 1.0))
        self.ent_ramp_denom = max(1e-8, float(rd_e) if rd_e is not None else max(0.3 * max_ep, 1.0))
        self.label_smoothing = float(self.config.method.get("label_smoothing", 0.0))

        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.net(imgs, is_source=False, return_kinematic=False)

    @staticmethod
    def _unpack_source_batch(batch):
        if len(batch) == 3:
            return batch[0], batch[1]
        if len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(f"Unexpected batch length {len(batch)}")

    def _trainable_parameters(self):
        return [p for p in self.net.parameters() if p.requires_grad]

    def train(self):
        max_epochs = int(self.config.method.epochs)
        lr = float(self.config.method.lr)
        optimizer_name = str(self.config.method.get("optimizer", "adam")).lower()
        weight_decay = float(self.config.method.get("weight_decay", 1e-4))
        backbone_lr_mult = float(self.config.method.get("backbone_lr_mult", 0.1))

        freeze_backbone = self._is_truthy(self.config.method.get("freeze_backbone", False))
        fc_params = list(self.net.backbone.fc.parameters())
        kin_params = list(self.net.kinematic_head.parameters())
        backbone_non_fc = [
            p
            for n, p in self.net.backbone.named_parameters()
            if not n.startswith("fc.") and p.requires_grad
        ]
        if freeze_backbone:
            param_groups = [{"params": fc_params + kin_params, "lr": lr}]
        else:
            param_groups = [
                {"params": backbone_non_fc, "lr": lr * backbone_lr_mult},
                {"params": fc_params, "lr": lr},
                {"params": kin_params, "lr": lr},
            ]
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True,
            )
        else:
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)

        total_iters = max(1, max_epochs * len(self.source_loader))

        def lr_lambda(step: int) -> float:
            progress = step / max(1, total_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self.register_training_state(optimizer=optimizer, scheduler=scheduler)

        best_acc = self._best_metric

        global_step = self._training_global_step
        logger.info(
            f"RVTC V2: λ_kin={self.lambda_kin} λ_ent={self.lambda_ent} kin_on_source={self.kin_on_source} "
            f"freeze_backbone={freeze_backbone} | epochs={max_epochs}"
        )

        for epoch in self._epoch_range(max_epochs):
            self.net.train()
            acc_meter = GpuLossAccumulator(device=self.device)
            tgt_iter = cycle(self.target_loader)
            w_kin = min(1.0, (epoch + 1) / self.kin_ramp_denom) * self.lambda_kin
            w_ent = min(1.0, (epoch + 1) / self.ent_ramp_denom) * self.lambda_ent

            for batch in self.source_loader:
                src_imgs, src_labels = self._unpack_source_batch(batch)
                tgt_imgs, _ = next(tgt_iter)
                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)
                tgt_imgs = self._to_device(tgt_imgs)

                self._zero_grad(optimizer)

                kin_t_val = 0.0
                kin_s_val = 0.0
                ent_t_val = 0.0

                with self._auto_cast():
                    logits_s, kin_s = self.net(src_imgs, is_source=True, return_kinematic=True)
                    logits_t, kin_t = self.net(tgt_imgs, is_source=False, return_kinematic=True)

                    loss_task = self.criterion_task(logits_s, src_labels)
                    loss = loss_task

                    if w_kin > 0:
                        loss = loss + w_kin * kin_t
                        kin_t_val = kin_t.detach()
                        if self.kin_on_source:
                            loss = loss + w_kin * kin_s
                            kin_s_val = kin_s.detach()

                    if w_ent > 0:
                        ent_t = _entropy_minimization(logits_t)
                        loss = loss + w_ent * ent_t
                        ent_t_val = ent_t.detach()

                self._optimizer_step_with_optional_clip(
                    loss,
                    optimizer,
                    clip_params=self._trainable_parameters(),
                    clip_max_norm=5.0,
                )
                scheduler.step()
                global_step += 1

                acc_meter.update("kin_t", kin_t_val)
                acc_meter.update("kin_s", kin_s_val)
                acc_meter.update("ent", ent_t_val)
                acc_meter.update("task", loss_task)
                acc_meter.update("total", loss)
                acc_meter.step()

            acc_val = self.evaluate()
            if acc_val > best_acc:
                best_acc = acc_val
            self._maybe_save_best(acc_val, epoch + 1)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics=acc_meter.compute(),
                extras={
                    "w_kin": w_kin,
                    "w_ent": w_ent,
                },
                score=acc_val,
                best_score=best_acc,
                score_name="Acc",
            )

        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")
