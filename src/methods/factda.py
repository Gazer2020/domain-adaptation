"""
FACT-DA: Fourier Augmented Co-Teacher adapted from DG to DA.

Core adaptation:
- Keep FACT's amplitude-mix augmentation and EMA teacher co-teaching.
- Use source labels for supervised classification (original + augmented).
- Add target-domain unsupervised co-teacher consistency on original/augmented views.
"""

import logging
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import GpuLossAccumulator, cycle


logger = logging.getLogger(__name__)


def _unwrap_weak_strong_from_maybe_tuple(tgt_imgs):
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) >= 2:
        return tgt_imgs[0], tgt_imgs[1]
    if isinstance(tgt_imgs, (tuple, list)) and len(tgt_imgs) == 1:
        return tgt_imgs[0], tgt_imgs[0]
    return tgt_imgs, tgt_imgs


def _sigmoid_rampup(current: float, rampup_length: float) -> float:
    if rampup_length <= 0:
        return 1.0
    current = float(max(0.0, min(current, rampup_length)))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def _entropy_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1).clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


@register_solver("factda")
class FACTDASolver(BaseSolver):
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
        label_smoothing = float(m.get("label_smoothing", 0.0))

        stu_model = get_backbone(backbone_name)
        if hasattr(stu_model, "fc"):
            stu_model.fc = nn.Linear(stu_model.fc.in_features, self.num_classes)
        self.stu_model = stu_model.to(self.device)

        tea_model = get_backbone(backbone_name)
        if hasattr(tea_model, "fc"):
            tea_model.fc = nn.Linear(tea_model.fc.in_features, self.num_classes)
        self.tea_model = tea_model.to(self.device)
        self.tea_model.load_state_dict(self.stu_model.state_dict())
        for p in self.tea_model.parameters():
            p.requires_grad_(False)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.temperature = float(m.get("temperature", 10.0))
        self.teacher_momentum = float(m.get("teacher_momentum", 0.9995))
        self.lambda_consistency = float(m.get("lambda_consistency", 2.0))
        self.lambda_target_consistency = float(m.get("lambda_target_consistency", 1.0))
        self.lambda_target_entropy = float(m.get("lambda_target_entropy", 0.0))
        self.amplitude_mix_strength = float(m.get("amplitude_mix_strength", 0.2))
        self.consistency_warmup_epochs = float(m.get("consistency_warmup_epochs", 5.0))
        self.grad_clip = float(m.get("grad_clip", 0.0))
        self.total_epochs = int(m.get("epochs", 20))
        self.epoch_steps_mode = str(m.get("epoch_steps_mode", "max")).strip().lower()

        logger.info(
            "FACT-DA build | backbone=%s T=%.3f ema=%.6f lam_cons=%.3f lam_tgt=%.3f "
            "lam_ent=%.3f mix=%.3f warmup=%.1f",
            backbone_name,
            self.temperature,
            self.teacher_momentum,
            self.lambda_consistency,
            self.lambda_target_consistency,
            self.lambda_target_entropy,
            self.amplitude_mix_strength,
            self.consistency_warmup_epochs,
        )

    def _build_optimizer(self):
        m = self.config.method
        lr = float(m.get("lr", 1e-3))
        optimizer_name = str(m.get("optimizer", "sgd")).strip().lower()
        weight_decay = float(m.get("weight_decay", 5e-4))
        momentum = float(m.get("momentum", 0.9))
        backbone_lr_mult = float(m.get("backbone_lr_mult", 1.0))

        if hasattr(self.stu_model, "fc") and hasattr(self.stu_model.fc, "parameters"):
            head_params = list(self.stu_model.fc.parameters())
            head_param_ids = {id(p) for p in head_params}
            backbone_params = [p for p in self.stu_model.parameters() if id(p) not in head_param_ids]
            param_groups = [
                {"params": backbone_params, "lr": lr * backbone_lr_mult},
                {"params": head_params, "lr": lr},
            ]
        else:
            param_groups = [{"params": self.stu_model.parameters(), "lr": lr}]

        if optimizer_name == "adam":
            optimizer = optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=self._is_truthy(m.get("nesterov", True)),
            )
        return optimizer

    def _update_teacher_ema(self):
        with torch.no_grad():
            for ps, pt in zip(self.stu_model.parameters(), self.tea_model.parameters()):
                pt.data.mul_(self.teacher_momentum).add_((1.0 - self.teacher_momentum) * ps.data)

    def _parse_source_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("FACT-DA expects source batch to provide at least (images, labels)")
        return batch[0], batch[1]

    def _parse_target_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt_imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        return _unwrap_weak_strong_from_maybe_tuple(tgt_imgs)

    def _amplitude_mix(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.ndim != 4 or imgs.size(0) <= 1 or self.amplitude_mix_strength <= 0:
            return imgs
        bsz = imgs.size(0)
        perm = torch.randperm(bsz, device=imgs.device)
        imgs_perm = imgs[perm]

        fft_a = torch.fft.fft2(imgs, dim=(-2, -1))
        fft_b = torch.fft.fft2(imgs_perm, dim=(-2, -1))
        amp_a = torch.abs(fft_a)
        amp_b = torch.abs(fft_b)
        pha_a = torch.angle(fft_a)

        lam = torch.rand(bsz, 1, 1, 1, device=imgs.device, dtype=imgs.dtype)
        lam = lam * float(self.amplitude_mix_strength)
        amp_mix = (1.0 - lam) * amp_a + lam * amp_b
        fft_mix = torch.polar(amp_mix, pha_a)
        imgs_mix = torch.fft.ifft2(fft_mix, dim=(-2, -1)).real
        return imgs_mix.to(imgs.dtype)

    def _dual_coteacher_kl(
        self,
        logits_ori_stu: torch.Tensor,
        logits_aug_stu: torch.Tensor,
        logits_ori_tea: torch.Tensor,
        logits_aug_tea: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = max(1e-6, self.temperature)
        p_ori = torch.softmax(logits_ori_stu / t, dim=1).clamp_min(1e-8)
        p_aug = torch.softmax(logits_aug_stu / t, dim=1).clamp_min(1e-8)
        p_ori_tea = torch.softmax(logits_ori_tea.detach() / t, dim=1).clamp_min(1e-8)
        p_aug_tea = torch.softmax(logits_aug_tea.detach() / t, dim=1).clamp_min(1e-8)

        loss_a2o = F.kl_div(p_aug.log(), p_ori_tea, reduction="batchmean")
        loss_o2a = F.kl_div(p_ori.log(), p_aug_tea, reduction="batchmean")
        return loss_a2o, loss_o2a

    def train(self):
        optimizer = self._build_optimizer()
        epoch_steps = self._resolve_epoch_steps()
        lr_drop_epoch = max(1, int(round(0.8 * self.total_epochs)))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_drop_epoch], gamma=0.1)
        self.register_training_state(optimizer=optimizer, scheduler=scheduler)

        best_acc = self._best_metric

        logger.info(
            "%s training | epoch_steps_mode=%s source_steps=%d target_steps=%d epoch_steps=%d",
            self._solver_display_name(),
            self.epoch_steps_mode,
            len(self.source_loader),
            len(self.target_loader),
            epoch_steps,
        )

        for epoch in self._epoch_range(self.total_epochs):
            self._set_train_mode()
            src_iter = cycle(self.source_loader)
            tgt_iter = cycle(self.target_loader)
            ramp = _sigmoid_rampup(epoch + 1.0, self.consistency_warmup_epochs)
            beta = self.lambda_consistency * ramp

            acc_meter = GpuLossAccumulator(device=self.device)

            for _ in range(epoch_steps):
                src_batch = next(src_iter)
                tgt_batch = next(tgt_iter)
                src_imgs, src_labels = self._parse_source_batch(src_batch)
                tgt_weak, _ = self._parse_target_batch(tgt_batch)

                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)
                tgt_weak = self._to_device(tgt_weak)

                src_aug = self._amplitude_mix(src_imgs)
                tgt_aug = self._amplitude_mix(tgt_weak)

                self._zero_grad(optimizer)

                with self._auto_cast():
                    logits_src_ori = self.stu_model(src_imgs)
                    logits_src_aug = self.stu_model(src_aug)
                    loss_src = 0.5 * (
                        self.criterion(logits_src_ori, src_labels)
                        + self.criterion(logits_src_aug, src_labels)
                    )

                    logits_tgt_ori = self.stu_model(tgt_weak)
                    logits_tgt_aug = self.stu_model(tgt_aug)

                    with torch.no_grad():
                        logits_src_ori_t = self.tea_model(src_imgs)
                        logits_src_aug_t = self.tea_model(src_aug)
                        logits_tgt_ori_t = self.tea_model(tgt_weak)
                        logits_tgt_aug_t = self.tea_model(tgt_aug)

                    src_a2o, src_o2a = self._dual_coteacher_kl(
                        logits_src_ori,
                        logits_src_aug,
                        logits_src_ori_t,
                        logits_src_aug_t,
                    )
                    loss_src_cons = src_a2o + src_o2a

                    tgt_a2o, tgt_o2a = self._dual_coteacher_kl(
                        logits_tgt_ori,
                        logits_tgt_aug,
                        logits_tgt_ori_t,
                        logits_tgt_aug_t,
                    )
                    loss_tgt_cons = tgt_a2o + tgt_o2a

                    loss_tgt_ent = _entropy_minimization_loss(logits_tgt_ori)

                    total_loss = (
                        loss_src
                        + beta * (loss_src_cons + self.lambda_target_consistency * loss_tgt_cons)
                        + self.lambda_target_entropy * loss_tgt_ent
                    )

                if self.grad_clip > 0:
                    self._optimizer_step_with_optional_clip(
                        total_loss,
                        optimizer,
                        clip_params=self.stu_model.parameters(),
                        clip_max_norm=self.grad_clip,
                    )
                else:
                    self._optimizer_step_with_optional_clip(total_loss, optimizer)
                self._update_teacher_ema()

                acc_meter.update("src", loss_src)
                acc_meter.update("src_cons", loss_src_cons)
                acc_meter.update("tgt_cons", loss_tgt_cons)
                acc_meter.update("tgt_ent", loss_tgt_ent)
                acc_meter.update("total", total_loss)
                acc_meter.update("beta", beta)
                acc_meter.step()

            scheduler.step()
            acc_val = self.evaluate()
            if acc_val > best_acc:
                best_acc = acc_val
            self._maybe_save_best(acc_val, epoch + 1)
            computed = acc_meter.compute()
            self._log_epoch_summary(
                epoch + 1,
                self.total_epochs,
                metrics={
                    "src": computed.get("src", 0),
                    "src_cons": computed.get("src_cons", 0),
                    "tgt_cons": computed.get("tgt_cons", 0),
                    "tgt_ent": computed.get("tgt_ent", 0),
                    "total": computed.get("total", 0),
                },
                extras={"beta": (computed.get("beta", 0), ".3f")},
                score=acc_val,
                best_score=best_acc,
                score_name="Acc",
            )

        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")

    def _set_train_mode(self):
        self.stu_model.train()
        self.tea_model.eval()

    def _set_eval_mode(self):
        self.stu_model.eval()
        self.tea_model.eval()

    def forward_for_eval(self, imgs):
        return self.stu_model(imgs)

    def save_checkpoint(self, path):
        self._save_named_modules_checkpoint(
            path,
            modules={
                "student_model": self.stu_model,
                "teacher_model": self.tea_model,
            },
        )

    def load_checkpoint(self, path):
        self._load_named_modules_checkpoint(
            path,
            modules={
                "student_model": self.stu_model,
                "teacher_model": self.tea_model,
            },
        )
