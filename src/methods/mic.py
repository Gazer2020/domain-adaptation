"""
MIC (Masked Image Consistency) solver for domain adaptation.

Implements teacher-student consistency training with masked images.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


@register_solver("mic")
class MICSolver(BaseSolver):
    """
    Masked Image Consistency (MIC) solver with teacher-student framework.
    
    Uses a teacher model (EMA) to generate pseudo-labels on full images,
    while the student learns to predict on masked images.
    """

    def build_model(self):
        """Build student and teacher models."""
        backbone_name = self.config.method.get("backbone", "resnet50")
        
        # Student model
        stu_model = get_backbone(backbone_name)
        if hasattr(stu_model, 'fc'):
            stu_model.fc = nn.Linear(stu_model.fc.in_features, self.num_classes)
        self.stu_model = stu_model.to(self.device)
        
        # Teacher model (EMA of student)
        tea_model = get_backbone(backbone_name)
        if hasattr(tea_model, 'fc'):
            tea_model.fc = nn.Linear(tea_model.fc.in_features, self.num_classes)
        self.tea_model = tea_model.to(self.device)
        
        # Compile models if available (PyTorch 2.0+)
        # Note: torch.compile is disabled on MPS due to backward pass issues
        enable_compile = bool(self.config.method.get("enable_compile", True))
        if enable_compile and hasattr(torch, 'compile') and self.device.type != 'mps':
            try:
                self.stu_model = torch.compile(self.stu_model)
                self.tea_model = torch.compile(self.tea_model)
            except Exception as e:
                logger.warning(f"torch.compile is unavailable for MIC in this runtime, fallback to eager mode: {e}")

        # MIC masking config
        self.mask_ratio = float(self.config.method.get("mask_ratio", 0.5))
        self.patch_size = int(self.config.method.get("patch_size", 32))
        self.apply_same_mask_to_batch = bool(self.config.method.get("apply_to_batch", True))

    def _generate_mask(self, images: torch.Tensor) -> torch.Tensor:
        """
        Generate a patch-wise binary mask resized to image resolution.

        Returns mask with shape [B, 1, H, W], where 0=masked and 1=kept.
        """
        bsz, _, height, width = images.shape
        h_patches = max(height // self.patch_size, 1)
        w_patches = max(width // self.patch_size, 1)
        num_patches = h_patches * w_patches

        num_keep = int(num_patches * (1.0 - self.mask_ratio))
        num_keep = max(0, min(num_keep, num_patches))

        if self.apply_same_mask_to_batch:
            noise = torch.rand(1, num_patches, device=images.device).repeat(bsz, 1)
        else:
            noise = torch.rand(bsz, num_patches, device=images.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros((bsz, num_patches), device=images.device)
        mask.scatter_(1, ids_shuffle[:, :num_keep], 1.0)
        mask = mask.view(bsz, 1, h_patches, w_patches)
        mask = F.interpolate(mask, size=(height, width), mode="nearest")
        return mask

    def _compute_mic_loss(self, target_images: torch.Tensor) -> torch.Tensor:
        """
        MIC consistency loss:
        teacher predicts pseudo labels on full images,
        student predicts on masked images.
        """
        with torch.no_grad():
            with self._auto_cast():
                teacher_logits = self.tea_model(target_images)
                pseudo_labels = torch.argmax(torch.softmax(teacher_logits, dim=1), dim=1)

        mask = self._generate_mask(target_images)
        masked_images = target_images * mask
        with self._auto_cast():
            student_logits = self.stu_model(masked_images)
            return F.cross_entropy(student_logits, pseudo_labels)

    def _get_trainable_params(self):
        """Return student model parameters for optimizer."""
        return self.stu_model.parameters()

    def _build_optimizer(self):
        """Build optimizer for training."""
        self.optimizer = optim.SGD(
            self._get_trainable_params(),
            lr=self.config.method.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

    def train(self):
        """Training loop with MIC consistency loss."""
        self._build_optimizer()
        
        max_epochs = self.config.method.epochs
        lambda_mic = self.config.method.get("lambda_mic", 0.5)
        ema_momentum = self.config.method.get("momentum", 0.999)

        logger.info(f"Start MIC training for {max_epochs} epochs...")

        for epoch in range(max_epochs):
            self._set_train_mode()

            tgt_iter = cycle(self.target_loader)
            sem_loss_meter = AverageMeter()
            mic_loss_meter = AverageMeter()
            tot_loss_meter = AverageMeter()

            for src_imgs, src_labels in self.source_loader:
                tgt_imgs, _ = next(tgt_iter)

                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)
                tgt_imgs = self._to_device(tgt_imgs)

                self._zero_grad(self.optimizer)

                with self._auto_cast():
                    # Semantic loss on source
                    src_pred = self.stu_model(src_imgs)
                    sem_loss = self.criterion(src_pred, src_labels)

                    # MIC consistency loss on target
                    mic_loss = self._compute_mic_loss(tgt_imgs)

                    # Combined loss
                    loss = sem_loss + lambda_mic * mic_loss

                self._optimizer_step_with_optional_clip(loss, self.optimizer)

                # Update teacher model with EMA
                self._update_teacher_ema(ema_momentum)

                sem_loss_meter.update(sem_loss.item())
                mic_loss_meter.update(mic_loss.item())
                tot_loss_meter.update(loss.item())

            acc = self.evaluate()
            logger.info(
                f"Epoch {epoch+1} finished. Loss: {tot_loss_meter.avg:.4f}, Target Acc: {acc:.2f}%"
            )

        logger.info("Training finished.")

    def _update_teacher_ema(self, momentum: float):
        """Update teacher model with exponential moving average of student."""
        with torch.no_grad():
            for param_s, param_t in zip(
                self.stu_model.parameters(), self.tea_model.parameters()
            ):
                param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)

    def _set_train_mode(self):
        """Set both models to training mode."""
        self.stu_model.train()
        self.tea_model.train()

    def _set_eval_mode(self):
        """Set student model to evaluation mode."""
        self.stu_model.eval()

    def forward_for_eval(self, imgs):
        """Use student model for evaluation."""
        return self.stu_model(imgs)

    def save_checkpoint(self, path):
        """Save student and teacher models to single checkpoint file."""
        torch.save({
            "method": "mic",
            "student_model": self.stu_model.state_dict(),
            "teacher_model": self.tea_model.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """Load student and teacher models from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both old and new checkpoint formats
        if "student_model" in checkpoint:
            self.stu_model.load_state_dict(checkpoint["student_model"])
            self.tea_model.load_state_dict(checkpoint["teacher_model"])
        else:
            # Old format: just model state dict
            self.stu_model.load_state_dict(checkpoint)
            self.tea_model.load_state_dict(checkpoint)
            
        logger.info(f"Model loaded from {path}")
