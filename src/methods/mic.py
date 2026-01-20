"""
MIC (Masked Image Consistency) solver for domain adaptation.

Implements teacher-student consistency training with masked images.
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from plugins import MICPlugin
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
        if hasattr(torch, 'compile') and self.device.type != 'mps':
            self.stu_model = torch.compile(self.stu_model)
            self.tea_model = torch.compile(self.tea_model)
        
        # MIC plugin for masked consistency training
        mask_ratio = self.config.method.get("mask_ratio", 0.5)
        patch_size = self.config.method.get("patch_size", 32)
        self.mic_plugin = MICPlugin(mask_ratio=mask_ratio, patch_size=patch_size).to(self.device)

    def _get_trainable_params(self):
        """Return student model parameters for optimizer."""
        return self.stu_model.parameters()

    def train(self):
        """Training loop with MIC consistency loss."""
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

            pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)

                self.optimizer.zero_grad()

                # Semantic loss on source
                src_pred = self.stu_model(src_imgs)
                sem_loss = self.criterion(src_pred, src_labels)

                # MIC consistency loss on target
                mic_loss = self.mic_plugin(self.stu_model, self.tea_model, tgt_imgs)

                # Combined loss
                loss = sem_loss + lambda_mic * mic_loss

                loss.backward()
                self.optimizer.step()

                # Update teacher model with EMA
                self._update_teacher_ema(ema_momentum)

                sem_loss_meter.update(sem_loss.item())
                mic_loss_meter.update(mic_loss.item())
                tot_loss_meter.update(loss.item())
                
                pbar.set_postfix({
                    "sem": sem_loss_meter.avg,
                    "mic": mic_loss_meter.avg,
                    "tot": tot_loss_meter.avg
                })

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

    def compute_loss(self, src_imgs, src_labels, tgt_imgs):
        """Compute combined loss (called if using default train loop)."""
        lambda_mic = self.config.method.get("lambda_mic", 0.5)
        
        src_pred = self.stu_model(src_imgs)
        sem_loss = self.criterion(src_pred, src_labels)
        mic_loss = self.mic_plugin(self.stu_model, self.tea_model, tgt_imgs)
        
        return sem_loss + lambda_mic * mic_loss

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
