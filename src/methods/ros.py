"""
Rotation-based self-supervised domain adaptation solver.

Implements ROS (Rotation for Open Set) which uses rotation prediction
as a pretext task for learning domain-invariant features.
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import RotationHead, SemanticHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


@register_solver("ros")
class RotationSolver(BaseSolver):
    """
    Rotation solver implementing two-stage training:
    1. Rotation pretraining on source + target
    2. Semantic finetuning on source
    """

    def build_model(self):
        """Build feature extractor, rotation head, and semantic classifier."""
        backbone = get_backbone(self.config.method.get("backbone", "resnet18"))
        
        # Split backbone into feature extractor and get feature dimension
        layers = list(backbone.children())
        self.feature_extractor = nn.Sequential(*(layers[:-1]), nn.Flatten())
        in_features = layers[-1].in_features
        
        # Build heads
        self.rotation_head = RotationHead(in_features=in_features, num_classes=4)
        self.semantic_head = SemanticHead(in_features=in_features, num_classes=self.num_classes)

        # Move to device
        self.feature_extractor.to(self.device)
        self.rotation_head.to(self.device)
        self.semantic_head.to(self.device)



    def _build_rotation_optimizer(self):
        """Build optimizer for rotation pretraining stage."""
        base_lr = self.config.method.lr
        self.rot_optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.rotation_head.parameters()),
            lr=base_lr,
            betas=(0.9, 0.9),
            eps=1e-08,
            weight_decay=5e-4,
        )

    def _build_semantic_optimizer(self):
        """Build optimizer for semantic finetuning stage."""
        base_lr = self.config.method.lr
        # Get feature extractor learning rate multiplier from config
        feature_lr_mult = self.config.method.get("feature_lr_mult", 0.1)
        
        params = [
            {
                "params": filter(lambda p: p.requires_grad, self.feature_extractor.parameters()),
                "lr": base_lr * feature_lr_mult,
            },
            {"params": self.semantic_head.parameters(), "lr": base_lr},
        ]
        self.sem_optimizer = optim.SGD(
            params,
            momentum=0.9,
            weight_decay=1e-4,
        )

    def _apply_rotation(self, imgs):
        """
        Apply random rotations to images.
        
        Args:
            imgs: Input images (B, C, H, W)
            
        Returns:
            Tuple of (rotated_images, rotation_labels)
            Labels: 0=0°, 1=90°, 2=180°, 3=270°
        """
        batch_size = imgs.size(0)
        rot_labels = torch.randint(0, 4, (batch_size,), device=self.device)
        rot_imgs = torch.stack([
            torch.rot90(imgs[i], k=rot_labels[i], dims=[-2, -1])
            for i in range(batch_size)
        ])
        return rot_imgs, rot_labels

    def train(self):
        """Two-stage training: rotation pretraining + semantic finetuning."""
        max_epochs = self.config.method.epochs
        logger.info(f"Start training for {max_epochs} epochs per stage...")

        # Stage 1: Rotation pretraining
        self._train_rotation_stage(max_epochs)
        
        # Freeze lower layers of feature extractor
        self._freeze_lower_layers()
        
        # Stage 2: Semantic finetuning
        self._train_semantic_stage(max_epochs)
        
        logger.info("Training finished.")

    def _train_rotation_stage(self, max_epochs):
        """Stage 1: Train rotation prediction."""
        logger.info("Stage 1: Rotation pretraining...")
        self._build_rotation_optimizer()

        for epoch in range(max_epochs):
            self.feature_extractor.train()
            self.rotation_head.train()

            loss_meter = AverageMeter()
            target_iter = cycle(self.target_loader)
            
            pbar = tqdm(self.source_loader, desc=f"Rotation {epoch+1}/{max_epochs}")
            for src_imgs, _ in pbar:
                self.rot_optimizer.zero_grad()

                tgt_imgs, _ = next(target_iter)
                all_imgs = torch.cat([src_imgs, tgt_imgs], dim=0).to(self.device)
                rot_imgs, rot_labels = self._apply_rotation(all_imgs)

                # Forward pass
                ori_feats = self.feature_extractor(all_imgs)
                rot_feats = self.feature_extractor(rot_imgs)
                rot_preds = self.rotation_head(ori_feats, rot_feats)

                loss = self.criterion(rot_preds, rot_labels)
                loss.backward()
                self.rot_optimizer.step()

                loss_meter.update(loss.item())
                pbar.set_postfix({"rot_loss": loss_meter.avg})

            acc = self.evaluate()
            logger.info(f"Rotation Epoch {epoch+1} finished. Target Acc: {acc:.2f}%")

    def _freeze_lower_layers(self):
        """Freeze lower layers of feature extractor for finetuning."""
        logger.info("Freezing lower layers of feature extractor...")
        modules = list(self.feature_extractor.children())
        for i in range(min(6, len(modules))):
            for param in modules[i].parameters():
                param.requires_grad = False

    def _train_semantic_stage(self, max_epochs):
        """Stage 2: Train semantic classification."""
        logger.info("Stage 2: Semantic finetuning...")
        self._build_semantic_optimizer()

        for epoch in range(max_epochs):
            self.feature_extractor.train()
            self.semantic_head.train()

            loss_meter = AverageMeter()
            
            pbar = tqdm(self.source_loader, desc=f"Semantic {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                self.sem_optimizer.zero_grad()

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)

                src_feats = self.feature_extractor(src_imgs)
                sem_preds = self.semantic_head(src_feats)

                loss = self.criterion(sem_preds, src_labels)
                loss.backward()
                self.sem_optimizer.step()

                loss_meter.update(loss.item())
                pbar.set_postfix({"sem_loss": loss_meter.avg})

            acc = self.evaluate()
            logger.info(f"Semantic Epoch {epoch+1} finished. Target Acc: {acc:.2f}%")

    # Note: compute_loss is not implemented - ROS uses custom train()

    def _set_train_mode(self):
        """Set all components to training mode."""
        self.feature_extractor.train()
        self.semantic_head.train()

    def _set_eval_mode(self):
        """Set all components to evaluation mode."""
        self.feature_extractor.eval()
        self.semantic_head.eval()

    def forward_for_eval(self, imgs):
        """Forward pass for evaluation using semantic classifier."""
        features = self.feature_extractor(imgs)
        return self.semantic_head(features)

    def save_checkpoint(self, path):
        """Save all model components to single checkpoint file."""
        torch.save({
            "method": "ros",
            "feature_extractor": self.feature_extractor.state_dict(),
            "rotation_head": self.rotation_head.state_dict(),
            "semantic_head": self.semantic_head.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        """Load all model components from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both old and new checkpoint formats
        if "feature_extractor" in checkpoint:
            # New format: single dict
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            if "semantic_head" in checkpoint:
                self.semantic_head.load_state_dict(checkpoint["semantic_head"])
            if "rotation_head" in checkpoint:
                self.rotation_head.load_state_dict(checkpoint["rotation_head"])
        else:
            # Old format: try loading from separate files
            from pathlib import Path
            path = Path(path)
            if path.with_suffix(".feature.pth").exists():
                self.feature_extractor.load_state_dict(
                    torch.load(path.with_suffix(".feature.pth"), map_location=self.device)
                )
            if path.with_suffix(".semantic.pth").exists():
                self.semantic_head.load_state_dict(
                    torch.load(path.with_suffix(".semantic.pth"), map_location=self.device)
                )
                
        logger.info(f"Model loaded from {path}")
