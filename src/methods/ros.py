"""
Rotation-based self-supervised domain adaptation solver.

Implements ROS (Rotation for Open Set) which uses rotation prediction
as a pretext task for learning domain-invariant features.
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim

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
        semantic_hidden_dim = self.config.method.get("semantic_hidden_dim", 256)
        self.semantic_head = SemanticHead(
            in_features=in_features, 
            num_classes=self.num_classes,
            hidden_dim=semantic_hidden_dim
        )

        # Move to device
        self.feature_extractor.to(self.device)
        self.rotation_head.to(self.device)
        self.semantic_head.to(self.device)



    def _build_rotation_optimizer(self):
        """Build optimizer for rotation pretraining stage."""
        base_lr = self.config.method.lr
        beta1 = float(self.config.method.get("rotation_beta1", 0.9))
        beta2 = float(self.config.method.get("rotation_beta2", 0.9))
        self.rot_optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.rotation_head.parameters()),
            lr=base_lr,
            betas=(beta1, beta2),
            eps=float(self.config.method.get("rotation_eps", 1e-8)),
            weight_decay=float(self.config.method.get("rotation_weight_decay", 5e-4)),
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
            momentum=float(self.config.method.get("semantic_momentum", 0.9)),
            weight_decay=float(self.config.method.get("semantic_weight_decay", 1e-4)),
            nesterov=self._is_truthy(self.config.method.get("semantic_nesterov", False)),
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
        rot_labels = torch.randint(0, 4, (batch_size,), device=imgs.device)
        rot_imgs = torch.empty_like(imgs)
        for rotation in range(4):
            selected = rot_labels == rotation
            if bool(selected.any()):
                rot_imgs[selected] = torch.rot90(
                    imgs[selected],
                    k=rotation,
                    dims=(-2, -1),
                )
        return rot_imgs, rot_labels

    def train(self):
        """Two-stage training: rotation pretraining + semantic finetuning."""
        max_epochs = self.config.method.epochs
        logger.info(f"Start training for {max_epochs} epochs per stage...")
        best_acc = self._best_metric
        global_epoch = self._resume_epoch

        # Stage 1: Rotation pretraining
        best_acc, global_epoch = self._train_rotation_stage(max_epochs, best_acc, global_epoch)
        
        # Freeze lower layers of feature extractor
        self._freeze_lower_layers()
        
        # Stage 2: Semantic finetuning
        best_acc, _ = self._train_semantic_stage(max_epochs, best_acc, global_epoch)
        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Acc")
        self._log_training_complete(best_score=best_acc, score_name="Acc")

    def _train_rotation_stage(self, max_epochs, best_acc, global_epoch):
        """Stage 1: Train rotation prediction."""
        logger.info("Stage 1: Rotation pretraining...")
        self._build_rotation_optimizer()
        self.register_training_state(rotation_optimizer=self.rot_optimizer)

        for epoch in self._epoch_range(max_epochs):
            self.feature_extractor.train()
            self.rotation_head.train()

            loss_meter = AverageMeter()
            target_iter = cycle(self.target_loader)
            
            for src_imgs, _ in self.source_loader:
                self._zero_grad(self.rot_optimizer)

                tgt_imgs, _ = next(target_iter)
                all_imgs = torch.cat([src_imgs, tgt_imgs], dim=0)
                all_imgs = self._to_device(all_imgs)
                rot_imgs, rot_labels = self._apply_rotation(all_imgs)

                # Forward pass
                with self._auto_cast():
                    ori_feats = self.feature_extractor(all_imgs)
                    rot_feats = self.feature_extractor(rot_imgs)
                    rot_preds = self.rotation_head(ori_feats, rot_feats)

                    loss = self.criterion(rot_preds, rot_labels)
                self._optimizer_step_with_optional_clip(loss, self.rot_optimizer)

                loss_meter.update(loss.item())

            acc = self.evaluate()
            global_epoch += 1
            if acc > best_acc:
                best_acc = acc
            self._maybe_save_best(acc, global_epoch)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics={"loss": loss_meter.avg},
                score=acc,
                best_score=best_acc,
                score_name="Acc",
                prefix="ROS Rotation",
            )
        return best_acc, global_epoch

    def _freeze_lower_layers(self):
        """Freeze lower layers of feature extractor for finetuning."""
        logger.info("Freezing lower layers of feature extractor...")
        modules = list(self.feature_extractor.children())
        for i in range(min(6, len(modules))):
            for param in modules[i].parameters():
                param.requires_grad = False

    def _train_semantic_stage(self, max_epochs, best_acc, global_epoch):
        """Stage 2: Train semantic classification."""
        logger.info("Stage 2: Semantic finetuning...")
        self._build_semantic_optimizer()
        self.register_training_state(semantic_optimizer=self.sem_optimizer)

        for epoch in self._epoch_range(max_epochs, offset=max_epochs):
            self.feature_extractor.train()
            self.semantic_head.train()

            loss_meter = AverageMeter()
            
            for src_imgs, src_labels in self.source_loader:
                self._zero_grad(self.sem_optimizer)

                src_imgs = self._to_device(src_imgs)
                src_labels = self._to_device(src_labels)

                with self._auto_cast():
                    src_feats = self.feature_extractor(src_imgs)
                    sem_preds = self.semantic_head(src_feats)

                    loss = self.criterion(sem_preds, src_labels)
                self._optimizer_step_with_optional_clip(loss, self.sem_optimizer)

                loss_meter.update(loss.item())

            acc = self.evaluate()
            global_epoch += 1
            if acc > best_acc:
                best_acc = acc
            self._maybe_save_best(acc, global_epoch)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics={"loss": loss_meter.avg},
                score=acc,
                best_score=best_acc,
                score_name="Acc",
                prefix="ROS Semantic",
            )
        return best_acc, global_epoch

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
        self._save_named_modules_checkpoint(
            path,
            modules={
                "feature_extractor": self.feature_extractor,
                "rotation_head": self.rotation_head,
                "semantic_head": self.semantic_head,
            },
        )

    def load_checkpoint(self, path):
        """Load all model components from checkpoint."""
        checkpoint = self._load_checkpoint_file(path)
        
        if "feature_extractor" not in checkpoint:
            raise ValueError(f"Invalid checkpoint format: {path}")
        
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        if "semantic_head" in checkpoint:
            self.semantic_head.load_state_dict(checkpoint["semantic_head"])
        if "rotation_head" in checkpoint:
            self.rotation_head.load_state_dict(checkpoint["rotation_head"])

        logger.info("%s checkpoint loaded from %s", self._solver_display_name(), path)
