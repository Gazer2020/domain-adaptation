"""
Base solver class for domain adaptation methods.

All domain adaptation methods should inherit from BaseSolver and implement
the required abstract methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbones import get_backbone
from utils import AverageMeter, cycle, get_device

from methods.registry import register_solver


logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """
    Abstract base solver class for domain adaptation.
    
    Subclasses must implement:
        - build_model(): Setup model architecture
        - compute_loss(): Compute training loss
        
    Subclasses may optionally override:
        - build_optimizer(): Customize optimizer
        - forward_for_eval(): Customize inference logic
        - train(): Completely customize training loop
    """

    def __init__(self, config, loaders: Tuple[DataLoader, DataLoader, DataLoader], 
                 class_info: dict = None):
        """
        Initialize the solver with config and data loaders.

        Args:
            config: OmegaConf configuration object
            loaders: Tuple of (source_loader, target_loader, target_test_loader)
            class_info: Dict containing class metadata for OSDA handling:
                - src_classes: List of source class indices
                - tgt_classes: List of target class indices
                - shared_classes: List of shared class indices
                - num_classes: Number of classifier output classes
                - unknown_label: Label for unknown classes (None for CSDA)
                - setting: DA setting string
        """
        self.config = config
        self.source_loader, self.target_loader, self.target_test_loader = loaders
        
        # Store class info for OSDA handling
        self.class_info = class_info if class_info else {}
        
        # Setup device (auto-detect if needed)
        device_str = get_device(config.device)
        self.device = torch.device(device_str)
        logger.info(f"Using device: {self.device}")
        
        # Setup number of classes based on setting
        self._setup_num_classes()
        
        # Build model (must be implemented by subclass)
        self.build_model()
        
        # Build optimizer (can be overridden)
        self.build_optimizer()
        
        # Default loss function
        self.criterion = nn.CrossEntropyLoss()

    def _setup_num_classes(self):
        """
        Setup number of classes based on class_info or config.
        
        For OSDA/UniDA: num_classes = len(src_classes) + 1 (includes unknown class)
        For CSDA/PDA: num_classes = len(src_classes)
        
        Subclasses can override this method to use custom num_classes logic.
        """
        if self.class_info and "num_classes" in self.class_info:
            # Use precomputed num_classes from loader
            base_num_classes = self.class_info["num_classes"]
            self.unknown_label = self.class_info.get("unknown_label")
            self.shared_classes = self.class_info.get("shared_classes", [])
            self.setting = self.class_info.get("setting", "csda")
            
            # For OSDA/UniDA, add 1 class for unknown
            if self.setting in ["osda", "unida"] and self.unknown_label is not None:
                self.num_classes = base_num_classes + 1
            else:
                self.num_classes = base_num_classes
        else:
            # Fallback for backward compatibility
            self.setting = self.config.method.get("setting", "csda")
            self.num_classes = self.config.dataset.num_classes
            self.unknown_label = None
            self.shared_classes = []
        
        # Unknown rejection threshold (for confidence-based rejection)
        self.unknown_threshold = self.config.method.get("unknown_threshold", 0.5)

    @abstractmethod
    def build_model(self):
        """
        Build the network architecture.
        
        Must be implemented by subclasses.
        Should set self.net or appropriate model attributes.
        """
        pass

    def build_optimizer(self):
        """
        Build the optimizer.
        
        Default implementation uses SGD. Override for custom optimizers.
        """
        lr = self.config.method.lr
        self.optimizer = optim.SGD(
            self._get_trainable_params(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )

    def _get_trainable_params(self):
        """
        Get trainable parameters for the optimizer.
        
        Override this if your model has multiple components.
        """
        if hasattr(self, 'net'):
            return self.net.parameters()
        raise NotImplementedError(
            "Subclass must either set self.net or override _get_trainable_params()"
        )

    def train(self):
        """
        Main training loop.
        
        Default implementation trains for max_epochs using source and target data.
        Override for custom training procedures (e.g., multi-stage training).
        """
        max_epochs = self.config.method.epochs

        logger.info(f"Start training for {max_epochs} epochs...")

        for epoch in range(max_epochs):
            self._set_train_mode()
            
            tgt_iter = cycle(self.target_loader)
            loss_meter = AverageMeter()

            pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)

                self.optimizer.zero_grad()
                
                loss = self.compute_loss(src_imgs, src_labels, tgt_imgs)
                
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item())
                pbar.set_postfix({"loss": loss_meter.avg})

            # Evaluation after each epoch
            acc = self.evaluate()
            logger.info(
                f"Epoch {epoch+1} finished. Avg Loss: {loss_meter.avg:.4f}, Target Acc: {acc:.2f}%"
            )

        logger.info("Training finished.")

    def _set_train_mode(self):
        """Set model to training mode. Override for multi-component models."""
        if hasattr(self, 'net'):
            self.net.train()

    def _set_eval_mode(self):
        """Set model to evaluation mode. Override for multi-component models."""
        if hasattr(self, 'net'):
            self.net.eval()

    @abstractmethod
    def compute_loss(self, src_imgs, src_labels, tgt_imgs):
        """
        Compute the loss for a batch.
        
        Args:
            src_imgs: Source domain images
            src_labels: Source domain labels
            tgt_imgs: Target domain images (labels typically not used in UDA)
            
        Returns:
            loss: The computed loss tensor
        """
        pass

    def forward_for_eval(self, imgs):
        """
        Forward pass for evaluation.
        
        Override this if your model has a different inference path.
        
        Args:
            imgs: Input images
            
        Returns:
            outputs: Model outputs (logits)
        """
        if hasattr(self, 'net'):
            return self.net(imgs)
        raise NotImplementedError(
            "Subclass must either set self.net or override forward_for_eval()"
        )

    def evaluate(self):
        """
        Evaluate on target test set.
        
        For OSDA/UniDA settings, computes:
        - Known Accuracy (OS*): Accuracy on shared classes
        - Unknown Accuracy: Rate of predicting unknown for target-private classes
        - H-score: Harmonic mean of known and unknown accuracy
        
        For CSDA, computes standard accuracy.
        
        Returns:
            acc: Overall accuracy (or H-score for OSDA)
        """
        self._set_eval_mode()
        
        all_preds = []
        all_labels = []
        all_probs = []  # For confidence-based rejection

        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                outputs = self.forward_for_eval(imgs)
                
                # Get predictions and confidence
                probs = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, dim=1)
                
                all_preds.append(predicted.cpu())
                all_labels.append(labels)
                all_probs.append(max_probs.cpu())
        
        # Concatenate all batches (more efficient than list + tensor conversion)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        # Check if we're in OSDA/UniDA mode
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            return self._compute_osda_metrics(all_preds, all_labels, all_probs)
        else:
            # Standard accuracy for CSDA/PDA
            correct = (all_preds == all_labels).sum().item()
            total = len(all_labels)
            acc = 100 * correct / total if total > 0 else 0
            return acc
    
    def _compute_osda_metrics(self, preds, labels, probs):
        """
        Compute OSDA metrics: Known Accuracy, Unknown Accuracy, and H-score.
        
        Uses confidence-based rejection: samples with max_prob < unknown_threshold
        are classified as unknown.
        
        Args:
            preds: Predicted labels (tensor) [N]
            labels: Ground truth labels (tensor) [N]
            probs: Maximum prediction probabilities (tensor) [N]
            
        Returns:
            hscore: H-score (harmonic mean of known and unknown accuracy)
        """
        unknown_label = self.unknown_label
        
        # Apply confidence-based rejection
        # Predictions with low confidence are classified as unknown
        rejected_mask = probs < self.unknown_threshold
        preds_with_rejection = preds.clone()
        preds_with_rejection[rejected_mask] = unknown_label
        
        # Separate known and unknown samples in ground truth
        known_mask = labels != unknown_label
        unknown_mask = labels == unknown_label
        
        # Known accuracy (OS*): accuracy on shared classes
        if known_mask.sum() > 0:
            known_preds = preds_with_rejection[known_mask]
            known_labels = labels[known_mask]
            known_correct = (known_preds == known_labels).sum().item()
            known_total = known_mask.sum().item()
            known_acc = known_correct / known_total
        else:
            known_acc = 0.0
        
        # Unknown accuracy: rate of correctly predicting unknown for unknown samples
        if unknown_mask.sum() > 0:
            unknown_preds = preds_with_rejection[unknown_mask]
            # Correct if predicted as unknown (== unknown_label)
            unknown_correct = (unknown_preds == unknown_label).sum().item()
            unknown_total = unknown_mask.sum().item()
            unknown_acc = unknown_correct / unknown_total
        else:
            unknown_acc = 0.0
        
        # H-score: harmonic mean
        if known_acc + unknown_acc > 0:
            hscore = 2 * known_acc * unknown_acc / (known_acc + unknown_acc)
        else:
            hscore = 0.0
        
        # Log detailed metrics
        logger.info(
            f"OSDA Metrics - Known Acc: {100*known_acc:.2f}%, "
            f"Unknown Acc: {100*unknown_acc:.2f}%, H-score: {100*hscore:.2f}%, "
            f"Rejection threshold: {self.unknown_threshold:.3f}"
        )
        
        return 100 * hscore

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Override if you have multiple components to save.
        """
        if hasattr(self, 'net'):
            # Save with metadata
            torch.save({
                "method": "base",
                "model": self.net.state_dict(),
            }, path)
            logger.info(f"Model saved to {path}")
        else:
            raise NotImplementedError(
                "Subclass must either set self.net or override save_checkpoint()"
            )

    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        
        Override if you have multiple components to load.
        """
        if hasattr(self, 'net'):
            checkpoint = torch.load(path, map_location=self.device)
            
            # Handle both old and new formats
            if "model" in checkpoint:
                self.net.load_state_dict(checkpoint["model"])
            else:
                # Old format: just state dict
                self.net.load_state_dict(checkpoint)
                
            logger.info(f"Model loaded from {path}")
        else:
            raise NotImplementedError(
                "Subclass must either set self.net or override load_checkpoint()"
            )


@register_solver("sourceonly")
class SourceOnlySolver(BaseSolver):
    """
    Source-only baseline solver.
    
    Trains only on source domain data without any domain adaptation.
    Useful as a baseline for comparison.
    """

    def build_model(self):
        """Build a simple classification network."""
        backbone = get_backbone(self.config.method.get("backbone", "resnet18"))
        
        # Replace the final FC layer
        if hasattr(backbone, 'fc'):
            backbone.fc = nn.Linear(backbone.fc.in_features, self.num_classes)
        
        self.net = backbone.to(self.device)

    def compute_loss(self, src_imgs, src_labels, tgt_imgs):
        """Compute source classification loss only."""
        src_logits = self.net(src_imgs)
        loss = self.criterion(src_logits, src_labels)
        return loss
