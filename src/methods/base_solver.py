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

    def __init__(self, config, loaders: Tuple[DataLoader, DataLoader, DataLoader]):
        """
        Initialize the solver with config and data loaders.

        Args:
            config: OmegaConf configuration object
            loaders: Tuple of (source_loader, target_loader, target_test_loader)
        """
        self.config = config
        self.source_loader, self.target_loader, self.target_test_loader = loaders
        
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
        """Setup number of classes based on the DA setting."""
        setting = self.config.method.setting
        if setting == "csda":
            self.num_classes = self.config.dataset.num_classes
        else:
            # For other settings (osda, pda, unida), use shared classes count
            # This can be overridden by subclasses for specific needs
            self.num_classes = self.config.dataset.num_classes

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
        
        Returns:
            acc: Accuracy percentage
        """
        self._set_eval_mode()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.forward_for_eval(imgs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total if total > 0 else 0
        return acc

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Override if you have multiple components to save.
        """
        if hasattr(self, 'net'):
            torch.save(self.net.state_dict(), path)
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
            self.net.load_state_dict(torch.load(path, map_location=self.device))
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
