import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import torch.nn.functional as F


class BaseSolver:
    def __init__(self, config, loaders):
        """
        Initialize the BaseSolver.

        Args:
            config: OmegaConf configuration object
            loaders: Tuple containing (source_loader, target_loader, target_test_loader)
        """
        self.config = config
        self.source_loader, self.target_loader, self.target_test_loader = loaders
        self.device = torch.device(config.device)

        if self.config.dataset.setting == "csda":
            self.num_classes = 31
        else:
            self.num_classes = 10

        # Build Model (Placeholder - should be overridden or configurable)
        # For base implementation, we'll use a simple ResNet-like backbone if possible,
        # or just a simple convnet for demonstration if torchvision is not preferred heavily here.
        # But usually DA uses ResNet50. Let's use a simple placeholder model.
        self.net = self.build_model().to(self.device)

        # Optimizer
        self.optimizer = self.build_optimizer()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self):
        """
        Build the network architecture.
        """
        # For simplicity in BaseSolver, let's use a torchvision resnet18
        # In a real scenario, this might come from a models/ backbone file
        from torchvision.models import resnet18, ResNet18_Weights

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace fc layer
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def build_optimizer(self):
        """
        Build the optimizer.
        """
        # Simple SGD or Adam
        lr = self.config.get("lr", 0.001)
        return optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    def train(self):
        """
        Main training loop.
        """
        max_epochs = self.config.method.epochs

        logger.info(f"Start training for {max_epochs} epochs...")

        for epoch in range(max_epochs):
            self.net.train()

            # Use zip to iterate over both domains (assuming same length or truncate)
            # In DA, we usually iterate max_iters per epoch or loop over source
            # For simplicity, we loop over source and cycle target if needed

            # Helper to cycle target loader
            def cycle(iterable):
                while True:
                    for x in iterable:
                        yield x

            tgt_iter = cycle(self.target_loader)

            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(
                    tgt_iter
                )  # We don't use target labels in training (Unsupervised DA)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)

                self.optimizer.zero_grad()

                # Forward & Loss calculation
                loss = self.compute_loss(src_imgs, src_labels, tgt_imgs)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": total_loss / num_batches})

            # Evaluation after each epoch
            acc = self.evaluate()
            logger.info(
                f"Epoch {epoch+1} finished. Avg Loss: {total_loss/num_batches:.4f}, Target Acc: {acc:.2f}%"
            )

        logger.info("Training finished.")

    def compute_loss(self, src_imgs, src_labels, tgt_imgs):
        """
        Compute the loss for a batch.
        In BaseSolver (Source Only), we only care about Source Classification Loss.
        Override this for DA methods (e.g. DANN, CDAN).
        """
        # Forward pass source
        src_logits = self.net(src_imgs)
        loss = self.criterion(src_logits, src_labels)
        return loss

    def evaluate(self):
        """
        Evaluate on target test set.
        """
        self.net.eval()
        correct = 0
        total = 0

        # H-Score or Accuracy?
        # For simple CSDA/OSDA, let's just track overall accuracy for known classes first.
        # Handling Unknowns (label == self.num_classes) depends on the setting.
        # If model outputs num_classes logits, it can't predict "Unknown" directly
        # unless we use thresholding or an extra class.
        # BaseSolver assumes Closed Set (CSDA) by default or just ignores Unknowns for simplicity unless overridden.

        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(imgs)
                _, predicted = torch.max(outputs.data, 1)

                # Basic accuracy
                # Note: If label is "unknown" (index >= num_classes), standard logic might fail
                # if we don't have that class in output.
                # Here we assume predicted is within [0, num_classes-1].

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total if total > 0 else 0
        return acc

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        logger.info(f"Model saved to {path}")
