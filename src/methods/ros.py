import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import AverageMeter
from models.backbones import get_resnet18
from methods.base_solver import BaseSolver


logger = logging.getLogger(__name__)


class RotationSolver(BaseSolver):
    """
    Rotation solver class implementing source only training with a rotation self supervised work.
    """

    def __init__(self, config, loaders):
        """
        Initialize the RotationSolver.

        Args:
            config: OmegaConf configuration object
            loaders: Tuple containing (source_loader, target_loader, target_test_loader)
        """
        self.config = config
        self.source_loader, self.target_loader, self.target_test_loader = loaders
        self.device = torch.device(config.device)

        if self.config.method.setting == "csda":
            self.num_classes = self.config.dataset.num_classes
        else:
            raise NotImplementedError("RotationSolver only supports csda setting.")

        self.build_model()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self):
        """
        Build the network architecture.
        """
        resnet18_backbone = get_resnet18()
        layers = list(resnet18_backbone.children())
        self.feature_extractor = nn.Sequential(*(layers[:-1]), nn.Flatten())
        in_features = layers[-1].in_features
        self.rotation_classifier = RotationModel(in_features=in_features, num_classes=4)
        self.semantic_classifier = SemanticModel(
            in_features=in_features, num_classes=self.num_classes
        )

        self.feature_extractor.to(self.device)
        self.rotation_classifier.to(self.device)
        self.semantic_classifier.to(self.device)

    def build_rotation_optimizer(self):
        """
        Build the optimizer for rotation classification.
        """
        base_lr = self.config.method.lr
        # pretrain optimizer for rotation
        self.rot_optimizer = optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.rotation_classifier.parameters()),
            lr=base_lr,
            betas=(0.9, 0.9),
            eps=1e-08,
            weight_decay=5e-4,
        )

    def build_semantic_optimizer(self):
        """
        Build the optimizer for semantic classification.
        """
        base_lr = self.config.method.lr
        # finetune optimizer for semantic classification
        params = [
            {
                "params": filter(
                    lambda p: p.requires_grad, self.feature_extractor.parameters()
                ),
                "lr": base_lr * 0.1,
            },
            {"params": self.semantic_classifier.parameters(), "lr": base_lr},
        ]
        self.sem_optimizer = optim.SGD(
            params,
            momentum=0.9,
            weight_decay=1e-4,
        )

    def rotation(self, imgs):
        """
        Apply random rotation to images and return rotated images and their labels.
        Labels: 0 - 0 degree, 1 - 90 degrees, 2 - 180 degrees, 3 - 270 degrees
        """
        batch_size = imgs.size(0)

        rot_labels = torch.randint(0, 4, (batch_size,), device=self.device)
        rot_imgs = torch.stack(
            [
                torch.rot90(imgs[i], k=rot_labels[i], dims=[-2, -1])  # type: ignore
                for i in range(batch_size)
            ]
        )

        return rot_imgs, rot_labels

    def train(self):
        """
        Main training loop.
        """
        max_epochs = self.config.method.epochs

        logger.info(f"Start training for {max_epochs} epochs...")

        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        self.build_rotation_optimizer()
        # stage 1
        logger.info("Start rotation training...")
        for epoch in range(max_epochs):
            self.feature_extractor.train()
            self.rotation_classifier.train()

            # meters
            rot_loss_meter = AverageMeter()

            target_iter = cycle(self.target_loader)
            pbar = tqdm(
                self.source_loader, desc=f"Rotation Epoch {epoch+1}/{max_epochs}"
            )
            for src_imgs, _ in pbar:
                self.rot_optimizer.zero_grad()

                tgt_imgs, _ = next(target_iter)

                ori_imgs = torch.cat([src_imgs, tgt_imgs], dim=0).to(self.device)
                rot_imgs, rot_labels = self.rotation(ori_imgs)

                ori_feats = self.feature_extractor(ori_imgs)
                rot_feats = self.feature_extractor(rot_imgs)

                rot_preds = self.rotation_classifier(ori_feats, rot_feats)

                loss_rot = self.criterion(rot_preds, rot_labels)
                loss_rot.backward()
                self.rot_optimizer.step()

                rot_loss_meter.update(loss_rot.item())

                pbar.set_postfix({"Rot Loss": rot_loss_meter.avg})

            acc = self.evaluate()
            logger.info(f"Rotation Epoch {epoch+1} finished. Target Acc: {acc:.2f}%")

        # freeze lower part of feature extractor
        logger.info("Freeze lower part of feature extractor...")
        modules = list(self.feature_extractor.children())
        for i in range(6):
            for param in modules[i].parameters():
                param.requires_grad = False

        self.build_semantic_optimizer()

        # stage 2
        logger.info("Start semantic training...")
        for epoch in range(max_epochs):
            self.feature_extractor.train()
            self.semantic_classifier.train()

            sem_loss_meter = AverageMeter()

            pbar = tqdm(
                self.source_loader, desc=f"Semantic Epoch {epoch+1}/{max_epochs}"
            )

            for src_imgs, src_labels in pbar:
                self.sem_optimizer.zero_grad()

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)

                src_feats = self.feature_extractor(src_imgs)
                sem_preds = self.semantic_classifier(src_feats)

                loss_sem = self.criterion(sem_preds, src_labels)
                loss_sem.backward()
                self.sem_optimizer.step()

                sem_loss_meter.update(loss_sem.item())

                pbar.set_postfix({"Sem Loss": sem_loss_meter.avg})

            # Evaluation after each epoch
            acc = self.evaluate()
            logger.info(f"Semantic Epoch {epoch+1} finished. Target Acc: {acc:.2f}%")

        logger.info("Training finished.")

    def evaluate(self):
        """
        Evaluate on target test set.
        """
        self.feature_extractor.eval()
        self.semantic_classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                features = self.feature_extractor(imgs)
                outputs = self.semantic_classifier(features)
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
        torch.save(self.feature_extractor.state_dict(), path.with_suffix(".feature.pth"))
        torch.save(self.semantic_classifier.state_dict(), path.with_suffix(".semantic.pth"))

        logger.info(f"Model saved to {path}")


class RotationModel(nn.Module):
    def __init__(self, in_features, num_classes=4):
        """
        number of classes should be 4 * number of semantic classes for source,
        and 4 for target.
        """
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, feat1, feat2):
        x = torch.cat((feat1, feat2), dim=1)
        return self.classifier(x)


class SemanticModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, feat):
        return self.classifier(feat)
