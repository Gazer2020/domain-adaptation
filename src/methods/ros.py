import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from utils import AverageMeter
from models.backbones import resnet18_backbone
from methods.base_solver import BaseSolver


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

        if self.config.dataset.setting == "csda":
            self.num_classes = self.config.dataset.num_classes
        else:
            raise NotImplementedError("RotationSolver only supports csda setting.")

        self.build_model()

        self.build_optimizer()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self):
        """
        Build the network architecture.
        """
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

    def build_optimizer(self):
        """
        Build the optimizer.
        """
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.rotation_classifier.parameters())
            + list(self.semantic_classifier.parameters()),
            lr=self.config.get("lr", 0.001),
            betas=(0.9, 0.9),
            eps=1e-08,
            weight_decay=5e-4,
        )

    def train(self):
        """
        Main training loop.
        """
        max_epochs = self.config.method.epochs

        source_rotation_loader = DataLoader(
            RotationDataset(self.source_loader.dataset),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.get("num_workers", 4)
        )
        target_rotation_loader = DataLoader(
            RotationDataset(self.target_loader.dataset),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.get("num_workers", 4),
        )

        logger.info(f"Start training for {max_epochs} epochs...")

        for epoch in range(max_epochs):
            self.feature_extractor.train()
            self.rotation_classifier.train()
            self.semantic_classifier.train()

            def cycle(iterable):
                while True:
                    for x in iterable:
                        yield x

            src_rot_iter = cycle(source_rotation_loader)
            tgt_rot_iter = cycle(target_rotation_loader)

            # meters
            total_loss_meter = AverageMeter()
            rot_loss_meter = AverageMeter()
            semantic_loss_meter = AverageMeter()

            pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                src_ori_imgs, src_rot_imgs, src_rot_labels = next(src_rot_iter)
                tgt_ori_imgs, tgt_rot_imgs, tgt_rot_labels = next(tgt_rot_iter)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)

                src_ori_imgs = src_ori_imgs.to(self.device)
                src_rot_imgs = src_rot_imgs.to(self.device)
                src_rot_labels = src_rot_labels.to(self.device)

                tgt_ori_imgs = tgt_ori_imgs.to(self.device)
                tgt_rot_imgs = tgt_rot_imgs.to(self.device)
                tgt_rot_labels = tgt_rot_labels.to(self.device)

                self.optimizer.zero_grad()

                # extract features
                src_features = self.feature_extractor(src_imgs)

                src_ori_features = self.feature_extractor(src_ori_imgs)
                src_rot_features = self.feature_extractor(src_rot_imgs)
                tgt_ori_features = self.feature_extractor(tgt_ori_imgs)
                tgt_rot_features = self.feature_extractor(tgt_rot_imgs)

                # classify rotation
                src_rot_preds = self.rotation_classifier(
                    src_ori_features, src_rot_features
                )
                tgt_rot_preds = self.rotation_classifier(
                    tgt_ori_features, tgt_rot_features
                )

                # classify semantic
                src_semantic_preds = self.semantic_classifier(src_features)

                src_rot_loss = self.criterion(src_rot_preds, src_rot_labels)
                tgt_rot_loss = self.criterion(tgt_rot_preds, tgt_rot_labels)
                semantic_loss = self.criterion(src_semantic_preds, src_labels)

                rot_loss = src_rot_loss + tgt_rot_loss
                loss = rot_loss + semantic_loss

                loss.backward()
                self.optimizer.step()

                # update meters
                total_loss_meter.update(loss.item())
                rot_loss_meter.update(rot_loss.item())
                semantic_loss_meter.update(semantic_loss.item())

                pbar.set_postfix({"loss": total_loss_meter.avg})

            # Evaluation after each epoch
            acc = self.evaluate()
            logger.info(
                f"Epoch {epoch+1} finished. "
                f"Avg Loss: {total_loss_meter.avg:.4f}, "
                f"Rot Loss: {rot_loss_meter.avg:.4f}, "
                f"Semantic Loss: {semantic_loss_meter.avg:.4f}, "
                f"Target Acc: {acc:.2f}%"
            )

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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.feature_extractor.state_dict(), path)
        torch.save(self.semantic_classifier.state_dict(), path.with_suffix('.semantic'))

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


class RotationDataset(Dataset):
    def __init__(self, dataset, num_classes=4):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = torch.randint(0, 4, (1,)).item()
        
        if label > 0:
            rot_img = torch.rot90(img, k=label, dims=[-2, -1])
        else:
            rot_img = img
            
        return (img, rot_img, label)
