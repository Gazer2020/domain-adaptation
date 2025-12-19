import os
from pathlib import Path
from loguru import logger
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def get_class_splits(dataset_cfg):
    """
    Return shared_classes, source_private_classes, target_private_classes
    based on the configuration.
    """
    setting = dataset_cfg.setting

    assert (
        setting in dataset_cfg.splits
    ), f"Setting {setting} not found in splits configuration."

    split_cfg = dataset_cfg.splits[setting]

    src_classes = split_cfg.source
    tgt_classes = split_cfg.target
    shared_classes = sorted(list(set(src_classes) & set(tgt_classes)))

    return src_classes, tgt_classes, shared_classes


class DomainDataset(Dataset):
    def __init__(self, root: Path, classes, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = classes
        self.class_names = []
        all_classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
        for c in classes:
            self.class_names.append(all_classes[c])

        for cls_name in self.class_names:
            cls_dir = self.root / cls_name

            for file in cls_dir.iterdir():
                if self._is_valid_file(file.name):
                    self.samples.append((str(file), self.class_names.index(cls_name)))

    def _is_valid_file(self, filename):
        return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


def get_dataloader(config):
    if not hasattr(config, "dataset"):
        raise ValueError("Config must contain 'dataset' section")

    dataset_name = config.dataset.name
    proj_path = Path(__file__).resolve().parent.parent.parent
    root_dir = (proj_path / config.dataset.root).resolve()

    source_domain = config.dataset.source
    target_domain = config.dataset.target

    batch_size = config.batch_size
    num_workers = config.get("num_workers", 4)

    # Determine classes
    src_classes, tgt_classes, shared_classes = get_class_splits(config.dataset)

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    src_path = root_dir / source_domain
    tgt_path = root_dir / target_domain

    source_dataset = DomainDataset(src_path, src_classes, transform=train_transform)
    target_dataset = DomainDataset(tgt_path, tgt_classes, transform=train_transform)
    target_test_dataset = DomainDataset(tgt_path, tgt_classes, transform=test_transform)

    # DataLoaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return source_loader, target_loader, target_test_loader
