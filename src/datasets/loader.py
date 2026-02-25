from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def get_class_splits(config):
    """
    Return src_classes, tgt_classes, shared_classes based on the configuration.
    
    Returns:
        src_classes: List of class indices present in source domain
        tgt_classes: List of class indices present in target domain  
        shared_classes: Sorted list of class indices common to both domains
    """
    setting = config.method.setting

    assert (
        setting in config.dataset.splits
    ), f"Setting {setting} not found in splits configuration."

    split_cfg = config.dataset.splits[setting]

    src_classes = split_cfg.source
    tgt_classes = split_cfg.target
    shared_classes = sorted(list(set(src_classes) & set(tgt_classes)))

    return src_classes, tgt_classes, shared_classes


def build_class_mapping(src_classes: List[int], tgt_classes: List[int], 
                        shared_classes: List[int], setting: str) -> Tuple[Dict[int, int], Dict[int, int], Optional[int]]:
    """
    Build class mappings for source and target domains.
    
    For OSDA/UniDA, we need to:
    1. Map source classes to consecutive labels [0, num_src_classes)
    2. Map shared target classes to same labels as source
    3. Map target-private classes to unknown_label = len(src_classes)
    
    Note: unknown_label is NOT included in the base num_classes count.
    The BaseSolver will add +1 to num_classes for OSDA/UniDA settings.
    
    Args:
        src_classes: List of original source class indices
        tgt_classes: List of original target class indices
        shared_classes: List of classes common to both domains
        setting: DA setting (csda, osda, pda, unida)
        
    Returns:
        src_mapping: Dict mapping original source class -> new label
        tgt_mapping: Dict mapping original target class -> new label (or unknown)
        unknown_label: Label for unknown classes (None for CSDA)
    """
    if setting == "csda":
        # CSDA: all classes are shared, use original indices
        mapping = {c: i for i, c in enumerate(sorted(src_classes))}
        return mapping, mapping, None
    
    # For OSDA/PDA/UniDA: map to source class space
    src_mapping = {c: i for i, c in enumerate(sorted(src_classes))}
    
    # Unknown label is the index after all source classes
    # This will be mapped to class index num_classes-1 after BaseSolver adds +1
    unknown_label = len(src_classes)
    
    # Target mapping: shared classes use same labels, private classes get unknown_label
    tgt_mapping = {}
    for c in tgt_classes:
        if c in shared_classes:
            # Use the same label as source for shared classes
            tgt_mapping[c] = src_mapping[c]
        else:
            # Target-private class -> unknown
            tgt_mapping[c] = unknown_label
    
    return src_mapping, tgt_mapping, unknown_label


class DomainDataset(Dataset):
    """
    Dataset for domain adaptation with support for class mapping.
    
    Args:
        root: Path to domain directory containing class subdirectories
        classes: List of original class indices to include
        transform: Image transforms to apply
        class_mapping: Dict mapping original class index -> new label
    """
    
    def __init__(self, root: Path, classes: List[int], transform=None,
                 class_mapping: Optional[Dict[int, int]] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = classes
        self.class_mapping = class_mapping
        self.class_names = []
        
        all_classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
        for c in classes:
            self.class_names.append(all_classes[c])

        for idx, orig_class in enumerate(classes):
            cls_name = self.class_names[idx]
            cls_dir = self.root / cls_name
            
            # Determine label
            if class_mapping is not None:
                label = class_mapping[orig_class]
            else:
                label = idx  # Fallback to sequential indexing

            for file in cls_dir.iterdir():
                if self._is_valid_file(file.name):
                    self.samples.append((str(file), label))

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
    """
    Create data loaders for domain adaptation.
    
    Returns:
        Tuple of (source_loader, target_loader, target_test_loader, class_info)
        
        class_info is a dict containing:
        - src_classes: List of original source class indices
        - tgt_classes: List of original target class indices
        - shared_classes: List of shared class indices
        - num_classes: Base number of classes (= len(src_classes), NOT including unknown)
        - unknown_label: Label for unknown classes (= len(src_classes) for OSDA)
        - setting: DA setting string
        
        Note: For OSDA/UniDA, BaseSolver will add +1 to num_classes to account for unknown.
    """
    if not hasattr(config, "dataset"):
        raise ValueError("Config must contain 'dataset' section")

    dataset_name = config.dataset.name
    proj_path = Path(__file__).resolve().parent.parent.parent
    root_dir = (proj_path / config.dataset.root).resolve()
    
    # Validate data directory exists
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Dataset root directory not found: {root_dir}\n"
            f"Please check your config file and ensure the data is downloaded."
        )

    source_domain = config.dataset.source
    target_domain = config.dataset.target
    
    # Validate domain directories exist
    src_path = root_dir / source_domain
    tgt_path = root_dir / target_domain
    
    if not src_path.exists():
        raise FileNotFoundError(
            f"Source domain directory not found: {src_path}\n"
            f"Available domains: {[d.name for d in root_dir.iterdir() if d.is_dir()]}"
        )
    if not tgt_path.exists():
        raise FileNotFoundError(
            f"Target domain directory not found: {tgt_path}\n"
            f"Available domains: {[d.name for d in root_dir.iterdir() if d.is_dir()]}"
        )

    batch_size = config.batch_size
    num_workers = config.num_workers

    # Determine classes
    src_classes, tgt_classes, shared_classes = get_class_splits(config)
    setting = config.method.setting
    
    # Build class mappings for proper label handling
    src_mapping, tgt_mapping, unknown_label = build_class_mapping(
        src_classes, tgt_classes, shared_classes, setting
    )

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
    
    # Strong Augmentation for DGA-Revamp
    strong_aug_enabled = getattr(config.method, "strong_aug", False)
    target_transform = train_transform
    
    if strong_aug_enabled:
        class WeakStrongAugment:
            def __init__(self, weak, strong):
                self.weak = weak
                self.strong = strong
            
            def __call__(self, x):
                return self.weak(x), self.strong(x)
        
        # Standard Weak
        weak_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Strong (RandAugment)
        strong_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        target_transform = WeakStrongAugment(weak_aug, strong_aug)

    # Datasets with proper class mappings
    src_path = root_dir / source_domain
    tgt_path = root_dir / target_domain

    source_dataset = DomainDataset(
        src_path, src_classes, transform=train_transform, class_mapping=src_mapping
    )
    
    # Target dataset uses special transform if enabled 
    target_dataset = DomainDataset(
        tgt_path, tgt_classes, transform=target_transform, class_mapping=tgt_mapping
    )
    
    target_test_dataset = DomainDataset(
        tgt_path, tgt_classes, transform=test_transform, class_mapping=tgt_mapping
    )

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
    
    # Class info for evaluation
    # Note: num_classes is the BASE count (source classes only)
    # BaseSolver will add +1 for OSDA/UniDA to account for unknown class
    class_info = {
        "src_classes": src_classes,
        "tgt_classes": tgt_classes,
        "shared_classes": shared_classes,
        "num_classes": len(src_classes),  # Base count, excluding unknown
        "unknown_label": unknown_label,
        "setting": setting,
    }

    return source_loader, target_loader, target_test_loader, class_info
