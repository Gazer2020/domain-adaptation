from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from utils import get_device


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_auto_bool(value, auto_value: bool) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return auto_value
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _class_sort_key(name: str):
    """
    Sort class folder names with numeric awareness.

    Example:
      "0","1","2",...,"10" -> numeric order instead of lexicographic order.
    """
    s = str(name)
    if s.isdigit():
        return (0, int(s))
    return (1, s.lower())


class ColorSpaceToTensorStack:
    """
    Convert an input PIL RGB image into multiple target color spaces, producing
    a stacked tensor suitable for multi-view inference/training.

    Output shape: [K, 3, H, W] where K=len(spaces)
    """

    def __init__(
        self,
        spaces: List[str],
        mean: List[float],
        std: List[float],
        random_erasing_p: float = 0.0,
    ):
        if not spaces:
            raise ValueError("spaces must be a non-empty list")

        self.spaces = [str(s).lower() for s in spaces]
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.random_erasing = (
            transforms.RandomErasing(p=float(random_erasing_p)) if random_erasing_p > 0 else None
        )

    def _convert(self, img: Image.Image, space: str) -> Image.Image:
        # PIL color modes: LAB, HSV, YCbCr, YUV are supported as 8-bit images.
        space = space.lower()
        if space == "rgb":
            return img.convert("RGB")
        if space == "lab":
            return img.convert("LAB")
        if space == "hsv":
            return img.convert("HSV")
        if space == "ycbcr":
            return img.convert("YCbCr")
        if space == "yuv":
            # PIL may not support "YUV" conversion mode reliably across versions.
            # Convert RGB->YUV manually (BT.601-like), then keep 3 channels as RGB mode.
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
            r = rgb[:, :, 0]
            g = rgb[:, :, 1]
            b = rgb[:, :, 2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
            v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0
            yuv = np.stack([y, u, v], axis=-1)
            yuv = np.clip(yuv, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(yuv, mode="RGB")
        if space == "gray":
            # Keep 3 channels so downstream code always sees [3,H,W].
            return img.convert("L").convert("RGB")
        raise ValueError(f"Unsupported color space: {space}")

    def __call__(self, img: Image.Image) -> torch.Tensor:
        views: List[torch.Tensor] = []
        for s in self.spaces:
            img_cs = self._convert(img, s)
            x = self.to_tensor(img_cs)  # [3,H,W], float in [0,1]
            x = self.normalize(x)
            if self.random_erasing is not None:
                # RandomErasing operates on [C,H,W] tensors.
                x = self.random_erasing(x)
            views.append(x)
        return torch.stack(views, dim=0)


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
    if setting in ("csda", "msda"):
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
        
        all_classes = sorted([p.name for p in root.iterdir() if p.is_dir()], key=_class_sort_key)
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


class TightCropByWhiteThreshold:
    """Crop an image to its non-white foreground bounding box."""

    def __init__(self, white_threshold: int = 245, padding: int = 2, min_foreground_pixels: int = 10):
        self.white_threshold = int(white_threshold)
        self.padding = int(padding)
        self.min_foreground_pixels = int(min_foreground_pixels)

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb)
        foreground = np.any(arr < self.white_threshold, axis=2)

        ys, xs = np.where(foreground)
        if len(xs) < self.min_foreground_pixels:
            return rgb

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        if self.padding > 0:
            x0 = max(0, x0 - self.padding)
            y0 = max(0, y0 - self.padding)
            x1 = min(arr.shape[1] - 1, x1 + self.padding)
            y1 = min(arr.shape[0] - 1, y1 + self.padding)

        return rgb.crop((x0, y0, x1 + 1, y1 + 1))


class RandomApplyTransform:
    """Apply a transform with probability p."""

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p >= 1.0 or torch.rand(1).item() < self.p:
            return self.transform(image)
        return image


class MultiSourceDomainDataset(Dataset):
    """
    Wrap multiple DomainDataset objects and return (img, label, domain_id).

    This dataset is meant for MSDA training where each source domain has an
    explicit domain id in [0..S-1].
    """

    def __init__(self, datasets: List[DomainDataset]):
        if not datasets:
            raise ValueError("datasets must be a non-empty list")
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._offsets = [0]
        for n in self._lengths[:-1]:
            self._offsets.append(self._offsets[-1] + n)
        self._total = sum(self._lengths)

    def __len__(self):
        return self._total

    def __getitem__(self, index):
        if index < 0:
            index = self._total + index
        if index < 0 or index >= self._total:
            raise IndexError("index out of range")
        # Find which dataset this index falls into
        # (linear scan is fine for small S; S is typically <= 4)
        for dom_id, (off, n) in enumerate(zip(self._offsets, self._lengths)):
            if off <= index < off + n:
                img, label = self.datasets[dom_id][index - off]
                return img, label, dom_id
        raise RuntimeError("Failed to map index to a dataset")


class _UniformDomainBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Batch sampler that yields batches with uniform domain contribution.

    Implementation detail:
    - Domains are visited in a shuffled round-robin order (uniform over steps).
    - Within each domain, indices are sampled without replacement via a random permutation,
      and when exhausted, the domain reshuffles and continues.
    """

    def __init__(self, domain_sizes: List[int], batch_size: int, steps_per_epoch: int, drop_last: bool = True):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if any(n <= 0 for n in domain_sizes):
            raise ValueError("All domains must have at least 1 sample")
        self.domain_sizes = domain_sizes
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.drop_last = bool(drop_last)

        # Precompute offsets into the concatenated MultiSourceDomainDataset index space
        self.offsets = [0]
        for n in domain_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + n)

    def __iter__(self):
        g = torch.Generator()
        # Ensure different per-epoch shuffles. Without this, a fresh Generator()
        # can repeat identical permutations every epoch.
        g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        num_domains = len(self.domain_sizes)
        # Per-domain cursors into a shuffled permutation
        perms = [torch.randperm(n, generator=g).tolist() for n in self.domain_sizes]
        cursors = [0 for _ in range(num_domains)]

        domain_order = torch.randperm(num_domains, generator=g).tolist()
        for step in range(self.steps_per_epoch):
            dom = domain_order[step % num_domains]
            n = self.domain_sizes[dom]
            off = self.offsets[dom]
            cur = cursors[dom]

            # Refill permutation if not enough for a full batch
            if cur + self.batch_size > n:
                perms[dom] = torch.randperm(n, generator=g).tolist()
                cur = 0

            batch_local = perms[dom][cur : cur + self.batch_size]
            cursors[dom] = cur + self.batch_size
            yield [off + i for i in batch_local]

    def __len__(self):
        return self.steps_per_epoch


class _StratifiedDomainBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Batch sampler that mixes samples from ALL source domains in every batch.

    Each batch allocates batch_size // num_domains samples per domain (with
    remainder distributed to the first domains). This gives the model exposure
    to all domains in every gradient step and enables cross-source interactions.
    """

    def __init__(self, domain_sizes: List[int], batch_size: int, steps_per_epoch: int, drop_last: bool = True):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if any(n <= 0 for n in domain_sizes):
            raise ValueError("All domains must have at least 1 sample")
        self.domain_sizes = domain_sizes
        self.num_domains = len(domain_sizes)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.drop_last = bool(drop_last)

        self.offsets = [0]
        for n in domain_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + n)

        per_dom = self.batch_size // self.num_domains
        remainder = self.batch_size % self.num_domains
        self.per_domain_counts = [per_dom + (1 if d < remainder else 0) for d in range(self.num_domains)]

    def __iter__(self):
        g = torch.Generator()
        # Ensure different per-epoch shuffles. Without this, a fresh Generator()
        # can repeat identical permutations every epoch.
        g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        perms = [torch.randperm(n, generator=g).tolist() for n in self.domain_sizes]
        cursors = [0] * self.num_domains

        for _ in range(self.steps_per_epoch):
            batch = []
            for dom in range(self.num_domains):
                need = self.per_domain_counts[dom]
                n = self.domain_sizes[dom]
                off = self.offsets[dom]
                cur = cursors[dom]

                if cur + need > n:
                    perms[dom] = torch.randperm(n, generator=g).tolist()
                    cur = 0

                batch.extend(off + perms[dom][j] for j in range(cur, cur + need))
                cursors[dom] = cur + need
            yield batch

    def __len__(self):
        return self.steps_per_epoch


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

    setting = config.method.setting

    source_domain = getattr(config.dataset, "source", None)
    source_domains = getattr(config.dataset, "sources", None)
    target_domain = config.dataset.target
    dataset_name_lower = str(dataset_name).strip().lower()
    target_domain_lower = str(target_domain).strip().lower().replace("_", " ")

    if setting == "msda":
        if source_domains is None:
            raise ValueError("For setting=msda, config.dataset.sources must be a non-empty list of source domains")
        try:
            source_domains = list(source_domains)
        except Exception as e:
            raise ValueError(f"For setting=msda, dataset.sources must be list-like; got: {type(source_domains)}") from e
        if len(source_domains) == 0:
            raise ValueError("For setting=msda, config.dataset.sources must be a non-empty list of source domains")
        if len(set(source_domains)) != len(source_domains):
            raise ValueError(f"Duplicate entries found in dataset.sources: {source_domains}")
    else:
        if source_domain is None:
            raise ValueError("Config must contain dataset.source for non-msda settings")
    
    # Validate domain directories exist
    tgt_path = root_dir / target_domain
    
    if setting == "msda":
        src_paths = [root_dir / d for d in source_domains]
        for p in src_paths:
            if not p.exists():
                raise FileNotFoundError(
                    f"Source domain directory not found: {p}\n"
                    f"Available domains: {[d.name for d in root_dir.iterdir() if d.is_dir()]}"
                )
    else:
        src_path = root_dir / source_domain
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
    # Only pin memory when CUDA is actually the selected device.
    device_str = getattr(config, "device", "auto")
    pin_memory = get_device(device_str) == "cuda"

    # Determine classes
    src_classes, tgt_classes, shared_classes = get_class_splits(config)
    
    # Build class mappings for proper label handling
    src_mapping, tgt_mapping, unknown_label = build_class_mapping(
        src_classes, tgt_classes, shared_classes, setting
    )

    # Transforms
    strong_train_aug = getattr(config.method, "strong_train_aug", False)
    source_aug_cfg = getattr(config.method, "source_aug", None)
    target_aug_cfg = getattr(config.method, "target_aug", None)
    clipart_focus_cfg = getattr(config.method, "clipart_focus", None)
    is_officehome_clipart_target = dataset_name_lower == "office-home" and target_domain_lower == "clipart"

    clipart_train_pre = []
    clipart_eval_pre = []
    if clipart_focus_cfg is not None:
        auto_enable = bool(is_officehome_clipart_target)
        clipart_focus_enabled = _resolve_auto_bool(getattr(clipart_focus_cfg, "enabled", False), auto_enable)
        if clipart_focus_enabled and is_officehome_clipart_target:
            cropper = TightCropByWhiteThreshold(
                white_threshold=int(getattr(clipart_focus_cfg, "white_threshold", 245)),
                padding=int(getattr(clipart_focus_cfg, "bbox_padding", 2)),
                min_foreground_pixels=int(getattr(clipart_focus_cfg, "min_foreground_pixels", 10)),
            )
            if _is_truthy(getattr(clipart_focus_cfg, "apply_on_train", True)):
                train_prob = float(getattr(clipart_focus_cfg, "train_prob", 0.8))
                clipart_train_pre.append(RandomApplyTransform(cropper, p=train_prob))
            if _is_truthy(getattr(clipart_focus_cfg, "apply_on_eval", True)):
                clipart_eval_pre.append(cropper)

    # Optional color-space stacking (used by `dcfm_cs`).
    color_space_cfg = getattr(config.method, "color_space", None)
    use_color_space = color_space_cfg is not None and bool(getattr(color_space_cfg, "enabled", False))

    if use_color_space:
        mode = str(getattr(color_space_cfg, "mode", "multi")).lower()
        if mode == "single":
            spaces = [str(getattr(color_space_cfg, "single", "rgb")).lower()]
        else:
            spaces = [
                str(s).lower()
                for s in list(
                    getattr(
                        color_space_cfg,
                        "spaces",
                        ["rgb", "lab", "hsv", "ycbcr", "yuv"],
                    )
                )
            ]

        mean = list(getattr(color_space_cfg, "mean", [0.485, 0.456, 0.406]))
        std = list(getattr(color_space_cfg, "std", [0.229, 0.224, 0.225]))

    if strong_train_aug:
        jitter_brightness = float(getattr(source_aug_cfg, "brightness", 0.4)) if source_aug_cfg is not None else 0.4
        jitter_contrast = float(getattr(source_aug_cfg, "contrast", 0.4)) if source_aug_cfg is not None else 0.4
        jitter_saturation = float(getattr(source_aug_cfg, "saturation", 0.4)) if source_aug_cfg is not None else 0.4
        jitter_hue = float(getattr(source_aug_cfg, "hue", 0.1)) if source_aug_cfg is not None else 0.1
        grayscale_p = float(getattr(source_aug_cfg, "random_grayscale_p", 0.1)) if source_aug_cfg is not None else 0.1
        geom_train = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
            ),
            transforms.RandomGrayscale(p=grayscale_p),
        ]
        random_erasing_p = (
            float(getattr(source_aug_cfg, "random_erasing_p", 0.25))
            if source_aug_cfg is not None
            else 0.25
        )
    else:
        geom_train = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        random_erasing_p = 0.0

    if use_color_space:
        train_color_stack = ColorSpaceToTensorStack(
            spaces=spaces,
            mean=mean,
            std=std,
            random_erasing_p=random_erasing_p,
        )
        train_transform = transforms.Compose(geom_train + [train_color_stack])
    else:
        if strong_train_aug:
            train_transform = transforms.Compose(
                geom_train
                + [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.25),
                ]
            )
        else:
            train_transform = transforms.Compose(
                geom_train
                + [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    geom_test = clipart_eval_pre + [transforms.Resize((224, 224))]
    if use_color_space:
        test_color_stack = ColorSpaceToTensorStack(
            spaces=spaces, mean=mean, std=std, random_erasing_p=0.0
        )
        test_transform = transforms.Compose(geom_test + [test_color_stack])
    else:
        test_transform = transforms.Compose(
            geom_test
            + [
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
        if use_color_space:
            weak_aug = transforms.Compose(
                clipart_train_pre
                + [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ColorSpaceToTensorStack(
                        spaces=spaces, mean=mean, std=std, random_erasing_p=0.0
                    ),
                ]
            )
        else:
            weak_aug = transforms.Compose(
                clipart_train_pre
                + [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        
        # Strong (RandAugment)
        if use_color_space:
            target_randaugment_ops = (
                int(getattr(target_aug_cfg, "randaugment_num_ops", 2))
                if target_aug_cfg is not None
                else 2
            )
            target_randaugment_mag = (
                int(getattr(target_aug_cfg, "randaugment_magnitude", 10))
                if target_aug_cfg is not None
                else 10
            )
            strong_aug = transforms.Compose(
                clipart_train_pre
                + [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(
                        num_ops=target_randaugment_ops,
                        magnitude=target_randaugment_mag,
                    ),
                    ColorSpaceToTensorStack(
                        spaces=spaces, mean=mean, std=std, random_erasing_p=0.0
                    ),
                ]
            )
        else:
            target_randaugment_ops = (
                int(getattr(target_aug_cfg, "randaugment_num_ops", 2))
                if target_aug_cfg is not None
                else 2
            )
            target_randaugment_mag = (
                int(getattr(target_aug_cfg, "randaugment_magnitude", 10))
                if target_aug_cfg is not None
                else 10
            )
            strong_aug = transforms.Compose(
                clipart_train_pre
                + [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(
                        num_ops=target_randaugment_ops,
                        magnitude=target_randaugment_mag,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        
        target_transform = WeakStrongAugment(weak_aug, strong_aug)
    elif clipart_train_pre:
        if use_color_space:
            target_transform = transforms.Compose(
                clipart_train_pre
                + geom_train
                + [
                    ColorSpaceToTensorStack(
                        spaces=spaces,
                        mean=mean,
                        std=std,
                        random_erasing_p=random_erasing_p,
                    )
                ]
            )
        else:
            target_tail = [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            if strong_train_aug:
                target_tail.append(transforms.RandomErasing(p=0.25))
            target_transform = transforms.Compose(clipart_train_pre + geom_train + target_tail)

    # Datasets with proper class mappings
    tgt_path = root_dir / target_domain

    if setting == "msda":
        source_datasets = [
            DomainDataset(p, src_classes, transform=train_transform, class_mapping=src_mapping) for p in src_paths
        ]
        source_dataset = MultiSourceDomainDataset(source_datasets)
    else:
        src_path = root_dir / source_domain
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
    if setting == "msda":
        domain_sizes = [len(d) for d in source_datasets]
        steps_per_epoch = sum(domain_sizes) // batch_size
        steps_per_epoch = max(1, int(steps_per_epoch))
        stratified = getattr(config.method, "stratified_batch", False)
        sampler_cls = _StratifiedDomainBatchSampler if stratified else _UniformDomainBatchSampler
        batch_sampler = sampler_cls(
            domain_sizes=domain_sizes,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            drop_last=True,
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
        )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
