import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import ImageFile, __version__ as PIL_VERSION

from datasets.samplers import (
    StratifiedDomainBatchSampler,
    UniformDomainBatchSampler,
)
from datasets.storage import (
    DomainDataset,
    LmdbDomainDataset,
    MultiSourceDomainDataset,
    resolve_lmdb_path,
)
from datasets.transforms import ColorSpaceToTensorStack, WeakStrongAugment
from utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_device,
    get_distributed_context,
)
from utils.config import is_truthy, resolve_auto_bool, resolve_int_or_auto

logger = logging.getLogger(__name__)

_PILLOW_RUNTIME_LOGGED = False
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _log_pillow_runtime_once():
    global _PILLOW_RUNTIME_LOGGED
    if _PILLOW_RUNTIME_LOGGED:
        return
    _PILLOW_RUNTIME_LOGGED = True
    version = str(PIL_VERSION)
    is_simd = "post" in version
    if is_simd:
        logger.info("Pillow runtime: pillow-simd detected (%s)", version)
    else:
        logger.warning(
            "Pillow runtime: standard Pillow detected (%s). "
            "For image decode/resize throughput, consider pillow-simd on x86 Linux.",
            version,
        )


def _build_loader_kwargs(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    worker_init_fn=None,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": int(max(0, num_workers)),
        "pin_memory": bool(pin_memory),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(max(1, prefetch_factor))
        if worker_init_fn is not None:
            kwargs["worker_init_fn"] = worker_init_fn
    return kwargs


class _WorkerThreadLimiter:
    def __init__(self, worker_threads: int):
        self.worker_threads = int(max(1, worker_threads))

    def __call__(self, _worker_id: int):
        worker_threads = self.worker_threads
        os.environ["OMP_NUM_THREADS"] = str(worker_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
        os.environ["MKL_NUM_THREADS"] = str(worker_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(worker_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(worker_threads)

        try:
            torch.set_num_threads(worker_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass


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
    method_name = str(getattr(config.method, "name", "")).strip().lower()

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

    _log_pillow_runtime_once()

    batch_size = int(config.batch_size)
    num_workers = int(config.num_workers)
    perf_cfg = getattr(config, "performance", None)
    dl_perf_cfg = getattr(perf_cfg, "dataloader", None) if perf_cfg is not None else None
    aug_perf_cfg = getattr(perf_cfg, "augmentation", None) if perf_cfg is not None else None
    target_tensor_v2_cfg = (
        getattr(aug_perf_cfg, "target_tensor_v2", "auto")
        if aug_perf_cfg is not None
        else "auto"
    )
    # Only pin memory when CUDA is actually the selected device.
    device_str = getattr(config, "device", "auto")
    is_cuda_device = get_device(device_str).startswith("cuda")
    distributed = get_distributed_context()
    pin_memory_cfg = getattr(perf_cfg, "pin_memory", "auto") if perf_cfg is not None else "auto"
    pin_memory = is_cuda_device if str(pin_memory_cfg).lower() == "auto" else is_truthy(pin_memory_cfg)
    non_blocking_transfer = (
        is_truthy(getattr(perf_cfg, "non_blocking_transfer", True))
        if perf_cfg is not None
        else True
    )
    if is_cuda_device and non_blocking_transfer and not pin_memory:
        logger.warning(
            "non_blocking_transfer=True but pin_memory=False. Async host->GPU copies may not be effective."
        )

    persistent_workers_default = (
        is_truthy(getattr(dl_perf_cfg, "persistent_workers", True))
        if dl_perf_cfg is not None
        else True
    )
    prefetch_factor_default = (
        int(getattr(dl_perf_cfg, "prefetch_factor", 4))
        if dl_perf_cfg is not None
        else 4
    )

    if num_workers <= 0:
        default_source_workers = 0
        default_target_workers = 0
        default_test_workers = 0
    else:
        default_source_workers = max(1, int(math.ceil(float(num_workers) / 2.0)))
        default_target_workers = max(1, int(num_workers - default_source_workers))
        default_test_workers = min(default_source_workers, 2)

    if dl_perf_cfg is not None:
        num_workers_source = resolve_int_or_auto(
            getattr(dl_perf_cfg, "num_workers_source", "auto"),
            default_source_workers,
        )
        num_workers_target = resolve_int_or_auto(
            getattr(dl_perf_cfg, "num_workers_target", "auto"),
            default_target_workers,
        )
        num_workers_test = resolve_int_or_auto(
            getattr(dl_perf_cfg, "num_workers_test", "auto"),
            default_test_workers,
        )
    else:
        num_workers_source = default_source_workers
        num_workers_target = default_target_workers
        num_workers_test = default_test_workers

    persistent_workers_source = (
        is_truthy(getattr(dl_perf_cfg, "persistent_workers_source", persistent_workers_default))
        if dl_perf_cfg is not None
        else persistent_workers_default
    )
    persistent_workers_target = (
        is_truthy(getattr(dl_perf_cfg, "persistent_workers_target", persistent_workers_default))
        if dl_perf_cfg is not None
        else persistent_workers_default
    )
    persistent_workers_test = (
        is_truthy(getattr(dl_perf_cfg, "persistent_workers_test", persistent_workers_target))
        if dl_perf_cfg is not None
        else persistent_workers_target
    )

    prefetch_factor_source = (
        int(getattr(dl_perf_cfg, "prefetch_factor_source", prefetch_factor_default))
        if dl_perf_cfg is not None
        else prefetch_factor_default
    )
    prefetch_factor_target = (
        int(getattr(dl_perf_cfg, "prefetch_factor_target", prefetch_factor_default))
        if dl_perf_cfg is not None
        else prefetch_factor_default
    )
    prefetch_factor_test = (
        int(getattr(dl_perf_cfg, "prefetch_factor_test", prefetch_factor_target))
        if dl_perf_cfg is not None
        else prefetch_factor_target
    )

    limit_worker_threads = (
        is_truthy(getattr(dl_perf_cfg, "limit_worker_threads", True))
        if dl_perf_cfg is not None
        else True
    )
    worker_threads = (
        int(getattr(dl_perf_cfg, "worker_threads", 1))
        if dl_perf_cfg is not None
        else 1
    )
    worker_init_fn = _WorkerThreadLimiter(worker_threads) if limit_worker_threads else None

    source_loader_kwargs = _build_loader_kwargs(
        num_workers=num_workers_source,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers_source,
        prefetch_factor=prefetch_factor_source,
        worker_init_fn=worker_init_fn,
    )
    target_loader_kwargs = _build_loader_kwargs(
        num_workers=num_workers_target,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers_target,
        prefetch_factor=prefetch_factor_target,
        worker_init_fn=worker_init_fn,
    )
    target_test_loader_kwargs = _build_loader_kwargs(
        num_workers=num_workers_test,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers_test,
        prefetch_factor=prefetch_factor_test,
        worker_init_fn=worker_init_fn,
    )

    storage_backend = (
        str(getattr(dl_perf_cfg, "storage_backend", "files")).strip().lower()
        if dl_perf_cfg is not None
        else "files"
    )
    if storage_backend not in {"files", "lmdb"}:
        raise ValueError(
            f"Unsupported storage_backend={storage_backend}. Expected one of: files, lmdb"
        )
    lmdb_root_cfg = getattr(dl_perf_cfg, "lmdb_root", None) if dl_perf_cfg is not None else None
    default_lmdb_root = (proj_path / "data" / "lmdb-cache").resolve()
    if lmdb_root_cfg is None or str(lmdb_root_cfg).strip().lower() in {"", "auto"}:
        lmdb_root = default_lmdb_root
    elif str(lmdb_root_cfg).strip().lower() in {"none"}:
        lmdb_root = None
    else:
        lmdb_root = Path(str(lmdb_root_cfg))
        if not lmdb_root.is_absolute():
            lmdb_root = (proj_path / lmdb_root).resolve()

    logger.info(
        "Dataloader runtime | backend=%s workers(src/tgt/test)=%d/%d/%d pin_memory=%s "
        "prefetch(src/tgt/test)=%d/%d/%d worker_threads=%s lmdb_root=%s",
        storage_backend,
        int(source_loader_kwargs["num_workers"]),
        int(target_loader_kwargs["num_workers"]),
        int(target_test_loader_kwargs["num_workers"]),
        pin_memory,
        int(source_loader_kwargs.get("prefetch_factor", 0)),
        int(target_loader_kwargs.get("prefetch_factor", 0)),
        int(target_test_loader_kwargs.get("prefetch_factor", 0)),
        worker_threads if limit_worker_threads else "off",
        str(lmdb_root) if lmdb_root is not None else "disabled",
    )

    # Determine classes
    src_classes, tgt_classes, shared_classes = get_class_splits(config)
    
    # Build class mappings for proper label handling
    src_mapping, tgt_mapping, unknown_label = build_class_mapping(
        src_classes, tgt_classes, shared_classes, setting
    )

    # Transforms
    strong_train_aug = is_truthy(getattr(config.method, "strong_train_aug", False))
    source_aug_cfg = getattr(config.method, "source_aug", None)
    target_aug_cfg = getattr(config.method, "target_aug", None)
    # Optional color-space stacking (used by `dcfm_cs`).
    color_space_cfg = getattr(config.method, "color_space", None)
    use_color_space = color_space_cfg is not None and is_truthy(getattr(color_space_cfg, "enabled", False))

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

        mean = list(getattr(color_space_cfg, "mean", IMAGENET_MEAN))
        std = list(getattr(color_space_cfg, "std", IMAGENET_STD))

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
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    transforms.RandomErasing(p=0.25),
                ]
            )
        else:
            train_transform = transforms.Compose(
                geom_train
                + [
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    geom_test = [transforms.Resize((224, 224))]
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
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    
    # Weak/strong target views for consistency-based methods.
    strong_aug_enabled = is_truthy(getattr(config.method, "strong_aug", False))
    target_transform = train_transform

    target_aug_backend = str(
        getattr(config.method, "target_aug_backend", "dataset")
    ).strip().lower()
    tensor_backend_requested = target_aug_backend == "tensor_v2"
    target_tensor_v2_auto = is_cuda_device and tensor_backend_requested
    target_tensor_v2_enabled = resolve_auto_bool(
        target_tensor_v2_cfg,
        target_tensor_v2_auto,
    )
    if target_tensor_v2_enabled and not tensor_backend_requested:
        logger.warning(
            "performance.augmentation.target_tensor_v2=True requires "
            "method.target_aug_backend=tensor_v2; falling back to dataset transforms "
            "for method=%s.",
            method_name or "<unknown>",
        )
        target_tensor_v2_enabled = False
    if target_tensor_v2_enabled and use_color_space:
        logger.warning(
            "target_tensor_v2 is incompatible with color_space.enabled=True; "
            "falling back to dataset weak/strong transforms."
        )
        target_tensor_v2_enabled = False
    if target_tensor_v2_enabled and (not strong_aug_enabled):
        logger.warning(
            "target_tensor_v2 requires method.strong_aug=True; "
            "falling back to default target transform."
        )
        target_tensor_v2_enabled = False

    if strong_aug_enabled:
        if target_tensor_v2_enabled:
            # Keep decode + deterministic resize in dataloader; apply weak/strong random ops
            # in solver on tensor path (v2, GPU-capable).
            target_transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.PILToTensor(),  # uint8 [C,H,W]
                ]
            )
            logger.info(
                "Target weak/strong augmentation: tensor path enabled (method=%s). "
                "Loader outputs uint8 tensors after resize; random weak/strong ops run in solver.",
                method_name or "<unknown>",
            )
        else:
            # Standard Weak
            if use_color_space:
                weak_aug = transforms.Compose(
                    [
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
                    [
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
                    [
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
                    [
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandAugment(
                            num_ops=target_randaugment_ops,
                            magnitude=target_randaugment_mag,
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                    ]
                )

            target_transform = WeakStrongAugment(weak_aug, strong_aug)

    logger.info(
        "Target augmentation runtime | strong_aug=%s backend=%s tensor_v2=%s color_space=%s",
        bool(strong_aug_enabled),
        target_aug_backend,
        bool(target_tensor_v2_enabled),
        bool(use_color_space),
    )

    def _build_domain_dataset(domain_path: Path, classes: List[int], transform, class_mapping: Optional[Dict[int, int]]):
        if storage_backend == "lmdb":
            lmdb_path = resolve_lmdb_path(domain_path, lmdb_root)
            return LmdbDomainDataset(
                lmdb_path,
                classes,
                transform=transform,
                class_mapping=class_mapping,
            )
        return DomainDataset(
            domain_path,
            classes,
            transform=transform,
            class_mapping=class_mapping,
        )

    # Datasets with proper class mappings
    tgt_path = root_dir / target_domain

    if setting == "msda":
        source_datasets = [
            _build_domain_dataset(p, src_classes, transform=train_transform, class_mapping=src_mapping)
            for p in src_paths
        ]
        source_dataset = MultiSourceDomainDataset(source_datasets)
    else:
        src_path = root_dir / source_domain
        source_dataset = _build_domain_dataset(
            src_path,
            src_classes,
            transform=train_transform,
            class_mapping=src_mapping,
        )
    
    # Target dataset uses special transform if enabled 
    target_dataset = _build_domain_dataset(
        tgt_path,
        tgt_classes,
        transform=target_transform,
        class_mapping=tgt_mapping,
    )
    
    target_test_dataset = _build_domain_dataset(
        tgt_path,
        tgt_classes,
        transform=test_transform,
        class_mapping=tgt_mapping,
    )

    # DataLoaders
    if setting == "msda":
        domain_sizes = [len(d) for d in source_datasets]
        steps_per_epoch = sum(domain_sizes) // batch_size
        steps_per_epoch = max(1, int(steps_per_epoch))
        stratified = getattr(config.method, "stratified_batch", False)
        sampler_cls = StratifiedDomainBatchSampler if stratified else UniformDomainBatchSampler
        batch_sampler = sampler_cls(
            domain_sizes=domain_sizes,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            drop_last=True,
            rank=distributed.rank,
            num_replicas=distributed.world_size,
            seed=int(config.get("seed", 42)),
        )
        source_loader = DataLoader(
            source_dataset,
            batch_sampler=batch_sampler,
            **source_loader_kwargs,
        )
    else:
        source_sampler = (
            DistributedSampler(
                source_dataset,
                num_replicas=distributed.world_size,
                rank=distributed.rank,
                shuffle=True,
                seed=int(config.get("seed", 42)),
                drop_last=True,
            )
            if distributed.enabled
            else None
        )
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=source_sampler is None,
            sampler=source_sampler,
            drop_last=True,
            **source_loader_kwargs,
        )
    target_sampler = (
        DistributedSampler(
            target_dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=True,
            seed=int(config.get("seed", 42)) + 1,
            drop_last=True,
        )
        if distributed.enabled
        else None
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=target_sampler is None,
        sampler=target_sampler,
        drop_last=True,
        **target_loader_kwargs,
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **target_test_loader_kwargs,
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
