import io
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile, __version__ as PIL_VERSION
import numpy as np
from utils import IMAGENET_MEAN, IMAGENET_STD, get_device
from utils.config import is_truthy, resolve_auto_bool, resolve_int_or_auto

logger = logging.getLogger(__name__)

_PILLOW_RUNTIME_LOGGED = False
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


class LmdbDomainDataset(Dataset):
    """
    LMDB-backed dataset with the same return contract as DomainDataset.

    Each LMDB sample value is expected to be a pickled tuple:
      (orig_class_index: int, image_bytes: bytes)

    Metadata key `__meta__` stores:
      - length: int
      - class_names: List[str]
      - indices_by_class: Dict[int, List[int]] (recommended)
    """

    _META_KEY = b"__meta__"
    _ENV_CACHE = {}

    def __init__(
        self,
        lmdb_path: Path,
        classes: List[int],
        transform=None,
        class_mapping: Optional[Dict[int, int]] = None,
    ):
        self.lmdb_path = Path(lmdb_path)
        self.transform = transform
        self.classes = list(classes)
        self.class_mapping = class_mapping
        self._env = None

        if not self.lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB path not found: {self.lmdb_path}. "
                "Build it first or set performance.dataloader.storage_backend=files."
            )

        meta = self._read_meta()
        self.class_names = list(meta.get("class_names", []))
        self.length = int(meta.get("length", 0))
        if len(self.class_names) == 0:
            raise ValueError(f"Invalid LMDB metadata in {self.lmdb_path}: missing class_names")

        for c in self.classes:
            if c < 0 or c >= len(self.class_names):
                raise ValueError(
                    f"Class index {c} is out of range [0, {len(self.class_names)-1}] in LMDB {self.lmdb_path}"
                )

        self.samples: List[Tuple[int, int]] = []
        local_label_by_orig: Dict[int, int] = {}
        for idx, orig_class in enumerate(self.classes):
            if self.class_mapping is not None:
                local_label_by_orig[orig_class] = int(self.class_mapping[orig_class])
            else:
                local_label_by_orig[orig_class] = idx

        indices_by_class = meta.get("indices_by_class", None)
        if isinstance(indices_by_class, dict):
            for orig_class in self.classes:
                class_indices = indices_by_class.get(orig_class, indices_by_class.get(str(orig_class), []))
                mapped_label = local_label_by_orig[orig_class]
                for sample_idx in class_indices:
                    self.samples.append((int(sample_idx), mapped_label))
        else:
            # Fallback for older LMDBs without indices_by_class metadata.
            env = self._open_env()
            with env.begin(write=False) as txn:
                for sample_idx in range(self.length):
                    key = f"{sample_idx:08d}".encode("ascii")
                    packed = txn.get(key)
                    if packed is None:
                        continue
                    orig_class, _ = pickle.loads(packed)
                    orig_class = int(orig_class)
                    if orig_class in local_label_by_orig:
                        self.samples.append((sample_idx, local_label_by_orig[orig_class]))

    def _open_env(self):
        if self._env is not None:
            return self._env
        cache_key = str(self.lmdb_path.resolve())
        cached_env = self._ENV_CACHE.get(cache_key, None)
        if cached_env is not None:
            self._env = cached_env
            return self._env
        try:
            import lmdb
        except ImportError as e:
            raise ImportError(
                "LMDB backend requested but `lmdb` is not installed. "
                "Install dependency `lmdb` or set performance.dataloader.storage_backend=files."
            ) from e

        self._env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=512,
            subdir=self.lmdb_path.is_dir(),
        )
        self._ENV_CACHE[cache_key] = self._env
        return self._env

    def _read_meta(self) -> Dict[str, object]:
        env = self._open_env()
        with env.begin(write=False) as txn:
            raw = txn.get(self._META_KEY)
            if raw is None:
                raise ValueError(f"LMDB {self.lmdb_path} is missing metadata key '__meta__'")
            meta = pickle.loads(raw)
            if not isinstance(meta, dict):
                raise ValueError(f"LMDB metadata at {self.lmdb_path} must be a dict")
            return meta

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None

    def __getitem__(self, index):
        sample_idx, label = self.samples[index]
        env = self._open_env()
        key = f"{sample_idx:08d}".encode("ascii")
        with env.begin(write=False) as txn:
            packed = txn.get(key)
        if packed is None:
            raise IndexError(f"Missing sample key {sample_idx} in LMDB {self.lmdb_path}")

        _, img_bytes = pickle.loads(packed)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


def _resolve_lmdb_path(domain_path: Path, lmdb_root: Optional[Path]) -> Path:
    if lmdb_root is None:
        return domain_path.with_suffix(".lmdb")
    return (lmdb_root / f"{domain_path.name}.lmdb").resolve()


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
    is_cuda_device = get_device(device_str) == "cuda"
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

    target_tensor_v2_methods = {"dcpr", "dcpr_alt"}
    target_tensor_v2_auto = is_cuda_device and method_name in target_tensor_v2_methods
    target_tensor_v2_enabled = resolve_auto_bool(target_tensor_v2_cfg, target_tensor_v2_auto)
    if target_tensor_v2_enabled and method_name not in target_tensor_v2_methods:
        logger.warning(
            "performance.augmentation.target_tensor_v2=True is currently wired for methods=%s only; "
            "falling back to dataset weak/strong transforms for method=%s.",
            sorted(target_tensor_v2_methods),
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
            class WeakStrongAugment:
                def __init__(self, weak, strong):
                    self.weak = weak
                    self.strong = strong

                def __call__(self, x):
                    return self.weak(x), self.strong(x)

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
        "Target augmentation runtime | strong_aug=%s tensor_v2=%s color_space=%s",
        bool(strong_aug_enabled),
        bool(target_tensor_v2_enabled),
        bool(use_color_space),
    )

    def _build_domain_dataset(domain_path: Path, classes: List[int], transform, class_mapping: Optional[Dict[int, int]]):
        if storage_backend == "lmdb":
            lmdb_path = _resolve_lmdb_path(domain_path, lmdb_root)
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
            **source_loader_kwargs,
        )
    else:
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **source_loader_kwargs,
        )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
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
