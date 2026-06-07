"""Dataset storage backends and process-local LMDB environment ownership."""

from __future__ import annotations

import io
import os
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


def class_sort_key(name: str):
    """Sort numeric class directory names numerically before text names."""
    value = str(name)
    if value.isdigit():
        return (0, int(value))
    return (1, value.lower())


class DomainDataset(Dataset):
    """Filesystem-backed domain dataset with explicit class mapping."""

    def __init__(
        self,
        root: Path,
        classes: List[int],
        transform=None,
        class_mapping: Optional[Dict[int, int]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = classes
        self.class_mapping = class_mapping
        self.class_names = []

        all_classes = sorted(
            [path.name for path in self.root.iterdir() if path.is_dir()],
            key=class_sort_key,
        )
        for class_index in classes:
            self.class_names.append(all_classes[class_index])

        for local_index, original_class in enumerate(classes):
            class_dir = self.root / self.class_names[local_index]
            label = (
                class_mapping[original_class]
                if class_mapping is not None
                else local_index
            )
            for file in class_dir.iterdir():
                if self._is_valid_file(file.name):
                    self.samples.append((str(file), label))

    @staticmethod
    def _is_valid_file(filename):
        return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)


class LmdbEnvironmentManager:
    """Own shared readonly LMDB environments within one process."""

    _lock = threading.RLock()
    _pid = os.getpid()
    _entries: dict[str, tuple[object, int]] = {}

    @classmethod
    def _reset_after_fork(cls) -> None:
        current_pid = os.getpid()
        if current_pid == cls._pid:
            return
        cls._entries = {}
        cls._pid = current_pid

    @classmethod
    def acquire(cls, path: Path):
        cache_key = str(path.resolve())
        with cls._lock:
            cls._reset_after_fork()
            cached = cls._entries.get(cache_key)
            if cached is not None:
                env, references = cached
                cls._entries[cache_key] = (env, references + 1)
                return cache_key, env

            try:
                import lmdb
            except ImportError as exc:
                raise ImportError(
                    "LMDB backend requested but `lmdb` is not installed. "
                    "Install dependency `lmdb` or set "
                    "performance.dataloader.storage_backend=files."
                ) from exc

            env = lmdb.open(
                str(path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=512,
                subdir=path.is_dir(),
            )
            cls._entries[cache_key] = (env, 1)
            return cache_key, env

    @classmethod
    def release(cls, cache_key: str | None) -> None:
        if cache_key is None:
            return
        with cls._lock:
            cls._reset_after_fork()
            cached = cls._entries.get(cache_key)
            if cached is None:
                return
            env, references = cached
            if references > 1:
                cls._entries[cache_key] = (env, references - 1)
                return
            cls._entries.pop(cache_key, None)
            env.close()

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            cls._reset_after_fork()
            entries = list(cls._entries.values())
            cls._entries.clear()
        for env, _references in entries:
            env.close()

    @classmethod
    def snapshot(cls) -> dict[str, int]:
        with cls._lock:
            cls._reset_after_fork()
            return {
                path: references
                for path, (_env, references) in cls._entries.items()
            }


class LmdbDomainDataset(Dataset):
    """LMDB-backed dataset with the same return contract as DomainDataset."""

    _META_KEY = b"__meta__"

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
        self._env_key: str | None = None

        if not self.lmdb_path.exists():
            raise FileNotFoundError(
                f"LMDB path not found: {self.lmdb_path}. "
                "Build it first or set performance.dataloader.storage_backend=files."
            )

        meta = self._read_meta()
        self.class_names = list(meta.get("class_names", []))
        self.length = int(meta.get("length", 0))
        if not self.class_names:
            raise ValueError(
                f"Invalid LMDB metadata in {self.lmdb_path}: missing class_names"
            )

        for class_index in self.classes:
            if class_index < 0 or class_index >= len(self.class_names):
                raise ValueError(
                    f"Class index {class_index} is out of range "
                    f"[0, {len(self.class_names) - 1}] in LMDB {self.lmdb_path}"
                )

        self.samples: List[Tuple[int, int]] = []
        local_label_by_original = {
            original_class: (
                int(self.class_mapping[original_class])
                if self.class_mapping is not None
                else local_index
            )
            for local_index, original_class in enumerate(self.classes)
        }

        indices_by_class = meta.get("indices_by_class")
        if isinstance(indices_by_class, dict):
            for original_class in self.classes:
                class_indices = indices_by_class.get(
                    original_class,
                    indices_by_class.get(str(original_class), []),
                )
                mapped_label = local_label_by_original[original_class]
                self.samples.extend(
                    (int(sample_index), mapped_label)
                    for sample_index in class_indices
                )
        else:
            env = self._open_env()
            with env.begin(write=False) as txn:
                for sample_index in range(self.length):
                    packed = txn.get(f"{sample_index:08d}".encode("ascii"))
                    if packed is None:
                        continue
                    original_class, _ = pickle.loads(packed)
                    original_class = int(original_class)
                    if original_class in local_label_by_original:
                        self.samples.append(
                            (sample_index, local_label_by_original[original_class])
                        )

    def _open_env(self):
        if self._env is None:
            self._env_key, self._env = LmdbEnvironmentManager.acquire(self.lmdb_path)
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

    def close(self) -> None:
        LmdbEnvironmentManager.release(self._env_key)
        self._env = None
        self._env_key = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        state["_env_key"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None
        self._env_key = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getitem__(self, index):
        sample_index, label = self.samples[index]
        env = self._open_env()
        key = f"{sample_index:08d}".encode("ascii")
        with env.begin(write=False) as txn:
            packed = txn.get(key)
        if packed is None:
            raise IndexError(
                f"Missing sample key {sample_index} in LMDB {self.lmdb_path}"
            )

        _, image_bytes = pickle.loads(packed)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)


def resolve_lmdb_path(domain_path: Path, lmdb_root: Optional[Path]) -> Path:
    if lmdb_root is None:
        return domain_path.with_suffix(".lmdb")
    return (lmdb_root / f"{domain_path.name}.lmdb").resolve()


class MultiSourceDomainDataset(Dataset):
    """Concatenate source domains and return an explicit domain id."""

    def __init__(self, datasets: List[Dataset]):
        if not datasets:
            raise ValueError("datasets must be a non-empty list")
        self.datasets = datasets
        self._lengths = [len(dataset) for dataset in datasets]
        self._offsets = [0]
        for length in self._lengths[:-1]:
            self._offsets.append(self._offsets[-1] + length)
        self._total = sum(self._lengths)

    def __len__(self):
        return self._total

    def __getitem__(self, index):
        if index < 0:
            index += self._total
        if index < 0 or index >= self._total:
            raise IndexError("index out of range")
        for domain_id, (offset, length) in enumerate(
            zip(self._offsets, self._lengths)
        ):
            if offset <= index < offset + length:
                image, label = self.datasets[domain_id][index - offset]
                return image, label, domain_id
        raise RuntimeError("Failed to map index to a dataset")
