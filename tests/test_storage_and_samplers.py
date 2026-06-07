from __future__ import annotations

import io
import pickle

import lmdb
from PIL import Image

from datasets.samplers import (
    StratifiedDomainBatchSampler,
    UniformDomainBatchSampler,
)
from datasets.storage import LmdbDomainDataset, LmdbEnvironmentManager


def _build_lmdb(path):
    image_buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(
        image_buffer, format="PNG"
    )
    env = lmdb.open(str(path), map_size=1 << 20)
    with env.begin(write=True) as txn:
        txn.put(
            b"__meta__",
            pickle.dumps(
                {
                    "class_names": ["zero"],
                    "length": 1,
                    "indices_by_class": {0: [0]},
                }
            ),
        )
        txn.put(
            b"00000000",
            pickle.dumps((0, image_buffer.getvalue())),
        )
    env.close()


def test_lmdb_environments_are_shared_and_reference_counted(tmp_path):
    lmdb_path = tmp_path / "domain.lmdb"
    _build_lmdb(lmdb_path)

    first = LmdbDomainDataset(lmdb_path, [0])
    second = LmdbDomainDataset(lmdb_path, [0])
    snapshot = LmdbEnvironmentManager.snapshot()
    assert list(snapshot.values()) == [2]

    image, label = second[0]
    assert image.size == (8, 8)
    assert label == 0

    first.close()
    assert list(LmdbEnvironmentManager.snapshot().values()) == [1]
    second.close()
    assert LmdbEnvironmentManager.snapshot() == {}


def test_distributed_domain_batch_samplers_keep_equal_rank_lengths():
    for sampler_cls in (
        UniformDomainBatchSampler,
        StratifiedDomainBatchSampler,
    ):
        rank_zero = sampler_cls(
            [8, 8],
            batch_size=4,
            steps_per_epoch=3,
            rank=0,
            num_replicas=2,
            seed=7,
        )
        rank_one = sampler_cls(
            [8, 8],
            batch_size=4,
            steps_per_epoch=3,
            rank=1,
            num_replicas=2,
            seed=7,
        )

        zero_batches = list(rank_zero)
        one_batches = list(rank_one)
        assert len(zero_batches) == len(one_batches) == 2
        assert all(len(batch) == 4 for batch in zero_batches + one_batches)

        rank_zero.set_epoch(1)
        assert list(rank_zero) != zero_batches
