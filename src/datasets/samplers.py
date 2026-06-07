"""Batch samplers used by multi-source domain adaptation."""

from __future__ import annotations

from typing import List
import math

import torch


class UniformDomainBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Yield full batches from domains in shuffled round-robin order."""

    def __init__(
        self,
        domain_sizes: List[int],
        batch_size: int,
        steps_per_epoch: int,
        drop_last: bool = True,
        rank: int = 0,
        num_replicas: int = 1,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if any(size <= 0 for size in domain_sizes):
            raise ValueError("All domains must have at least 1 sample")
        self.domain_sizes = domain_sizes
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.seed = int(seed)
        self.epoch = 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.offsets = [0]
        for size in domain_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        num_domains = len(self.domain_sizes)
        permutations = [
            torch.randperm(size, generator=generator).tolist()
            for size in self.domain_sizes
        ]
        cursors = [0 for _ in range(num_domains)]
        domain_order = torch.randperm(num_domains, generator=generator).tolist()

        local_steps = len(self)
        for step in range(local_steps * self.num_replicas):
            domain = domain_order[step % num_domains]
            size = self.domain_sizes[domain]
            offset = self.offsets[domain]
            cursor = cursors[domain]
            if cursor + self.batch_size > size:
                permutations[domain] = torch.randperm(
                    size, generator=generator
                ).tolist()
                cursor = 0
            local_indices = permutations[domain][cursor : cursor + self.batch_size]
            cursors[domain] = cursor + self.batch_size
            if step % self.num_replicas == self.rank:
                yield [offset + index for index in local_indices]

    def __len__(self):
        return int(math.ceil(self.steps_per_epoch / self.num_replicas))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


class StratifiedDomainBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Mix samples from every source domain in each batch."""

    def __init__(
        self,
        domain_sizes: List[int],
        batch_size: int,
        steps_per_epoch: int,
        drop_last: bool = True,
        rank: int = 0,
        num_replicas: int = 1,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        if any(size <= 0 for size in domain_sizes):
            raise ValueError("All domains must have at least 1 sample")
        self.domain_sizes = domain_sizes
        self.num_domains = len(domain_sizes)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.seed = int(seed)
        self.epoch = 0
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.offsets = [0]
        for size in domain_sizes[:-1]:
            self.offsets.append(self.offsets[-1] + size)

        per_domain = self.batch_size // self.num_domains
        remainder = self.batch_size % self.num_domains
        self.per_domain_counts = [
            per_domain + (1 if domain < remainder else 0)
            for domain in range(self.num_domains)
        ]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutations = [
            torch.randperm(size, generator=generator).tolist()
            for size in self.domain_sizes
        ]
        cursors = [0] * self.num_domains

        local_steps = len(self)
        for step in range(local_steps * self.num_replicas):
            batch = []
            for domain in range(self.num_domains):
                needed = self.per_domain_counts[domain]
                size = self.domain_sizes[domain]
                offset = self.offsets[domain]
                cursor = cursors[domain]
                if cursor + needed > size:
                    permutations[domain] = torch.randperm(
                        size, generator=generator
                    ).tolist()
                    cursor = 0
                batch.extend(
                    offset + permutations[domain][index]
                    for index in range(cursor, cursor + needed)
                )
                cursors[domain] = cursor + needed
            if step % self.num_replicas == self.rank:
                yield batch

    def __len__(self):
        return int(math.ceil(self.steps_per_epoch / self.num_replicas))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


_UniformDomainBatchSampler = UniformDomainBatchSampler
_StratifiedDomainBatchSampler = StratifiedDomainBatchSampler
