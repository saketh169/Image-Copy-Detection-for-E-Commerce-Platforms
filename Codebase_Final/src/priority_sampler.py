"""Deterministic priority sampler for manifest-based training."""

from __future__ import annotations

import os
import random

from torch.utils.data import Sampler


class PrioritySampler(Sampler[int]):
    """Yield every manifest index once, prioritizing difficult examples.

    A seed makes reruns reproducible.  Priorities only influence ordering; no
    pair is dropped, so full manifest coverage is preserved.
    """

    def __init__(self, manifest: list[dict], seed: int | None = None):
        self.manifest = manifest
        self.seed = int(os.environ.get("BTP_SAMPLER_SEED", "20260721")) if seed is None else seed

    def __iter__(self):
        rng = random.Random(self.seed)
        ranked = []
        for index, row in enumerate(self.manifest):
            priority = float(row.get("priority", 0.0))
            ranked.append((priority, rng.random(), index))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return iter(item[2] for item in ranked)

    def __len__(self) -> int:
        return len(self.manifest)
