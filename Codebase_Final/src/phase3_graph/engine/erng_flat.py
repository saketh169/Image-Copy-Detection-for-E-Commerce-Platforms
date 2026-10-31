"""Flat Exponential-Rank Navigation Graph (ERNG) construction.

ERNG is deliberately different from HNSW: it has one graph layer and uses
multiple deterministic starts at query time.  Each node keeps close local
neighbours plus links to exponentially ranked neighbours in its distance
ordering.  No HNSW library or hidden shortcut layer is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ERNGIndex:
    centroids: np.ndarray
    adjacency: np.ndarray
    degrees: np.ndarray
    starts: np.ndarray
    max_degree: int
    local_degree: int
    skip_count: int


def _pairwise_squared_distances(centroids: np.ndarray) -> np.ndarray:
    """Compute the centroid distance matrix in float32 without a graph library."""
    x = np.ascontiguousarray(centroids, dtype=np.float32)
    gram = x @ x.T
    d2 = gram.diagonal()[:, None] + gram.diagonal()[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, np.inf)
    return d2


def build_erng(
    centroids: np.ndarray,
    *,
    local_degree: int,
    skip_count: int,
    probes: int,
    seed: int = 20260915,
    reciprocal: bool = True,
) -> ERNGIndex:
    """Build ERNG using local K-NN plus rank 1,2,4,... skip links.

    Rank is one-based in the algorithm description.  Thus rank 1 is the
    nearest neighbour, rank 2 is the second nearest, rank 4 is the fourth,
    and so on.  Duplicate links are removed before the degree cap is applied.
    """
    x = np.ascontiguousarray(centroids, dtype=np.float32)
    n = int(x.shape[0])
    if n < 2:
        raise ValueError("ERNG requires at least two centroids")
    local_degree = max(1, min(int(local_degree), n - 1))
    skip_count = max(0, int(skip_count))
    probes = max(1, min(int(probes), n))
    d2 = _pairwise_squared_distances(x)
    ranked = np.argsort(d2, axis=1, kind="stable")
    candidates: list[list[int]] = [[] for _ in range(n)]
    for node in range(n):
        row = ranked[node]
        picked: list[int] = []
        for other in row[:local_degree]:
            other = int(other)
            if other not in picked:
                picked.append(other)
        # Walk power-of-two ranks until skip_count genuinely new links have
        # been found.  Ranks 1, 2, 4 may already be local links, so simply
        # taking the first skip_count powers would under-fill the graph.
        rank = 1
        added_skips = 0
        while rank <= n - 1 and added_skips < skip_count:
            other = int(row[rank - 1])
            if other not in picked:
                picked.append(other)
                added_skips += 1
            rank <<= 1
        candidates[node] = picked

    if reciprocal:
        for node, neighbours in enumerate(list(candidates)):
            for other in neighbours:
                if node not in candidates[other]:
                    candidates[other].append(node)

    max_degree = local_degree + skip_count
    adjacency = np.full((n, max_degree), -1, dtype=np.int32)
    degrees = np.zeros(n, dtype=np.int32)
    for node in range(n):
        # Keep local edges first, then nearest reciprocal additions.  This
        # preserves the intended ERNG semantics when reciprocal links overflow.
        local = [int(v) for v in ranked[node, :local_degree]]
        remaining = [int(v) for v in candidates[node] if int(v) not in local]
        remaining.sort(key=lambda v: float(d2[node, v]))
        chosen: list[int] = []
        for other in local + remaining:
            if other != node and other not in chosen:
                chosen.append(other)
            if len(chosen) >= max_degree:
                break
        degrees[node] = len(chosen)
        if chosen:
            adjacency[node, : len(chosen)] = np.asarray(chosen, dtype=np.int32)

    # Fixed seeded random starts are reproducible, independent of the query,
    # and match the planned multi-probe ERNG definition.
    rng = np.random.default_rng(seed)
    starts = np.asarray(rng.choice(n, size=probes, replace=False), dtype=np.int32)
    return ERNGIndex(
        centroids=x,
        adjacency=np.ascontiguousarray(adjacency),
        degrees=np.ascontiguousarray(degrees),
        starts=np.ascontiguousarray(starts),
        max_degree=int(max_degree),
        local_degree=int(local_degree),
        skip_count=int(skip_count),
    )


def query_starts(index: ERNGIndex, query_count: int) -> np.ndarray:
    """Repeat deterministic graph starts for the C++ query kernel."""
    return np.tile(index.starts[None, :], (int(query_count), 1)).astype(np.int32, copy=False)


def save_erng(index: ERNGIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centroids=index.centroids,
        adjacency=index.adjacency,
        degrees=index.degrees,
        starts=index.starts,
        max_degree=np.asarray([index.max_degree], dtype=np.int32),
        local_degree=np.asarray([index.local_degree], dtype=np.int32),
        skip_count=np.asarray([index.skip_count], dtype=np.int32),
    )
