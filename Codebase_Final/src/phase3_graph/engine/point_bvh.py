"""Exact point-level BVH used only to certify unresolved HNSW negatives."""
from __future__ import annotations

import numpy as np


def build_point_bvh(points: np.ndarray, leaf_size: int = 64) -> tuple[np.ndarray, ...]:
    """Return a conservative AABB tree over individual bank vectors.

    A leaf stores a contiguous range in ``order``.  Nodes are split by the
    widest coordinate range, so rejecting a node by its AABB lower bound can
    never reject a true threshold match.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    if not len(points):
        raise ValueError("cannot build a point BVH for an empty bank")
    order: list[int] = []
    lo_nodes: list[np.ndarray] = []
    hi_nodes: list[np.ndarray] = []
    left: list[int] = []
    right: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    def visit(indices: np.ndarray) -> int:
        node = len(left)
        subset = points[indices]
        lo_nodes.append(subset.min(axis=0))
        hi_nodes.append(subset.max(axis=0))
        left.append(-1); right.append(-1); starts.append(-1); ends.append(-1)
        if len(indices) <= leaf_size:
            starts[node] = len(order)
            order.extend(indices.tolist())
            ends[node] = len(order)
            return node
        axis = int(np.argmax(hi_nodes[node] - lo_nodes[node]))
        ranked = indices[np.argsort(points[indices, axis], kind="stable")]
        middle = len(ranked) // 2
        left[node] = visit(ranked[:middle])
        right[node] = visit(ranked[middle:])
        return node

    visit(np.arange(len(points), dtype=np.int32))
    return (
        np.ascontiguousarray(np.stack(lo_nodes), dtype=np.float32),
        np.ascontiguousarray(np.stack(hi_nodes), dtype=np.float32),
        np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32),
        np.asarray(starts, dtype=np.int32), np.asarray(ends, dtype=np.int32),
        np.asarray(order, dtype=np.int32),
    )
