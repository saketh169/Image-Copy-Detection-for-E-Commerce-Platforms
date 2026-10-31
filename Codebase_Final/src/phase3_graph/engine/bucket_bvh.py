"""Conservative AABB hierarchy over centroid buckets for exact certification."""
from __future__ import annotations

import numpy as np


def build_bucket_bvh(centroids: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, ...]:
    centers = np.asarray(centroids, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    if centers.ndim != 2 or radii.shape != (len(centers),):
        raise ValueError("centroids/radii shape mismatch")

    # A bucket sphere is enclosed by center +/- radius on every coordinate.
    lo = np.nextafter(centers - radii[:, None], np.float32(-np.inf))
    hi = np.nextafter(centers + radii[:, None], np.float32(np.inf))
    node_lo: list[np.ndarray] = []
    node_hi: list[np.ndarray] = []
    left: list[int] = []
    right: list[int] = []
    bucket: list[int] = []

    def build(ids: np.ndarray) -> int:
        idx = len(left)
        node_lo.append(np.min(lo[ids], axis=0))
        node_hi.append(np.max(hi[ids], axis=0))
        left.append(-1)
        right.append(-1)
        bucket.append(-1)
        if len(ids) == 1:
            bucket[idx] = int(ids[0])
            return idx
        spread = np.ptp(centers[ids], axis=0)
        axis = int(np.argmax(spread))
        order = ids[np.argsort(centers[ids, axis], kind="stable")]
        mid = len(order) // 2
        left[idx] = build(order[:mid])
        right[idx] = build(order[mid:])
        return idx

    build(np.arange(len(centers), dtype=np.int32))
    return (
        np.ascontiguousarray(np.stack(node_lo).astype(np.float32)),
        np.ascontiguousarray(np.stack(node_hi).astype(np.float32)),
        np.ascontiguousarray(left, dtype=np.int32),
        np.ascontiguousarray(right, dtype=np.int32),
        np.ascontiguousarray(bucket, dtype=np.int32),
    )
