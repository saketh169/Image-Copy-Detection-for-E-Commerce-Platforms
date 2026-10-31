"""Shared, single-query timing primitives for Phase-3 experiments."""
from __future__ import annotations

import statistics

import numpy as np


def trimmed_median(values_seconds: list[float]) -> float:
    """Drop one endpoint when possible and return a robust median in seconds."""
    ordered = sorted(values_seconds)
    if len(ordered) >= 5:
        ordered = ordered[1:-1]
    return float(statistics.median(ordered))


def latency_summary(values_seconds: list[float]) -> dict[str, float]:
    """Return millisecond latency statistics for per-query robust measurements."""
    values = np.asarray(values_seconds, dtype=np.float64) * 1000.0
    mean = float(np.mean(values))
    return {
        "min_ms": float(np.min(values)),
        "mean_ms": mean,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "max_ms": float(np.max(values)),
        "cv_pct": float(np.std(values) / max(mean, 1e-12) * 100.0),
    }
