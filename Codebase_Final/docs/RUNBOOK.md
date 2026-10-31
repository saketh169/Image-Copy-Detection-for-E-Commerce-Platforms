# Runbook

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set local artifact and dataset paths.
4. Export `PYTHONPATH=<project-root>/src` before invoking source scripts.
5. Run `python tools/validate_repository.py`.

## Canonical order

1. Build or validate a pair manifest with `src/data/manifests/`.
2. Train a backbone through `src/training/model_selection/`.
3. Fit the diagonal metric with `src/training/calibration/search_diagonal_metric.py`.
4. Build reference stores using `src/data/stores/`.
5. Select balanced queries in `src/query_selection/`.
6. Evaluate direct search and attack tables under `src/evaluation/`.
7. Run full-Q HNSW candidate sweeps with
   `src/phase3_graph/experiments/hnsw_full_q_sweep.py`.

## Acceptance rules

For a selected graph candidate, require direct-search agreement for every
query at each configured Q size and all 19 attacks. Record per-query minimum,
mean, and maximum latency. Prefer a middle-range centroid count when it meets
exactness, tail-latency, Q-size stability, and speed goals together.

## What this repository does not contain

It excludes proprietary/raw image data, generated attack images, embeddings,
checkpoints, compiled kernels, and completed result JSON files. These belong
under the configured artifact root and are referenced, never duplicated.
