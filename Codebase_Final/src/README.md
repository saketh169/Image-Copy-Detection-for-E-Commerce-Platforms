# Image Copy Detection: Final Codebase

This is the curated source-only codebase for the project through Phase 3. It
keeps the familiar project structure while removing dated experiment copies.
It contains no image data, embeddings, checkpoints, generated attacks, logs,
compiled libraries, or result JSON files.

## Pipeline

```text
data -> models -> training -> evaluation -> query_selection -> phase3_graph
```

## Folders

| Folder | Responsibility |
|---|---|
| `configs/` | Shared paths and project defaults. |
| `data/` | Loaders, manifests, pools, pair inputs, and embedding stores. |
| `models/` | ConvNeXt, ResNet, ViT, and the metric head. |
| `training/` | Model selection, pair metric learning, hard mining, and calibration. |
| `evaluation/` | Direct-bank evaluation and attack-table code. |
| `query_selection/` | Final balanced and graph query-set selection. |
| `phase3_graph/` | Custom centroid routing, HNSW, ERNG, C++ kernels, and full-Q HNSW sweep. |
| `timing/` | Timing primitives and the authoritative timing protocol. |

## Important Phase-3 Files

| File | Purpose |
|---|---|
| `phase3_graph/engine/hnsw_hierarchical.py` | Custom hierarchical HNSW construction and traversal over centroids. |
| `phase3_graph/engine/erng_flat.py` | Flat ERNG baseline with local and rank-based links. |
| `phase3_graph/engine/bucket_bvh.py` | Exactness-preserving bucket bound traversal. |
| `phase3_graph/engine/graph_search_kernels.cpp` | C++ direct and graph query kernels. |
| `phase3_graph/experiments/hnsw_full_q_sweep.py` | Current all-Q, full-exact HNSW candidate evaluator. |
| `timing/TIMING_PROTOCOL.md` | Authoritative timing and exactness rules. |

Run `python main.py` to print the stable entry points. Read
`timing/TIMING_PROTOCOL.md` before running Phase 3.
