# Final Selected HNSW Artifacts

This folder packages the four HNSW configurations selected for the final
report. Each configuration reproduced the direct-search decision on every
tested query for all 19 attacks.

## Contents

| Path | Purpose |
|---|---|
| `selected_hnsw_configurations.yaml` | Report-backed final choice for UCID, Flickr, CIFAR, and Amazon. |
| `source/graph_search_kernels.cpp` | Final C++ exact distance and graph-search kernel source. |
| `source/hnsw_hierarchical.py` | Hierarchical HNSW construction and routing implementation. |
| `source/hnsw_full_q_sweep.py` | Full-Q candidate sweep and validation driver. |
| `validation/` | Kernel smoke test and hierarchical-search test. |
| `selected_indexes/` | Serialized centroid layouts, HNSW adjacency graphs, overlap memberships, and bucket radii for the four selected configurations. |

The selected indexes were constructed over the final calibrated 384-D ConvNeXt
embedding banks. Model checkpoints remain in the separate canonical
`../model_checkpoints/diagonal_metric/` archive and are intentionally not
duplicated in this graph package.
The full timing tables and validation evidence are recorded in the sibling
`../../Timing.md` report.
