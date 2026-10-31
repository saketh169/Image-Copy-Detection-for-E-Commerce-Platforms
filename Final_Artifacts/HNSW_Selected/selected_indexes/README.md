# Serialized Selected HNSW Indexes

This directory contains the serialized index components from the four final
report selections. Each bundle was built over the calibrated 384-dimensional
ConvNeXt embedding bank recorded in `../selected_hnsw_configurations.yaml`.
Model checkpoints are deliberately not duplicated here.

Each dataset directory contains:

| File kind | Contents |
|---|---|
| `*_cpp_centroids_*.npz` | Calibrated centroids and the primary centroid label for every bank vector. |
| `*_hnsw_*.npz` | Fixed-width centroid graph adjacency matrix from the final timing campaign. |
| `*_overlap_*.npz` | Overlapping bucket membership when the selected overlap is greater than one. |
| `*_radii_*.npy` | Bucket radii used by the lower-bound candidate pruning stage. |

For Amazon, `overlap=1`; its bucket membership is reconstructed exactly from
the `labels` array in `amazon_cpp_centroids_c2048.npz`. A separate overlap
file is neither required nor produced for this single-assignment case.

See `index_manifest.json` for the selected parameters, file inventory, and
bank dimensions. Run `../validation/verify_selected_index_artifacts.py` to
validate this package's selected graph components. The separate full
hierarchical implementation is in `../source/hnsw_hierarchical.py`; it must
be rebuilt and revalidated when its graph construction parameters change.
