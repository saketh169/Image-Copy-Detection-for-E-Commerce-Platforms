# Final Artifacts Structure

`Final_Artifacts/` contains the final reproducibility evidence, selected HNSW
index components, evaluation records, checkpoints, manifests, and visual attack
examples. It is separate from `Codebase_Final/`, which contains runnable source.

```text
Final_Artifacts/
|-- README.md                              # Artifact-package overview
|-- HNSW_Selected_Final/                   # Final report-selected graph artifacts
|   |-- README.md
|   |-- selected_hnsw_configurations.yaml  # Chosen C/overlap/entries/degree/ef values
|   |-- source/
|   |   |-- hnsw_hierarchical.py           # Hierarchical HNSW reference implementation
|   |   |-- hnsw_full_q_sweep.py           # Candidate sweep driver
|   |   `-- graph_search_kernels.cpp       # C++ search kernel source
|   |-- selected_indexes/                  # Exact graph components from final timing sweep
|   |   |-- README.md
|   |   |-- index_manifest.json            # Dataset-specific graph inventory
|   |   |-- ucid/
|   |   |   |-- ucid_cpp_centroids_c256.npz
|   |   |   |-- ucid_hnsw_c256_d24.npz
|   |   |   |-- ucid_overlap_c256_r2.npz
|   |   |   `-- ucid_radii_c256_o2.npy
|   |   |-- flickr/
|   |   |   |-- flickr_cpp_centroids_c2048.npz
|   |   |   |-- flickr_hnsw_c2048_d12.npz
|   |   |   |-- flickr_overlap_c2048_r3.npz
|   |   |   `-- flickr_radii_c2048_o3.npy
|   |   |-- cifar/
|   |   |   |-- cifar_cpp_centroids_c1536.npz
|   |   |   |-- cifar_hnsw_c1536_d32.npz
|   |   |   |-- cifar_overlap_c1536_r3.npz
|   |   |   `-- cifar_radii_c1536_o3.npy
|   |   `-- amazon/
|   |       |-- amazon_cpp_centroids_c2048.npz
|   |       |-- amazon_hnsw_c2048_d32.npz
|   |       `-- amazon_radii_c2048_o1.npy  # Labels in centroid file define ov=1 buckets
|   |-- validation/
|   |   |-- verify_selected_index_artifacts.py
|   |   |-- test_hnsw_hierarchical.py
|   |   `-- hnsw_kernel_exactness_smoke.cpp
|   `-- metric_checkpoints/                # Existing checkpoint-link archive; not graph data
|-- model_checkpoints/
|   `-- diagonal_metric/                   # Canonical Stage-II calibration checkpoints
|-- evaluations/
|   |-- timing/Timing.md                   # Final direct versus HNSW timing record
|   |-- graph_selection/
|   |   `-- candidate_rankings/            # Ranked sweep candidates and final choices
|   |-- query_sets/
|   |   `-- selected_q_sets/
|   |       |-- README.md
|   |       |-- ucid/, flickr/, cifar/, amazon/
|   |       `-- report-level query and 19-condition results
|   `-- attack_validation/
|       |-- trusted_attack_checks/
|       `-- watermark_variants/
|-- Images/
|   |-- Q100_Selected_Images/              # 50 selected samples per dataset
|   |   |-- ucid/, flickr/, cifar/, amazon/
|   |   |-- logs/
|   |   `-- each sample folder: original plus final attack variants
|   `-- Watermark_Collections/
|       |-- README.md
|       |-- watermark_selection_summary.json
|       `-- Raw_Asset_Logos_Used/
|           `-- ucid/, flickr/, cifar/, amazon/
|-- provenance/
|   `-- manifests/result_manifests/         # Checkpoint/result provenance and file lists
|-- data_catalog/
|   `-- external_data_locations.md          # External source-data locations
`-- reports/
    `-- manuscript_source/                  # Manuscript-related material
```

The selected index validator checks the actual package dimensions and
memberships: UCID 4,500, Flickr 10,000, CIFAR 17,000, and Amazon 35,000
calibrated 384-D embedding-bank entries.
