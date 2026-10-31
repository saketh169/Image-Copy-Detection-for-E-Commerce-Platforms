# Codebase Structure

`Codebase_Final/` is the runnable consolidated implementation of the image-copy
detection pipeline. It contains dataset construction, Siamese ConvNeXt metric
learning, calibration, evaluation, query selection, and Phase-III graph search.
Historical scripts are retained under `legacy/` for traceability and are not the
default implementation.

```text
Codebase_Final/
|-- README.md                              # Setup, entry points, and workflow
|-- configs/                               # Final fixed experiment settings
|   |-- attacks_final19.json               # The 19-condition attack suite
|   |-- datasets.yaml                      # Dataset locations and split metadata
|   `-- final_graph_choices.yaml           # Report-selected HNSW choices
|-- src/                                   # Runnable project source
|   |-- main.py                            # Pipeline entry point
|   |-- config.py                          # Shared runtime configuration
|   |-- dataloader.py                      # Compatibility data loader entry
|   |-- transforms.py                      # Image attack and preprocessing transforms
|   |-- loss.py                            # Contrastive-loss utilities
|   |-- priority_sampler.py                # Pair difficulty tracking
|   |-- train_logger.py                    # Training diagnostics
|   |-- configs/
|   |   `-- project_config.py              # Structured project configuration
|   |-- data/
|   |   |-- loaders/
|   |   |   |-- image_dataset_loader.py    # Image and split loading
|   |   |   `-- training_pair_loader.py    # Soft/hard pair construction
|   |   |-- manifests/
|   |   |   |-- generate_dataset_manifest.py
|   |   |   `-- generate_ucid_manifest.py
|   |   |-- pools/
|   |   |   |-- build_training_pool.py
|   |   |   |-- build_evaluation_pool.py
|   |   |   |-- build_known_unknown_pool.py
|   |   |   `-- expand_attack_aware_pool.py
|   |   `-- stores/
|   |       |-- build_embedding_store.py   # Calibrated embedding-bank creation
|   |       |-- build_ranked_store.py
|   |       `-- build_target_store.py
|   |-- models/
|   |   |-- convnext_backbone.py           # ConvNeXt feature extractor
|   |   |-- metric_head.py                 # Global/local projection heads
|   |   |-- model_convnext_metric.py       # Final Siamese metric model
|   |   |-- model_convnext.py              # ConvNeXt baseline
|   |   |-- resnet_backbone.py
|   |   |-- model_resnet.py
|   |   |-- vit_backbone.py
|   |   `-- model_vit.py
|   |-- training/
|   |   |-- metric_learning/
|   |   |   |-- train_hard_pair_metric.py  # Stage-I contrastive training
|   |   |   `-- train_global_tau_hard_mining.py
|   |   |-- calibration/
|   |   |   |-- search_diagonal_metric.py  # Stage-II diagonal calibration
|   |   |   `-- apply_global_threshold.py  # Dataset threshold application
|   |   `-- model_selection/
|   |       |-- train_convnext.py
|   |       |-- train_resnet.py
|   |       |-- train_vit.py
|   |       |-- train_baseline.py
|   |       |-- evaluate_checkpoint.py
|   |       `-- compare_convnext_checkpoints.py
|   |-- evaluation/
|   |   |-- baselines/evaluate_direct_bank.py
|   |   `-- attack_tables/
|   |       |-- attacks_manifest.json
|   |       `-- build_attack_tables.py
|   |-- query_selection/
|   |   |-- select_balanced_query_sets.py
|   |   |-- select_graph_query_sets.py
|   |   |-- select_nearest_bank_query_sets.py
|   |   `-- rescore_query_set_stability.py
|   |-- phase3_graph/
|   |   |-- engine/
|   |   |   |-- hnsw_hierarchical.py        # Hierarchical centroid-graph builder/search
|   |   |   |-- graph_search_kernels.cpp    # C++ distance and graph-query kernels
|   |   |   |-- graph_benchmark.py          # Index construction and timing runner
|   |   |   |-- bucket_bvh.py               # Optional lower-bound bucket pruning
|   |   |   |-- point_bvh.py                # Optional point-level pruning
|   |   |   |-- erng_flat.py                # Historical ERNG comparison path
|   |   |   `-- timing.py
|   |   |-- experiments/
|   |   |   |-- hnsw_full_q_sweep.py        # Full-Q HNSW candidate sweep
|   |   |   `-- amazon_certificate_preflight.py
|   |   `-- tests/
|   |       |-- test_hnsw_hierarchical.py
|   |       `-- hnsw_kernel_exactness_smoke.cpp
|   |-- reporting/build_final_tables_report.py
|   |-- timing/
|   |   |-- timing.py
|   |   `-- TIMING_PROTOCOL.md
|   |-- utils/training_logger.py
|   `-- docs/source_map.md
|-- tests/
|   `-- test_repository_layout.py           # Consolidated-layout guard
|-- tools/
|   `-- operations/
|       |-- dataset_materialization/        # Amazon benchmark build/transfer utilities
|       `-- visual_validation/              # Q100 and watermark image-export utilities
|-- docs/project_records/                   # Historical work notes and provenance
`-- legacy/                                # Read-only archived pre-consolidation work
    |-- training/, data/, models/, evaluation/, query_selection/
    |-- phase3_graph/                       # Earlier HNSW/ERNG experiments
    `-- historical_experiments/             # Superseded experiment snapshots
```

The selected final graph settings and serialized graph artifacts are documented
in [`Final_Artifacts/HNSW_Selected_Final/`](Final_Artifacts/HNSW_Selected_Final/).
