# Image Copy Detection Repository

This repository contains the complete final implementation and evidence for
image-copy detection in e-commerce product imagery. The system learns a
Siamese ConvNeXt representation, calibrates its metric, verifies edited image
copies, and accelerates large-bank search with centroid-indexed HNSW retrieval.

## Repository Contents

```text
Repository/
  Codebase_Final/
    configs/                 final datasets, 19 attacks, and graph choices
    docs/                    architecture, runbook, layout, and source inventory
    src/
      data/                  manifests, loaders, pools, stores, augmentation,
                             and Amazon image materialization
      models/                ConvNeXt, ResNet, ViT, and metric heads
      training/              model selection, metric learning, and calibration
      evaluation/            direct-bank and per-attack evaluation
      query_selection/       balanced, nearest-bank, and graph query selection
      phase3_graph/          centroid construction, HNSW, kernels, sweeps, timing
      reporting/              final result-table generation
      timing/                timing protocol and measurement helpers
    tools/                   validation, visual-validation, and launch utilities
    tests/                   final contract and repository checks
  Final_Artifacts/
    HNSW_Selected/           serialized final indexes, kernels, and verification
    model_checkpoints/       approved diagonal-metric checkpoints
    evaluations/             query sets, attack checks, graph results, and timing
    Images/                  selected Q100 images and watermark collections
    provenance/              result and checkpoint manifests
    data_catalog/            external benchmark and dataset locations
    reports/                 final reports and supporting evidence
```

## Processing Phases

1. **Dataset preparation:** build manifests, known/unknown pools, training
   pairs, embedding stores, and compact Amazon derivatives.
2. **Representation learning:** train and compare Siamese ConvNeXt, ResNet,
   and ViT models with shared-weight pair training.
3. **Metric calibration:** learn the diagonal feature scale, apply the global
   threshold, and store the calibrated embedding configuration.
4. **Evaluation and retrieval:** evaluate the approved 19 attacks, select
   fixed query sets, build centroid buckets, route queries through HNSW, prune
   candidates, and perform exact L2 verification.

## Datasets

The final evaluation covers **UCID, Flickr, CIFAR-10, and the curated Amazon
benchmark**. The selected embedding banks contain 4,500, 10,000, 17,000, and
35,000 vectors respectively. Each dataset uses its approved Q-size set and the
same final 19-attack protocol, including geometric edits, crops, photometric
changes, resampling/compression, watermarks, and compound attacks.

## Selected HNSW Configurations

The authoritative values are in
[`Codebase_Final/configs/final_graph_choices.yaml`](Codebase_Final/configs/final_graph_choices.yaml).
All four selected configurations were validated against direct search with
100% reported full-workload agreement.

| Dataset | Bank | Centroids | Overlap | Entries | Degree | ef | efConstruction |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCID | 4,500 | 256 | 2 | 1 | 24 | 16 | 96 |
| Flickr | 10,000 | 2,048 | 3 | 1 | 12 | 12 | 64 |
| CIFAR-10 | 17,000 | 1,536 | 3 | 2 | 32 | 16 | 128 |
| Amazon | 35,000 | 2,048 | 1 | 3 | 32 | 32 | 128 |

## Validation

From `Codebase_Final`:

```powershell
python tools\validate_repository.py
python -m unittest discover -s tests -v
```

To verify the serialized four-dataset indexes:

```powershell
python ..\Final_Artifacts\HNSW_Selected\validation\verify_selected_index_artifacts.py
```

Raw images, generated attacks, embeddings, and checkpoints are intentionally
kept in `Final_Artifacts` or external data locations rather than duplicated in
the source tree.
