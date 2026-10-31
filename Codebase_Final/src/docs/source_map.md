# Source Map

The files in this folder were retained from the working project sources and
renamed only to make the pipeline readable. No generated data are copied here.

## Data

- `data/loaders/`: image decoding and pair-loader logic.
- `data/augmentation/attack_augmentor.py`: approved 19-attack in-memory generator.
- `data/materialization/`: compact Amazon derivative image generation.
- `manifests/`: dataset and UCID manifest builders.
- `pools/`: known/unknown, evaluation, and training-pool builders.
- `data/stores/`: persistent embedding, ranked, and target store builders.

## Models And Training

- `models/convnext_backbone.py`: final representation backbone.
- `models/metric_head.py`: learned metric projection head.
- `training/`: ConvNeXt training and checkpoint evaluation.
- `metric_learning/`: pair loss, hard-pair mining, and global threshold training.
- `calibration/`: diagonal metric and threshold application.
- `evaluation/`: direct-bank evaluation and attack table construction.

## Phase 3 Graph Search

- `phase3_graph/engine/hnsw_hierarchical.py`: custom HNSW hierarchy.
- `phase3_graph/engine/erng_flat.py`: comparison graph only; it is not selected in the
  current HNSW-only sweep.
- `phase3_graph/engine/graph_benchmark.py`: kernel loading, direct truth, layouts, and calls.
- `phase3_graph/engine/graph_search_kernels.cpp`: C++ kernel implementation.
- `phase3_graph/experiments/hnsw_full_q_sweep.py`: current active evaluator used for the
  1,053/1,215-candidate full-Q campaign.

Historical experiments are outside this delivery tree and are not part of the
supported code path.
