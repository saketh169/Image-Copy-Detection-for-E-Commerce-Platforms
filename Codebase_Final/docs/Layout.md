# Delivery Layout

`Codebase_Final` is the runnable source delivery. `Final_Artifacts` is the
matching evidence delivery. They are deliberately separate so that code can
be versioned and validated without bundling large generated outputs.

## Code Organization

| Area | Location | Responsibility |
|---|---|---|
| Dataset preparation | `src/data/` | source manifests, loaders, known/unknown pools, training pools, embedding stores, and materialization |
| Image generation | `src/data/augmentation/`, `src/data/materialization/` | in-memory attacks and compact Amazon derivative creation |
| Model development | `src/models/` | ConvNeXt, ResNet, ViT, and the metric projection head |
| Training | `src/training/model_selection/` | baseline and backbone comparisons |
| Metric learning | `src/training/metric_learning/` | pair loss, hard mining, and global-threshold training |
| Calibration | `src/training/calibration/` | diagonal metric search and threshold application |
| Verification | `src/evaluation/` | direct-bank baselines and per-attack results |
| Query protocol | `src/query_selection/` | balanced, nearest-bank, and graph query-set selection |
| Retrieval | `src/phase3_graph/` | centroid construction, hierarchical HNSW, exact candidate verification, kernels, and sweeps |
| Timing | `src/timing/` | measurement protocol and timing helpers |
| Operations | `tools/operations/` | Amazon materialization and visual-validation utilities |
| Tests | `tests/` | repository-layout and HNSW correctness checks |

## Artifact Organization

| Artifact type | Location under `Final_Artifacts` |
|---|---|
| Final diagonal checkpoints | `model_checkpoints/diagonal_metric/` |
| Q-set selections | `evaluations/query_sets/selected_q_sets/` |
| HNSW candidate rankings | `evaluations/graph_selection/candidate_rankings/` |
| Attack and watermark validation | `evaluations/attack_validation/` |
| Timing records | `evaluations/timing/` |
| Final 19-attack visual panels | `visual_validation/final19_attack_panels/` |
| Watermark examples | `visual_validation/watermark_samples/` |
| Result manifests | `provenance/manifests/` |
| Manuscript source and figures | `reports/` |

## Validation

```powershell
cd C:\Users\saket\Test\Repository\Codebase_Final
python tools\validate_repository.py
python -m unittest discover -s tests
```

The validation script checks the approved 19-attack panel for all four
datasets, compiles canonical Python source, and records file hashes in
`docs/source_inventory.json`.
