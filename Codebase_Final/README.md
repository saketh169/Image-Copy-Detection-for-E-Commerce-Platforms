# Image Copy Detection Final Codebase

This is the complete, organized final source repository for the Image Copy
Detection project. It contains one maintained implementation for dataset
preparation, model training, metric calibration, evaluation, and HNSW search.
Historical experiments are excluded from this delivery tree.

## Quick Start

```powershell
cd C:\Users\saket\Test\Codebase_Final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python tools\validate_repository.py
python main.py
```

Configure local artifact paths in `.env` using `.env.example` as the template.
The project does not bundle raw images, embeddings, checkpoints, generated
attacks, or previous timing output. Those assets remain in the external
artifact location selected by environment variables.

## Repository Map

```text
Codebase_Final/
  configs/      final-19 attack manifest, dataset definitions, graph choices
  docs/         architecture, runbook, project layout, and source inventory
  src/          canonical implementation
    data/       loaders, manifests, pools, stores, augmentation, and image materialization
    models/     Siamese ConvNeXt, ResNet, ViT, and metric layer
    training/   training, hard mining, and diagonal calibration
    evaluation/ direct-bank and attack-table evaluation
    query_selection/
    phase3_graph/ custom centroid HNSW, kernels, and full-Q sweeps
  tools/        launchers, visual validation, and inventory generation
  tests/        fast repository checks
```

Read [the architecture](docs/ARCHITECTURE.md) and [the runbook](docs/RUNBOOK.md)
before starting a training or HNSW sweep.

## Final Protocol

- Four datasets: UCID, Flickr, CIFAR, and Amazon.
- One approved 19-attack panel at every configured Q size.
- 384-D descriptors: 256-D global plus 128-D local features.
- Calibrated diagonal metric stored alongside the model checkpoint.
- Hierarchical HNSW routes over centroid nodes; exact L2 verification keeps
  accepted graph configurations aligned with direct search.
- Timing reports individual-query min, mean, and max latency. A graph choice
  must be exact, improve tail latency, remain stable across Q sizes, use a
  sensible centroid count, and deliver measured speedup.
For a complete source-tree map, see [`../Code.md`](../Code.md). Final selected
results and serialized index artifacts are described in [`../Artifacts.md`](../Artifacts.md).

The approved graph choices are recorded in
[`configs/final_graph_choices.yaml`](configs/final_graph_choices.yaml). Run
`python tools/validate_repository.py` before packaging or launching a sweep.

