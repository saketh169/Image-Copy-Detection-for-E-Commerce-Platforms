# Final Artifacts

For the complete final-artifact tree, see [`../Artifacts.md`](../Artifacts.md).

This folder contains the final reproducibility evidence for the image-copy
detection pipeline. The runnable implementation is kept separately in the
sibling `C:\Users\saket\Test\Codebase_Final` folder.

## Contents

| Directory | Contents |
|---|---|
| `HNSW_Selected_Final/` | The four report-selected HNSW configurations, serialized centroid HNSW indexes, final C++ kernel, Python implementation, and validation code. |
| `model_checkpoints/diagonal_metric/` | Canonical local diagonal-metric checkpoints for UCID, Flickr, CIFAR, and Amazon. |
| `evaluations/` | Fixed query sets, attack validation evidence, graph-selection results, and timing outputs. |
| `Images/Q100_Selected_Images/` | Fifty selected image folders per dataset; each contains the final 19 attacks. |
| `Images/Watermark_Collections/Raw_Asset_Logos_Used/` | The 73 downloaded raw overlay assets used for selected watermark/logo attacks. |
| `provenance/` | Result and checkpoint manifests. |
| `data_catalog/` | Locations and descriptions of external benchmark and curated data. |
