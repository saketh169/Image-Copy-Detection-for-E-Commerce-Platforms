# Architecture

The canonical implementation is in `src/`. It is the only supported code path.
Historical scripts are excluded from this final delivery.

```text
images and manifests
        |
        v
Siamese ConvNeXt / ResNet / ViT training
        |
        v
384-D [global(256) | local(128)] embedding
        |
        v
diagonal metric calibration and threshold selection
        |
        v
reference embedding store
        |
        v
MiniBatchKMeans centroid buckets -> hierarchical HNSW over centroids
        |
        v
bucket lower-bound pruning -> exact L2 verification
```

The final attack protocol is stored in `configs/attacks_final19.json`. It has
19 attacks for every dataset and Q size; `flip_h` is included and the retired
`flip` name is not part of the protocol.

The graph layer is genuine hierarchical HNSW routing over centroid nodes:
`src/phase3_graph/engine/hnsw_hierarchical.py` constructs upper layers and
greedily descends them before a bottom-layer search. Bucket radius bounds and
exact candidate verification are separate mechanisms used to retain direct
search agreement; they do not replace HNSW.
