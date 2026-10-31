# Final Diagonal Checkpoints

These four local files are the final self-contained diagonal/metric checkpoints.
They are renamed to match the final training/adjustment/diagonal numbering table.

| Dataset | Final clean name | Equivalent original working name |
|---|---|---|
| CIFAR | `cifar-epoch-019-diagonal.pt` | `cifar-generalized-gap-accepted-003.pt` |
| Flickr | `flickr-epoch-028-diagonal.pt` | `flickr-fresh-bank-metric-layer-best.pt` |
| UCID | `ucid-epoch-040-diagonal.pt` | `ucid-fresh-bank-metric-layer-best.pt` |
| Amazon | `amazon-epoch-037-diagonal.pt` | `amazon-fresh-bank-metric-layer-best.pt` |

## GPU Final Named Paths

| Dataset | GPU final diagonal path |
|---|---|
| CIFAR | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/cifar/cifar-epoch-019-diagonal.pt` |
| Flickr | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/flickr/flickr-epoch-028-diagonal.pt` |
| UCID | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/ucid/ucid-epoch-040-diagonal.pt` |
| Amazon | `/home/saketh/Codebase/checkpoints/final_named_checkpoints_20260601/amazon/amazon-epoch-037-diagonal.pt` |

## GPU Original Working Paths

| Dataset | GPU original working checkpoint path |
|---|---|
| CIFAR | `/home/saketh/Codebase/checkpoints/generalized_gap_guard_diagonal_search_20260717/cifar/cifar-generalized-gap-accepted-003.pt` |
| Flickr | `/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/flickr/flickr-fresh-bank-metric-layer-best.pt` |
| UCID | `/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/ucid/ucid-fresh-bank-metric-layer-best.pt` |
| Amazon | `/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/amazon/amazon-fresh-bank-metric-layer-best.pt` |

## Verified Code Paths

| Purpose | GPU path |
|---|---|
| Trusted q100 original-only verifier | `/home/saketh/Codebase/trusted_q100_original_only_verify.py` |
| Latest resize / resize-compress / flip / watermark verifier | `/home/saketh/Codebase/trusted_q100_latest5_no_distance_cache_verify.py` |
| Latest text/asset watermark variant sweep | `/home/saketh/Codebase/trusted_q100_watermark_variant_sweep.py` |

## Verified Output Folders

| Purpose | GPU output folder |
|---|---|
| Trusted q100 original-only verification | `/home/saketh/json/final_named_artifacts_20260601/trusted_q100_original_only_verify_20260727` |
| Latest 5 attack verification | `/home/saketh/json/final_named_artifacts_20260601/trusted_q100_latest5_no_distance_cache_verify_20260727` |
| Text/asset watermark variant sweep | `/home/saketh/json/final_named_artifacts_20260601/trusted_q100_watermark_variant_sweep_20260727` |

## Runtime Meaning

Each checkpoint is self-contained for inference:

```text
image -> transform -> final diagonal checkpoint -> embedding
```

The graph/index/store then compares that embedding against the final bank/store.
The checkpoint does not need to load another checkpoint at runtime.


`/home/saketh/Codebase/trusted_q100_latest5_no_distance_cache_verify.py` is the verifier for the **latest 5 updated attacks**:

```text
flip_h
resize_latest
resize_compress_latest
watermark_text_latest
watermark_asset_latest
```

What it does:
1. Loads the trusted q100 selected image paths.
2. Loads the final checkpoint for that dataset.
3. Loads the complete bank/store `.npz`.
4. Applies each attack freshly to the q100 images.
5. Computes new embeddings using:
   ```python
   model(x, x)[0]
   ```
6. Compares those embeddings against the full bank/store.
7. Uses tau to decide duplicate/non-duplicate.
8. Writes results to:
   ```text
   /home/saketh/json/final_named_artifacts_20260601/trusted_q100_latest5_no_distance_cache_verify_20260727
   ```


Amazon 50K split
The archived category summary records 50,000 images total: 35,000 known and 15,000 unknown. The known split contains baby products (4,275), cosmetics (6,331), electronics (7,754), home (12,399), and sports (4,241). The unknown split contains sports (8,347) and wearables (6,653). So sports appears in both splits; wearables is unknown-only. This differs from the older category counts you pasted, so those older counts should not be reused without checking their manifest.