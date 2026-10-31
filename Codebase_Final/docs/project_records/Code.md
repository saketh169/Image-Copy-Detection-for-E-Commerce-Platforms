You’re right. The previous list was too short. Use this fuller important-code list. This covers training, manifests, stores, eval, repair, reports, and Phase-3 graph query.

**Core Files**
```text
/home/saketh/Codebase/config.py
/home/saketh/Codebase/dataloader.py
/home/saketh/Codebase/transforms.py
/home/saketh/Codebase/loss.py
/home/saketh/Codebase/priority_sampler.py
/home/saketh/Codebase/train_logger.py
/home/saketh/Codebase/embeddings.py
/home/saketh/Codebase/reembed_store.py
/home/saketh/Codebase/query_embedding_db.py
/home/saketh/Codebase/build_embedding_db.py
/home/saketh/Codebase/model.py
```

**Model Files**
```text
/home/saketh/Codebase/models/model_convnext.py
/home/saketh/Codebase/models/model_resnet.py
/home/saketh/Codebase/models/model_vit.py
```

**Training Files**
```text
/home/saketh/Codebase/train/train_convnext.py
/home/saketh/Codebase/train/train_convnext_ucid.py
/home/saketh/Codebase/train/train_resnet.py
/home/saketh/Codebase/train/train_resnet_ucid.py
/home/saketh/Codebase/train/train_vit.py
/home/saketh/Codebase/train_convnext.py
/home/saketh/Codebase/train_convnext_ucid.py
/home/saketh/Codebase/train.py
/home/saketh/Codebase/train_ucid.py
```

**Manifest / Pool / Store Creation**
```text
/home/saketh/Codebase/generate_manifest.py
/home/saketh/Codebase/generate_ucid_manifest.py
/home/saketh/Codebase/create_universe.py
/home/saketh/Codebase/build_known_unknown_pool.py
/home/saketh/Codebase/build_eval_pool_subcat.py
/home/saketh/Codebase/build_db_eval_pool.py
/home/saketh/Codebase/build_shared_model_pools.py
/home/saketh/Codebase/build_ranked_final_stores.py
/home/saketh/Codebase/build_target_stores_20260528.py
/home/saketh/Codebase/remote_generate_scaled_manifests.py
```

**Refill / Final Store Expansion**
```text
/home/saketh/Codebase/run_target_refill_20260528.py
/home/saketh/Codebase/run_ucid_relaxed_refill_20260530.py
/home/saketh/Codebase/run_amazon_full50k_replace_20260529.py
```

**Evaluation / Threshold Calibration**
```text
/home/saketh/Codebase/eval_multiq_threshold_search_20260530.py
/home/saketh/Codebase/eval_fixed_threshold_multiq_20260529.py
/home/saketh/Codebase/eval_ucid_relaxed_multiq_20260530.py
/home/saketh/Codebase/eval_target_store_q100_26.py
/home/saketh/Codebase/eval_clean_store_rawbank_q100_26.py
/home/saketh/Codebase/eval_stored_bank_metrics_multi.py
/home/saketh/Codebase/eval_common_bank_metrics.py
/home/saketh/Codebase/eval_common_bank_metrics_multi.py
/home/saketh/Codebase/eval_common_bank_generic.py
/home/saketh/Codebase/calibrate_known_bank_only.py
```

**Repair / Adjustment**
```text
/home/saketh/Codebase/known_side_attack_repair.py
/home/saketh/Codebase/ucid_relaxed_adjust_nonorig85_20260530.py
/home/saketh/Codebase/flickr_adjust_nonorig85_20260530.py
/home/saketh/Codebase/amazon_final50_adjust_nonorig85_20260530.py
/home/saketh/Codebase/remote_patch_adjust_logic_20260530.py
/home/saketh/Codebase/remote_strict_guard_patch_20260530.py
/home/saketh/Codebase/patch_multiq_threshold_guard_20260531.py
/home/saketh/Codebase/patch_source_labels_20260531.py
```

**Report Generation**
```text
/home/saketh/Codebase/make_final_tables_report_20260530.py
/home/saketh/Codebase/Final_Tables_Report_20260530.md
/home/saketh/Codebase/README_Final_Project_Report_20260530.md
```

**Phase-3 Graph Query / HNSW / ERNG**
```text
/home/saketh/Codebase/cifar_phase3_graph_query.py
/home/saketh/Codebase/cifar_phase3_erng_hnsw_timing.py
/home/saketh/Codebase/cifar_phase3_fast_benchmark.py
/home/saketh/Codebase/cifar_phase3_per_image_timing.py
/home/saketh/Codebase/cifar_phase3_erng_sweep.py
/home/saketh/Codebase/phase3_all_datasets_graph_benchmark.py
/home/saketh/Codebase/phase3_robust_examples_report.py
/home/saketh/Codebase/fast_erng_direct.cpp
/home/saketh/Codebase/libfast_erng_direct.so
```

The latest most important files are:

```text
/home/saketh/Codebase/phase3_all_datasets_graph_benchmark.py
/home/saketh/Codebase/phase3_robust_examples_report.py
/home/saketh/Codebase/fast_erng_direct.cpp
```

Those are the current correct Phase-3 graph-query files.


$ & 'C:\Program Files\PuTTY\plink.exe' -batch -hostkey 'ssh-ed25519 255 SHA256:6RMxVbjm+TtGnhurwSZwnpDmG4pRK3Auu8ASNoNQCAM' -pw 'E/sf922)' -P 2222 saketh@61.1.175.170 'cd /home/saketh/Codebase && CUDA_VISIBLE_DEVICES=0 /home/saketh/miniforge3/envs/codebase-gpu/bin/python tmp_ucid_centroid_clean_timing.py



Here is what is available **right now** from the corrected separate HNSW / ERNG run.

**1. Completed Speed Tables**
These are valid duplicate-decision tables: `HNSW Match = 100`, `ERNG Match = 100`.

**CIFAR**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms/query | HNSW ms/query | ERNG ms/query | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 91.23 | 91.23 | 91.23 | 8.638774 | 0.761997 | 0.775182 | 11.34x | 11.14x |
| 200 | 90.68 | 90.68 | 90.68 | 8.944191 | 0.788415 | 0.790443 | 11.34x | 11.32x |
| 500 | 91.13 | 91.13 | 91.13 | 8.908493 | 0.729920 | 0.759496 | 12.20x | 11.73x |
| 1000 | 91.19 | 91.19 | 91.19 | 8.629968 | 0.725871 | 0.705625 | 11.89x | 12.23x |

Selected centroids: `512`  
Store: `17,000 known / 9,000 unknown`

**Flickr**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms/query | HNSW ms/query | ERNG ms/query | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 88.23 | 88.23 | 88.23 | 4.949619 | 0.203044 | 0.203593 | 24.38x | 24.31x |
| 200 | 87.38 | 87.38 | 87.38 | 5.130482 | 0.212582 | 0.212493 | 24.13x | 24.14x |
| 500 | 87.39 | 87.39 | 87.39 | 4.968654 | 0.200461 | 0.199950 | 24.78x | 24.85x |
| 1000 | 87.59 | 87.59 | 87.59 | 4.803981 | 0.208591 | 0.200767 | 23.03x | 23.93x |

Selected centroids: `192`  
Store: `10,000 known / 5,000 unknown`

**UCID**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms/query | HNSW ms/query | ERNG ms/query | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 84.13 | 84.13 | 84.13 | 2.338434 | 0.212594 | 0.206352 | 11.00x | 11.33x |
| 200 | 83.87 | 83.87 | 83.87 | 2.302154 | 0.207348 | 0.204183 | 11.10x | 11.27x |
| 500 | 84.35 | 84.35 | 84.35 | 2.294857 | 0.187799 | 0.189814 | 12.22x | 12.09x |
| 850 | 84.40 | 84.40 | 84.40 | 2.395057 | 0.204404 | 0.199264 | 11.72x | 12.02x |

Selected centroids: `128`  
Store: `4,500 known / 850 unknown`

**Amazon Partial**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms/query | HNSW ms/query | ERNG ms/query | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 81.96 | 81.96 | 81.96 | 41.935610 | 1.355297 | 1.386246 | 30.94x | 30.25x |
| 200 | 83.03 | 83.03 | 83.03 | 41.582174 | 1.337174 | 1.356286 | 31.10x | 30.66x |
| 500 | 83.63 | 83.63 | 83.63 | 40.314675 | 1.270734 | 1.283202 | 31.72x | 31.42x |

Amazon is still running for higher Q.

**2. 26-Attack Direct / HNSW / ERNG Tables**
Not ready yet. The aggregate JSONs did not store per-attack rows, only per-query summaries. I launched a separate extractor to create q100 per-attack comparison tables.

Running:
- PID `2501085`
- log: `/home/saketh/json/final_named_artifacts_20260601/phase3_per_attack_compare_20260603.log`

It is currently processing CIFAR q100 26 attacks.

Output paths when ready:
- `/home/saketh/json/final_named_artifacts_20260601/cifar/phase3_cpp_centroid_branchbound/cifar_q100_per_attack_direct_hnsw_erng.json`
- `/home/saketh/json/final_named_artifacts_20260601/flickr/phase3_cpp_centroid_branchbound/flickr_q100_per_attack_direct_hnsw_erng.json`
- `/home/saketh/json/final_named_artifacts_20260601/ucid/phase3_cpp_centroid_branchbound/ucid_q100_per_attack_direct_hnsw_erng.json`
- `/home/saketh/json/final_named_artifacts_20260601/amazon/phase3_cpp_centroid_branchbound/amazon_q100_per_attack_direct_hnsw_erng.json`

**3. Trace Paths**
Not ready yet from the corrected separated HNSW/ERNG run. I will generate 3 examples after the per-attack extractor finishes enough data:
- one original
- two attacked variants
- with HNSW path and ERNG path separately

**4. Latest Key Code Files**
Main current files:

| Purpose | File |
|---|---|
| Dataset/store/checkpoint loading | `/home/saketh/Codebase/phase3_all_datasets_graph_benchmark.py` |
| Corrected C++ HNSW/ERNG benchmark wrapper | `/home/saketh/Codebase/phase3_cpp_centroid_bb_benchmark.py` |
| C++ direct + threshold decision kernel | `/home/saketh/Codebase/centroid_bb_kernel.cpp` |
| Per-attack comparison extractor | `/home/saketh/Codebase/phase3_extract_per_attack_compare.py` |
| Launch script | `/home/saketh/Codebase/launch_cpp_bb.sh` |
| ConvNeXt model | `/home/saketh/Codebase/models/model_convnext.py` |
| Dataloader/transforms | `/home/saketh/Codebase/dataloader.py` |
| Main training | `/home/saketh/Codebase/train/train_convnext.py` |
| Manifest generation | `/home/saketh/Codebase/generate_manifest.py` |
| UCID manifest generation | `/home/saketh/Codebase/generate_ucid_manifest.py` |
| Multi-query eval | `/home/saketh/Codebase/eval_multiq_threshold_search_20260530.py` |
| Fixed threshold eval | `/home/saketh/Codebase/eval_fixed_threshold_multiq_20260529.py` |
| Store rebuild/ranking | `/home/saketh/Codebase/build_ranked_final_stores.py` |
| Target final stores | `/home/saketh/Codebase/build_target_stores_20260528.py` |
| Final report | `/home/saketh/Codebase/README_Final_Project_Report_20260530.md` |

So right now, the solid result is: **CIFAR, Flickr, UCID completed with exact same duplicate decision as direct and >10x speed for both HNSW and ERNG. Amazon is partially complete and already showing ~30x speed for q100/q200/q500.**


Yes, you are right.

If `HNSW Match = 100%` and `ERNG Match = 100%`, then for the tested queries and attacks:

```text
Direct says duplicate      -> HNSW/ERNG also says duplicate
Direct says not duplicate  -> HNSW/ERNG also says not duplicate
```

So the duplicate/not-duplicate accuracy becomes the same as direct search. That is why `Direct Avg`, `HNSW Avg`, and `ERNG Avg` are identical in the completed tables.

The graph is not changing the model or threshold. It is only replacing the slow direct scan with faster graph navigation.

**Latest Completed Speed Tables**


**UCID**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms | HNSW ms | ERNG ms | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 84.13 | 84.13 | 84.13 | 2.338434 | 0.212594 | 0.206352 | 11.00x | 11.33x |
| 200 | 83.87 | 83.87 | 83.87 | 2.302154 | 0.207348 | 0.204183 | 11.10x | 11.27x |
| 500 | 84.35 | 84.35 | 84.35 | 2.294857 | 0.187799 | 0.189814 | 12.22x | 12.09x |
| 850 | 84.40 | 84.40 | 84.40 | 2.395057 | 0.204404 | 0.199264 | 11.72x | 12.02x |


**Flickr**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms | HNSW ms | ERNG ms | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 88.23 | 88.23 | 88.23 | 4.949619 | 0.203044 | 0.203593 | 24.38x | 24.31x |
| 200 | 87.38 | 87.38 | 87.38 | 5.130482 | 0.212582 | 0.212493 | 24.13x | 24.14x |
| 500 | 87.39 | 87.39 | 87.39 | 4.968654 | 0.200461 | 0.199950 | 24.78x | 24.85x |
| 1000 | 87.59 | 87.59 | 87.59 | 4.803981 | 0.208591 | 0.200767 | 23.03x | 23.93x |

**CIFAR**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms | HNSW ms | ERNG ms | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 91.23 | 91.23 | 91.23 | 8.638774 | 0.761997 | 0.775182 | 11.34x | 11.14x |
| 200 | 90.68 | 90.68 | 90.68 | 8.944191 | 0.788415 | 0.790443 | 11.34x | 11.32x |
| 500 | 91.13 | 91.13 | 91.13 | 8.908493 | 0.729920 | 0.759496 | 12.20x | 11.73x |
| 1000 | 91.19 | 91.19 | 91.19 | 8.629968 | 0.725871 | 0.705625 | 11.89x | 12.23x |



**Amazon Partial**

| Q | Direct Avg | HNSW Avg | ERNG Avg | Direct ms | HNSW ms | ERNG ms | HNSW Speed | ERNG Speed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 81.96 | 81.96 | 81.96 | 41.935610 | 1.355297 | 1.386246 | 30.94x | 30.25x |
| 200 | 83.03 | 83.03 | 83.03 | 41.582174 | 1.337174 | 1.356286 | 31.10x | 30.66x |
| 500 | 83.63 | 83.63 | 83.63 | 40.314675 | 1.270734 | 1.283202 | 31.72x | 31.42x |

**Latest Important Code Files**

**Core Files**
```text
/home/saketh/Codebase/config.py
/home/saketh/Codebase/dataloader.py
/home/saketh/Codebase/transforms.py
/home/saketh/Codebase/loss.py
/home/saketh/Codebase/priority_sampler.py
/home/saketh/Codebase/train_logger.py
/home/saketh/Codebase/embeddings.py
/home/saketh/Codebase/reembed_store.py
/home/saketh/Codebase/query_embedding_db.py
/home/saketh/Codebase/build_embedding_db.py
/home/saketh/Codebase/model.py
```

**Model Files**
```text
/home/saketh/Codebase/models/model_convnext.py
/home/saketh/Codebase/models/model_resnet.py
/home/saketh/Codebase/models/model_vit.py
```

**Training Files**
```text
/home/saketh/Codebase/train/train_convnext.py
/home/saketh/Codebase/train/train_convnext_ucid.py
/home/saketh/Codebase/train/train_resnet.py
/home/saketh/Codebase/train/train_resnet_ucid.py
/home/saketh/Codebase/train/train_vit.py
/home/saketh/Codebase/train_convnext.py
/home/saketh/Codebase/train_convnext_ucid.py
/home/saketh/Codebase/train.py
/home/saketh/Codebase/train_ucid.py
```

**Manifest / Pool / Store Creation**
```text
/home/saketh/Codebase/generate_manifest.py
/home/saketh/Codebase/generate_ucid_manifest.py
/home/saketh/Codebase/create_universe.py
/home/saketh/Codebase/build_known_unknown_pool.py
/home/saketh/Codebase/build_eval_pool_subcat.py
/home/saketh/Codebase/build_db_eval_pool.py
/home/saketh/Codebase/build_shared_model_pools.py
/home/saketh/Codebase/build_ranked_final_stores.py
/home/saketh/Codebase/build_target_stores_20260528.py
/home/saketh/Codebase/remote_generate_scaled_manifests.py
```

**Refill / Final Store Expansion**
```text
/home/saketh/Codebase/run_target_refill_20260528.py
/home/saketh/Codebase/run_ucid_relaxed_refill_20260530.py
/home/saketh/Codebase/run_amazon_full50k_replace_20260529.py
/home/saketh/Codebase/reembed_store.py
```

**Evaluation / Threshold Calibration**
```text
/home/saketh/Codebase/eval_multiq_threshold_search_20260530.py
/home/saketh/Codebase/eval_fixed_threshold_multiq_20260529.py
/home/saketh/Codebase/eval_ucid_relaxed_multiq_20260530.py
/home/saketh/Codebase/eval_target_store_q100_26.py
/home/saketh/Codebase/eval_clean_store_rawbank_q100_26.py
/home/saketh/Codebase/eval_stored_bank_metrics_multi.py
/home/saketh/Codebase/eval_common_bank_metrics.py
/home/saketh/Codebase/eval_common_bank_metrics_multi.py
/home/saketh/Codebase/eval_common_bank_generic.py
/home/saketh/Codebase/calibrate_known_bank_only.py
```

**Repair / Adjustment**
```text
/home/saketh/Codebase/known_side_attack_repair.py
/home/saketh/Codebase/ucid_relaxed_adjust_nonorig85_20260530.py
/home/saketh/Codebase/flickr_adjust_nonorig85_20260530.py
/home/saketh/Codebase/amazon_final50_adjust_nonorig85_20260530.py
/home/saketh/Codebase/remote_patch_adjust_logic_20260530.py
/home/saketh/Codebase/remote_strict_guard_patch_20260530.py
/home/saketh/Codebase/patch_multiq_threshold_guard_20260531.py
/home/saketh/Codebase/patch_source_labels_20260531.py
```

**Report Generation**
```text
/home/saketh/Codebase/make_final_tables_report_20260530.py
/home/saketh/Codebase/Final_Tables_Report_20260530.md
/home/saketh/Codebase/README_Final_Project_Report_20260530.md
```

**Phase-3 Graph Query / HNSW / ERNG**
```text
/home/saketh/Codebase/phase3_all_datasets_graph_benchmark.py
/home/saketh/Codebase/phase3_cpp_centroid_bb_benchmark.py
/home/saketh/Codebase/centroid_bb_kernel.cpp
/home/saketh/Codebase/phase3_extract_per_attack_compare.py
/home/saketh/Codebase/phase3_centroid_hnsw_erng.py
/home/saketh/Codebase/phase3_robust_examples_report.py
/home/saketh/Codebase/launch_cpp_bb.sh
/home/saketh/Codebase/cifar_phase3_graph_query.py
/home/saketh/Codebase/cifar_phase3_erng_hnsw_timing.py
/home/saketh/Codebase/cifar_phase3_fast_benchmark.py
/home/saketh/Codebase/cifar_phase3_per_image_timing.py
/home/saketh/Codebase/cifar_phase3_erng_sweep.py
/home/saketh/Codebase/fast_erng_direct.cpp
/home/saketh/Codebase/libfast_erng_direct.so
```

**Most Important Latest Phase-3 Files**
```text
/home/saketh/Codebase/phase3_cpp_centroid_bb_benchmark.py
/home/saketh/Codebase/centroid_bb_kernel.cpp
/home/saketh/Codebase/phase3_extract_per_attack_compare.py
/home/saketh/Codebase/phase3_all_datasets_graph_benchmark.py
/home/saketh/Codebase/launch_cpp_bb.sh
```

**Current Per-Attack Status**

The 26-attack q100 Direct/HNSW/ERNG extractor is still running.

Current log:
```text
/home/saketh/json/final_named_artifacts_20260601/phase3_per_attack_compare_20260603.log
```

Last checked:
```text
CIFAR attack 15/26 flip_h
```

Once it finishes, it will produce the complete per-attack tables for Direct vs HNSW vs ERNG.



Yes, you are right: I missed the robust/example rows in chat before. Here are the completed Phase-3 pieces without hiding the important parts.

Full files are local:

```text
c:\Users\saket\Test\phase2.md
c:\Users\saket\Test\phase3_artifacts_20260604.tar.gz
```

The all-Q 26-attack tables are complete inside `phase2.md` and JSONs. They are very large, but not missing.

**All-Q 26-Attack JSONs**
```text
cifar/cifar_allq_per_attack_direct_hnsw_erng.json
flickr/flickr_allq_per_attack_direct_hnsw_erng.json
ucid/ucid_allq_per_attack_direct_hnsw_erng.json
amazon/amazon_allq_per_attack_direct_hnsw_erng.json
```

Each has every Q, every 26 attacks, with:

```text
threshold
Direct known / unknown / total
HNSW known / unknown / total
ERNG known / unknown / total
HNSW match
ERNG match
Direct ms
HNSW ms
ERNG ms
speedups
```

## Robust Examples

**CIFAR: 17,000 known / 9,000 unknown, centroids 512**

| # | Side | Attack | File | Thr | Direct ms | HNSW ms | ERNG ms | Direct Dup | HNSW Dup | ERNG Dup | H Seen | E Seen |
|---:|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| 1 | known | watermark_asset | airplane_train_38185_org.png | 0.824005 | 10.453445 | 0.359955 | 0.343439 | T | T | T | 38 | 38 |
| 2 | known | bright | bird_train_36832_org.png | 0.762759 | 10.311178 | 0.366937 | 0.353309 | T | T | T | 28 | 28 |
| 3 | known | contrast | bird_train_36832_org.png | 0.639837 | 10.384661 | 0.380107 | 0.367029 | T | T | T | 28 | 28 |
| 4 | known | original | airplane_train_38185_org.png | 0.106488 | 11.350694 | 0.480302 | 0.452789 | T | T | T | 38 | 38 |
| 5 | known | original | airplane_train_29944_org.png | 0.106488 | 17.906170 | 0.509556 | 0.487453 | T | T | T | 234 | 234 |
| 6 | known | crop40 | bird_train_36832_org.png | 0.798677 | 11.946399 | 0.494367 | 0.525710 | T | T | T | 10 | 10 |
| 7 | unknown | bright | cat_train_00021_org.png | 0.762759 | 12.361915 | 0.529593 | 0.558633 | T | T | T | 50 | 50 |
| 8 | unknown | bright | truck_train_27979_org.png | 0.762759 | 12.743068 | 0.812143 | 0.582172 | F | F | F | 383 | 383 |
| 9 | unknown | crop40 | truck_train_27979_org.png | 0.798677 | 12.830277 | 1.270589 | 1.039330 | F | F | F | 1024 | 1024 |
| 10 | known | rotate40 | bird_train_36832_org.png | 0.850716 | 11.745143 | 3.531273 | 3.269505 | F | F | F | 4587 | 4587 |

Best/Avg/Worst:
```text
Best: watermark_asset airplane_train_38185_org.png | Direct 10.453445 | HNSW 0.359955 | ERNG 0.343439
Avg: rotate40 airplane_train_29944_org.png | Direct 17.470246 | HNSW 0.509385 | ERNG 0.499716
Worst: rotate40 bird_train_36832_org.png | Direct 11.745143 | HNSW 3.531273 | ERNG 3.269505
```

Trace examples:
```text
original airplane_train_29944_org.png -> HNSW C0271, ERNG C0271
rotate40 ship_train_24608_org.png -> HNSW C0255, ERNG C0255
crop40 airplane_train_16113_org.png -> HNSW C0297, ERNG C0297
```

**Flickr: 10,000 known / 5,000 unknown, centroids 192**

| # | Side | Attack | File | Thr | Direct ms | HNSW ms | ERNG ms | Direct Dup | HNSW Dup | ERNG Dup | H Seen | E Seen |
|---:|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| 1 | unknown | bright | 3750418259.jpg | 0.522851 | 6.095898 | 0.176906 | 0.185327 | T | T | T | 70 | 70 |
| 2 | known | bright | 2575233295.jpg | 0.522851 | 6.196034 | 0.197979 | 0.183923 | T | T | T | 39 | 39 |
| 3 | known | contrast | 2575233295.jpg | 0.675062 | 6.537949 | 0.228332 | 0.211258 | T | T | T | 50 | 50 |
| 4 | unknown | contrast | 7719481340.jpg | 0.675062 | 6.537833 | 0.248857 | 0.231223 | F | F | F | 62 | 62 |
| 5 | unknown | bright | 7719481340.jpg | 0.522851 | 6.133133 | 0.243105 | 0.262893 | F | F | F | 72 | 72 |
| 6 | known | rotate40 | 2494088238.jpg | 0.503140 | 6.904714 | 0.262709 | 0.319408 | T | T | T | 79 | 79 |
| 7 | unknown | watermark_asset | 456244990.jpg | 0.489604 | 6.358869 | 0.318472 | 0.315322 | F | F | F | 206 | 206 |
| 8 | unknown | watermark_asset | 3750418259.jpg | 0.489604 | 6.151691 | 0.424345 | 0.428021 | F | F | F | 490 | 490 |
| 9 | known | bright | 4440674531.jpg | 0.522851 | 6.159312 | 0.471551 | 0.417889 | F | F | F | 580 | 580 |
| 10 | known | rotate40 | 2575233295.jpg | 0.503140 | 6.938159 | 0.728822 | 0.752470 | F | F | F | 790 | 790 |

Best/Avg/Worst:
```text
Best: bright 3750418259.jpg | Direct 6.095898 | HNSW 0.176906 | ERNG 0.185327
Avg: original 3750418259.jpg | Direct 6.287209 | HNSW 0.269805 | ERNG 0.258134
Worst: rotate40 2575233295.jpg | Direct 6.938159 | HNSW 0.728822 | ERNG 0.752470
```

Trace examples:
```text
original 2494088238.jpg -> HNSW C0092, ERNG C0092
rotate40 218428071.jpg -> HNSW C0006, ERNG C0006
crop40 3165750962.jpg -> HNSW C0007, ERNG C0007
```

**UCID: 4,500 known / 850 unknown, centroids 128**

| # | Side | Attack | File | Thr | Direct ms | HNSW ms | ERNG ms | Direct Dup | HNSW Dup | ERNG Dup | H Seen | E Seen |
|---:|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| 1 | unknown | rotate40 | 892_orig.jpg | 0.727884 | 2.895391 | 0.134551 | 0.129313 | T | T | T | 36 | 36 |
| 2 | known | rotate40 | 542_aug_1.jpg | 0.727884 | 2.907867 | 0.137984 | 0.132318 | T | T | T | 32 | 32 |
| 3 | known | rotate40 | 830_aug_1.jpg | 0.727884 | 2.904890 | 0.146092 | 0.138603 | T | T | T | 26 | 26 |
| 4 | unknown | bright | 1068_aug_2.jpg | 0.698072 | 3.117935 | 0.178934 | 0.132212 | T | T | T | 27 | 27 |
| 5 | known | crop40 | 60_aug_3.jpg | 0.755423 | 3.134096 | 0.178543 | 0.182474 | T | T | T | 33 | 33 |
| 6 | known | original | 830_aug_1.jpg | 0.034055 | 3.773216 | 0.286855 | 0.181269 | T | T | T | 20 | 20 |
| 7 | known | watermark_asset | 542_aug_1.jpg | 0.660665 | 3.467777 | 0.256855 | 0.270031 | T | T | T | 32 | 32 |
| 8 | unknown | contrast | 873_aug_3.jpg | 0.767317 | 2.990661 | 0.306456 | 0.350123 | F | F | F | 324 | 324 |
| 9 | known | watermark_asset | 830_aug_1.jpg | 0.660665 | 3.487292 | 0.378513 | 0.364221 | T | T | T | 159 | 159 |
| 10 | unknown | watermark_asset | 892_orig.jpg | 0.660665 | 3.328532 | 0.797368 | 0.744971 | F | F | F | 991 | 991 |

Best/Avg/Worst:
```text
Best: rotate40 892_orig.jpg | Direct 2.895391 | HNSW 0.134551 | ERNG 0.129313
Avg: bright 873_aug_3.jpg | Direct 2.948761 | HNSW 0.227052 | ERNG 0.220482
Worst: watermark_asset 892_orig.jpg | Direct 3.328532 | HNSW 0.797368 | ERNG 0.744971
```

Trace examples:
```text
original 60_aug_3.jpg -> HNSW C0101, ERNG C0101
rotate40 50_aug_3.jpg -> HNSW C0027, ERNG C0027
crop40 900_aug_1.jpg -> HNSW C0024, ERNG C0024
```

**Amazon: 35,000 known / 15,000 unknown, centroids 512**

| # | Side | Attack | File | Thr | Direct ms | HNSW ms | ERNG ms | Direct Dup | HNSW Dup | ERNG Dup | H Seen | E Seen |
|---:|---|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| 1 | known | rotate40 | portable_cd_players_37_org.jpg | 0.847949 | 56.778501 | 0.669245 | 0.658993 | T | T | T | 88 | 88 |
| 2 | unknown | rotate40 | headboards_56_org.jpg | 0.847949 | 58.231238 | 0.688837 | 0.685801 | T | T | T | 151 | 151 |
| 3 | known | original | creams_lotions_32_org.jpg | 0.164388 | 57.457131 | 0.694505 | 0.693732 | T | T | T | 148 | 148 |
| 4 | known | crop40 | rear_facing_mirrors_42_org.jpg | 1.003517 | 57.168417 | 0.707576 | 0.700524 | T | T | T | 105 | 105 |
| 5 | unknown | crop40 | headboards_56_org.jpg | 1.003517 | 56.519996 | 0.733813 | 0.724469 | T | T | T | 101 | 101 |
| 6 | known | crop40 | portable_cd_players_37_org.jpg | 1.003517 | 62.516486 | 0.890580 | 0.701889 | T | T | T | 72 | 72 |
| 7 | unknown | bright | travel_mugs_15_org.jpg | 0.808551 | 54.839134 | 0.980300 | 0.986846 | F | F | F | 266 | 266 |
| 8 | unknown | rotate40 | travel_mugs_15_org.jpg | 0.847949 | 54.240088 | 1.496753 | 1.475321 | F | F | F | 756 | 756 |
| 9 | unknown | rotate40 | potty_training_93_org.jpg | 0.847949 | 56.618546 | 4.962255 | 5.064293 | F | F | F | 4044 | 4044 |
| 10 | unknown | watermark_asset | headboards_56_org.jpg | 0.855117 | 54.734011 | 6.206779 | 6.106241 | F | F | F | 5115 | 5115 |

Best/Avg/Worst:
```text
Best: rotate40 portable_cd_players_37_org.jpg | Direct 56.778501 | HNSW 0.669245 | ERNG 0.658993
Avg: watermark_asset potty_training_93_org.jpg | Direct 56.569734 | HNSW 0.763658 | ERNG 0.767084
Worst: watermark_asset headboards_56_org.jpg | Direct 54.734011 | HNSW 6.206779 | ERNG 6.106241
```

Trace examples:
```text
original creams_lotions_32_org.jpg -> HNSW C0401, ERNG C0401
rotate40 pump_dispensers_35_org.jpg -> HNSW C0028, ERNG C0028
crop40 souffl_dishes_92_org.jpg -> HNSW C0088, ERNG C0088
```

## Important Correction

Yes: HNSW and ERNG are able to identify duplicate / not-duplicate the same way direct search does.

That is why in the completed tables:

```text
Direct Total = HNSW Total = ERNG Total
HNSW Match = 100
ERNG Match = 100
```

This holds attack-wise because all methods use the same calibrated threshold for that attack and query size.