## HNSW Timing Clarification


Your intended rule is:

- Sweep the planned centroid ranges.
- As dataset size increases, the useful centroid count and graph parameters should generally increase.
- For each dataset, search time should improve up to a best region, then worsen.
- Select the stable minimum-time region, not simply the largest graph.
- HNSW and ERNG may have different winning parameters, but their selected configurations should be comparable.
- Timing should remain nearly stable across q sizes.
- First finish all direct timings; then use them as the correct denominators for HNSW and ERNG.

Amazon q1000 was the final pending direct job in that archived snapshot.



This is how the current HNSW and ERNG implementations are actually built.

### Common preparation

For each dataset:

1. Load the full embedding bank.
2. Select a centroid count `C`.
3. Load or build `C` centroids.
4. Assign bank embeddings to centroid buckets.
5. Compute the centroid overlap layout and bucket radii.
6. Route each query through the graph to obtain candidate buckets.
7. Compare candidates using the same threshold `tau` as direct search.
8. Compare graph decisions against direct decisions.

### HNSW

HNSW is a genuine hierarchical graph.

1. Each centroid receives a deterministic random level.
2. The graph has layers:

```text
Layer 2: sparse upper navigation
Layer 1: intermediate navigation
Layer 0: dense base graph
```

3. Centroids are inserted in shuffled deterministic order.
4. During insertion, the algorithm:
   - greedily descends from the current upper entry;
   - searches the appropriate layer with `ef_construction`;
   - selects diverse neighbors;
   - adds reverse links;
   - prunes neighbors when degree limits are exceeded.
5. At query time, search begins from the selected top-level entry points.
6. It descends layer by layer using greedy navigation.
7. At layer 0, best-first search with `ef` explores candidate centroids.
8. The resulting centroid buckets are searched for the final duplicate decision.

The HNSW parameters are:

```text
M               upper-layer degree cap
M0              layer-zero degree cap (2M in this builder)
ef_construction construction SearchLayer width
entry_count     optional multi-entry extension; canonical HNSW uses 1
ef_search       layer-zero query SearchLayer width
```

The graph kernel also performs exact radius-bound bucket certification when the graph route has not found a duplicate. That work is part of HNSW query latency; routing-only time must not be presented as end-to-end search time.

### ERNG

ERNG is a flat, single-layer graph. It does not use HNSW layers.

For every centroid:

1. Compute distances to all other centroids.
2. Add the closest `local_degree` neighbors.
3. Add additional rank-based links at ranks:

```text
1, 2, 4, 8, 16, ...
```

4. Remove duplicate links.
5. Add reciprocal links.
6. Keep the closest links within the degree cap.
7. Select deterministic starting centroids using the configured probe count.

At query time:

1. Start from multiple deterministic probe centroids.
2. Each probe greedily follows neighbors that reduce query-to-centroid distance.
3. Each probe stops at a local minimum.
4. The best result from all probes is selected.
5. The selected centroid buckets are searched for the duplicate decision.

The ERNG parameters are:

```text
local_degree   number of local nearest-neighbor links
skip_count     number of rank-based additional links
probes         number of starting centroids
```

### Main Difference

```text
HNSW  = hierarchical layers + layer-by-layer descent
ERNG  = one flat graph + multiple greedy probes
```

Both use the same centroids, bank, buckets, threshold, and direct truth. Only the graph construction and centroid-routing strategy differ.


Yes. The complete dataset is staged on the GPU server at:

`/home/saketh/final_dataset_images_all4_known_unknown_20260916`

Counts:

| Dataset | Known | Unknown |
|---|---:|---:|
| UCID | 4,500 | 850 |
| CIFAR | 17,000 | 9,000 |
| Flickr | 10,000 | 5,000 |
| Amazon | 35,000 | 15,000 |

Total: **95,350 images**. It is ready for copying to local; it is not yet copied locally.


The centroid ranges and parameter lists are complete. The current runner evaluates them using a one-factor-at-a-time sweep around the dataset-scaled baseline, not the full Cartesian product of every parameter combination.

**Final Core Files**

1. [benchmark_graphs.py](/C:/Users/saket/Test/phase-3/code/benchmark_graphs.py)  
   Shared dataset loading, centroid construction, overlap buckets, radii, direct comparison, timing, sweep ranges, and validation calls.

2. [hnsw_hierarchical.py](/C:/Users/saket/Test/phase-3/code/hnsw_hierarchical.py)  
   Custom hierarchical HNSW construction, multi-layer links, entry points, reachability repair, and invariant validation.

3. [erng_flat.py](/C:/Users/saket/Test/phase-3/code/erng_flat.py)  
   Custom flat ERNG construction, local KNN edges, exponential-rank skip edges, probes, reachability repair, and edge validation.

4. [graph_search_kernels.cpp](/C:/Users/saket/Test/phase-3/code/graph_search_kernels.cpp)  
   Compiled direct, HNSW, and ERNG query kernels with exact image verification and single-query OpenMP overhead control.

5. [generalized_phase3_runner_fast.py](/C:/Users/saket/Test/phase-3/code/generalized_phase3_runner_fast.py)  
   Final sweep orchestrator, direct baseline reuse, all-q validation, robust timing, and candidate selection.


Your main expectations are:

1. **Exactness:** one HNSW configuration must match direct search on every query at Q=100, 200, 400, and 500, across the current 19 attacks.
2. **Lower worst-case latency:** reduce the slowest query times and outliers, while keeping per-query timing measurements honest.
3. **Stable Q-size averages:** mean latency should stay as similar as possible across the four Q sizes.
4. **A sensible centroid count:** prefer a middle-range C where search is efficient, not an arbitrary smallest or largest C.
5. **Good speed:** aim near the 3-4x speedup you saw before, but verify it rather than assume it.

For timing, **min / average / max** mean:

- **Min:** fastest individual query in the measured set.
- **Average:** mean time per query.
- **Max:** slowest individual query in the measured set.



// UCID

**UCID Sweep**

Centroids tested: `16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024` with 81 configurations per C: `1,053` total.

All `1,053 / 1,053` received full-Q validation; `645` achieved 100% direct agreement across all 19 attacks and Q=`100, 200, 400, 500`.

**Direct Timings**

| Q | Direct min (ms) | Direct mean (ms) | Direct max (ms) |
|---:|---:|---:|---:|
| 100 | 0.401666 | 0.413678 | 0.477084 |
| 200 | 0.404671 | 0.459645 | 0.585211 |
| 400 | 0.405984 | 0.451599 | 0.531802 |
| 500 | 0.404778 | 0.453264 | 0.575124 |

**Best Exact Candidate At Each C**

`O/E/D/ef` means overlap / entries / degree / search depth. Profile uses all full-Q queries; panel is robust repeated query timing.

| C | Config | Full-Q profile mean / max (ms) | Panel mean / max (ms) | Speedup min / avg / max | Avg hops / visited / bank checks | Q spread |
|---:|---|---:|---:|---:|---:|---:|
| 16 | `2/6/24/2` | 0.04561 / 0.21109 | 0.17359 / 0.30356 | 2.480x / 2.560x / 2.588x | 0.00 / 16.00 / 376.0 | 6.29% |
| 24 | `1/3/48/4` | 0.04156 / 0.17585 | 0.17049 / 0.28780 | 2.515x / 2.606x / 2.657x | 1.00 / 24.00 / 337.6 | 5.31% |
| 32 | `4/3/6/2` | 0.04080 / 0.23962 | 0.16957 / 0.32127 | 2.514x / 2.621x / 2.664x | 0.83 / 25.32 / 336.2 | 4.71% |
| 48 | `4/1/12/2` | 0.02741 / 0.13193 | 0.15588 / 0.23997 | 2.736x / 2.851x / 2.938x | 1.17 / 18.59 / 229.4 | 4.69% |
| 64 | `4/1/8/4` | 0.02925 / 0.16693 | 0.15713 / 0.26281 | 2.666x / 2.829x / 2.911x | 1.87 / 16.34 / 239.3 | 1.99% |
| 96 | `1/2/16/12` | 0.03332 / 0.15300 | 0.16011 / 0.24745 | 2.645x / 2.776x / 2.865x | 1.10 / 49.86 / 170.6 | 4.80% |
| 128 | `4/1/32/6` | 0.02883 / 0.12006 | 0.15768 / 0.23214 | 2.655x / 2.819x / 2.901x | 0.87 / 50.15 / 185.2 | 1.67% |
| 192 | `3/1/12/12` | 0.03000 / 0.14889 | 0.15914 / 0.22663 | 2.647x / 2.793x / 2.866x | 1.64 / 37.74 / 188.3 | 2.75% |
| **256** | **`2/1/24/16`** | **0.02987 / 0.09879** | **0.16104 / 0.22088** | **2.615x / 2.760x / 2.829x** | **1.27 / 56.65 / 149.3** | **3.06%** |
| 384 | `2/1/24/16` | 0.03553 / 0.10523 | 0.16713 / 0.20231 | 2.466x / 2.660x / 2.751x | 1.51 / 95.90 / 109.0 | 0.68% |
| 512 | `3/1/6/32` | 0.03642 / 0.12127 | 0.16380 / 0.23544 | 2.551x / 2.714x / 2.801x | 2.59 / 60.52 / 159.0 | 4.15% |
| 768 | `1/1/48/48` | 0.05866 / 0.12624 | 0.19163 / 0.23133 | 2.172x / 2.320x / 2.398x | 0.98 / 184.59 / 73.5 | 2.69% |
| 1024 | `4/2/48/64` | 0.14525 / 0.28855 | 0.27791 / 0.36962 | 1.518x / 1.599x / 1.632x | 1.24 / 369.71 / 155.3 | 3.31% |

**Selected Under Full-Q Tail-Latency Priority**

```text
C = 256
overlap = 2
entries = 1
degree = 24
ef = 16
efConstruction = 96
```

`C=256` has the lowest full-Q slowest-query latency: `0.098795 ms`. It is exact on every full-Q query.

| Q | Direct mean (ms) | HNSW min (ms) | HNSW mean (ms) | HNSW max (ms) | Speedup | Full-workload match |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.413678 | 0.134249 | 0.158165 | 0.203942 | 2.615x | 100% (3,800 / 3,800) |
| 200 | 0.459645 | 0.135020 | 0.162491 | 0.209774 | 2.829x | 100% (7,600 / 7,600) |
| 400 | 0.451599 | 0.133469 | 0.160405 | 0.210658 | 2.815x | 100% (15,200 / 15,200) |
| 500 | 0.453264 | 0.133244 | 0.163095 | 0.220879 | 2.779x | 100% (19,000 / 19,000) |

Across Q sizes, speedup is **2.615x / 2.760x / 2.829x**. The Q-mean spread is **3.06%**. The robust sampled-panel maximum is **0.220879 ms**; the exhaustive full-Q worst query is **0.098795 ms**.


// FLICKR

**Flickr Sweep**

Centroids tested: `32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096` with 81 configurations per C: `1,215` total.

All `1,215 / 1,215` received full-Q validation; `975` achieved 100% direct agreement across all 19 attacks and Q=`100, 200, 500, 1000`.

**Direct Timings**

| Q | Direct min (ms) | Direct mean (ms) | Direct max (ms) |
|---:|---:|---:|---:|
| 100 | 1.740888 | 1.769407 | 1.806591 |
| 200 | 1.731916 | 1.766221 | 1.797852 |
| 500 | 1.731911 | 1.830175 | 1.962801 |
| 1000 | 1.274671 | 1.817681 | 1.957364 |

**Best Exact Candidate At Each C**

| C | Config `O/E/D/ef` | Full-Q profile mean / max (ms) | Panel mean / max (ms) | Speedup min / avg / max | Avg hops / visited / bank checks | Q spread |
|---:|---|---:|---:|---:|---:|---:|
| 32 | `1/4/24/4` | 0.18330 / 0.42000 | 0.21267 / 0.32737 | 8.358x / 8.445x / 8.496x | 0.00 / 32.00 / 807.7 | 5.21% |
| 48 | `1/6/12/4` | 0.09014 / 0.22737 | 0.19366 / 0.27041 | 9.073x / 9.274x / 9.436x | 2.50 / 43.91 / 555.4 | 2.57% |
| 64 | `3/4/16/2` | 0.07493 / 0.24093 | 0.18414 / 0.30127 | 9.556x / 9.753x / 9.931x | 1.07 / 54.37 / 488.2 | 3.80% |
| 96 | `4/1/12/2` | 0.05269 / 0.20243 | 0.17085 / 0.26559 | 10.336x / 10.511x / 10.679x | 1.53 / 23.39 / 425.8 | 0.72% |
| 128 | `1/1/32/8` | 0.04876 / 0.18504 | 0.16135 / 0.21570 | 11.019x / 11.130x / 11.232x | 0.91 / 63.28 / 276.1 | 1.95% |
| 192 | `1/1/32/8` | 0.03881 / 0.13546 | 0.15526 / 0.19848 | 11.348x / 11.567x / 11.769x | 0.79 / 69.69 / 187.7 | 0.47% |
| 256 | `1/1/32/8` | 0.03250 / 0.11809 | 0.15001 / 0.19472 | 11.698x / 11.973x / 12.325x | 0.95 / 73.86 / 140.4 | 2.53% |
| 384 | `4/1/8/4` | 0.02802 / 0.11728 | 0.14992 / 0.19480 | 11.717x / 11.980x / 12.225x | 2.38 / 29.45 / 177.1 | 1.55% |
| 512 | `1/1/32/8` | 0.02626 / 0.12384 | 0.14612 / 0.17237 | 11.962x / 12.292x / 12.521x | 1.51 / 85.52 / 66.4 | 1.89% |
| 768 | `4/1/8/4` | 0.02040 / 0.09979 | 0.14137 / 0.16643 | 12.363x / 12.704x / 13.015x | 2.47 / 32.74 / 89.3 | 2.02% |
| 1024 | `4/1/32/6` | 0.02736 / 0.10316 | 0.15017 / 0.17747 | 11.722x / 11.959x / 12.161x | 1.77 / 82.87 / 95.5 | 1.80% |
| 1536 | `4/1/32/6` | 0.02501 / 0.10069 | 0.14858 / 0.17048 | 11.778x / 12.088x / 12.316x | 1.71 / 91.81 / 63.4 | 1.41% |
| **2048** | **`3/1/12/12`** | **0.02619 / 0.08705** | **0.14654 / 0.17168** | **12.062x / 12.255x / 12.475x** | **2.61 / 68.86 / 65.1** | **1.53%** |
| 3072 | `3/1/12/12` | 0.02389 / 0.09027 | 0.14596 / 0.16673 | 12.094x / 12.304x / 12.577x | 2.99 / 70.63 / 443.3 | 1.20% |
| 4096 | `3/1/12/12` | 0.02157 / 0.09149 | 0.14629 / 0.16335 | 12.021x / 12.277x / 12.585x | 3.39 / 70.44 / 331.3 | 2.86% |

**Selected Under Full-Q Tail-Latency Priority**

```text
C = 2048
overlap = 3
entries = 1
degree = 12
ef = 12
efConstruction = 64
```

`C=2048` has the lowest exhaustive full-Q slowest-query latency: **0.087046 ms**. It is not the largest tested C, and it is exact for every full-Q query.

| Q | Direct mean (ms) | HNSW min (ms) | HNSW mean (ms) | HNSW max (ms) | Speedup | Full-workload match |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.769407 | 0.129985 | 0.146698 | 0.163953 | 12.062x | 100% (3,800 / 3,800) |
| 200 | 1.766221 | 0.129346 | 0.145776 | 0.165371 | 12.116x | 100% (7,600 / 7,600) |
| 500 | 1.830175 | 0.130586 | 0.147961 | 0.170574 | 12.369x | 100% (19,000 / 19,000) |
| 1000 | 1.817681 | 0.128684 | 0.145712 | 0.171684 | 12.475x | 100% (38,000 / 38,000) |

Across Q sizes, speedup is **12.062x / 12.255x / 12.475x**. Q-mean spread is **1.53%**. The robust panel maximum is **0.171684 ms**; the exhaustive full-Q worst query is **0.087046 ms**.


**CIFAR Sweep**

Centroids tested: `32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048` with 81 configurations per C: `1,053` total.

All `1,053 / 1,053` received full-Q validation; `584` achieved 100% direct agreement across all 19 attacks and Q=`100, 200, 500, 1000`.

**Direct Timings**

| Q | Direct min (ms) | Direct mean (ms) | Direct max (ms) |
|---:|---:|---:|---:|
| 100 | 2.058077 | 2.213496 | 2.347074 |
| 200 | 2.016837 | 2.212187 | 2.318241 |
| 500 | 1.999674 | 2.151239 | 2.203766 |
| 1000 | 2.047025 | 2.173764 | 2.207718 |

**Best Exact Candidate At Each C**

| C | Config `O/E/D/ef` | Full-Q profile mean / max (ms) | Panel mean / max (ms) | Speedup min / avg / max | Avg hops / visited / bank checks | Q spread |
|---:|---|---:|---:|---:|---:|---:|
| 32 | `2/2/8/6` | 0.41150 / 6.78189 | 0.40356 / 0.84147 | 5.257x / 5.423x / 5.492x | 0.89 / 24.48 / 2761.4 | 7.22% |
| 48 | `1/2/16/12` | 0.25676 / 1.00948 | 0.32412 / 0.74632 | 6.551x / 6.753x / 6.900x | 0.54 / 41.28 / 1927.2 | 6.99% |
| 64 | `1/2/16/12` | 0.18799 / 4.48506 | 0.28635 / 0.60855 | 7.487x / 7.641x / 7.850x | 1.26 / 48.43 / 1518.5 | 2.81% |
| 96 | `4/1/8/4` | 0.10990 / 0.69042 | 0.22064 / 0.51249 | 9.772x / 9.917x / 10.100x | 1.68 / 23.45 / 898.1 | 6.07% |
| 128 | `4/1/32/6` | 0.12365 / 0.63164 | 0.23416 / 0.43087 | 9.279x / 9.343x / 9.434x | 0.87 / 63.26 / 961.5 | 3.14% |
| 192 | `4/1/32/6` | 0.09636 / 0.43313 | 0.20939 / 0.36880 | 10.417x / 10.448x / 10.492x | 0.96 / 76.82 / 681.5 | 2.43% |
| 256 | `2/1/24/16` | 0.11198 / 0.48075 | 0.22433 / 0.36405 | 9.584x / 9.755x / 10.093x | 1.60 / 91.98 / 758.4 | 5.14% |
| 384 | `4/1/32/6` | 0.05705 / 0.22877 | 0.17859 / 0.27170 | 12.224x / 12.250x / 12.265x | 1.36 / 84.34 / 380.7 | 2.78% |
| 512 | `3/1/12/12` | 0.06127 / 0.23242 | 0.18204 / 0.26061 | 11.865x / 12.018x / 12.258x | 2.11 / 67.84 / 421.4 | 3.17% |
| 768 | `2/1/24/16` | 0.05477 / 0.19419 | 0.17820 / 0.23626 | 12.138x / 12.276x / 12.489x | 1.54 / 120.41 / 289.4 | 1.63% |
| 1024 | `2/1/16/24` | 0.06281 / 0.19465 | 0.18386 / 0.24512 | 11.786x / 11.898x / 12.022x | 2.47 / 129.50 / 318.2 | 1.27% |
| **1536** | **`3/2/32/16`** | **0.07554 / 0.17376** | **0.20278 / 0.25313** | **10.745x / 10.789x / 10.912x** | **1.65 / 224.54 / 198.6** | **2.83%** |
| 2048 | `4/2/16/32` | 0.10066 / 0.23618 | 0.22178 / 0.31387 | 9.716x / 9.865x / 10.072x | 2.28 / 211.11 / 340.7 | 2.58% |

**Selected Under Full-Q Tail-Latency Priority**

```text
C = 1536
overlap = 3
entries = 2
degree = 32
ef = 16
efConstruction = 128
```

`C=1536` has the lowest exhaustive full-Q slowest-query latency: **0.173755 ms**. It is exact on every full-Q query.

| Q | Direct mean (ms) | HNSW min (ms) | HNSW mean (ms) | HNSW max (ms) | Speedup | Full-workload match |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2.213496 | 0.158348 | 0.202850 | 0.253131 | 10.912x | 100% (3,800 / 3,800) |
| 200 | 2.212187 | 0.166405 | 0.205877 | 0.247524 | 10.745x | 100% (7,600 / 7,600) |
| 500 | 2.151239 | 0.160043 | 0.200141 | 0.238069 | 10.749x | 100% (19,000 / 19,000) |
| 1000 | 2.173764 | 0.158541 | 0.202243 | 0.246336 | 10.748x | 100% (38,000 / 38,000) |

Across Q sizes, speedup is **10.745x / 10.789x / 10.912x**. Q-mean spread is **2.83%**. The robust panel maximum is **0.253131 ms**; the exhaustive full-Q worst query is **0.173755 ms**.


// Amazon

**Amazon Sweep**

Centroids tested: `64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192` with 81 configurations per C: `1,215` total. The strength-2 pairwise-covering design used overlap `1-4`, entries `1, 2, 3, 4, 6, 8, 12, 16`, degree `6, 8, 12, 16, 24, 32, 48`, and ef `2, 4, 6, 8, 12, 16, 24, 32, 48, 64`.

All `1,215 / 1,215` candidates received full-Q validation; `620` achieved 100% direct agreement across all 19 attacks and Q=`100, 200, 500, 1000`. The known-reference bank contains `35,000` calibrated Amazon embeddings.

**Direct Timings**

| Q | Direct min (ms) | Direct mean (ms) | Direct max (ms) |
|---:|---:|---:|---:|
| 100 | 4.615310 | 4.668980 | 4.794705 |
| 200 | 4.609284 | 4.620603 | 4.631328 |
| 500 | 4.618882 | 4.681730 | 4.896991 |
| 1000 | 4.615928 | 4.672992 | 4.848405 |

**Best Exact Candidate At Each C**

`O/E/D/ef` means overlap / entries / degree / search depth. Full-Q profile timing is the exhaustive, internal per-query measurement; panel timing is the fixed sampled wall-clock measurement shared by every candidate.

| C | Config `O/E/D/ef` | Full-Q profile mean / max (ms) | Panel mean / max (ms) | Speedup min / avg / max | Avg hops / visited / bank checks | Q spread |
|---:|---|---:|---:|---:|---:|---:|
| 64 | `1/3/48/4` | 0.28921 / 0.88980 | 0.33803 / 0.88252 | 13.192x / 13.798x / 14.058x | 0.45 / 61.22 / 1471.2 | 6.54% |
| 96 | `4/1/12/2` | 0.24649 / 0.84794 | 0.30191 / 0.73163 | 14.929x / 15.454x / 16.150x | 1.22 / 19.02 / 1299.3 | 8.91% |
| 128 | `1/1/32/8` | 0.24969 / 0.87005 | 0.29423 / 0.69494 | 15.357x / 15.850x / 16.199x | 0.79 / 48.25 / 1327.8 | 6.39% |
| 192 | `2/2/8/6` | 0.20415 / 0.60960 | 0.26430 / 0.54806 | 17.108x / 17.645x / 18.128x | 1.59 / 47.94 / 1022.4 | 6.91% |
| 256 | `1/8/48/12` | 0.28035 / 0.69846 | 0.35530 / 0.62794 | 12.927x / 13.121x / 13.305x | 0.98 / 247.38 / 909.7 | 3.92% |
| 384 | `4/1/32/6` | 0.14758 / 0.55297 | 0.23559 / 0.45341 | 18.973x / 19.798x / 20.170x | 1.28 / 64.45 / 766.5 | 7.31% |
| 512 | `3/1/12/12` | 0.18478 / 0.55306 | 0.25047 / 0.46040 | 18.072x / 18.616x / 18.975x | 1.99 / 46.46 / 909.6 | 5.20% |
| 768 | `4/3/8/12` | 0.17255 / 0.53292 | 0.24964 / 0.47110 | 17.878x / 18.684x / 18.995x | 5.21 / 114.13 / 737.4 | 7.26% |
| 1024 | `2/1/24/16` | 0.10883 / 0.36387 | 0.21331 / 0.32196 | 21.378x / 21.857x / 22.163x | 2.26 / 95.05 / 499.0 | 4.70% |
| 1536 | `1/4/12/32` | 0.16834 / 0.43372 | 0.27767 / 0.42849 | 16.514x / 16.792x / 17.208x | 4.66 / 261.65 / 369.1 | 5.12% |
| **2048** | **`1/3/32/32`** | **0.18150 / 0.33563** | **0.30157 / 0.38964** | **15.262x / 15.458x / 15.706x** | **1.92 / 418.87 / 287.4** | **2.59%** |
| 3072 | `4/2/48/24` | 0.14698 / 0.40718 | 0.24534 / 0.38373 | 18.718x / 19.002x / 19.360x | 1.81 / 224.66 / 405.8 | 4.48% |
| 4096 | `4/2/16/32` | 0.21116 / 0.48653 | 0.30583 / 0.46450 | 14.910x / 15.244x / 15.553x | 3.13 / 465.17 / 390.5 | 4.15% |
| 6144 | `4/3/16/64` | 0.30999 / 0.62107 | 0.40822 / 0.57934 | 11.280x / 11.419x / 11.592x | 2.84 / 580.41 / 456.5 | 2.56% |
| 8192 | `4/2/48/64` | 0.40776 / 0.96733 | 0.60640 / 1.00546 | 7.538x / 7.688x / 7.770x | 1.89 / 658.32 / 374.9 | 2.91% |

**Selected Under Full-Q Tail-Latency Priority**

```text
C = 2048
overlap = 1
entries = 3
degree = 32
ef = 32
efConstruction = 128
```

`C=2048` has the lowest exhaustive full-Q slowest-query latency among all 620 exact candidates: **0.335629 ms**. Its Q-mean spread is **2.59%**, and it verifies an average of **287.4** bank embeddings per query from the 35,000-image reference bank.

| Q | Direct mean (ms) | HNSW min (ms) | HNSW mean (ms) | HNSW max (ms) | Speedup | Full-workload match |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4.668980 | 0.196407 | 0.305913 | 0.389639 | 15.262x | 100% (3,800 / 3,800) |
| 200 | 4.620603 | 0.195224 | 0.302231 | 0.383369 | 15.288x | 100% (7,600 / 7,600) |
| 500 | 4.681730 | 0.179348 | 0.298090 | 0.385049 | 15.706x | 100% (19,000 / 19,000) |
| 1000 | 4.672992 | 0.193299 | 0.300035 | 0.374712 | 15.575x | 100% (38,000 / 38,000) |

Across Q sizes, speedup is **15.262x / 15.458x / 15.706x**. The robust panel maximum is **0.389639 ms**; the exhaustive full-Q worst query is **0.335629 ms**.


1. **UCID:** 4,500 embeddings, selected `C=256`, average speedup `2.760x`.
2. **Flickr:** 10,000 embeddings, selected `C=2048`, average speedup `12.255x`.
3. **CIFAR:** 17,000 embeddings, selected `C=1536`, average speedup `10.789x`.
4. **Amazon:** 35,000 embeddings, selected `C=2048`, average speedup `15.458x`.

**Comparisons**

- Dataset size: `UCID 4,500 < Flickr 10,000 < CIFAR 17,000 < Amazon 35,000`
- Selected centroids: `UCID 256 < CIFAR 1536 < Flickr 2048 = Amazon 2048`
- Average speedup: `UCID 2.760x < CIFAR 10.789x < Flickr 12.255x < Amazon 15.458x`

**Selection Rules Considered**

- Exact agreement with direct search for every full-Q query and all 19 attacks.
- Lowest exhaustive full-Q worst-case query latency.
- Low outlier and tail latency in the sampled timing panel.
- Stable mean latency across all Q sizes.
- A practical centroid count, avoiding an arbitrary smallest or largest value.
- Measured end-to-end speedup against the matching direct-search baseline.
- Same embedding bank, calibrated distance threshold, candidate-bucket verification, and direct decision rule for each dataset.


