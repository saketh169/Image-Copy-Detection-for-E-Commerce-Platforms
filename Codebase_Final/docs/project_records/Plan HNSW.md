# HNSW Graph Plan for Image Copy Detection

## **1. INTRODUCTION**

### **Why HNSW Is Used**

After the Siamese ConvNeXt model converts an image into a final 384-dimensional
embedding, direct search compares that query against every known-bank image.
This is exact but becomes slower as the bank grows.

HNSW makes retrieval faster by first navigating between **centroid nodes** and
then exactly checking only the relevant image buckets. It does not replace the
copy-detection threshold rule.

```text
Query image
    -> ConvNeXt + folded diagonal metric
    -> 384-D query embedding q
    -> Hierarchical HNSW traversal over centroids
    -> Candidate image buckets
    -> Exact L2 verification using tau
    -> Duplicate / non-duplicate
```

### **Direct Search Reference**

For bank embedding x_i and query q:

```text
d2(q, x_i) = sum_k (q_k - x_i,k)^2

duplicate if min_i d2(q, x_i) <= tau^2
non-duplicate otherwise
```

The direct result is the correctness reference. HNSW must use the same bank,
same embeddings, same queries, and same tau. It only reduces unnecessary
distance comparisons.

---

## **2. ARCHITECTURE - CENTROID BUCKET CREATION**

### **Step 1: Build C Centroids**

K-means is run on the verified known-bank embeddings.

```text
Known bank: N embeddings, each 384-D
             |
             v
K-means with C clusters
             |
             v
C centroids = C HNSW graph nodes
```

Example: UCID has 4,500 known-bank images. If C=192, the graph has only 192
nodes, while each centroid bucket holds a smaller local group of bank images.

The centroid cache is accepted only when the bank fingerprint, embedding
dimension, centroid count, and overlap match the current experiment.

### **Step 2: Create Overlapping Buckets**

Each bank image goes into its `ov` nearest centroid buckets.

```text
Image x_37
  nearest centroid 1 = z_12
  nearest centroid 2 = z_19

With overlap=2:
  x_37 is stored in bucket P_12 and bucket P_19
```

Overlap protects images near a cluster boundary. It can improve coverage, but
it can also increase exact verification work, so it is measured in the sweep.

For each bucket P_j, save its radius:

```text
r_j = maximum distance from centroid z_j to an image in P_j
```

For a query q, the safe lower bound is:

```text
LB(q, j) = max(0, ||q-z_j||_2 - r_j)^2
```

If LB(q,j) is above tau squared, bucket j cannot contain a duplicate and is
not scanned.

---

## **3. ARCHITECTURE - HNSW GRAPH CONSTRUCTION**

### **Step 1: Random Layer Assignment**

HNSW does **not** place centroids in layers by centrality. Every centroid gets
a seeded random level from an exponential distribution:

```text
u ~ Uniform(0,1)
level = floor(-log(u) / log(M))
```

Most centroids receive level 0. Fewer appear in level 1, fewer again in level
2, and so on.

```text
Layer 3:   very few centroids, long jumps
Layer 2:   regional navigation
Layer 1:   more detailed navigation
Layer 0:   all C centroids, local navigation
```

If a centroid is assigned level 2, it belongs to layers 0, 1, and 2.

### **Step 2: Insert Centroids One at a Time**

Centroids are inserted in a deterministic shuffled order. For a new centroid
v, construction performs this sequence:

```text
1. Start from the current top-layer entry centroid.
2. Greedily move in upper layers until no neighbor is closer to v.
3. At layers where v exists, run SearchLayer(v, entry, efConstruction).
4. Select a diverse set of neighbors.
5. Add reciprocal links.
6. Prune links if a degree cap is exceeded.
```

SearchLayer uses a visited set, a queue of centroids still to explore, and a
bounded set of the best candidates. `efConstruction` is only for graph build
quality; it is separate from query-time ef.

### **Step 3: Diverse Neighbor Selection**

HNSW does not keep only the nearest neighbors. It avoids redundant edges that
all point in one direction.

For candidate c and already selected neighbor s of centroid v:

```text
keep c when distance(c,s) >= distance(v,c)
```

This creates routes into different visual regions. If diversity pruning leaves
free slots, the nearest rejected neighbors fill the remaining slots.

### **Step 4: Degree Limits**

| Layer | Maximum degree | Reason |
|---|---:|---|
| Upper layers | M | Sparse long-range routing |
| Layer 0 | M0 = 2M | More local connections for detailed search |

When a reciprocal edge makes a node exceed the limit, its old and new links
are combined and diversity-pruned again. The graph therefore remains bounded.

### **Step 5: Entry Points**

Canonical HNSW has one top-level entry point. When `entries > 1` is evaluated,
the first remains canonical and extra entries are chosen by farthest-point
sampling in centroid space.

```text
Entry 1: canonical highest-level node
Entry 2: centroid farthest from Entry 1
Entry 3: centroid farthest from Entries 1 and 2
...
```

This makes multi-entry probes spatially diverse instead of clustered together.

---

## **4. ARCHITECTURE - QUERY SEARCH**

### **Step 1: Top-Down Routing**

For each entry point, start at its highest usable layer.

```text
Current centroid = entry

At current layer:
  inspect all neighbors
  move to the neighbor closest to q if it improves distance
  otherwise descend to the next layer
```

The final node in one layer becomes the starting node of the next lower layer.

### **Step 2: Layer-0 Best-First Search**

At layer 0, HNSW keeps up to ef promising centroids rather than following only
one greedy path.

```text
candidate queue = current centroid
best centroid set = current centroid

while candidates remain:
  take nearest candidate
  inspect unvisited neighbors
  retain at most ef best centroids
```

Results from all entry points are merged and deduplicated.

### **Step 3: Exact Bucket Verification**

The compiled C++ kernel ranks routed buckets using the lower bound, discards
impossible buckets, deduplicates image IDs across overlapping buckets, and
computes exact L2 against retained bank images.

```text
HNSW centroid candidates
    -> lower-bound bucket ranking
    -> unique bank-image IDs
    -> exact d2(q,x_i)
    -> compare with tau^2
```

The conservative certificate path is included when required to preserve direct
decisions. The graph alone never labels a query as duplicate.

---

## **5. PARAMETERS TO SWEEP**

| Parameter | Values | What it controls |
|---|---|---|
| C | Dataset-specific centroid range | Number of graph nodes and bucket size |
| overlap | 1, 2, 3, 4 | Boundary coverage versus posting-list work |
| degree M | 6, 8, 12, 16, 24, 32, 48 | Graph connectivity |
| entries | 1, 2, 3, 4, 6, 8, 12, 16 | Number of spatially diverse starts |
| ef | 2, 4, 6, 8, 12, 16, 24, 32, 48, 64 | Layer-0 candidate budget |
| efConstruction | At least 4 x degree | Construction search quality |

Every candidate is evaluated independently. A setting that is best at one
centroid count is not assumed best at another centroid count.

---

## **6. EVALUATION RULES**

For every candidate and every Q size, report:

1. Full-query agreement with direct search across all current attacks.
2. Known, unknown, and total decisions.
3. Direct and HNSW mean, median, p95, and maximum per-query latency.
4. Speedup = direct mean latency / HNSW mean latency.
5. Graph nodes visited, hops, buckets checked, and unique bank images scanned.
6. Q-size mean-latency spread.

Build time is reported separately from query time. Increasing C may reduce
bucket work before graph overhead becomes dominant; this shape is measured by
the sweep and is not forced.

---

## **7. IMPLEMENTATION FILES**

| File | Role |
|---|---|
| `hnsw_hierarchical.py` | Hierarchy, insertion, diverse pruning, entries |
| `benchmark_graphs.py` | Centroids, buckets, direct truth, C++ wrapper |
| `bucket_bvh.py` | Conservative bucket-bound hierarchy |
| `graph_kernels.cpp` | Compiled direct and HNSW retrieval kernels |
| `TIMING_PROTOCOL.md` | Shared timing protocol |
