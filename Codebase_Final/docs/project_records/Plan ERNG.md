# ERNG Graph Plan for Image Copy Detection

## **1. INTRODUCTION**

### **Why ERNG Is Used**

ERNG means **Exponential-Rank Navigation Graph**. It is the flat-graph
alternative to hierarchical HNSW. It routes over centroid embeddings, then
uses exact L2 verification against bank-image embeddings.

```text
Query image
    -> ConvNeXt + folded diagonal metric
    -> 384-D query embedding q
    -> Flat ERNG multi-probe traversal over centroids
    -> Candidate image buckets
    -> Exact L2 verification using tau
    -> Duplicate / non-duplicate
```

ERNG does not have HNSW levels. Its navigation comes from local links,
exponential-rank skip links, and multiple probe starts.

### **Direct Search Reference**

```text
d2(q, x_i) = sum_k (q_k - x_i,k)^2

duplicate if min_i d2(q, x_i) <= tau^2
non-duplicate otherwise
```

ERNG uses the same bank, embeddings, threshold, and query set as direct search
and HNSW. It accelerates routing only; it cannot change the final rule.

---

## **2. ARCHITECTURE - CENTROID BUCKET CREATION**

### **Step 1: Create C Centroids**

Deterministic K-means clusters the verified known-bank embeddings.

```text
Known bank embeddings
       |
       v
K-means with C clusters
       |
       v
C centroids = C ERNG nodes
```

The centroid cache is reused only when its bank fingerprint, embedding
dimension, C, and overlap exactly match the present experiment.

### **Step 2: Create Overlapping Buckets**

Each bank image is put into its `ov` nearest centroid posting lists.

```text
Image x_37
  nearest centroids: z_12, z_19, z_31

With overlap=2:
  x_37 is stored in P_12 and P_19
```

For every bucket P_j, save its radius r_j. For query q, calculate:

```text
LB(q,j) = max(0, ||q-z_j||_2-r_j)^2
```

When this lower bound is above tau squared, bucket j cannot contain a duplicate
and can be skipped before exact image comparisons.

---

## **3. ARCHITECTURE - FLAT ERNG CONSTRUCTION**

### **Step 1: Calculate Centroid Distance Ranks**

For every centroid z_i, ERNG calculates squared distance to every other
centroid and sorts the results from nearest to farthest.

```text
For centroid z_i:
  Rank 1  -> nearest centroid
  Rank 2  -> second nearest centroid
  Rank 4  -> fourth nearest centroid
  Rank 8  -> eighth nearest centroid
  ...
```

ERNG has one graph layer only. There is no random level assignment.

### **Step 2: Add Local KNN Links**

Each centroid connects to its k_local nearest centroids.

```text
If k_local=6:
  z_i -> rank 1, rank 2, rank 3, rank 4, rank 5, rank 6
```

These edges support precise movement within a local visual region.

### **Step 3: Add Exponential-Rank Skip Links**

ERNG next visits ranks 1, 2, 4, 8, 16, 32, and so on. A link is added only
when it is a genuinely new centroid, not a local link already selected.

```text
Example: k_local=6 and k_skip=3

Ranks 1,2,4 are already local.
Continue to rank 8 -> add first skip link
Continue to rank 16 -> add second skip link
Continue to rank 32 -> add third skip link
```

This gives deterministic medium/longer-distance navigation options without
using another hierarchy.

### **Step 4: Make Edges Reciprocal**

If z_i chooses z_j, the construction can also add z_i to z_j's candidate
neighbors. Reciprocity helps avoid one-way dead ends in the flat graph.

### **Step 5: Apply the Degree Cap**

The maximum stored degree is:

```text
max_degree = k_local + k_skip
```

For every centroid row:

1. Keep the nearest local links first.
2. Gather remaining reciprocal and skip candidates.
3. Sort them by centroid distance.
4. Add them until max_degree is reached.
5. Remove duplicate links and self-links.

The graph is stored as a fixed-width adjacency matrix. Unused slots are -1;
a separate degree vector says how many links in each row are valid.

---

## **4. ARCHITECTURE - QUERY SEARCH**

### **Step 1: Choose Probe Starts**

ERNG does not use one HNSW top entry. It selects R deterministic start
centroids with a fixed random seed when the graph is created.

```text
Probe 1 starts at z_a
Probe 2 starts at z_b
Probe 3 starts at z_c
Probe 4 starts at z_d
```

The same seeded starts are reused for every query in the experiment, making the
run reproducible.

### **Step 2: Greedy Navigation for Each Probe**

```text
current = probe start

repeat:
  inspect every valid neighbor of current
  best = neighbor nearest to query q

  if best is closer than current:
      current = best
  else:
      stop this probe
```

Each probe ends at a local centroid endpoint.

### **Step 3: Merge and Expand Candidates**

The endpoint centroids from all probes are merged and deduplicated. ERNG uses
the ef budget to retain and expand a bounded flat candidate set.

```text
Probe endpoints -> merged centroid candidates -> bounded expansion with ef
```

More probes provide wider initial coverage. Larger ef explores more candidate
centroids. Both can improve retrieval but add query work.

### **Step 4: Exact Bucket Verification**

The compiled C++ kernel performs the final work:

```text
ERNG centroid candidates
    -> lower-bound bucket ranking
    -> remove impossible buckets
    -> deduplicate bank-image IDs from overlap
    -> exact d2(q,x_i) comparisons
    -> compare with tau^2
```

ERNG traversal never directly labels a query. A duplicate is returned only
after an exact bank-image distance satisfies the threshold. Results are checked
against direct search for every query.

---

## **5. PARAMETERS TO SWEEP**

| Parameter | Values | What it controls |
|---|---|---|
| C | Dataset-specific centroid range | Number of flat graph nodes and bucket size |
| overlap | 1, 2, 3, 4 | Boundary coverage versus exact bucket work |
| degree | 6, 8, 12, 16, 24, 32 | Flat graph connectivity budget |
| k_local | Derived and saved per degree | Local KNN edge count |
| k_skip | Derived and saved per degree | Exponential-rank edge count |
| probes | 1, 2, 3, 4, 6, 8, 12 | Number of deterministic starts |
| ef | 2, 4, 6, 8, 12, 16, 24, 32, 48, 64 | Flat candidate-expansion budget |

Every candidate is evaluated independently. The degree-to-local/skip mapping
is saved with the result; it is part of the ERNG configuration.

---

## **6. EVALUATION RULES**

For every candidate and every Q size, record:

1. Full-query agreement with direct search across all current attacks.
2. Known, unknown, and total decisions.
3. Direct and ERNG mean, median, p95, and maximum per-query latency.
4. Speedup = direct mean latency / ERNG mean latency.
5. Visited graph nodes, hops, buckets checked, and unique bank images scanned.
6. Q-size mean-latency spread, build cost, and memory use.

Build time is separate from retrieval latency. ERNG is a flat-graph baseline;
it must be compared fairly using the same centroid layout, overlap, threshold,
query set, timing panel, and single-thread policy as HNSW.

---

## **7. IMPLEMENTATION FILES**

| File | Role |
|---|---|
| `erng_flat.py` | Local links, exponential-rank skips, reciprocity, probes |
| `benchmark_graphs.py` | Centroids, buckets, direct truth, C++ wrapper |
| `graph_kernels.cpp` | Compiled ERNG and direct retrieval kernels |
| `TIMING_PROTOCOL.md` | Shared timing protocol |

## **8. HONEST SCOPE**

ERNG combines local KNN links, power-of-two rank skips, reciprocal links, and
multiple flat-graph starts. These are known graph-search ingredients. In this
project ERNG is an application-specific centroid-routing baseline, not a claim
of a new general-purpose ANN algorithm.
