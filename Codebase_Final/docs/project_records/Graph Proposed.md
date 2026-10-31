# RESEARCH REVIEW DOCUMENT
## Graph-Based Approximate Nearest Neighbor Search — Past to Present to Our Proposal

---

# SECTION A: WHAT IS THE PROBLEM?

- You have **millions or billions of vectors** (numbers representing images, text, etc.).
- Given a new query vector, you need to find the **most similar vector** in the database.
- Checking every single vector (**brute force**) is too slow.
- So we build a **graph** (a network of connected nodes) and use it as a shortcut to find the answer fast.
- This is called **Approximate Nearest Neighbor Search (ANNS)**.

---

# SECTION B: HISTORY OF GRAPH-BASED ANN (Paper by Paper)

---

## 📄 Paper 1: Kleinberg (2000) — The Theory That Started It All

- **Title:** "The Small-World Phenomenon: An Algorithmic Perspective"
- **Venue:** STOC 2000 (top theory conference)
- **What he did:**
  - Proved mathematically WHY some networks can be searched fast using only local information.
  - Showed that if you add random "shortcut" links with the right probability, a greedy search can find any target in O(log² n) steps.
  - The probability must follow a **power law**: closer nodes are more likely to get links than far nodes.
- **Key concept:** **Navigability** — the idea that greedy search (always move to the closest neighbor) actually works.
- **Why it matters for us:**
  - This is the **theoretical foundation** of every graph-based ANN method that came after.
  - HNSW, NSW, and even our ERNG all rely on this principle.

---

## 📄 Paper 2: NN-Descent (2011) — Fast kNN Graph Construction

- **Title:** "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures"
- **Authors:** Dong, Moses, Li
- **Venue:** WWW 2011
- **What they did:**
  - Invented a fast way to build a **k-Nearest Neighbor (kNN) graph**.
  - Key idea: **"neighbors of neighbors are likely neighbors."**
  - Start with random connections → iteratively improve by checking neighbors-of-neighbors.
- **Key concepts:**
  - Local join: compare every pair of neighbors of each node.
  - Converges quickly — empirically O(n^1.14) distance computations.
- **Limitation:**
  - Produces a kNN graph, but **not a search index** — no guarantee that greedy search works on it.
- **Why it matters for us:**
  - NN-Descent is used as **Step 1** (initial graph construction) in many later methods including NSG, Vamana, and our proposed method.

---

## 📄 Paper 3: NSW (2014) — The First Navigable Graph for ANN

- **Title:** "Approximate nearest neighbor algorithm based on navigable small world graphs"
- **Authors:** Malkov, Ponomarenko, Logvinov, Krylov
- **Venue:** Information Systems, 2014
- **What they did:**
  - Built a **single-layer graph** where nodes are data points and edges connect similar points.
  - Construction: insert nodes one by one; for each new node, do a greedy search to find neighbors, then link to them.
  - Early-inserted nodes naturally get **long-range links** (because the graph is sparse when they are added).
- **Key concepts:**
  - **Small-world property:** short average path length + high clustering.
  - **Greedy search:** always move to the neighbor closest to the query.
  - **Incremental insertion:** add one point at a time.
- **Limitations:**
  - Single entry point → search quality depends heavily on where you start.
  - As the graph grows, early nodes get too many edges, late nodes get too few.
  - No hierarchy → search can be slow on large datasets.
- **Why it matters for us:**
  - This is the **direct ancestor** of HNSW.
  - Shows that a flat graph CAN work, but has scalability problems.

---

## 📄 Paper 4: HNSW (2016/2020) — The Industry Standard

- **Title:** "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- **Authors:** Malkov, Yashunin
- **Venue:** arXiv 2016, IEEE TPAMI 2020
- **What they did:**
  - Added a **multi-layer hierarchy** on top of NSW (inspired by skip lists).
  - Top layers: few nodes with long-range links → "express highway."
  - Bottom layer: all nodes with short-range links → "local roads."
  - Each node is randomly assigned a level (exponential distribution).
- **Key concepts:**
  - **Hierarchical layers:** coarse search at top → fine search at bottom.
  - **Diversity heuristic:** when selecting neighbors, prefer nodes that are spread out in different directions (not all clustered together).
  - **Fixed degree M:** every node gets exactly M connections per layer (e.g., M=16).
  - **Single global entry point** at the highest layer.
- **Strengths:**
  - O(log n) search time.
  - Very high recall.
  - Easy to implement (hnswlib is the standard library).
- **Limitations:**
  - ⚠️ **Fixed degree M for ALL nodes** — dense regions waste edges, sparse regions are starved.
  - Multi-layer structure uses extra memory.
  - No formal worst-case guarantees (until HENN in 2025).
- **Why it matters for us:**
  - HNSW is the **#1 baseline** we must beat.
  - Its weakness (uniform degree M) is exactly the gap we exploit.

---

## 📄 Paper 5: FANNG (2016) — Occlusion-Based Pruning

- **Title:** "FANNG: Fast Approximate Nearest Neighbour Graphs"
- **Authors:** Harwood, Drummond
- **Venue:** CVPR 2016
- **What they did:**
  - Build kNN graph using random projection forests.
  - Prune edges using an **occlusion rule**: remove edge (u,v) if another neighbor w "blocks" v (i.e., w is between u and v angularly).
- **Key concepts:**
  - **Angular diversity** in edge selection.
  - Similar principle to RNG (Relative Neighborhood Graph).
- **Limitation:**
  - Fixed degree bound. No adaptive allocation.
- **Why it matters for us:**
  - Shows that **how you prune edges** matters a lot — not just how many you keep.

---

## 📄 Paper 6: EFANNA (2016) — KD-Tree + Graph Hybrid

- **Title:** "EFANNA: An Extremely Fast Approximate Nearest Neighbor Search Algorithm"
- **Authors:** Fu, Cai
- **Venue:** arXiv 2016
- **What they did:**
  - Use KD-trees to get an initial rough kNN graph.
    - *(Note: A KD-Tree is an older, tree-based data structure that splits space into halves iteratively. It is fast for low dimensions but struggles in high dimensions).*
  - Refine it using NN-Descent.
  - Search using beam search on the refined graph.
- **Key concepts:**
  - **Hybrid approach:** tree initialization + graph refinement.
  - Shows that **initialization quality** affects final graph quality.
- **Limitation:**
  - KD-trees don't work well in very high dimensions.
  - Fixed degree.
- **Why it matters for us:**
  - Confirms that the **initial graph quality** matters — supports our approach of building an initial graph first, then improving it.

---

## 📄 Paper 7: NSG (2019) — The Sparse Navigable Graph

- **Title:** "Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph"
- **Authors:** Fu, Xiang, Yu, Cai
- **Venue:** PVLDB 2019
- **What they did:**
  - Approximate the **Monotonic Relative Neighborhood Graph (MRNG)**.
  - MRNG rule: keep edge (u,v) ONLY if no other neighbor w is closer to v than u is.
  - This produces a very **sparse** graph with guaranteed navigability.
  - Single "navigating node" (similar to medoid) as entry point.
    - *(Note: A 'medoid' is simply the data point that is closest to the mathematical center of the entire dataset).*
- **Key concepts:**
  - **MRNG pruning:** aggressive edge removal for sparsity.
  - **Single-layer, flat graph** (no hierarchy like HNSW).
  - **Fixed degree R** for all nodes.
- **Strengths:**
  - Very memory efficient.
  - Fast search.
- **Limitations:**
  - ⚠️ **Fixed degree R for ALL nodes.**
  - Single entry point can be a bottleneck.
- **Why it matters for us:**
  - NSG proves that flat graphs CAN compete with HNSW.
  - But it still uses uniform degree — our gap applies here too.

---

## 📄 Paper 8: SSG (2019/2021) — Relaxed Angular Pruning

- **Title:** "High Dimensional Similarity Search with Satellite System Graph"
- **Authors:** Fu, Wang, Cai
- **Venue:** IEEE TPAMI 2021
- **What they did:**
  - Like NSG but with **less aggressive pruning**.
  - Use angular diversity instead of strict MRNG rule.
  - Result: denser graph → more robust but uses more memory.
- **Key concepts:**
  - **Relaxed pruning** = keep more edges = better recall but more memory.
  - Trade-off between sparsity and search quality.
- **Limitation:**
  - Still fixed degree R for all nodes.
- **Why it matters for us:**
  - Shows the **tension between sparsity and quality** — our method resolves this by being smart about WHERE to put the edges.

---

## 📄 Paper 9: Vamana / DiskANN (2019) — The Flat Graph Champion

- **Title:** "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
- **Authors:** Subramanya, Devvrit, Simhadri, Krishnawamy, Kadekodi
- **Venue:** NeurIPS 2019
- **What they did:**
  - Built a **flat, single-layer graph** (no hierarchy).
  - Invented **RobustPrune**: an edge selection algorithm that balances local and long-range edges.
  - Key parameter **α** (alpha): controls how aggressively long-range edges are added.
    - α = 1: only local edges.
    - α > 1 (e.g., 1.2): forces inclusion of long-range shortcuts.
  - Two-pass construction: first pass local (α=1), second pass adds long-range (α>1).
  - Entry point: medoid (center of dataset).
- **Key concepts:**
  - **RobustPrune:** the most important edge selection algorithm in modern ANN.
  - **α parameter:** global control for local vs. long-range balance.
  - **Fixed degree R** (e.g., R=64) for ALL nodes.
  - **Disk-aware:** designed to work when the index is on SSD.
- **Strengths:**
  - Handles billion-scale datasets on a single machine.
  - Flat graph → fewer random disk reads than HNSW.
- **Limitations:**
  - ⚠️ **α is a GLOBAL parameter** — same aggressiveness for every node.
  - ⚠️ **Fixed degree R for ALL nodes.**
- **Why it matters for us:**
  - Vamana is the **closest existing method** to what we want to build.
  - We will use RobustPrune as our edge selection algorithm.
  - But we fix its two biggest weaknesses: uniform R and global α.

---

## 📄 Paper 10: Wang et al. Survey (2021) — The Comprehensive Benchmark

- **Title:** "A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search"
- **Authors:** Wang, Xu, Yue, Wang
- **Venue:** PVLDB 2021
- **What they did:**
  - Benchmarked **13 graph-based ANN algorithms** on 8 real + 12 synthetic datasets.
  - Created a taxonomy of construction and search pipelines.
  - Provided rule-of-thumb recommendations for practitioners.
- **Key finding:**
  - **No single method dominates** across all datasets and metrics.
  - The choice of pruning strategy and degree bound significantly affects performance.
- **Why it matters for us:**
  - Confirms that **the field is not "solved"** — there is room for improvement.
  - We will use the same benchmark datasets they used (SIFT1M, GIST1M, GloVe).

---

## 📄 Paper 11: Dynamic HNSW / DATE 2025 — Density-Based Degree Adaptation

- **Title:** "Efficient Approximate Nearest Neighbor Search via Data-Adaptive Parameter Adjustment"
- **Venue:** DATE 2025
- **What they did:**
  - Adjusted M (degree) and ef (search depth) based on **local data density**.
  - Dense regions get lower M; sparse regions get higher M.
- **Key concepts:**
  - **Density-based adaptation** — a static data property.
- **Critical distinction:**
  - ⚠️ They adapt by **density** (a property of the DATA).
  - We adapt by **navigational difficulty** (a property of the SEARCH ALGORITHM ON THE GRAPH).
  - Density ≠ Difficulty. A node can be in a dense region but still hard to reach if the graph topology is bad.

---

# SECTION C: THE MASTER COMPARISON TABLE

| Paper | Year | Degree | Adaptive? | What adapts? | Layers | Edge Selection |
|-------|------|--------|-----------|-------------|--------|---------------|
| Kleinberg | 2000 | Theory | N/A | N/A | 1 | Power-law probability |
| NN-Descent | 2011 | Fixed k | ❌ | Nothing | 1 | Distance-based |
| NSW | 2014 | Variable (uncontrolled) | ❌ | Nothing | 1 | Greedy insertion |
| HNSW | 2016 | **Fixed M** | ❌ | Nothing | Multi | Diversity heuristic |
| FANNG | 2016 | Fixed | ❌ | Nothing | 1 | Occlusion/angular |
| EFANNA | 2016 | Fixed k | ❌ | Nothing | 1 | KD-tree + NN-Descent |
| NSG | 2019 | **Fixed R** | ❌ | Nothing | 1 | MRNG rule |
| SSG | 2019 | **Fixed R** | ❌ | Nothing | 1 | Angular diversity |
| Vamana | 2019 | **Fixed R** | ❌ | Nothing | 1 | RobustPrune (global α) |
| DATE 2025 | 2025 | Variable M | ⚠️ By density | Local density | Multi | Standard HNSW |
| **Our Proposed Method** | **2026** | **Variable per node** | **✅ Per-node** | **Navigational difficulty** | **1** | **Budget-aware RobustPrune** |

---

# SECTION D: THE GAP

After reviewing ALL of the above papers, here is what we found:

## What EVERY existing method does:
- Most standard methods give every node the **same maximum degree** (same number of edges).
- Whether it's M=16 in HNSW, R=64 in Vamana, or R=32 in NSG — **every node gets the same limit**.

## What NO existing method does:
- **No method measures how hard each node is to reach via greedy search during the initial flat graph construction in a fixed-budget manner.**
- **No method explicitly redistributes a fixed edge budget based directly on search failure rates (difficulty).**

---

# SECTION E: WHY OUR ERNG IS NOT NOVEL (Honest Assessment)

## Individual components of ERNG:

| ERNG Feature | Already exists in |
|---|---|
| Local kNN edges | Every method since 2011 |
| Power-of-two skip links | Kleinberg (2000), HNSW hierarchy |
| Flat single-layer graph | NSW (2014), Vamana (2019) |
| Multiple entry points | HNSW extensions, Vamana restarts |
| Reciprocal/reverse edges | Vamana |
| Connectivity repair | NSG, Vamana |
| Fixed degree limit | Every method |

## Is the COMBINATION novel?

**No.** The combination is essentially:
> "Vamana's flat-graph approach + Kleinberg's power-of-two link selection"

A reviewer would say: *"This is Vamana with a different edge selection heuristic."*

## Additional problem:
- ERNG operates on **centroids** (cluster centers), not raw data points.
- This makes it an **application-specific tool** (image copy detection), not a general-purpose ANN algorithm.
- A general-purpose ANN paper MUST work on standard benchmarks (SIFT1M, GIST1M, GloVe).

## Verdict:
> **ERNG as-is cannot be published as a novel research contribution.**

---

# SECTION F: OUR PROPOSED METHOD

## Full Name:
**Navigational Difficulty-Aware Adaptive Degree Allocation for Graph-Based Approximate Nearest Neighbor Search**

## One-Sentence Summary:
> Give more graph edges to nodes that are hard to reach via greedy search, and fewer edges to nodes that are easy to reach, under a fixed total edge budget.

---

## How It Works (Step by Step):

### Step 1: Build a Draft Graph
- **Algorithm:** NN-Descent (Paper 2 above)
- **What happens:** Every node gets connected to its k nearest neighbors.
- **Purpose:** This gives us a rough "road network" to test.

### Step 2: Measure Navigational Difficulty
- **Algorithm:** Multi-start Greedy Search simulation
- **What happens:**
  - For EVERY node in the graph, we pretend it is the target.
  - We drop random starting points and run greedy search trying to reach it.
  - We measure:
    - **Failure rate (F):** How often does the search fail to find this node?
    - **Path length (L):** When it succeeds, how many hops does it take?
  - We combine into a single difficulty score: `d = (α * F_i) + (β * (L_i / L_max))`
- **Why this is important:**
  - We measure **search failure** (dynamic graph+algorithm property).
  - **Density ≠ Difficulty.** A node can be in a dense region but still hard to reach.

### Step 3: Redistribute Edge Budget
- **What happens:**
  - Total budget B = N × M_avg (e.g., 1000 nodes × 32 avg degree = 32,000 total edges).
  - This budget is **FIXED** — we never add or remove total edges.
  - For each node: `Budget_i = M_avg * (d_i / average_difficulty)`
  - Hard nodes get MORE edges (e.g., 60).
  - Easy nodes get FEWER edges (e.g., 15).
  - Clip between minimum (8) and maximum (128): `E_i = MAX( M_min, MIN(M_max, E_i) )`
  - Rescale so the total exactly equals B: `Sum of all E_i = B = (N * M_avg)`

### Step 4: Select Edges Using Custom Budgets
- **Algorithm:** RobustPrune (from Vamana, Paper 9)
- **What happens:**
  - For each node, we run RobustPrune but stop at the **custom budget** for that node (not a global R).
  - For high-difficulty nodes: use larger α → bias toward long-range edges.
  - For low-difficulty nodes: use smaller α → bias toward local edges.

### Step 5: Connectivity Repair
- **Algorithm:** DFS (Depth-First Search) reachability check
  - *(Note: DFS is an algorithm that starts at a single node and attempts to visit every single other node in the graph. If it misses some, we know the graph is broken into disconnected islands).*
- **What happens:**
  - Run DFS from a central node.
  - If any nodes are unreachable, add bridge edges.
  - Guarantees 100% connectivity.
- **Why it is needed:**
  - Because we took edges away from easy nodes, some might become isolated.
  - The repair step guarantees the graph is still fully connected.

### Step 6 (Optional): Iterative Refinement
- **What happens:**
  - After building the final graph, repeat Steps 2-4.
  - The difficulty scores change because the graph changed.
  - Each iteration produces a better graph.
- **When to stop:** When difficulty scores stabilize (convergence).

---

# SECTION G: WHY THIS IS NOVEL (The Novelty Argument)

## Three specific novelty claims:

### Claim 1: The Feedback Loop
- No existing standard method defines a direct feedback loop between **greedy-search failure rate** and **edge budget redistribution**.
- Existing methods look at geometry (distances) to build edges.
- We look at **algorithmic failure** to build edges.

### Claim 2: Fixed-Budget Redistribution
- We don't just add edges (that's trivial — more edges = better search = obvious).
- We **redistribute** a strictly fixed budget.
- This means we prove that **WHERE you put edges matters more than HOW MANY you have**.

### Claim 3: Search-Aware vs. Data-Aware
- DATE 2025 adapts by density → **data-aware** (static property of the points).
- We adapt by navigational difficulty → **search-aware** (dynamic property of the graph + algorithm).
- A node in a dense region can still be hard to reach if the graph topology around it is poor.
- Our method captures this; density-based methods cannot.

---

# SECTION H: ROADMAP (What We Plan To Do)

## Phase 1: Literature Confirmation (✅ DONE)
- [x] Review major graph ANN papers (2000–2025).
- [x] Identify the gap (uniform degree everywhere).

## Phase 2: Mathematical Formalization (NEXT)
- [ ] Formally define the navigational difficulty metric
- [ ] Prove that the budget redistribution is well-defined (conservation law)
- [ ] Analyze the convergence of the iterative refinement loop
- [ ] Write the formal problem statement

## Phase 3: Implementation
- [ ] Implement NN-Descent for initial graph (can use existing library)
- [ ] Implement multi-start greedy search for difficulty estimation
- [ ] Implement budget allocation with clipping and normalization
- [ ] Implement modified RobustPrune with per-node budget
- [ ] Implement DFS connectivity repair
- [ ] Build the full construction pipeline

## Phase 4: Experiments
- [ ] **Datasets:** SIFT1M, GIST1M, GloVe-100, Deep1M
- [ ] **Baselines:** HNSW (hnswlib), NSG, Vamana/DiskANN
- [ ] **Key comparison:** Same total edge budget → compare recall and latency
- [ ] **Ablation studies:**
  - The method with uniform degree (to prove adaptation helps)
  - The method without long-range bias (to prove it helps hard nodes)
  - The method with density instead of difficulty (to prove difficulty > density)
- [ ] **Metrics:** Recall@1/10/100, QPS, p95/p99 latency, memory

## Phase 5: Paper/Project Writing
- [ ] Write introduction, related work, method, experiments, conclusion
- [ ] Expected timeline: 3-4 months for implementation + experiments

---

# SECTION I: FINAL SUMMARY

| Question | Answer |
|----------|--------|
| Is ERNG novel? | ❌ No — combination of known techniques |
| Is the gap real? | ✅ Yes — standard methods do not do per-node degree by navigational difficulty |
| Is the Proposed Method novel? | ✅ Yes — strongly differentiated from standard HNSW and Vamana |
| What makes it novel? | The feedback loop: search failures → degree redistribution |
| What datasets? | SIFT1M, GIST1M, GloVe-100, Deep1M |
| What baselines? | HNSW, NSG, Vamana |
| What do we keep from ERNG? | The flat-graph philosophy and power-of-two skip link idea as inspiration |
| What do we change? | Replace fixed degree with difficulty-aware adaptive degree |
| What is the risk? | Improvement may be small on easy datasets; construction overhead of difficulty estimation |
| What is the expected contribution? | New problem formulation + new mechanism + general-purpose algorithm |
