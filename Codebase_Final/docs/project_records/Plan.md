# Image Copy Detection Strategy for E-Commerce Platforms

**Team Members:** Saketh Pabbu, Nerella Venkata Sriram

Deep Learning Network–based image copy detection system for e-commerce platforms to identify fake sellers, prevent fraudulent listings.

---

## **1. INTRODUCTION**

### **Main Problem**

E-commerce platforms (Amazon, Flipkart, Alibaba, Google Shopping) are flooded with duplicate product listings. Fake sellers steal images from genuine sellers and use them to sell counterfeit/low-quality products.

### **How Fake Sellers Operate**

1. Find popular products with good reviews (e.g., Nike shoes)
2. Download original seller's product images
3. Make small modifications to avoid detection
4. Upload as their own product listing
5. Sell fake/counterfeit products using stolen images

### **Common Modifications That Should Be Detected**

| Modification | Example | Detection Requirement |
| :--- | :--- | :--- |
| **Rotation** | Shoe facing left → rotated to face right (same shoe) | Detect invariance to rotation |
| **Cropping** | Full handbag photo → cropped to show only bag (same handbag) | Handle partial visibility |
| **Color Changes/Filters** | Bright t-shirt → Instagram filter applied (same t-shirt) | Ignore color variations |
| **Flipping (Mirror)** | Watch on left wrist → flipped to right wrist (same watch) | Detect mirror symmetry |
| **Resizing/Compression** | High quality 4K → compressed low quality (same image) | Robust to resolution changes |
| **Adding Watermark/Text** | Clean photo → "SALE 50% OFF" added (same photo) | Ignore overlays |

### **What Counts as "Copy"?**

Any of the above modifications applied to the original product image without changing the core product identity.

---

## **2. PUBLIC DATASETS & REFERENCES**

### **Data Source**

**Primary Source:** Amazon Product Metadata 2023 (McAuley Lab, UCSD)
- **Fashion & Apparel:** Clothing, Shoes and Jewelry, Amazon Fashion
- **Electronics:** Cell Phones and Accessories, Consumer Electronics
- **Home:** Home and Kitchen categories
- **Beauty:** Beauty and Personal Care
- **Sports:** Sports and Outdoors

**Dataset Extraction:** 6-pillar extraction across Baby Products, Cosmetics, Fashion, Electronics, Sports, and Home categories

### **Supporting Benchmarks**

- **Copydays Dataset** - Near-duplicate image detection benchmark
- **DISC21 (Facebook AI Image Similarity Challenge)** - Image similarity benchmark
- **Stanford Online Products Dataset** - Fine-grained product retrieval
- **DeepFashion** - Fashion-specific product images

### **Related Papers & References**

| Paper | Key Insights |
| :--- | :--- |
| **Paper 1: Adversarial Graph Pairwise Training for Robust Visual Recommendation** | Inspired adversarial training approach; augmentations (cropping, watermarks, color jitter, JPEG compression, rotation/flip, blur/noise) applied to ResNet-50 inputs for robust positive pairs |
| **Paper 2: Siamese Coding Network for Near-Duplicate Image Detection** | Parallel CNN architecture using Siamese networks with shared ResNet-50 weights to extract global (256-dim) and local (128-dim) features |
| **Paper 3: Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs** | HNSW graph construction enabling O(log K) queries via efficient graph traversal |
| **Paper 4: Fast Transfer Learning Method Using Random Layer Freezing and Feature Refinement Strategy** | Random layer freezing to reduce overhead and accelerate adaptation (integration in progress) |
| **Paper 5: Cross-ViT: Cross-attention Vision Transformer for Image Duplicate Detection** | Cross-attention mechanisms for duplicate detection; exploration of ViT backbone replacement |

---

## **3. ARCHITECTURE - LAYER I: DATASET PREPARATION**

### **Dataset Structure**

- **Organization:** 750 refined categories × ~300 images per category = ~225,000 total images
- **Structure:** Category-wise folders with hierarchical organization
- **Storage:** `Final Dataset/[Category]/[Product_Folder]/`

### **Pair Generation Strategy**

**Total Universe:** 225,000 products managed via **Master Manifest (training_universe.json)**
**Total Pairs Generated:** ~815,440 image pairs (all stored as JSON metadata, NOT physical images)

**Distribution:**
- **Positive Pairs:** 235,500 soft + 329,220 hard = 564,720 (69%)
- **Negative Pairs:** 308,000 soft + 99,720 hard = 407,720 (50%)
- **Total:** 815,440 pairs

### **The Master Manifest (training_universe.json)**

**Purpose:** Single 180MB JSON file containing all 1,186,940 pair definitions with loss tracking

**Manifest Entry Structure:**
```json
{
  "row_id": 1,
  "path_1": "Final Dataset/Electronics/phone_001/_org.jpg",
  "path_2": "Final Dataset/Electronics/phone_001/_duplicate_1.jpg",
  "label": 1,
  "pair_type": "soft_pos",
  "priority": 0.0,
  "times_trained": 0
}
```

**Benefits:**
- ✅ Zero disk waste for hard positives (generated in RAM)
- ✅ Dynamic priority tracking (loss scores after each epoch)
- ✅ Automatic queue shrinkage (delete when mastered)
- ✅ Scalable to billions of pairs with minimal storage

### **4-Pillar Pair Strategy**

#### **1. SOFT POSITIVES** (450,000 pairs = 225,000 products × 2 variants)**

- **Definition:** Same product with one identifiable, single attack per variant
- **Variants per Product:**
  - Variant 1: 80% center crop (removes peripheral details)
  - Variant 2: 90° rotation (spatial reorientation)
  - Variant 3: Hue shift ±30° (color manipulation)
- **Generation:** Pre-computed once during dataset setup
- **Storage:** Physically saved as `_duplicate_1.jpg`, `_duplicate_2.jpg`, `_duplicate_3.jpg` (3 files per product)
- **Manifest Entry:** `path_1 = _org.jpg`, `path_2 = _dup_1/2/3.jpg`, `label = 1`, `pair_type = soft_pos`, `priority = 0.0`
- **Purpose:** Base identity learning with single, identifiable manipulations
- **Training:** Fixed attacks per variant (consistent across epochs)
- **Disk Storage:** Heavy (235,500 extra files physically stored)
- **Delta:** Adds ~706,500 MB (~707 GB) to dataset size

#### **2. HARD POSITIVES** (329,220 pairs = 78,500 products × ~4.2 repetitions)**

- **Definition:** Same product with extreme, combined attack **generated in RAM each epoch**
- **Attack Combinations (random selections per epoch):**
  - Epoch 1 Example: 50% crop + 90° rotation + grayscale + watermark
  - Epoch 2 Example: Blur + hue shift + heavy JPEG + white border
  - Epoch 3 Example: Noise + vertical flip + brightness jitter + overlay
  - Epoch N: Different random combo each time
- **Generation:** During training (on-the-fly in memory)
- **Storage:** NOT physically stored; only referenced via manifest
- **Manifest Entry:** `path_1 = _org.jpg`, `path_2 = _org.jpg` (**SAME FILE**), `label = 1`, `pair_type = hard_pos`, `priority = 0.50`
- **Ghost Row Logic:** Both paths point to same source file, but attack is generated in RAM before training
- **Purpose:** Product "DNA" learning; model learns invariant features across **diverse attack combinations**
- **Training:** Different random attacks each epoch (prevents memorization of specific attacks)
- **Disk Storage:** Zero (all attacks generated in memory; never saved to disk)
- **Key Insight:** No disk waste despite 329k pairs; achieved via in-memory augmentation

#### **3. SOFT NEGATIVES** (307,720 pairs = category cross-pairing)**

- **Definition:** Cross-category pairing strategy
- **Calculation:** 785 categories × 784 other categories ÷ 2 = 307,720 theoretical pairs
- **Actual Pairs:** Stochastic sampling to create balanced diverse negatives
- **Example Pairs:** Electronics vs. Home & Kitchen, Cosmetics vs. Sports, Fashion vs. Baby
- **Generation:** Once during manifest creation (pre-sampled)
- **Storage:** Logically paired; both image files exist in different category folders
- **Manifest Entry:** `path_1 = category_A/product_X/_org.jpg`, `path_2 = category_B/product_Y/_org.jpg`, `label = 0`, `pair_type = soft_neg`, `priority = 0.0`
- **Purpose:** Easy baseline; model learns "obviously different" visual patterns
- **Training:** Consistent pairs throughout training (same images every epoch)
- **Disk Storage:** Uses existing files (no additional storage)
- **Priority Behavior:** Always starts at 0.0 (easily mastered); typically deleted after Epoch 1-2

#### **4. HARD NEGATIVES** (100,000 pairs = stochastic same-category negatives)**

- **Definition:** Same-category pairing strategy: different products within same category
- **Sampling Strategy:** ~127 hard negatives per category (stochastic random selection)
- **Example Pairs:** 
  - Electronics: Phone A vs. Phone B, Laptop A vs. Laptop B
  - Sports: Shoe A vs. Shoe B, Ball A vs. Ball B
- **Generation:** Once during manifest creation
- **Storage:** Logically paired; both image files exist but different product folders
- **Manifest Entry:** `path_1 = category/product_A/_org.jpg`, `path_2 = category/product_B/_org.jpg`, `label = 0`, `pair_type = hard_neg`, `priority = 0.15-0.25`
- **Purpose:** Difficult baseline; model learns fine-grained distinction between similar products
- **Training:** Consistent pairs throughout training (same images every epoch)
- **Disk Storage:** Uses existing files (no additional storage)
- **Priority Behavior:** Starts at 0.15-0.25 (harder than soft negatives); pruned gradually as loss decreases

**Pair Distribution Summary Table:**

| Type | Count | Storage | Path_1 | Path_2 | Label | Initial Priority | Created When | When Pruned |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **Soft Pos** | 235,500 | Disk ✅ | _org | _dup_1/2/3 | 1 | 0.0 | Before training | Never (fixed) |
| **Hard Pos** | 329,220 | RAM only ❌ | _org | _org (same) | 1 | 0.50 | During training | When priority < 0.05 |
| **Soft Neg** | 307,720 | Disk ✅ | Cat A | Cat B | 0 | 0.0 | Before training | Epoch 1-2 |
| **Hard Neg** | 100,000 | Disk ✅ | Prod A | Prod B (same cat) | 0 | 0.15-0.25 | Before training | Gradually (Epoch 2+) |
| **TOTAL** | **815,440** | **~707GB + JSON** | - | - | - | - | - | Progressive shrinkage |

### **Augmentations Applied**

Randomly applied per image during data loading:

| Augmentation | Purpose | Parameters |
| :--- | :--- | :--- |
| **RandomResizedCrop** | Simulate cropping attacks | Scale: (0.8-1.0), Ratio: (0.75-1.33) |
| **ColorJitter** | Simulate color/filter attacks | Brightness: 0.2, Contrast: 0.2, Saturation: 0.2 |
| **GaussianBlur** | Simulate blur attacks | Kernel size: 3-5, Sigma: 0.1-2.0 |
| **JPEG Compression** | Simulate compression artifacts | Quality: 30-70 |
| **RandomHorizontalFlip** | Simulate mirroring attacks | Probability: 0.5 |
| **Normalize** | Standardize input | ImageNet Stats |

### **Priority Queue System (Dynamic Loss-Based Sorting)**

**Architecture:** Single unified priority queue stored in manifest with loss-based prioritization

**Initialization at Training Start:**
```
soft_positives: priority = 0.0         (easiest)
soft_negatives: priority = 0.0         (easiest)
hard_negatives: priority = 0.15-0.25   (medium)
hard_positives: priority = 0.50        (hardest)
```

**Batch Formation (Epoch N):**
```
1. Read all 815,440 pairs from manifest
2. Sort by priority (descending: highest loss first)
3. Pull top 64 pairs:
   - ~32 Hard Positives (priority: 0.45-0.85)
   - ~16 Hard Negatives (priority: 0.10-0.30)
   - ~8 Soft Positives (priority: 0.00-0.05)
   - ~8 Soft Negatives (priority: 0.00-0.02)
4. Train on this batch
5. Update priority scores in manifest
6. Re-sort for next batch
```

**Balanced Training:** Hard pairs trained first; easy pairs naturally deprioritized

### **Pair Evolution Rules (Prevent Overfitting & Ensure Convergence)**

#### **Rule 1: Direct Loss-Based Priority**
```
After each epoch:
  manifest[pair_id]["priority"] = calculated_loss
  
Example progression:
  Epoch 1: priority = 0.50 → Loss calculated = 0.42 → priority = 0.42
  Epoch 2: priority = 0.42 → Loss calculated = 0.38 → priority = 0.38
  Epoch 3: priority = 0.38 → Loss calculated = 0.35 → priority = 0.35
```

#### **Rule 2: Decay Factor (Prevent Over-Training)**
```
Applied ONLY to pairs trained multiple times:
  adjusted_priority = loss × (decay_factor ^ times_trained)
  
Where:
  - decay_factor = 0.9
  - times_trained = number of epochs this pair has been trained
  - Penalizes pairs trained repeatedly (reduces memorization)
  
Example (same pair over 3 epochs):
  Epoch 1: loss=0.45, times_trained=1
    priority = 0.45 × (0.9 ^ 1) = 0.45 × 0.9 = 0.405
    
  Epoch 2: loss=0.38, times_trained=2
    priority = 0.38 × (0.9 ^ 2) = 0.38 × 0.81 = 0.308
    
  Epoch 3: loss=0.35, times_trained=3
    priority = 0.35 × (0.9 ^ 3) = 0.35 × 0.729 = 0.255
```

#### **Rule 3: Queue Pruning (Delete When Mastered)**
```
After each epoch:

if priority < 0.05:  # Pair has been mastered
    DELETE row from manifest
    Add to learned_pairs archive
    Free up space in training queue
else:
    KEEP in manifest
    Increment times_trained counter
    Continue training next epoch
```

#### **Rule 4: Hard Positive Persistence**
```
Hard positives are retained longer because:
- They generate new random attacks each epoch
- priority never reaches 0 (always have variance)
- Training stops only when priority stabilizes or epoch limit reached
- Ensures robust "DNA" learning throughout
```

#### **Pseudocode Implementation**
```python
# After each training epoch
for pair_id, pair_data in manifest.items():
    calculated_loss = train_on_pair(pair_data)
    
    # Apply decay if trained multiple times
    times_trained = pair_data["times_trained"]
    if times_trained > 0:
        decay_penalty = decay_factor ^ times_trained
        adjusted_loss = calculated_loss * decay_penalty
    else:
        adjusted_loss = calculated_loss
    
    # Pruning decision
    if adjusted_loss < 0.05:
        # Pair mastered - delete from manifest
        learned_pairs.append(pair_id)
        manifest.delete(pair_id)
    else:
        # Pair still challenging - keep and update
        manifest[pair_id]["priority"] = adjusted_loss
        manifest[pair_id]["times_trained"] += 1
        
# Save updated manifest back to disk
save_manifest(manifest, "training_universe.json")
```

### **Dynamic Queue Shrinkage Over Training**

As training progresses, the manifest automatically shrinks as pairs are mastered:

| Epoch | Manifest Size | Total Pairs | Soft Neg Remaining | Hard Neg Remaining | Soft Pos Remaining | Hard Pos Remaining | Batches |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Start** | 815,440 rows | 815,440 | 307,720 | 100,000 | 235,500 | 329,220 | 12,741 |
| **After Ep 1** | ~765,440 | 765,440 | ~257,720 (**-50k**) | 100,000 | ~233,000 | 329,220 | 11,960 |
| **After Ep 2** | ~735,440 | 735,440 | ~227,720 (**-30k**) | ~95,000 | ~232,000 | 329,220 | 11,491 |
| **After Ep 3** | ~715,440 | 715,440 | ~207,720 (**-20k**) | ~90,000 | ~231,000 | 329,220 | 11,179 |
| **After Ep 5** | ~635,000 | 635,000 | ~127,000 | ~75,000 | ~230,000 | 329,220 | 9,922 |
| **Convergence** | ~50,000-100,000 | ~50,000-100,000 | ~0 (all deleted) | ~10,000-20,000 | ~10,000-20,000 | ~329,220 (residue) | ~781-1,563 |

**Deletion Pattern:**
1. **Soft negatives** deleted first (easiest; priority = 0.0)
2. **Soft positives** deleted second (easy; priority = 0.0-0.05)
3. **Hard negatives** deleted third (medium; priority = 0.15-0.25)
4. **Hard positives** retained longest (hardest; priority = 0.50+ regenerated each epoch)

**Final State:** Only hard positives remain (or hard negatives + hard positives), focusing on the most challenging product identity distinctions

### **Hard Positive Evolution Examples**

Each hard positive always has same file reference (`path_1 = path_2`), but attacks change each epoch:

| Epoch | Manifest | Random Attacks Applied (in RAM) | Loss | Priority (with decay) | Times_Trained |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | path_1=_org, path_2=_org | 50% crop + 90° rotate + grayscale | 0.42 | 0.42 | 1 |
| **2** | path_1=_org, path_2=_org | Blur + hue shift + JPEG compression | 0.38 | 0.34 (0.38 × 0.9) | 2 |
| **3** | path_1=_org, path_2=_org | Noise + flip + border + watermark | 0.35 | 0.26 (0.35 × 0.81) | 3 |
| **4** | path_1=_org, path_2=_org | Color jitter + gaussian blur + sepia | 0.32 | 0.21 (0.32 × 0.729) | 4 |
| **5** | path_1=_org, path_2=_org | Different random combo again... | 0.30 | 0.18 (0.30 × 0.656) | 5 |

**Key Insight:** No disk writes for hard positives; attacks generated in-memory each epoch, preventing memorization while maintaining zero storage overhead

---

## **4. ARCHITECTURE - LAYER II: TRAINING PHASE**

### **Siamese Network Architecture**

**Name:** ResNet-50-based Siamese Neural Network

**Total Layers:** 50 layers (16 bottleneck blocks across 4 layers)

**Output:** 384-dimensional feature vector

### **ResNet-50 Layer Architecture**

```
────────────────────────────────────────────────
Conv1: 7×7 conv, stride 2 → MaxPool
│
Output: 56×56×64
│
Layer 1: 3 Bottleneck blocks (1×1, 3×3, 1×1)
│ Output: 56×56×256
│ Focus: Edges, textures, colors
│
Layer 2: 4 Bottleneck blocks (1×1, 3×3, 1×1)
│ Output: 28×28×512
│ Focus: Shapes, patterns, parts
│
Layer 3: 6 Bottleneck blocks (1×1, 3×3, 1×1)
│ Output: 14×14×1024
│ Purpose: LOCAL FEATURES (128-dim)
│
Layer 4: 3 Bottleneck blocks (1×1, 3×3, 1×1)
│ Output: 7×7×2048
│ Purpose: GLOBAL FEATURES (256-dim)
│
────────────────────────────────────────────────
```

### **Feature Extraction Strategy**

| Feature Type | Source Layer | Spatial Size | Technique | Dimensions | Focus Area |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **LOCAL** | Layer 3 | 14×14 | 4×4 grid pool + FC | 128-dim | Textures, stitching details |
| **GLOBAL** | Layer 4 | 7×7 | AvgPool + FC | 256-dim | Silhouette, shape, overall form |
| **COMBINED** | Both | - | Concatenation | 384-dim | Final forensic signature |

### **Dual-Branch Siamese Architecture**

```
Input Image 1          Input Image 2
     ↓                      ↓
  ResNet-50 (shared weights; identical architecture)
     ↓                      ↓
  Layer 3 (14×14×1024)   Layer 3 (14×14×1024)
     ↓                      ↓
  Grid Pool + FC → 128-dim LOCAL  128-dim LOCAL
     ↓                      ↓
  Layer 4 (7×7×2048)     Layer 4 (7×7×2048)
     ↓                      ↓
  AvgPool + FC → 256-dim GLOBAL   256-dim GLOBAL
     ↓                      ↓
  Concatenate [256 + 128]  Concatenate [256 + 128]
     ↓                      ↓
   384-dim embedding     384-dim embedding
     ↓─────────────────────↓
         Compare Distance
           (Contrastive Loss)
```

### **Training & Usage**

Siamese network with two parallel ResNet-50 branches sharing identical weights, trained via **contrastive loss** on image pairs to:
- Minimize distances between similar images (positive pairs)
- Maximize distances between dissimilar images (negative pairs)

### **Loss Calculation**

#### **Total Loss Formula**
```
TOTAL_LOSS = GLOBAL_LOSS + λ × LOCAL_LOSS

Where:
  λ (local_weight) = 0.5
  Global Loss: weight = 1.0 (shape and silhouette critical)
  Local Loss: weight = 0.5 (texture details secondary)
```

#### **Contrastive Loss Formula**

For a pair (img1, img2) with label Y:
- Y = 1 → Same image (positive pair)
- Y = 0 → Different images (negative pair)

**STEP 1: COMPUTE EUCLIDEAN DISTANCE**
```
D = ||emb1 - emb2||₂ = √(Σ(emb1ᵢ - emb2ᵢ)²)
```

**STEP 2: COMPUTE CONTRASTIVE LOSS**
```
L = Y × D² + (1 - Y) × max(0, margin - D)²

Where margin = 1.0
```

#### **Behavior on Different Pair Types**

**POSITIVE PAIRS: Push TOGETHER**
```
Before Optimization:
●─────────────●
 D = 2.0, L = 4.0

After Optimization:
●──●
 D = 0.2, L = 0.04
```

**NEGATIVE PAIRS: Push APART (beyond margin)**
```
Before Optimization:
●──●
 D = 0.3, L = 0.49

After Optimization:
●─────────────●
 D = 1.5, L = 0
 (within margin boundary)
```

### **Queue Shrinkage Over Epochs**

| Epoch | Pairs at Start | Batches | Learned & Removed | Remaining Pairs |
| :--- | :---: | :---: | :---: | :---: |
| **Epoch 1** | 500,000 | 7,812 | 100,000 | 400,000 |
| **Epoch 2** | 400,000 | 6,250 | 50,000 | 350,000 |
| **Epoch 3** | 350,000 | ~5,468 | 36,000 | 314,000 |
| **... Epoch N** | Decreasing | Decreasing | Decreasing | Decreasing |

**Convergence Condition:** Hard pairs stabilize or learning stops

---

## **5. ARCHITECTURE - LAYER III: INDEX/GRAPH CONSTRUCTION**

### **Post-Training: Freeze Model**

```
1. Set model to evaluation mode
2. Disable gradient updates
3. Lock trained weights
```

### **Compute & Save Embeddings**

```python
for img_id, img in all_images:
    global_feat, local_feat = model(img)
    embeddings[img_id] = (global_feat, local_feat)
```

Store embeddings in a persistent file for indexing.

### **Part I: Clustering Similar Images**

#### **Step 1: Random Initialization (384-dim Embeddings)**

```
Pick random embeddings as centroids:
  Ci = [2.0, 3.0, 4.0, ...] # 384-dim
  Cj = [9.5, 8.8, 9.1, ...] # 384-dim
  Ck = [5.0, 5.5, 5.2, ...] # 384-dim
```

#### **Step 2: Assign All N Embeddings**

For each 384-dim embedding x:
```
Assign x → nearest of {Ci, Cj, Ck}
```

#### **Step 3: Recompute Centroid (Mean Update)**

```
Ci_new = mean(all 384-dim embeddings assigned to Ci)
Similarly compute for Cj_new and Ck_new
```

#### **Step 4: Stop Condition**

```
Stop when:
  ||Ci_new − Ci_old|| < ε
  for all centroids i
```

#### **Number of Centroids Calculation**

```
K ≈ √N

Example: For N = 50,000
  K ≈ √50,000 ≈ 223
  Use K = 256 (nearest power of 2)
  Clusters: ~50,000 / 256 ≈ 195 images per cluster
  
For N = 78,500 (current dataset):
  K ≈ √78,500 ≈ 280
  Use K = 256 or 512
  Clusters: 100-180 images per cluster (average)
```

---

### **Part II - Approach 1: HNSW (Hierarchical Navigable Small World Graph)**

#### **Step 1: Layer Assignment**

Calculate centrality for each centroid:
```
Centrality(Ci) = sum of distances to all other centroids
```

Assign centroids to layers based on centrality:
```
Most central centroids → Top layer (entry points)
Moderate centrality → Middle layers
All centroids → Bottom layer

Rule: If Ci in Layer L → Ci exists in all layers ≤ L
```

#### **Step 2: Building Connections**

For each layer independently:
```
For each centroid Ci in that layer:
  - Find M nearest neighbors within the layer
  - Connect Ci to those neighbors
```

**Example:**
```
Layer 2: [Ci, Cj, Ck] exist
  Ci connects to: Cj, Ck (only 2 neighbors)

Layer 1: [Ci, Cj, Ck, Cp, Cq, Cr] exist
  Ci connects to: Cj, Cp, Cq (more neighbors available!)

Layer 0: ALL centroids exist
  Ci connects to: Cj, Cs, Ct (maximum neighbors)
```

#### **Step 3: HNSW Query Search**

**1. Entry Point Selection:**
```
Compare query to all centroids in top layer
Start at closest one (e.g., Ci)
```

**2. Layer-by-Layer Greedy Descent:**
```
Current layer = Top
Current centroid = Ci

While not at bottom layer:
  Look at Ci's neighbors in current layer: {Cj, Ck, ...}
  Find best = closest neighbor to query
  
  If best is closer than current:
    Move to best
    Repeat in same layer
  Else:
    Stuck in this layer
    Drop to next layer below
    (Ci now has MORE neighbors to explore)
```

**3. Bottom layer search finds final centroid**

**4. Search within that centroid's cluster for duplicates**

#### **Connection Strategy Details**

For each centroid Ci:
```
Calculate distances to all other centroids
Sort by distance:
  Rank 1: Cj (closest)
  Rank 2: Ck (2nd closest)
  Rank 3: Cp (3rd closest)
  Rank 4: Cq (4th closest)
  ...
  Rank N: Cz (farthest)
```

**Local Connections (K-NN):**
```
Ranks: {1, 2, 3, ..., M}
Example: Ci → Cj, Ck, Cp
```

**Exponential Skip Connections:**
```
Ranks: {2^k where k = 0, 1, 2, ..., ⌈log₂(K)⌉}
     = {1, 2, 4, 8, 16, ..., K}
Example: Ci → Cq (rank 4), Cr (rank 8)
```

**Combined (remove overlaps):**
```
Ci's full neighbor set = {Cj, Ck, Cp, Cq, Cr, ...}
Each Ci connects to different centroids (based on own distance ranking)
```

---

### **Part II - Approach 2: ERNG (Exponential Rank Navigation Graph)**

#### **Step 1: Graph Construction**

Build a layered graph structure where each centroid is connected to:
- **Local K-NN:** Nearest M neighbors
- **Exponential Skip Connections:** Ranks {1, 2, 4, 8, 16, ...}

#### **Step 2: Connection Strategy**

Same as HNSW but without strict layering; more flexible topology.

#### **Step 3: ERNG Query Search**

**1. Launch R Probes from Random Centroids:**
```
Probe 1: Start at Ci
Probe 2: Start at Cj
Probe 3: Start at Ck
```

**2. Each Probe: Greedy Search**
```
Current = starting centroid

While True:
  Check neighbors (M local + exponential skips)
  Best = neighbor closest to query
  
  If best closer than current:
    Move to best
  Else:
    Stop → return current
```

**3. Select Best Result:**
```
Probe 1 → Cp (distance d1)
Probe 2 → Cq (distance d2)
Probe 3 → Cr (distance d3)

Final = min(d1, d2, d3)
```

**4. Search within final centroid's cluster for duplicates**

---

## **6. ARCHITECTURE - LAYER IV: QUERY PHASE**

### **Query Pipeline**

```
Query Image
    ↓
ResNet-50 (frozen, trained weights)
    ↓
Extract Features:
  - Global: 256-dim vector (from Layer4 avg pool)
  - Local: 128-dim vector (from Layer3 grid pool)
    ↓
Combined: 384-dim embedding (global + local concatenated)
    ↓
CHOOSE SEARCH APPROACH:
    ↓
┌───────────────────────────────────────┐
│ Option A: HNSW Search                 │
│ Navigate through hierarchical layers  │
│ Start at √K entry points in top layer │
│ Drop through layers until final found │
├───────────────────────────────────────┤
│ Option B: ERNG Search                 │
│ Launch R=3 probes from random clusters│
│ Each probe greedy navigation           │
│ Take best result from all probes      │
└───────────────────────────────────────┘
    ↓
Find Final Centroid (~100-180 images cluster)
    ↓
For each candidate image:
  Calculate embedding distance (Euclidean)
    ↓
Rank by embedding distance
    ↓
Return top matches above threshold → DUPLICATES DETECTED
```

### **Detailed Query Steps**

#### **Step 1: Feature Extraction**
```
Input: Query image
Extract:
  - Global 256-dim from Layer 4
  - Local 128-dim from Layer 3
  - Concatenate → 384-dim embedding
```

#### **Step 2: Find Nearest Centroid**
```
Use HNSW or ERNG to navigate to closest centroid cluster
Complexity: O(log K) for both approaches
```

#### **Step 3: Search Cluster & Verify Duplicates**
```
For each image in final centroid cluster:
  Distance = ||query_embedding - image_embedding||₂
  
If distance < threshold:
  Label as DUPLICATE
  
Return ranked list of duplicates
```

---

## **7. FUTURE SCOPE**

### **Collage and Image Blending Detection**
- Detect and split collages (multiple images combined)
- Handle blended/superimposed images
- Check each part separately
- Prevent fraudsters from hiding copied content in stitched images

### **Vision Transformer (ViT) Backbone**
- Replace current CNN (ResNet-50) backbone
- Leverage self-attention mechanisms
- Capture global patch-level relationships
- Improve detection of complex copy-edits spanning multiple regions

### **GAN-Enhanced Protection**
- Train against AI-generated realistic modifications
- Use GANs to create sophisticated attacks
- Ensure robustness beyond simple mathematical perturbations
- Handle future fraud techniques

### **Additional Improvements**
- Cross-modal search (text + image)
- Temporal consistency for video frames
- Real-time detection at scale
- Multi-region forensic analysis

---

## **8. CONCLUSION & PROJECT STATUS**

### **Complete System Architecture**

This document outlines a **production-grade image copy detection system** designed for e-commerce fraud prevention. The architecture spans 4 interconnected layers:

**Layer I - Dataset Preparation (815,440 Pairs via Master Manifest)**
- Source: Amazon 2023 metadata (6 product categories)
- Refined to: 785 categories, 78,500 unique products
- Pair generation: 4-pillar strategy (soft pos, hard pos, soft neg, hard neg)
- Storage: 150MB JSON manifest + 707GB soft positive images
- Hard positives: Generated in-RAM (zero disk overhead)

**Layer II - Training Phase (Siamese ResNet-50 Network)**
- Architecture: Dual-branch ResNet-50 with shared weights
- Output: 384-dim hybrid embedding (256 global + 128 local)
- Loss: Contrastive loss with weighted global/local components
- Optimization: Priority queue with loss-based sorting
- Queue management: Dynamic shrinkage via decay factor & pruning

**Layer III - Indexing/Graph Construction (HNSW or ERNG)**
- Post-training embedding extraction and clustering
- K-means clustering: ~256 centroids from 78,500 embeddings
- Graph structure: HNSW (hierarchical) or ERNG (exponential rank)
- Query complexity: O(log K) for millions of products
- Cluster size: 100-180 images per centroid

**Layer IV - Query/Detection Phase (Real-Time Duplicate Search)**
- Input: Query product image
- Processing: Extract 384-dim embedding (frozen model)
- Navigation: HNSW/ERNG graph traversal to nearest centroid
- Verification: Euclidean distance ranking within cluster
- Output: Ranked list of duplicates above threshold

### **Key Design Innovations**

| Innovation | Benefit | Implementation |
| :--- | :--- | :--- |
| **Master Manifest JSON** | Zero disk waste for 329k hard positives | Ghost Row logic (path_1 = path_2) |
| **In-Memory Attacks** | Random attacks each epoch (prevents memorization) | Augmentation pipeline in dataloader |
| **Priority Queue** | Train hardest pairs first (efficient convergence) | Loss-based sorting with decay factor |
| **Queue Shrinkage** | Auto-delete mastered pairs (reduces overhead) | Threshold < 0.05 triggers deletion |
| **Hybrid Embeddings** | Captures both fine & coarse features | 128-dim local + 256-dim global |
| **Logarithmic Search** | Fast querying for scale | HNSW/ERNG graph structure |

### **Project Completion Status**

#### **✅ FULLY COMPLETED**

| Component | Details | Verification |
| :--- | :--- | :--- |
| **Dataset Extraction** | 6-pillar mining from Amazon 2023 metadata | 78,500 products verified |
| **Category Refinement** | Collapsed redundant categories → 785 master categories | Category_mapping.json created |
| **Soft Positives** | 3 variants per product (crop, rotate, hue shift) | 235,500 pairs, 3 files per product |
| **Hard Positive Manifest** | 329,220 pairs stored as JSON pointers | training_universe.json (150MB) |
| **Negative Pair Generation** | 307,720 soft + 100,000 hard negatives | Stochastic sampling documented |
| **Attack Dictionary** | 54 forensic attack variations catalogued | attack_dictionary.md published |
| **Siamese Architecture** | ResNet-50 defined (50 layers, 384-dim output) | Architecture diagrams documented |
| **Loss Function** | Contrastive loss with weighted components | Mathematical formulas specified |
| **Priority Queue Logic** | Loss-based sorting with decay factor | Pseudocode provided |
| **Queue Shrinkage Rules** | Threshold < 0.05 pruning, decay calculations | Detailed tables with epoch progression |
| **Graph Indexing** | HNSW and ERNG strategies documented | Connection strategies specified |
| **Query Pipeline** | Complete 4-step querying process defined | Feature extraction to ranking |
| **Documentation** | 5 comprehensive markdown files | All layers thoroughly explained |

#### **🔄 READY FOR IMPLEMENTATION**

| Component | Status | Next Steps |
| :--- | :--- | :--- |
| **Training Code** | Architecture fully defined | Implement training loop with manifest |
| **Loss Optimization** | Formula specified; weights determined | Validate loss calculations |
| **Graph Construction** | Strategy detailed; K-means approach set | Build centroid clustering pipeline |
| **Query Engine** | Complete logic specified | Implement HNSW/ERNG traversal |
| **Evaluation** | Metrics to track performance | Run on validation set |

### **Final Statistics**

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Total Categories** | 785 | Refined from 6 pillars |
| **Total Products** | 78,500 | From Amazon 2023 metadata |
| **Soft Positive Images** | 235,500 × 3 files | = 707 GB storage |
| **Total Pairs in Manifest** | 815,440 | All tracked in JSON |
| **Hard Positives** | 329,220 | Generated in-memory (0 GB) |
| **Negative Pairs** | 407,720 | Soft (307k) + Hard (100k) |
| **Manifest Size** | ~150 MB | Contains all metadata |
| **Batch Size** | 64 pairs | 32 pos + 32 neg per batch |
| **Centroid Count** | ~256 | From K-means (√78,500) |
| **Embedding Dimension** | 384 | 256 global + 128 local |
| **Query Complexity** | O(log K) | Via HNSW/ERNG navigation |

### **Attack Variations Covered**

The system is designed to detect **54+ specific forensic attacks** organized into 5 categories:

| Category | Examples | Detection Method |
| :--- | :--- | :--- |
| **Geometric** | Cropping, aspect ratio, rotation, perspective | Invariant feature learning |
| **Photometric** | Color shifts, saturation, exposure, contrast | Color-agnostic embeddings |
| **Signal Quality** | Noise, blur, JPEG compression | Multi-scale feature extraction |
| **Structural** | Watermarks, overlays, badges, backgrounds | Region-agnostic embeddings |
| **Reflection** | Horizontal/vertical flips, mirrors | Positional embedding invariance |

---

## **FINAL SUMMARY**

The **Image Copy Detection System** is now fully architected and documented. Every component from data preparation through query execution has been:

1. ✅ **Designed** - Complete technical specifications
2. ✅ **Justified** - Rationale for each design choice provided
3. ✅ **Documented** - 5+ supporting documents with tables, formulas, examples
4. ✅ **Validated** - Dataset verified (78,500 products, 785 categories)
5. ✅ **Optimized** - Zero-waste hard positive storage, O(log K) queries

The system is **ready for implementation** and will detect image copies across e-commerce platforms with robustness against sophisticated manipulation attacks.

**Technology Stack:**
- Deep Learning: PyTorch/TensorFlow (Siamese ResNet-50)
- Data Structure: JSON manifest + image file hierarchy
- Graph Search: HNSW/ERNG for efficient nearest neighbor queries
- Scalability: Designed for millions of products

**Next Phase:** Training, validation, and deployment on Amazon/e-commerce platform data


I installed these libraries in the GPU environment:

Environment:

/home/saketh/miniforge3/envs/codebase-gpu
Libraries installed:

torch 1.12.1+cu113
torchvision 0.13.1+cu113
numpy 1.26.4
pillow 12.2.0
opencv-python-headless 4.11.0.86
Supporting packages that came along with them:

requests 2.33.1
typing-extensions 4.15.0
certifi 2026.2.25
charset_normalizer 3.4.7
idna 3.11
urllib3 2.6.3
I also installed Miniforge at:

/home/saketh/miniforge3
And created the Conda environment:

codebase-gpu
Python 3.10.20
CUDA verification passed with PyTorch:

torch CUDA build: 11.3
GPU detected: Tesla K80
CUDA available: True