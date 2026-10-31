"""
model.py — Siamese ResNet-50 with Dual-Branch Feature Extraction
==================================================================

Architecture (from Architecture.md and Plan.md):

  Input Image (224×224×3)
       │
       ▼
  ResNet-50 backbone (BUILT FROM SCRATCH, random init, shared weights)
       │
       ├──── Layer 3 output (14×14×1024) ──→ LOCAL BRANCH  ──→ 128-dim
       │                                      (texture, watermarks, stitching)
       │
       └──── Layer 4 output (7×7×2048)  ──→ GLOBAL BRANCH ──→ 256-dim
                                               (shape, silhouette, overall form)
       │
       ▼
  Concatenate [256 global + 128 local] = 384-dim embedding


Siamese Setup:
  Image A ──→ same ResNet ──→ 384-dim embedding A
  Image B ──→ same ResNet ──→ 384-dim embedding B
                                      │
                              Compare distance
                          (Contrastive Loss in loss.py)

Why from scratch?
  - No pretrained ImageNet weights — train purely on our forensic data
  - Kaiming Normal initialization for stable gradient flow
  - Same architecture as standard ResNet-50 (Bottleneck blocks, expansion=4)
"""

import torch
import torch.nn as nn

from config import GLOBAL_DIM, LOCAL_DIM, DEVICE


# ═══════════════════════════════════════════════════════════
# ResNet-50 — Built From Scratch
# ═══════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck block (3-layer deep).

    Structure:
      Input (in_channels)
        │
        ├─→ 1×1 Conv (squeeze to mid_channels)
        │   → BatchNorm → ReLU
        ├─→ 3×3 Conv (mid_channels, stride)
        │   → BatchNorm → ReLU
        ├─→ 1×1 Conv (expand to mid_channels × 4)
        │   → BatchNorm
        │
        + Skip connection (identity or 1×1 projection)
        │
        → ReLU

    Expansion factor = 4:
      e.g. Bottleneck(64) produces 64×4 = 256 output channels
    """
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: input feature map channels
            mid_channels: bottleneck squeeze channels
            stride: spatial downsampling stride (1 = keep size, 2 = halve)
            downsample: optional 1×1 conv to match dimensions for skip connection
        """
        super().__init__()

        out_channels = mid_channels * self.expansion

        # 1×1 squeeze
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3×3 spatial convolution (this is where stride applies)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1×1 expand
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Skip connection: match dimensions if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50Backbone(nn.Module):
    """
    Full ResNet-50 built from scratch with Kaiming initialization.

    Layer breakdown (standard ResNet-50):
      conv1:  7×7 conv, stride 2  →  224→112, channels 3→64
      bn1 + relu + maxpool        →  112→56
      layer1: 3 Bottleneck blocks →  56×56×256
      layer2: 4 Bottleneck blocks →  28×28×512    (stride 2 on first block)
      layer3: 6 Bottleneck blocks →  14×14×1024   (stride 2 on first block)  ← LOCAL
      layer4: 3 Bottleneck blocks →  7×7×2048     (stride 2 on first block)  ← GLOBAL

    All weights initialized via Kaiming Normal (He initialization).
    """

    def __init__(self):
        super().__init__()

        self.in_channels = 64  # Tracks current channel count during layer building

        # ── Stem ──
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Residual Layers ──
        #                        mid_ch, num_blocks, stride
        self.layer1 = self._make_layer(64,   3, stride=1)  # 56×56×256
        self.layer2 = self._make_layer(128,  4, stride=2)  # 28×28×512
        self.layer3 = self._make_layer(256,  6, stride=2)  # 14×14×1024
        self.layer4 = self._make_layer(512,  3, stride=2)  # 7×7×2048

        # ── Initialize all weights ──
        self._init_weights()

    def _make_layer(self, mid_channels, num_blocks, stride=1):
        """
        Build a layer of Bottleneck blocks.

        First block may downsample (stride > 1 or channel mismatch).
        Remaining blocks keep the same spatial dimensions.
        """
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None

        # Need projection if channels change or spatial size changes
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # First block: may downsample
        layers.append(Bottleneck(self.in_channels, mid_channels, stride, downsample))
        self.in_channels = out_channels

        # Remaining blocks: no downsampling
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Kaiming Normal initialization for all layers.
        
        - Conv2d: Kaiming Normal (fan_out, ReLU mode)
        - BatchNorm2d: weight=1, bias=0
        - Linear: Kaiming Normal
        
        This gives stable gradients even at depth 50.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ═══════════════════════════════════════════════════════════
# Feature Extraction Branches
# ═══════════════════════════════════════════════════════════

class LocalBranch(nn.Module):
    """
    Extracts LOCAL features from ResNet Layer 3 output.

    Input:  14×14×1024 feature map (from ResNet Layer 3)
    Output: 128-dim vector

    Process:
      1. AdaptiveAvgPool to 4×4 grid (captures spatial detail at 4 positions)
      2. Flatten: 4×4×1024 = 16,384
      3. FC layer: 16,384 → 128

    What it captures: fine textures, stitching patterns, watermark edges
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 14×14 → 4×4 grid
        self.fc = nn.Sequential(
            nn.Flatten(),                    # 4×4×1024 → 16384
            nn.Linear(1024 * 4 * 4, LOCAL_DIM),  # 16384 → 128
            nn.ReLU(),
            nn.BatchNorm1d(LOCAL_DIM),       # Stabilize training
        )

    def forward(self, x):
        # x shape: (batch, 1024, 14, 14)
        x = self.pool(x)   # → (batch, 1024, 4, 4)
        x = self.fc(x)     # → (batch, 128)
        return x


class GlobalBranch(nn.Module):
    """
    Extracts GLOBAL features from ResNet Layer 4 output.

    Input:  7×7×2048 feature map (from ResNet Layer 4)
    Output: 256-dim vector

    Process:
      1. AdaptiveAvgPool to 1×1 (compresses all spatial info into one vector)
      2. Flatten: 1×1×2048 = 2048
      3. FC layer: 2048 → 256

    What it captures: overall shape, silhouette, brand identity
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 7×7 → 1×1
        self.fc = nn.Sequential(
            nn.Flatten(),                      # 1×1×2048 → 2048
            nn.Linear(2048, GLOBAL_DIM),       # 2048 → 256
            nn.ReLU(),
            nn.BatchNorm1d(GLOBAL_DIM),        # Stabilize training
        )

    def forward(self, x):
        # x shape: (batch, 2048, 7, 7)
        x = self.pool(x)   # → (batch, 2048, 1, 1)
        x = self.fc(x)     # → (batch, 256)
        return x


# ═══════════════════════════════════════════════════════════
# Siamese Network
# ═══════════════════════════════════════════════════════════

class SiameseResNet50(nn.Module):
    """
    Full Siamese Network with from-scratch ResNet-50.

    Two images go through the SAME backbone (shared weights).
    Each produces a 384-dim embedding (256 global + 128 local).

    Usage:
      model = SiameseResNet50()
      emb1, emb2, g1, g2, l1, l2 = model(img1_batch, img2_batch)
      # emb1, emb2 are both (batch_size, 384)
    """

    def __init__(self):
        super().__init__()

        # ── Build ResNet-50 from scratch ──
        backbone = ResNet50Backbone()

        # ── Split backbone for dual-branch extraction ──
        # Everything up to and including Layer 3
        self.backbone_to_layer3 = nn.Sequential(
            backbone.conv1,      # 7×7 conv, stride 2: 224→112, channels 3→64
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,    # MaxPool: 112→56
            backbone.layer1,     # 3 bottleneck blocks: 56×56×256
            backbone.layer2,     # 4 bottleneck blocks: 28×28×512
            backbone.layer3,     # 6 bottleneck blocks: 14×14×1024  ← LOCAL features
        )

        # Layer 4 only (takes Layer 3 output as input)
        self.layer4 = backbone.layer4  # 3 bottleneck blocks: 7×7×2048  ← GLOBAL features

        # ── Feature extraction branches ──
        self.local_branch  = LocalBranch()   # 14×14×1024 → 128-dim
        self.global_branch = GlobalBranch()  # 7×7×2048  → 256-dim
        self.known_head = nn.Linear(GLOBAL_DIM + LOCAL_DIM, 1)

    def extract_embedding(self, x):
        """
        Extract 384-dim embedding from a single image batch.

        Flow:
          x (224×224×3)
            → backbone_to_layer3 → layer3_out (14×14×1024)
            → layer4             → layer4_out (7×7×2048)

          layer3_out → local_branch  → 128-dim local
          layer4_out → global_branch → 256-dim global

          concatenate → 384-dim embedding
        """
        # Forward through backbone up to Layer 3
        layer3_out = self.backbone_to_layer3(x)  # (batch, 1024, 14, 14)

        # Continue to Layer 4
        layer4_out = self.layer4(layer3_out)      # (batch, 2048, 7, 7)

        # Extract features from both layers
        local_feat  = self.local_branch(layer3_out)   # (batch, 128)
        global_feat = self.global_branch(layer4_out)  # (batch, 256)

        # Concatenate: [global 256 | local 128] = 384-dim
        embedding = torch.cat([global_feat, local_feat], dim=1)  # (batch, 384)

        return embedding, global_feat, local_feat

    def forward(self, img1, img2):
        """
        Forward pass for a pair of images.

        Both images go through the SAME backbone (shared weights).
        This is what makes it "Siamese" — twin networks with identical weights.

        Returns:
          emb1, emb2: full 384-dim embeddings
          global1, global2: 256-dim global features (for global loss)
          local1, local2: 128-dim local features (for local loss)
        """
        emb1, global1, local1 = self.extract_embedding(img1)
        emb2, global2, local2 = self.extract_embedding(img2)

        return emb1, emb2, global1, global2, local1, local2


# ─── Quick Test ──────────────────────────────────────────
if __name__ == "__main__":
    model = SiameseResNet50().to(DEVICE)

    # Fake batch of 4 image pairs
    img1 = torch.randn(4, 3, 224, 224).to(DEVICE)
    img2 = torch.randn(4, 3, 224, 224).to(DEVICE)

    emb1, emb2, g1, g2, l1, l2 = model(img1, img2)

    print(f"Full embedding: {emb1.shape}")    # → (4, 384)
    print(f"Global branch:  {g1.shape}")      # → (4, 256)
    print(f"Local branch:   {l1.shape}")      # → (4, 128)
    print(f"✅ Model works! Total params: {sum(p.numel() for p in model.parameters()):,}")
