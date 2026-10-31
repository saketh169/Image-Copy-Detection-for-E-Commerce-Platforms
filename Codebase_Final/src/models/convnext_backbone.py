"""
model_convnext.py — Siamese ConvNeXt-B with Dual-Branch Feature Extraction
============================================================================

ConvNeXt (Convolution-next) is a modernized CNN architecture that improves upon
ResNet by incorporating design insights from Vision Transformers while maintaining
the efficiency of pure CNNs.

Architecture (modernized convolution design):

  Input Image (224×224×3)
       │
       ▼
  Stem: 4×4 Conv (stride 4) → (56×56×96) feature map
       │
       ├─── Stage 1: 3 ConvNeXt blocks  (56×56×96)
       │
       ├─── Downsample + Stage 2: 3 ConvNeXt blocks  (28×28×192)
       │
       ├─── Downsample + Stage 3: 9 ConvNeXt blocks  (14×14×384) ← LOCAL BRANCH
       │                                              (captures fine details)
       │
       └─── Downsample + Stage 4: 3 ConvNeXt blocks  (7×7×768)  ← GLOBAL BRANCH
            (captures high-level semantics)
       │
       ▼
  Concatenate [256 global + 128 local] = 384-dim embedding

Key Differences from ResNet-50:
  1. ConvNeXt blocks instead of Bottleneck blocks
  2. Depthwise separable convolutions (spatial at fixed channels, then 1×1)
  3. No explicit bottleneck squeeze (modern CNNs don't squeeze as much)
  4. GELU activation instead of ReLU
  5. Layer-wise learning rate scaling (LR × (depth_ratio ^ layer_index))
  6. Output: Same 384-dim dual-branch as ResNet for compatibility

Mathematical Components:

  ConvNeXt Block:
    x → DwConv(7×7) → norm → Linear(up) → GELU → Linear(down) → + skip

  Where:
    - DwConv: depthwise convolution (1 kernel per channel)
    - norm: LayerNorm (not BatchNorm)
    - Linear layers implement inverted bottleneck (C → 4C → C)
    - GELU: smooth activation better than ReLU
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
# ConvNeXt-B Backbone
# ═══════════════════════════════════════════════════════════

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt building block (modern CNN design).

    Structure:
      Input (C channels)
        │
        ├─→ DwConv(7×7) — depthwise convolution (groups=C)
        │   → LayerNorm
        ├─→ 1×1 Conv (Linear up: C → 4C)
        │   → GELU activation
        ├─→ 1×1 Conv (Linear down: 4C → C)
        │   → Stochastic Depth (optional, for regularization)
        │
        + Skip connection (identity)
        │
        Output (C channels)

    Key: Uses LayerNorm (per-channel) instead of BatchNorm (per-batch).
         This is more stable for modern CNN training.
    """
    def __init__(self, dim, drop_path=0.0):
        """
        Args:
            dim: number of input/output channels
            drop_path: stochastic depth rate (0.0 = no dropping)
        """
        super().__init__()
        
        # Depthwise convolution: 7×7 kernels, one per input channel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # Layer normalization (per-channel, not per-batch like BatchNorm)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # MLP block (inverted bottleneck: dim → 4×dim → dim)
        # This is where most parameters are
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, kernel_size=1),  # Linear expansion
            nn.GELU(),  # Smooth activation (better than ReLU for modern CNNs)
            nn.Conv2d(4 * dim, dim, kernel_size=1),  # Linear projection back
        )
        
        # Stochastic depth for regularization (higher layers drop more)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x shape: (batch, dim, height, width)
        shortcut = x
        
        # Depthwise convolution with layer norm
        x = self.dwconv(x)
        # LayerNorm expects (batch, channels, height, width) 
        x = x.permute(0, 2, 3, 1)  # → (batch, height, width, channels)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # → (batch, channels, height, width)
        
        # MLP block (inverted bottleneck)
        x = self.mlp(x)
        
        # Skip connection with stochastic depth
        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth: randomly drop entire residual connections during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * random_tensor / keep_prob if keep_prob > 0 else x


class ConvNeXtDownsample(nn.Module):
    """Downsampling layer between ConvNeXt stages (2x spatial reduction)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # x: (batch, channels, height, width)
        # Permute for LayerNorm
        x = x.permute(0, 2, 3, 1)  # → (batch, height, width, channels)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # → (batch, channels, height, width)
        # Downsample with conv
        x = self.conv(x)
        return x



class ConvNeXtBackbone(nn.Module):
    """
    Full ConvNeXt-B backbone built from scratch.

    Stages breakdown:
      Stem:    4×4 conv, stride 4  →  224→56, channels 3→96
      Stage 1: 3 ConvNeXt blocks   →  56×56×96
      Stage 2: 3 ConvNeXt blocks   →  28×28×192  (downsampled by 2)
      Stage 3: 9 ConvNeXt blocks   →  14×14×384  (downsampled by 2)  ← LOCAL
      Stage 4: 3 ConvNeXt blocks   →  7×7×768    (downsampled by 2)  ← GLOBAL

    The 'B' in ConvNeXt-B refers to this specific block configuration.
    """

    def __init__(self, drop_path_rate=0.0):
        super().__init__()

        # Stage depths and dimensions for ConvNeXt-B
        depths = [3, 3, 9, 3]  # Blocks per stage
        dims = [96, 192, 384, 768]  # Channels per stage

        # ── Stem (initial feature extraction) ──
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),  # 224→56
            # Note: Output is (batch, 96, 56, 56)
            # LayerNorm will be applied in ConvNeXtBlock fashion
        )

        # ── Stages ──
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j]))
            self.stages.append(nn.Sequential(*stage_blocks))
            
            # Downsampling between stages (except after last)
            if i < 3:
                self.stages.append(ConvNeXtDownsample(dims[i], dims[i+1]))
            cur += depths[i]

        # Stage outputs for extraction
        self.stage1_dim = dims[0]   # 96
        self.stage2_dim = dims[1]   # 192
        self.stage3_dim = dims[2]   # 384 ← LOCAL
        self.stage4_dim = dims[3]   # 768 ← GLOBAL

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for ConvNeXt."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Returns:
            stage3_feat: (batch, 384, 14, 14) — local features
            stage4_feat: (batch, 768, 7, 7)   — global features
        """
        # Stem
        x = self.stem(x)  # (batch, 96, 56, 56)
        
        # Stage 1
        x = self.stages[0](x)  # (batch, 96, 56, 56)
        
        # Downsample + Stage 2
        x = self.stages[1](x)  # downsample → (batch, 192, 28, 28)
        x = self.stages[2](x)  # (batch, 192, 28, 28)
        
        # Downsample + Stage 3 (LOCAL)
        x = self.stages[3](x)  # downsample → (batch, 384, 14, 14)
        stage3_feat = x  # Save for local branch
        x = self.stages[4](x)  # (batch, 384, 14, 14)
        
        # Downsample + Stage 4 (GLOBAL)
        x = self.stages[5](x)  # downsample → (batch, 768, 7, 7)
        stage4_feat = x  # Save for global branch
        x = self.stages[6](x)  # (batch, 768, 7, 7)
        
        return stage3_feat, x  # (local, global) feature maps


# ═══════════════════════════════════════════════════════════
# Feature Extraction Branches (same as ResNet but for ConvNeXt dims)
# ═══════════════════════════════════════════════════════════

class ConvNeXtLocalBranch(nn.Module):
    """
    Extracts LOCAL features from ConvNeXt Stage 3 output.

    Input:  14×14×384 feature map
    Output: 128-dim vector

    Process matches ResNet local branch for consistency.
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384 * 4 * 4, 128),  # 6144 → 128
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        # x: (batch, 384, 14, 14)
        x = self.pool(x)   # → (batch, 384, 4, 4)
        x = self.fc(x)     # → (batch, 128)
        return x


class ConvNeXtGlobalBranch(nn.Module):
    """
    Extracts GLOBAL features from ConvNeXt Stage 4 output.

    Input:  7×7×768 feature map
    Output: 256-dim vector

    Adapted for ConvNeXt-B's higher dimensional features.
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),  # 768 → 256
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

    def forward(self, x):
        # x: (batch, 768, 7, 7)
        x = self.pool(x)   # → (batch, 768, 1, 1)
        x = self.fc(x)     # → (batch, 256)
        return x


# ═══════════════════════════════════════════════════════════
# Siamese ConvNeXt Network
# ═══════════════════════════════════════════════════════════

class SiameseConvNeXtB(nn.Module):
    """
    Full Siamese Network with ConvNeXt-B backbone.

    Maintains same 384-dim output as ResNet for loss function compatibility.
    Two images through SAME backbone (shared weights).

    Output: (emb1, emb2, global1, global2, local1, local2)
            all with same dimensions as SiameseResNet50
    """

    def __init__(self):
        super().__init__()

        # Build ConvNeXt-B backbone
        backbone = ConvNeXtBackbone()
        
        # Store stage outputs
        self.backbone = backbone
        
        # Branch heads
        self.local_branch = ConvNeXtLocalBranch()
        self.global_branch = ConvNeXtGlobalBranch()

    def forward(self, img1, img2):
        """
        Args:
            img1: (batch, 3, 224, 224)
            img2: (batch, 3, 224, 224)

        Returns:
            emb1, emb2: (batch, 384) — concatenated embeddings
            global1, global2: (batch, 256) — global branch
            local1, local2: (batch, 128) — local branch
        """
        # Image 1 through shared backbone
        local_feat1, global_feat1 = self.backbone(img1)
        local1 = self.local_branch(local_feat1)    # (batch, 128)
        global1 = self.global_branch(global_feat1)  # (batch, 256)
        emb1 = torch.cat([global1, local1], dim=1)  # (batch, 384)

        # Image 2 through same backbone (shared weights)
        local_feat2, global_feat2 = self.backbone(img2)
        local2 = self.local_branch(local_feat2)    # (batch, 128)
        global2 = self.global_branch(global_feat2)  # (batch, 256)
        emb2 = torch.cat([global2, local2], dim=1)  # (batch, 384)

        return emb1, emb2, global1, global2, local1, local2
