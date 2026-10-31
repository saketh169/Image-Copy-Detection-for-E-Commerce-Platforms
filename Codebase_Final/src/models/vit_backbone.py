"""
model_vit.py — Siamese Vision Transformer (ViT) with Dual-Branch Feature Extraction
====================================================================================

Vision Transformer (ViT) replaces convolutions with pure transformer architecture.
Instead of sliding windows, it works with image patches and self-attention.

Architecture (patch-based transformer design):

  Input Image (224×224×3)
       │
       ▼
  Divide into patches: 16×16 patches → 14×14 = 196 patches
       │
       ├─→ Linear projection: each patch → 768-dim vector
       │   + positional encoding + class token
       │
       ├─→ 12 Transformer blocks (each with Multi-Head Self-Attention + MLP)
       │   at intermediate layer (after 6 blocks) ──→ LOCAL BRANCH (196, 768)
       │                                               (captures local patch interactions)
       │
       └─→ after 12 blocks → GLOBAL BRANCH (CLS token, 768-dim)
                              (captures global semantic information)
       │
       ▼
  Concatenate [256 global + 128 local aggregated] = 384-dim embedding

Key Differences from CNN:
  1. Input: Grid of patches, not sliding convolutions
  2. Attention: Self-attention over ALL patches simultaneously
  3. No spatial inductive bias (unlike convolutions) — learns from scratch
  4. Linear complexity with patch count, not image dimensions
  5. Better for large images and transfer learning
  6. Similar embedding output (384-dim) for compatibility

Mathematical Components:

  Patch embedding: 16×16×3 → Linear(768)
  Positional encoding: learned 1D position embeddings for each patch
  Transformer block:
    attn_out = MultiHeadSelfAttention(LayerNorm(x))
    ff_out = MLP(LayerNorm(attn_out))
    output = x + attn_out + ff_out  (residuals)

  Multi-Head Attention:
    Q, K, V = Linear projections of input
    scores = softmax((Q @ K^T) / √d_k)  @ V
    MultiHead = Concat[head1, ..., head8] @ W_o
"""

import torch
import torch.nn as nn
import math


# ═══════════════════════════════════════════════════════════
# Vision Transformer Building Blocks
# ═══════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Convert image into patch embeddings.

    Process:
      1. Divide image into non-overlapping 16×16 patches
      2. Flatten each patch: 16×16×3 = 768 values
      3. Project to embedding dimension: 768 → 768
      4. Add learnable positional encodings

    Mathematical:
      patch_embed = Linear(patch_pixels)  [converts 768 → 768]
      patch_with_pos = patch_embed + pos_encoding[patch_index]

    For 224×224 image:
      num_patches = (224 / 16)² = 14² = 196 patches
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Project patch pixels to embedding dimension
        # Each patch: patch_size × patch_size × in_channels
        patch_dim = patch_size * patch_size * in_channels  # 16×16×3 = 768
        self.linear = nn.Linear(patch_dim, embed_dim)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token (learnable special token, like BERT's [CLS])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize with small random values
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        
        Returns:
            embeddings: (batch, 197, 768)  [196 patches + 1 CLS token]
        """
        # x: (batch, 3, 224, 224)
        batch_size = x.shape[0]
        
        # Reshape into patches: (batch, 196, 16, 16, 3)
        x = x.reshape(
            batch_size,
            3,
            self.img_size // self.patch_size,
            self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size
        )
        
        # Reorganize: (batch, 196, 16*16*3)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(batch_size, self.num_patches, -1)  # (batch, 196, 768)
        
        # Project patches to embedding dimension
        x = self.linear(x)  # (batch, 196, 768)
        
        # Prepend class token to each sample
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 197, 768)
        
        # Add positional encoding
        x = x + self.pos_embed  # (batch, 197, 768)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (core of Transformer).

    Mathematical formula:
      Attention(Q, K, V) = softmax((Q @ K^T) / √d_k) @ V

    For multi-head:
      head_i = Attention(Q_i, K_i, V_i)
      Multi-Head = Concat[head_1, ..., head_h] @ W_o

    Args:
        dim: embedding dimension (768)
        num_heads: number of attention heads (12)
        head_dim: dimension per head (768 / 12 = 64)

    Computational:
      - Q, K, V projections: (batch, 197, 768) → (batch, 197, 768)
      - Attention scores: (batch, 12, 197, 197)  [one per head]
      - Softmax + matmul with V values
      - Concat and project: (batch, 197, 768)
    """
    def __init__(self, dim=768, num_heads=12, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / √d_k
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3)  # Project to Q, K, V simultaneously
        
        # Attention dropout (randomly set scores to -inf before softmax)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim) = (batch, 197, 768)
        
        Returns:
            out: (batch, seq_len, dim) = (batch, 197, 768)
            attn_weights: (batch, num_heads, seq_len, seq_len)  [for visualization]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V: (batch, 197, 768*3) → 3 of (batch, 197, 768)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores: Q @ K^T
        scores = (queries @ keys.transpose(-2, -1)) * self.scale  # (batch, heads, 197, 197)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, heads, 197, 197)
        attn_weights = self.attn_drop(attn_weights)
        
        # Apply to values
        out = (attn_weights @ values)  # (batch, heads, 197, head_dim)
        out = out.transpose(1, 2).contiguous()  # (batch, 197, heads, head_dim)
        out = out.reshape(batch_size, seq_len, self.dim)  # (batch, 197, 768)
        
        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, attn_weights


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block (Vision Transformer layer).

    Structure (Pre-normalization):
      x_norm = LayerNorm(x)
      attn_out = MultiHeadSelfAttention(x_norm) + x
      
      ffn_norm = LayerNorm(attn_out)
      ffn_out = MLP(ffn_norm) + attn_out
      
      output = ffn_out

    Why pre-norm? More stable training and better gradient flow.

    MLP (Feed-Forward Network):
      Linear(768 → 3072) → GELU → Dropout → Linear(3072 → 768)
      Expansion factor: 3072 / 768 = 4x (hidden_mult parameter)
    """
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # Attention
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # MLP: 768 → 3072 → 768
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        
        Returns:
            out: (batch, seq_len, dim)
            attn_weights: attention map from self-attention
        """
        # Pre-norm attention
        y, attn_weights = self.attn(self.norm1(x))
        x = x + y
        
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


# ═══════════════════════════════════════════════════════════
# Vision Transformer Backbone
# ═══════════════════════════════════════════════════════════

class ViTBackbone(nn.Module):
    """
    Full Vision Transformer backbone (ViT-B/16 equivalent).

    Configuration (ViT-Base):
      - Patch size: 16×16 (14×14 patches for 224×224 image)
      - Hidden dim: 768
      - Num heads: 12
      - Depth (num blocks): 12
      - MLP ratio: 4 (hidden_dim = 768 × 4 = 3072)

    Outputs:
      - intermediate_features (after 6 blocks): (batch, 196, 768) for local
      - final_features (after 12 blocks, CLS token): (batch, 768) for global

    Total parameters: ≈ 86M
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_heads=12, depth=12, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch embedding + positional encoding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # For extracting intermediate features at depth/2
        self.intermediate_depth = depth // 2

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        
        Returns:
            intermediate: (batch, 196, 768) — local features from 6th block
            final_cls: (batch, 768) — global CLS token after 12 blocks
        """
        # Patch embedding + pos encoding
        x = self.patch_embed(x)  # (batch, 197, 768)
        
        # Transformer blocks
        attn_weights_list = []
        for i, block in enumerate(self.blocks):
            x, attn_w = block(x)
            attn_weights_list.append(attn_w)
            
            # Capture intermediate features (after 6 blocks, before final 6)
            if i == self.intermediate_depth - 1:
                intermediate = x[:, 1:, :]  # (batch, 196, 768) [exclude CLS token]
        
        # Final norm
        x = self.norm(x)  # (batch, 197, 768)
        
        # Extract CLS token (first token, global representation)
        final_cls = x[:, 0, :]  # (batch, 768)
        
        return intermediate, final_cls


# ═══════════════════════════════════════════════════════════
# Feature Extraction Branches
# ═══════════════════════════════════════════════════════════

class ViTLocalBranch(nn.Module):
    """
    Extracts LOCAL features from intermediate ViT layer.

    Input:  (batch, 196, 768) — all patch tokens except CLS
    Output: (batch, 128)

    Process:
      1. Spatial pooling: aggregate 196 patches to 4×4 grid
      2. Reshape back to 2D: (batch, 768, 4, 4)
      3. Project: 768 → 128

    This mimics the "local" receptive field of CNNs.
    """
    def __init__(self):
        super().__init__()
        
        # Reshape patches back to grid: 14×14 → 4×4
        self.pool = nn.AvgPool2d(kernel_size=(14//4), stride=(14//4))
        
        # Project from flattened pooled patches to local embedding
        # After pooling: (batch, 768, 4, 4) → flatten → (batch, 768*16) = (batch, 12288)
        self.fc = nn.Sequential(
            nn.Linear(768*16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 196, 768) — patch embeddings
        
        Returns:
            out: (batch, 128)
        """
        batch_size = x.shape[0]
        
        # Reshape patches to spatial grid: (batch, 196, 768) → (batch, 768, 14, 14)
        x = x.reshape(batch_size, 14, 14, 768).permute(0, 3, 1, 2)
        
        # Pool to 4×4: (batch, 768, 4, 4)
        x = self.pool(x)
        
        # Flatten and project: (batch, 768*4*4) → (batch, 128)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        
        return x


class ViTGlobalBranch(nn.Module):
    """
    Extracts GLOBAL features from ViT CLS token.

    Input:  (batch, 768) — CLS token after transformer
    Output: (batch, 256)

    The CLS token serves similar role as global average pooling in CNNs:
    it aggregates information from entire image through self-attention.

    Process:
      1. Project CLS: 768 → 256

    This is already global, so minimal processing needed.
    """
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 768) — CLS token
        
        Returns:
            out: (batch, 256)
        """
        return self.fc(x)


# ═══════════════════════════════════════════════════════════
# Siamese Vision Transformer
# ═══════════════════════════════════════════════════════════

class SiameseViT(nn.Module):
    """
    Full Siamese Vision Transformer network.

    Maintains same 384-dim output as ResNet for loss function compatibility.
    Two images through SAME backbone (shared weights).

    Output: (emb1, emb2, global1, global2, local1, local2)
            384-dim embeddings using dual-branch architecture
    """

    def __init__(self):
        super().__init__()

        # ViT backbone
        self.backbone = ViTBackbone(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            num_heads=12,
            depth=12,
            mlp_ratio=4.0
        )
        
        # Feature branches
        self.local_branch = ViTLocalBranch()
        self.global_branch = ViTGlobalBranch()

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
        intermediate1, cls1 = self.backbone(img1)
        local1 = self.local_branch(intermediate1)    # (batch, 128)
        global1 = self.global_branch(cls1)           # (batch, 256)
        emb1 = torch.cat([global1, local1], dim=1)   # (batch, 384)

        # Image 2 through same backbone (shared weights)
        intermediate2, cls2 = self.backbone(img2)
        local2 = self.local_branch(intermediate2)    # (batch, 128)
        global2 = self.global_branch(cls2)           # (batch, 256)
        emb2 = torch.cat([global2, local2], dim=1)   # (batch, 384)

        return emb1, emb2, global1, global2, local1, local2
