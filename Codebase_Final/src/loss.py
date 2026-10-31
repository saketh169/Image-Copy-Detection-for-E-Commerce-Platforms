"""Dual-head contrastive objective used by all canonical Siamese trainers."""

from __future__ import annotations

import torch
from torch import nn

from config import LOCAL_WEIGHT, MARGIN


class DualBranchContrastiveLoss(nn.Module):
    """Apply contrastive loss to global and local feature branches.

    Labels use ``1`` for copies and ``0`` for distinct images.  The returned
    distance is the concatenated descriptor distance for manifest diagnostics.
    """

    def __init__(self, margin: float = MARGIN, local_weight: float = LOCAL_WEIGHT):
        super().__init__()
        self.margin = float(margin)
        self.local_weight = float(local_weight)

    def _branch_loss(
        self, first: torch.Tensor, second: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distance = torch.linalg.vector_norm(first - second, ord=2, dim=1)
        positive = labels * distance.square()
        negative = (1.0 - labels) * torch.relu(self.margin - distance).square()
        return positive + negative, distance

    def forward(
        self,
        global_first: torch.Tensor,
        global_second: torch.Tensor,
        local_first: torch.Tensor,
        local_second: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels.float().reshape(-1)
        global_per_item, _ = self._branch_loss(global_first, global_second, labels)
        local_per_item, _ = self._branch_loss(local_first, local_second, labels)
        if sample_weights is None:
            weights = torch.ones_like(labels)
        else:
            weights = sample_weights.float().reshape(-1).clamp_min(0.0)
        normalizer = weights.sum().clamp_min(1e-12)
        global_loss = (global_per_item * weights).sum() / normalizer
        local_loss = (local_per_item * weights).sum() / normalizer
        total_loss = global_loss + self.local_weight * local_loss
        pair_distance = torch.linalg.vector_norm(
            torch.cat((global_first, local_first), dim=1)
            - torch.cat((global_second, local_second), dim=1),
            ord=2,
            dim=1,
        )
        return total_loss, global_loss, local_loss, pair_distance
