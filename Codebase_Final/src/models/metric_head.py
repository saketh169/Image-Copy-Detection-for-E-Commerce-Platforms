from __future__ import annotations

import torch
import torch.nn as nn

from models.model_convnext import SiameseConvNeXtB


class SiameseConvNeXtBMetric(nn.Module):
    """ConvNeXt Siamese model with a trainable diagonal metric layer.

    The final embedding returned by this model is:
        raw_embedding * metric_scale

    Plain L2 distance on this final embedding is equivalent to weighted L2
    with diagonal weights metric_scale**2. This lets us store/query exactly like
    a normal checkpoint while keeping the diagonal search result inside the
    model.
    """

    def __init__(self, init_scale=None, trainable_metric=True):
        super().__init__()
        self.base = SiameseConvNeXtB()
        if init_scale is None:
            init = torch.ones(384, dtype=torch.float32)
        else:
            init = torch.as_tensor(init_scale, dtype=torch.float32).flatten()
            if init.numel() != 384:
                raise ValueError(f"metric scale must have 384 dims, got {init.numel()}")
        self.metric_scale = nn.Parameter(init.clone(), requires_grad=trainable_metric)

    def load_base_state_dict(self, state_dict, strict=True):
        return self.base.load_state_dict(state_dict, strict=strict)

    def final_embedding(self, emb):
        return emb * self.metric_scale.to(device=emb.device, dtype=emb.dtype)

    def forward(self, x1, x2):
        out = self.base(x1, x2)
        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError("Unexpected SiameseConvNeXtB forward output")
        out = list(out)
        out[0] = self.final_embedding(out[0])
        out[1] = self.final_embedding(out[1])
        return tuple(out)