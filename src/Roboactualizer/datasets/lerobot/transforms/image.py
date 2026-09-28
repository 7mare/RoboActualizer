import torch
import torch.nn as nn


class ToTensor(nn.Module):
    """uint8 image -> float32 in [0, 1]."""

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        return x.to(torch.float32) / 255.0
