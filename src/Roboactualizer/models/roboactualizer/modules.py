"""Basic layers."""

import torch
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """RMSNorm, used as q/k norm."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class FeedForward(nn.Module):
    """Linear -> GELU -> Linear; the block applies the pre-FFN norm."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)
