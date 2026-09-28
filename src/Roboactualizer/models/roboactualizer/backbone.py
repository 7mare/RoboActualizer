"""Frozen V-JEPA 2.1 visual backbone."""

import torch
from torch import nn

from .vjepa import build_vjepa_encoder, parse_model_size


class Backbone(nn.Module):
    """Minimal interface the policy relies on."""

    embed_dim: int
    num_patches: int
    patch_grid: tuple

    def encode_clip_last_group(self, clip):
        raise NotImplementedError


class VJEPABackbone(Backbone):
    """Frozen V-JEPA 2.1 encoder; multi-camera images arrive pre-tiled into one frame."""

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        ckpt_path: str,
        *,
        model_name: str = "vjepa2_1_vit_large_384",
        model_size: str | None = None,
        torch_hub_dir: str | None = None,
        image_size=(384, 320),
        patch_size: int = 16,
        num_cameras: int = 2,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_size = model_size or parse_model_size(model_name)
        self.image_size = tuple(int(s) for s in image_size)
        self.patch_size = int(patch_size)
        self.patch_grid = (self.image_size[0] // self.patch_size, self.image_size[1] // self.patch_size)
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]
        self.num_cameras = int(num_cameras)
        self._dtype = dtype

        self.encoder = build_vjepa_encoder(
            ckpt_path=ckpt_path,
            model_size=self.model_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            torch_hub_dir=torch_hub_dir,
        )
        self.embed_dim = int(self.encoder.embed_dim)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.to(device=device, dtype=dtype)
        self.register_buffer("_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1).to(device))
        self.register_buffer("_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1).to(device))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode_clip_last_group(self, clip: torch.Tensor) -> torch.Tensor:
        """Video-branch encode of ``[B, T, C, H, W]`` in [-1, 1]; returns the last tubelet group [B, N, D]."""
        b, t, c, hi, wi = clip.shape
        if t < 2 or t % 2 != 0:
            raise ValueError(f"clip length must be even and >= 2 (tubelet=2), got {t}")
        x = clip.reshape(b * t, c, hi, wi).to(dtype=self._dtype)
        x = (x + 1.0) * 0.5
        x = (x - self._mean) / self._std
        x = x.reshape(b, t, c, hi, wi).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        z = self.encoder(x)  # [B, (T/2)*N, D]
        return z[:, -self.num_patches:, :]
