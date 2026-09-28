"""V-JEPA 2.1 encoder construction from the cached official source (github.com/facebookresearch/vjepa2)."""

import os
import sys

import torch
from torch import nn

# Size -> official factory. giant/gigantic must use the xformers variants (different head counts).
_VIT_FACTORY = {
    "base": "vit_base",
    "large": "vit_large",
    "giant": "vit_giant_xformers",
    "gigantic": "vit_gigantic_xformers",
}

_MODEL_SIZES = {
    "vjepa2_1_vit_base_384": "base",
    "vjepa2_1_vit_large_384": "large",
    "vjepa2_1_vit_giant_384": "giant",
    "vjepa2_1_vit_gigantic_384": "gigantic",
}

# Size -> official weight file stem.
_CKPT_STEM = {
    "base": "vjepa2_1_vitb_dist_vitG_384",
    "large": "vjepa2_1_vitl_dist_vitG_384",
    "giant": "vjepa2_1_vitg_384",
    "gigantic": "vjepa2_1_vitG_384",
}

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"

# Encoder state_dict key differs across official releases.
_CKPT_STATE_KEYS = ("ema_encoder", "target_encoder", "encoder")


def resolve_hub_dir(torch_hub_dir: str | None = None) -> str:
    hub_dir = torch_hub_dir or os.path.join(torch.hub.get_dir(), "facebookresearch_vjepa2_main")
    if not os.path.isdir(hub_dir):
        raise FileNotFoundError(
            f"V-JEPA2 repo snapshot not found at {hub_dir}. "
            "Run `torch.hub.list('facebookresearch/vjepa2')` once to cache it."
        )
    return hub_dir


def parse_model_size(model_name: str) -> str:
    if model_name in _MODEL_SIZES:
        return _MODEL_SIZES[model_name]
    for size in sorted(_VIT_FACTORY, key=len, reverse=True):  # "gigantic" before "giant"
        if f"vit_{size}" in model_name or f"_{size}_" in model_name or model_name.endswith(f"_{size}"):
            return size
    raise ValueError(f"Cannot infer V-JEPA size from '{model_name}'; pass model_size.")


def official_ckpt_filename(model_name: str) -> str:
    return f"{_CKPT_STEM[parse_model_size(model_name)]}.pt"


def official_ckpt_url(model_name: str) -> str:
    return f"{VJEPA_BASE_URL}/{official_ckpt_filename(model_name)}"


def build_vjepa_encoder(
    *,
    ckpt_path: str,
    model_size: str,
    image_size: tuple[int, int],
    patch_size: int = 16,
    torch_hub_dir: str | None = None,
) -> nn.Module:
    """Build the encoder for ``model_size`` and load local weights strictly."""
    hub_dir = resolve_hub_dir(torch_hub_dir)
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)
    from app.vjepa_2_1.models import vision_transformer as vit  # noqa: E402

    vit_factory = getattr(vit, _VIT_FACTORY[model_size])
    img = (int(image_size[0]), int(image_size[0]))  # square RoPE init; forward accepts non-square
    encoder = vit_factory(
        patch_size=int(patch_size),
        img_size=img,
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    picked = next((k for k in _CKPT_STATE_KEYS if k in ckpt), None)
    if picked is None:
        raise KeyError(f"No encoder state_dict in {ckpt_path}; keys: {sorted(ckpt)}")
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in ckpt[picked].items()}
    encoder.load_state_dict(sd, strict=True)
    return encoder
