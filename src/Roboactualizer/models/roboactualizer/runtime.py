"""Factories: frozen V-JEPA backbone, Wan text encoder, and the Roboactualizer policy."""

import os

import torch
from omegaconf import DictConfig, OmegaConf

from .backbone import VJEPABackbone
from .policy import FlowPolicy
from .vjepa import official_ckpt_url


def _ensure_vjepa_repo() -> str:
    """Cache the V-JEPA2 source snapshot in torch hub (code only)."""
    hub_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_vjepa2_main")
    if not os.path.isdir(hub_dir):
        torch.hub.list("facebookresearch/vjepa2", trust_repo=True)
    return hub_dir


def load_vjepa2_1_backbone(
    *,
    ckpt_path: str,
    model_name: str = "vjepa2_1_vit_large_384",
    model_size: str | None = None,
    cache_dir: str | None = None,
    torch_hub_dir: str | None = None,
    download_if_missing: bool = False,
    image_size=(384, 320),
    patch_size: int = 16,
    num_cameras: int = 2,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    hub_dir = torch_hub_dir or _ensure_vjepa_repo()
    if not os.path.exists(ckpt_path):
        if not download_if_missing:
            raise FileNotFoundError(
                f"V-JEPA2.1 checkpoint not found: {ckpt_path}. Set backbone.download_if_missing=true to fetch it."
            )
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        torch.hub.download_url_to_file(official_ckpt_url(model_name), ckpt_path)

    backbone = VJEPABackbone(
        ckpt_path=ckpt_path,
        model_name=model_name,
        model_size=model_size,
        torch_hub_dir=hub_dir,
        image_size=image_size,
        patch_size=patch_size,
        num_cameras=num_cameras,
        device=device,
        dtype=dtype,
    )
    metadata = {
        "model_name": model_name,
        "model_size": backbone.model_size,
        "ckpt_path": os.path.abspath(ckpt_path),
        "cache_dir": cache_dir,
        "patch_grid": list(backbone.patch_grid),
        "num_patches": backbone.num_patches,
        "num_cameras": backbone.num_cameras,
        "embed_dim": backbone.embed_dim,
        "patch_size": patch_size,
        "image_size": list(image_size),
    }
    return backbone, metadata


def load_text_encoder_components(
    *,
    model_id: str,
    tokenizer_model_id: str,
    context_len: int,
    redirect_common_files: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Wan umT5 text encoder + tokenizer for online prompt encoding."""
    from Roboactualizer.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from Roboactualizer.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    text_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=dtype, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=context_len, clean="whitespace")
    return text_encoder, tokenizer


# DiT presets; depth 12 and head dim 64 for both.
_DIT_PRESETS = {
    "s": {"d_model": 384, "num_heads": 6, "mlp_dim": 1536},
    "b": {"d_model": 768, "num_heads": 12, "mlp_dim": 3072},
}


def create_joint_predict_fm(
    *,
    backbone,
    chunk_size: int,
    future_size: int,
    video_freq_ratio: int,
    action_dim: int,
    raw_action_dim: int,
    text_dim: int,
    proprio_dim: int,
    dit_size: str = "s",
    num_layers: int = 12,
    dim_head: int = 64,
    dropout: float = 0.0,
    freq_dim: int = 256,
    num_train_timesteps: int = 1000,
    shift: float = 5.0,
    num_inference_steps: int = 10,
    action_weight: float = 1.0,
    image_predict_weight: float = 1.0,
    rgb_latent_dtype: str = "float16",
    latent_norm_stats_path: str | None = None,
    eval_encode_fp32: bool = False,
    num_cameras: int = 2,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    **extra_cfg,
):
    """Hydra factory. ``extra_cfg`` holds deploy-only keys and is recorded in metadata."""
    dims = _DIT_PRESETS[dit_size]
    if isinstance(backbone, DictConfig):
        backbone = OmegaConf.to_container(backbone, resolve=True)
    backbone_cfg = dict(backbone)
    backbone_cfg.pop("_target_", None)
    vbackbone, backbone_meta = load_vjepa2_1_backbone(
        device=device, dtype=torch.float32, num_cameras=num_cameras, **backbone_cfg
    )
    if action_dim != raw_action_dim:
        raise ValueError(f"action_dim ({action_dim}) must equal raw_action_dim ({raw_action_dim}).")

    policy = FlowPolicy(
        vbackbone,
        chunk_size=chunk_size,
        future_size=future_size,
        video_freq_ratio=video_freq_ratio,
        action_dim=action_dim,
        text_dim=text_dim,
        proprio_dim=proprio_dim,
        d_model=dims["d_model"],
        num_layers=num_layers,
        num_heads=dims["num_heads"],
        dim_head=dim_head,
        mlp_dim=dims["mlp_dim"],
        dropout=dropout,
        freq_dim=freq_dim,
        num_train_timesteps=num_train_timesteps,
        shift=shift,
        num_inference_steps=num_inference_steps,
        action_weight=action_weight,
        image_predict_weight=image_predict_weight,
        rgb_latent_dtype=rgb_latent_dtype,
        latent_norm_stats_path=latent_norm_stats_path,
        eval_encode_fp32=eval_encode_fp32,
    )
    # Trainable modules in model_dtype; the frozen backbone stays fp32.
    for name in ("image_context_builder", "action_context_builder", "image_token_encoder",
                 "action_token_encoder", "image_time_embedder", "action_time_embedder",
                 "joint_expert", "image_decoder", "action_decoder"):
        getattr(policy, name).to(device=device, dtype=model_dtype)
    policy.to(device=device)  # non-persistent buffers (RoPE freqs) keep their dtype

    policy.metadata = {
        "backbone": backbone_meta,
        "raw_action_dim": raw_action_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "future_size": future_size,
        "video_freq_ratio": video_freq_ratio,
        "num_cameras": vbackbone.num_cameras,
        "patch_grid": list(vbackbone.patch_grid),
        "num_patches": vbackbone.num_patches,
        "dit_size": dit_size,
        "rgb_latent_dtype": rgb_latent_dtype,
        "latent_norm_stats": policy.latent_norm_stats_meta,
        "shift": shift,
        **extra_cfg,
    }
    policy.device = device
    policy.torch_dtype = model_dtype
    return policy
