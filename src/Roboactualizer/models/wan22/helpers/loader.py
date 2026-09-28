"""Hash-checked loading of the Wan umT5 text encoder (DiffSynth-style)."""

import torch

from .io import ModelConfig, hash_model_file, load_state_dict
from ..wan_video_text_encoder import WanTextEncoder

WAN22_MODEL_REGISTRY = [
    {
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": WanTextEncoder,
    },
]


def _load_registered_model(path, model_name: str, torch_dtype: torch.dtype, device: str):
    model_hash = hash_model_file(path)
    matched = next(
        (c for c in WAN22_MODEL_REGISTRY if c["model_hash"] == model_hash and c["model_name"] == model_name), None
    )
    if matched is None:
        raise ValueError(f"Cannot detect model type for {model_name}. File: {path}. Hash: {model_hash}.")
    model = matched["model_class"]()
    model.load_state_dict(load_state_dict(path, torch_dtype=torch_dtype, device="cpu"), strict=False)
    return model.to(device=device, dtype=torch_dtype)


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    """-> (text encoder config, tokenizer config)."""
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")
    if redirect_common_files:
        text_config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        text_config.origin_file_pattern = "models_t5_umt5-xxl-enc-bf16.safetensors"
    return text_config, tokenizer_config
