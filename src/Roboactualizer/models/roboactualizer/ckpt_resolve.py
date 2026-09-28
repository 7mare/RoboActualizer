"""Checks between an eval config and a checkpoint's metadata, shared by the sim entry points."""

import hashlib
import os
from pathlib import Path

import torch
from omegaconf import DictConfig


def check_ckpt_matches_cfg(ckpt: str, cfg: DictConfig) -> None:
    """Fail if the config's architecture differs from what the checkpoint was trained with."""
    meta = torch.load(ckpt, map_location="cpu", weights_only=False).get("metadata", {})
    m = cfg.model
    expected = {
        "dit_size": (meta.get("dit_size", "s"), m.dit_size),
        "chunk_size": (meta.get("chunk_size"), m.chunk_size),
        "future_size": (meta.get("future_size", m.future_size), m.future_size),
        "video_freq_ratio": (meta.get("video_freq_ratio", m.video_freq_ratio), m.video_freq_ratio),
        "clip_len": (meta.get("clip_len"), m.clip_len),
        "frame_interval": (meta.get("frame_interval"), m.frame_interval),
        "backbone": (meta.get("backbone", {}).get("model_name"), m.backbone.model_name),
        "image_size": (list(meta.get("backbone", {}).get("image_size", [])), list(m.backbone.image_size)),
    }
    bad = {k: v for k, v in expected.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"Config does not match checkpoint {ckpt} (ckpt, config): {bad}")


def resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    """EVALUATION.dataset_stats_path, else dataset_stats.json next to the ckpt; checked against its hash."""
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    ckpt = Path(os.path.expanduser(str(cfg.ckpt)))
    candidates = ([Path(os.path.expanduser(str(explicit)))] if explicit else []) + [
        p / "dataset_stats.json" for p in list(ckpt.parents)[:4]
    ]
    path = next((p.resolve() for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("dataset_stats.json not found; pass EVALUATION.dataset_stats_path.")
    want = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("metadata", {}).get("dataset_stats_sha256")
    if want and hashlib.sha256(path.read_bytes()).hexdigest() != want:
        raise ValueError(f"{path} is not the dataset_stats.json this checkpoint was trained with.")
    return path
