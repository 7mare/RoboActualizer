import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .models.roboactualizer.runtime import create_joint_predict_fm  # noqa: F401 (hydra _target_)
from .utils.logging_config import get_logger, setup_logging
from .utils import misc

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        _normalize_mixed_precision(mixed_precision)
    ]


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    # fp32_master_weights: keep trainable params in fp32 regardless of mixed_precision.
    fp32_master_weights = bool(cfg.get("fp32_master_weights", False))
    if fp32_master_weights:
        model_dtype = torch.float32
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    model.metadata["fp32_master_weights"] = fp32_master_weights
    train_ds, val_ds = build_datasets(cfg.data)

    from .trainer_jointfm import JointPredictFMTrainer
    JointPredictFMTrainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds).train()
