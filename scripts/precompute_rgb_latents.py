"""Precompute V-JEPA clip latents (the rgb_latent_cache_dir used in training).

For every frame f, the clip [f-(clip_len-1)*interval, ..., f] (clamped at 0) goes through the video
branch; the last tubelet group becomes latents[f], aligned with parquet row f. Pixels follow the
training pipeline (processor transforms, camera concat/mosaic, resize/crop/normalize).

    python scripts/precompute_rgb_latents.py task=libero_dits model.latent_norm_stats_path=null \
        +rgb_latent_cache_dir=./data/int2_clip4/libero
    python scripts/precompute_rgb_latents.py task=robotwin_dits model.latent_norm_stats_path=null \
        +rgb_latent_cache_dir=./data/int4_clip4/robotwin

Writes <cache_dir>/<suite>/episode_XXXXXX.pt and latents_meta.json per suite.
"""

import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from Roboactualizer.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from Roboactualizer.datasets.robotwin_image import concat_cameras_robotwin
from Roboactualizer.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from Roboactualizer.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata
from Roboactualizer.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _concat_cameras(pixels: torch.Tensor, mode: str) -> torch.Tensor:
    """[num_cam, T, C, H, W] -> [T, C, H', W'], same rule as training."""
    num_cam = pixels.shape[0]
    if mode == "robotwin":
        return concat_cameras_robotwin(pixels)
    if num_cam == 1:
        return pixels.squeeze(0)
    if mode == "horizontal":
        return torch.cat([pixels[i] for i in range(num_cam)], dim=-1)
    if mode == "vertical":
        return torch.cat([pixels[i] for i in range(num_cam)], dim=-2)
    raise ValueError(f"Unsupported concat_multi_camera: {mode}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg = cfg.data.train

    cache_root = cfg.get("rgb_latent_cache_dir")
    if not cache_root:
        raise ValueError("Please pass `rgb_latent_cache_dir=...` (output directory).")
    cache_root = Path(str(cache_root))
    enc_batch = int(cfg.get("encode_batch_size", 32))
    limit_eps = int(cfg.get("max_episodes_per_suite", 0))  # >0: first N episodes only (debug)

    frame_interval = int(cfg.model.get("frame_interval", 2))
    clip_len = int(cfg.model.get("clip_len", 4))
    latent_dtype = str(cfg.model.get("rgb_latent_dtype", "float16"))
    latent_dtypes = {"float16": torch.float16, "float32": torch.float32}
    save_dtype = latent_dtypes[latent_dtype]
    if frame_interval < 1:
        raise ValueError(f"frame_interval must be >= 1, got {frame_interval}")
    if clip_len < 2 or clip_len % 2 != 0:
        raise ValueError(f"clip_len must be an even int >= 2 (V-JEPA tubelet=2), got {clip_len}")

    # Only the frozen fp32 backbone is used.
    model = instantiate(cfg.model, model_dtype=torch.float32, device=device)
    backbone = model.backbone.eval()
    num_patches, embed_dim = int(backbone.num_patches), int(backbone.embed_dim)
    backbone_name = str(getattr(backbone, "model_name", "vjepa2_1_vit_large_384"))
    backbone_size = str(getattr(backbone, "model_size", "large"))

    processor = instantiate(data_cfg.processor).train()
    img_transforms = processor.train_transforms
    image_meta = OmegaConf.to_container(data_cfg.shape_meta.images, resolve=True)
    cam_keys = [m["key"] for m in image_meta]
    concat_mode = str(data_cfg.concat_multi_camera)
    vh, vw = int(data_cfg.video_size[0]), int(data_cfg.video_size[1])
    resize_t = ResizeSmallestSideAspectPreserving(args={"img_w": vw, "img_h": vh})
    crop_t = CenterCrop(args={"img_w": vw, "img_h": vh})
    norm_t = Normalize(args={"mean": 0.5, "std": 0.5})

    # Optional episode subset; the same file as data.train.episode_ids_file.
    episode_ids = None
    ids_file = data_cfg.get("episode_ids_file", None)
    if ids_file:
        with open(Path(str(ids_file)).expanduser()) as f:
            episode_ids = sorted({int(i) for i in json.load(f)["episode_ids"]})
        logger.info("episode subset %s -> %d", ids_file, len(episode_ids))

    logger.info(
        "RGB clip-latent precompute | backbone=%s(%s) patches=%d dim=%d image_size=%s cams=%s concat=%s "
        "frame_interval=%d clip_len=%d out=%s dtype=%s",
        backbone_name, backbone_size, num_patches, embed_dim, [vh, vw], cam_keys, concat_mode,
        frame_interval, clip_len, cache_root, save_dtype,
    )

    for ds_dir in data_cfg.dataset_dirs:
        ds_dir = str(ds_dir)
        meta = LeRobotDatasetMetadata(repo_id=ds_dir, root=Path(ds_dir))
        suite = Path(ds_dir).name
        fps = meta.fps
        tol = 1.0 / fps
        out_dir = cache_root / suite
        out_dir.mkdir(parents=True, exist_ok=True)

        if episode_ids is not None:
            bad = [i for i in episode_ids if not 0 <= i < meta.total_episodes]
            if bad:
                raise ValueError(f"[{suite}] episode ids out of range: {bad[:5]} (total {meta.total_episodes})")
            ep_list = list(episode_ids)
            logger.info("[%s] encoding %d / %d episodes", suite, len(ep_list), meta.total_episodes)
        else:
            n = meta.total_episodes if limit_eps <= 0 else min(limit_eps, meta.total_episodes)
            ep_list = list(range(n))
        if limit_eps > 0 and episode_ids is not None:
            ep_list = ep_list[:limit_eps]
        ep_lengths = {}
        for ep in tqdm(ep_list, desc=f"[{suite}] episodes"):
            T = int(meta.episodes[ep]["length"])
            timestamps = [i / fps for i in range(T)]

            cam_imgs = []
            for ck in cam_keys:
                vp = meta.root / meta.get_video_file_path(ep, f"observation.images.{ck}")
                frames = decode_video_frames(vp, timestamps, tol)  # [T,C,H,W] float[0,1]
                if frames.shape[0] != T:
                    raise RuntimeError(f"{suite} ep{ep} cam {ck}: decoded {frames.shape[0]} != length {T}")
                x = (frames * 255).to(torch.uint8)  # as in BaseLerobotDataset._get_image
                for tr in img_transforms:           # ToTensor(/255) + per-camera Resize
                    x = tr(x)
                cam_imgs.append(x)

            pixels = torch.stack(cam_imgs, dim=0)            # [num_cam, T, C, H, W]
            video = _concat_cameras(pixels, concat_mode)     # [T, C, vh, vw]
            video = norm_t(crop_t(resize_t(video)))          # [T, C, vh, vw] in [-1,1]

            # Clip indices per target frame, oldest first, clamped at 0.
            clip_idx = torch.tensor(
                [[max(f - (clip_len - 1 - j) * frame_interval, 0) for j in range(clip_len)]
                 for f in range(T)],
                dtype=torch.long,
            )  # [T, clip_len]

            latents = []
            with torch.no_grad():
                for s in range(0, T, enc_batch):
                    idx = clip_idx[s:s + enc_batch]                   # [b, clip_len]
                    clip = video[idx].to(device)                     # [b, clip_len, C, H, W]
                    z = backbone.encode_clip_last_group(clip)        # [b, N, D]
                    latents.append(z.to(save_dtype).cpu())
            latents = torch.cat(latents, dim=0)              # [T, N, D]

            torch.save(
                {"latents": latents, "episode_index": ep, "fps": fps,
                 "num_patches": num_patches, "embed_dim": embed_dim,
                 "backbone_model_name": backbone_name, "backbone_model_size": backbone_size,
                 "frame_interval": frame_interval, "clip_len": clip_len},
                out_dir / f"episode_{ep:06d}.pt",
            )
            ep_lengths[str(ep)] = T

        with open(out_dir / "latents_meta.json", "w") as f:
            json.dump(
                {"suite": suite, "dataset_dir": ds_dir, "fps": fps,
                 "episode_ids_file": str(ids_file) if ids_file else None,
                 "num_episodes_encoded": len(ep_lengths),
                 "num_patches": num_patches, "embed_dim": embed_dim,
                 "backbone_model_name": backbone_name, "backbone_model_size": backbone_size,
                 "image_size": [vh, vw], "concat_multi_camera": concat_mode,
                 "cam_keys": cam_keys, "latent_dtype": str(save_dtype).split(".")[-1],
                 "frame_interval": frame_interval, "clip_len": clip_len,
                 "episodes": ep_lengths},
                f, indent=2,
            )
        logger.info("Done suite=%s episodes=%d -> %s", suite, len(ep_list), out_dir)


if __name__ == "__main__":
    main()
