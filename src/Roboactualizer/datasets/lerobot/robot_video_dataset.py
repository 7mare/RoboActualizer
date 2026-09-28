import hashlib
import os
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from Roboactualizer.utils.logging_config import get_logger
from Roboactualizer.utils import misc
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        clip_len: int = 1,
        frame_interval: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        rgb_latent_cache_dir: Optional[str] = None, # precomputed V-JEPA latents; skips decoding and encoding
        episode_ids_file: Optional[str] = None, # optional episode subset; must match the latent precompute
    ):
        self.dataset_dirs = list(dataset_dirs)
        self.rgb_latent_cache_dir = rgb_latent_cache_dir
        self.clip_len = int(clip_len)
        self.frame_interval = int(frame_interval)
        # Only the pixel path needs past frames to build each target's clip.
        self.past_image_size = 0 if self.rgb_latent_cache_dir is not None else (self.clip_len - 1) * self.frame_interval
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            past_image_size=self.past_image_size,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            episode_ids_file=episode_ids_file,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(self.rgb_latent_cache_dir is None)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    @staticmethod
    def _online_clip_indices(
        video_sample_indices,
        past_image_size,
        clip_len,
        frame_interval,
        image_is_pad=None,
    ):
        """Target indices and their causal clip indices, clamped to the episode like the cache."""
        targets = torch.tensor(
            [past_image_size + i for i in video_sample_indices],
            dtype=torch.long,
        )
        valid_indices = (~image_is_pad.bool()).nonzero(as_tuple=False).flatten()
        first_valid, last_valid = valid_indices[0], valid_indices[-1]
        clip_targets = targets.clamp(min=first_valid, max=last_valid)
        offsets = torch.arange(
            -(clip_len - 1) * frame_interval,
            1,
            frame_interval,
            dtype=torch.long,
        )
        clip_indices = clip_targets[:, None] + offsets[None, :]
        return targets, clip_indices.clamp(min=first_valid, max=last_valid)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"][self.past_image_size:]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        if self.rgb_latent_cache_dir is not None:
            video = self._load_video_latent(sample)          # [T_video, N, D]
            image_is_pad = image_is_pad[self.video_sample_indices]
            return self._assemble_sample(sample, video, image_is_pad, t_video=video.shape[0], video_key="video_latent")

        video = sample["pixel_values"]  # [T_all,C,H,W] or [num_cameras,T_all,C,H,W]
        num_cameras = 1
        if video.ndim == 5:
            num_cameras, T_all, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            T_all, C, H, W = video.shape

        video = video.view(num_cameras, T_all, C, H, W)  # [num_cameras,T_all,C,H,W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_all,C,H,W]

        target_indices, clip_indices = self._online_clip_indices(
            self.video_sample_indices,
            self.past_image_size,
            self.clip_len,
            self.frame_interval,
            image_is_pad,
        )
        image_is_pad = image_is_pad[target_indices]
        video = video[clip_indices]  # [T_video,clip_len,C,H,W]
        return self._assemble_sample(
            sample, video, image_is_pad, t_video=len(self.video_sample_indices), video_key="video"
        )

    def _load_video_latent(self, sample) -> torch.Tensor:
        """Read the cached latents of this window's sampled frames (clamped to the episode)."""
        ds_idx = int(sample["dataset_index"])
        ep = int(sample["episode_index"])
        f0 = int(sample["frame_index"])
        suite = os.path.basename(os.path.normpath(self.dataset_dirs[ds_idx]))
        path = os.path.join(self.rgb_latent_cache_dir, suite, f"episode_{ep:06d}.pt")
        latents = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["latents"]  # [T_ep,N,D]
        t_ep = latents.shape[0]
        frames = [min(max(f0 + i, 0), t_ep - 1) for i in self.video_sample_indices]
        return latents[frames].clone()  # [T_video, N, D]

    def _assemble_sample(self, sample, video, image_is_pad, *, t_video, video_key):
        """Align action/proprio and attach the cached text context."""
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim], aligned with action

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        # Keep the real token mask for cross-attn; padded embeddings are zeroed.
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0

        return {
            video_key: video,
            "action": sample["action"],
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
