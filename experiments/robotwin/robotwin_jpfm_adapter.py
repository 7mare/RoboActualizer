"""RoboTwin adapter: builds the 3-camera mosaic like training, buffers z0 clips, calls ``generate``."""

from collections import deque
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig

from Roboactualizer.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from Roboactualizer.datasets.robotwin_image import build_mosaic_from_frames
from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components

# shape_meta.images order -> simulator camera; also the mosaic layout (top, bottom-left, bottom-right).
_CAM_KEY_TO_SIM_NAME = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


class RoboTwinJointPredictFMAdapter:
    """Wraps ``FlowPolicy`` with ``infer_action`` and per-step ``update_history``."""

    def __init__(self, model, model_cfg: DictConfig, data_cfg: DictConfig, processor, device: str,
                 model_dtype: torch.dtype) -> None:
        self.model = model
        self.device = device
        self.torch_dtype = model_dtype
        self.processor = processor
        self.pixel_dtype = torch.float32 if bool(model_cfg.eval_encode_fp32) else model_dtype
        self.clip_len = int(model_cfg.clip_len)
        self.frame_interval = int(model_cfg.frame_interval)
        self.context_len = int(model_cfg.tokenizer_max_len)
        self.accel = {k: bool(model_cfg[k]) for k in
                      ("use_kv_cache", "use_cuda_graph", "use_adaln_cache", "use_cross_kv_cache")}

        # T5 stays bf16 whatever the policy dtype: the training text cache was encoded in bf16.
        self.text_encoder, self.tokenizer = load_text_encoder_components(
            model_id=str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
            tokenizer_model_id=str(model_cfg.tokenizer_model_id),
            context_len=self.context_len,
            device=device,
            dtype=torch.bfloat16,
        )

        # Same pixel pipeline as RobotVideoDataset: per-camera val transforms, mosaic, resize/crop/normalize.
        self._img_transforms = processor.val_transforms
        vh, vw = int(data_cfg.video_size[0]), int(data_cfg.video_size[1])
        self._resize_t = ResizeSmallestSideAspectPreserving(args={"img_w": vw, "img_h": vh})
        self._crop_t = CenterCrop(args={"img_w": vw, "img_h": vh})
        self._norm_t = Normalize(args={"mean": 0.5, "std": 0.5})
        self._cam_keys = [str(m["key"]) for m in data_cfg.shape_meta.images]
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=(self.clip_len - 1) * self.frame_interval + 1)

    def reset_history(self) -> None:
        self.frame_buffer.clear()

    def obs_to_mosaic(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Simulator observation -> [3, 384, 320] mosaic in [-1, 1]."""
        obs_data = observation["observation"]
        cams = [
            torch.from_numpy(np.asarray(obs_data[_CAM_KEY_TO_SIM_NAME[k]]["rgb"]).astype(np.uint8))
            .permute(2, 0, 1).unsqueeze(0)
            for k in self._cam_keys
        ]
        video = build_mosaic_from_frames(
            torch.stack(cams, dim=0), self._img_transforms, self._resize_t, self._crop_t, self._norm_t
        )
        return video[0].to(device=self.device, dtype=self.pixel_dtype)

    def update_history(self, observation: Dict[str, Any]) -> None:
        """Append the current observation; call after every take_action."""
        self.frame_buffer.append(self.obs_to_mosaic(observation))

    def _z0_clip_pixels(self) -> torch.Tensor:
        """clip_len frames spaced frame_interval apart, ending at the current frame (clamped at 0)."""
        frames = list(self.frame_buffer)
        idxs = [max(0, len(frames) - 1 - i * self.frame_interval) for i in range(self.clip_len)]
        return torch.stack([frames[i] for i in reversed(idxs)], dim=0).unsqueeze(0)

    def _encode_instruction(self, prompt: str):
        ids, mask = self.tokenizer([prompt], return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        context = self.text_encoder(ids, mask)
        context[~mask] = 0.0  # zero padding, as in the training cache
        return context.to(self.torch_dtype), mask

    def normalize_proprio(self, state: np.ndarray) -> torch.Tensor:
        key = self.processor.shape_meta["state"][0]["key"]
        batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        return self.processor.normalizer.forward(batch)["state"][key]

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        key = self.processor.shape_meta["action"][0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def infer_action(self, *, instruction: str, observation: Dict[str, Any], num_inference_steps: int,
                     text_cfg_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        """Normalized action chunk [1, chunk_size, raw_action_dim]; history must already hold this frame."""
        pixels = self._z0_clip_pixels()
        proprio = self.normalize_proprio(np.asarray(observation["joint_action"]["vector"], dtype=np.float32))
        context, text_mask = self._encode_instruction(DEFAULT_PROMPT.format(task=instruction))
        extra = {}
        if float(text_cfg_scale) != 1.0:
            u_ctx, u_mask = self._encode_instruction(DEFAULT_PROMPT.format(task=""))
            extra = {"cfg_scale": float(text_cfg_scale), "uncond_text_context": u_ctx, "uncond_text_mask": u_mask}
        with torch.autocast("cuda", dtype=self.torch_dtype):
            actions = self.model.generate(
                pixels, context, text_mask, proprio.to(self.device),
                num_inference_steps=num_inference_steps, **extra, **self.accel,
            )
        return {"action": actions}
