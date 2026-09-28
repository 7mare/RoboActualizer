"""LIBERO adapter: buffers env frames into z0 clips, encodes the prompt online, calls ``generate``."""

from collections import deque
from typing import Dict

import torch
from omegaconf import DictConfig

from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components

class LiberoJointPredictFMAdapter:
    """Wraps ``FlowPolicy`` as ``infer_action(**kwargs) -> {"action": [1, T, raw_dim]}``."""

    def __init__(self, model, model_cfg: DictConfig, device: str, model_dtype: torch.dtype) -> None:
        self.model = model
        self.device = device
        self.torch_dtype = model_dtype
        # Keep pixels in fp32 when z0 is encoded in fp32.
        self.pixel_dtype = torch.float32 if bool(model_cfg.eval_encode_fp32) else model_dtype
        self.clip_len = int(model_cfg.clip_len)
        self.frame_interval = int(model_cfg.frame_interval)
        self.context_len = int(model_cfg.tokenizer_max_len)
        self.accel = {k: bool(model_cfg[k]) for k in
                      ("use_kv_cache", "use_cuda_graph", "use_adaln_cache", "use_cross_kv_cache")}

        self.text_encoder, self.tokenizer = load_text_encoder_components(
            model_id=str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B")),
            tokenizer_model_id=str(model_cfg.tokenizer_model_id),
            context_len=self.context_len,
            device=device,
            dtype=model_dtype,
        )
        # A clip spans (clip_len - 1) * frame_interval + 1 env steps.
        self.frame_buffer: deque[torch.Tensor] = deque(maxlen=(self.clip_len - 1) * self.frame_interval + 1)

    def reset_history(self) -> None:
        self.frame_buffer.clear()

    def eval(self):
        self.model.eval()
        return self

    def update_history(self, input_image: torch.Tensor) -> None:
        """Append the current [1, 3, H, W] frame; call after every env step."""
        self.frame_buffer.append(input_image.detach().squeeze(0))

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

    @torch.no_grad()
    def infer_action(self, *, prompt: str, proprio: torch.Tensor, num_inference_steps: int,
                     text_cfg_scale: float = 1.0, **_) -> Dict[str, torch.Tensor]:
        pixels = self._z0_clip_pixels()
        context, text_mask = self._encode_instruction(prompt)
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
