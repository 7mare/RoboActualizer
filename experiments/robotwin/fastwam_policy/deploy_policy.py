"""RoboTwin policy hooks (get_model / eval / reset_model) for a Roboactualizer checkpoint."""

import contextlib
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Roboactualizer.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from Roboactualizer.models.roboactualizer.ckpt_resolve import check_ckpt_matches_cfg
from Roboactualizer.utils.pytorch_utils import set_global_seed

from experiments.robotwin.robotwin_jpfm_adapter import RoboTwinJointPredictFMAdapter

logger = logging.getLogger(__name__)

_MODEL_DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _compose_sim_cfg(sim_cfg_path: str, sim_task: Optional[str]) -> DictConfig:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    config_name = Path(str(sim_cfg_path)).resolve().relative_to(configs_root).as_posix()
    overrides = [f"task={sim_task}"] if sim_task else []
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        return compose(config_name=config_name, overrides=overrides)


class JointPredictFMRobotWinPolicy:
    """Action queue around the adapter.

    Every take_action step feeds an observation: z0 is a clip that needs the per-step frame history.
    """

    def __init__(self, adapter, replan_steps: int, num_inference_steps: int, text_cfg_scale: float) -> None:
        self.adapter = adapter
        self.replan_steps = int(max(1, replan_steps))
        self.num_inference_steps = int(num_inference_steps)
        self.text_cfg_scale = float(text_cfg_scale)
        self.pending_actions: deque[np.ndarray] = deque()

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        pred = self.adapter.infer_action(
            instruction=instruction,
            observation=observation,
            num_inference_steps=self.num_inference_steps,
            text_cfg_scale=self.text_cfg_scale,
        )
        chunk = self.adapter.denormalize_action(pred["action"])[0]  # [K, 14] absolute joint targets
        for i in range(min(self.replan_steps, chunk.shape[0])):
            self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return True

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        # Record the frame first: z0 always ends with the view seen before this step's action.
        self.adapter.update_history(observation)
        if not self.pending_actions:
            self._fill_action_queue(observation, task_env.get_instruction())
        task_env.take_action(self.pending_actions.popleft(), action_type="qpos")

    def reset(self) -> None:
        self.pending_actions.clear()
        self.adapter.reset_history()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


@contextlib.contextmanager
def _cwd(path: Path):
    """RoboTwin runs from third_party/RoboTwin; model loading needs repo-relative paths."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def get_model(usr_args: Dict[str, Any]):
    cfg = _compose_sim_cfg(usr_args["sim_cfg_path"], usr_args.get("sim_task"))
    checkpoint_path = str(usr_args["ckpt_setting"])
    device = str(usr_args.get("device") or cfg.EVALUATION.device)
    model_dtype = _MODEL_DTYPES[str(usr_args.get("mixed_precision") or cfg.mixed_precision)]

    # Seed torch: FlowPolicy.generate draws its initial noise from the global RNG.
    set_global_seed(int(usr_args["seed"]), get_worker_init_fn=False)

    with _cwd(PROJECT_ROOT):
        check_ckpt_matches_cfg(checkpoint_path, cfg)
        model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
        model.load_checkpoint(checkpoint_path)
        model = model.to(device).eval()

        processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(usr_args["dataset_stats_path"])))
        adapter = RoboTwinJointPredictFMAdapter(
            model=model, model_cfg=cfg.model, data_cfg=cfg.data.train, processor=processor,
            device=device, model_dtype=model_dtype,
        ).eval()

    return JointPredictFMRobotWinPolicy(
        adapter=adapter,
        replan_steps=int(usr_args.get("replan_steps") or cfg.EVALUATION.replan_steps),
        num_inference_steps=int(usr_args.get("num_inference_steps") or cfg.EVALUATION.num_inference_steps),
        text_cfg_scale=float(usr_args.get("text_cfg_scale", cfg.EVALUATION.text_cfg_scale)),
    )


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()
