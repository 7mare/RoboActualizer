"""Evaluate one LIBERO / LIBERO-Plus task (all its trials) with a Roboactualizer checkpoint."""

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import PartialState

# LIBERO init-state files hold numpy globals; torch>=2.6 needs weights_only=False to read them.
_orig_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_full
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_rollout_video,
)
from Roboactualizer.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from Roboactualizer.models.roboactualizer.ckpt_resolve import check_ckpt_matches_cfg, resolve_dataset_stats_path
from Roboactualizer.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark
from experiments.libero.joint_fm_adapter import LiberoJointPredictFMAdapter

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_MODEL_DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(proprio: np.ndarray, processor: FastWAMProcessor) -> torch.Tensor:
    state_key = processor.shape_meta["state"][0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_image(obs: dict, processor: FastWAMProcessor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Agent-view and wrist images, each resized, tiled side by side and scaled to [-1, 1]."""
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    primary = _center_crop_resize(imgs["image"], width=image_meta[0]["shape"][2], height=image_meta[0]["shape"][1])
    wrist = _center_crop_resize(imgs["wrist_image"], width=image_meta[1]["shape"][2], height=image_meta[1]["shape"][1])
    rgb = np.concatenate([primary, wrist], axis=1)
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return x * (2.0 / 255.0) - 1.0


def _update_model_history(model, obs: dict, processor: FastWAMProcessor, model_device: str) -> None:
    model.update_history(_obs_to_model_image(obs, processor, model_device, model.pixel_dtype))


def _extract_sim_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    action_key = processor.shape_meta["action"][0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()


def _predict_action_chunk(obs: dict, task_description: str, model, processor: FastWAMProcessor, cfg: DictConfig,
                          model_device: str) -> tuple[np.ndarray, dict]:
    proprio = _normalize_proprio(_extract_sim_state(obs), processor)
    with torch.no_grad():
        pred = model.infer_action(
            prompt=DEFAULT_PROMPT.format(task=task_description),
            proprio=proprio,
            num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
            text_cfg_scale=float(cfg.EVALUATION.text_cfg_scale),
        )
    action = _denormalize_action(pred["action"], processor)[0]  # [T, D]
    # Training data flips the gripper to 0=close/1=open; map back to -1=open/+1=close.
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.binarize_gripper):
        action[..., -1] = np.sign(action[..., -1])
    return action, get_libero_image(obs)


def _get_max_steps(task_suite_name: str) -> int:
    return {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 700, "libero_90": 700}[
        task_suite_name
    ]


def run_single_episode(env, initial_state, task_description: str, model, processor: FastWAMProcessor,
                       cfg: DictConfig, episode_idx: int, model_device: str) -> tuple[bool, list]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.replan_steps)
    num_steps_wait = int(cfg.EVALUATION.num_steps_wait)

    env.reset()
    obs = env.set_init_state(initial_state)
    model.reset_history()
    _update_model_history(model, obs, processor, model_device)  # the initial state is the first observation

    replay_images = []
    pending_actions: list[list[float]] = []
    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            _update_model_history(model, obs, processor, model_device)
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs = _predict_action_chunk(obs, task_description, model, processor, cfg, model_device)
            pending_actions = action_chunk[:replan_steps].tolist()
        else:
            imgs = get_libero_image(obs)
        replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        _update_model_history(model, obs, processor, model_device)
        if done:
            break
        t += 1
    pbar.close()
    return bool(done), replay_images


def run_single_task(task, initial_states, model, processor: FastWAMProcessor, cfg: DictConfig, video_dir: Path,
                    model_device: str) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    results = {"successes": 0, "failure_episodes": [], "success_episodes": [], "task_description": task_description}
    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images = run_single_episode(
            env, initial_states[trial_idx], task_description, model, processor, cfg, trial_idx, model_device
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        save_rollout_video(
            video_dir, replay_images, f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success, task_description=task_description,
        )
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_libero_dits.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    PartialState().config = cfg
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    check_ckpt_matches_cfg(str(cfg.ckpt), cfg)
    model_device = str(cfg.EVALUATION.device)
    model_dtype = _MODEL_DTYPES[str(cfg.mixed_precision)]
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    model.load_checkpoint(str(cfg.ckpt))
    model = LiberoJointPredictFMAdapter(model.to(model_device).eval(), cfg.model, model_device, model_dtype).eval()

    dataset_stats_path = resolve_dataset_stats_path(cfg)
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(dataset_stats_path)))
    logging.info("Using dataset stats: %s", dataset_stats_path)

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    task_suite = benchmark.get_benchmark_dict()[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    num_trials = int(cfg.EVALUATION.num_trials)
    while len(initial_states) < num_trials:
        initial_states.extend(initial_states[: (num_trials - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": num_trials,
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }
    results.update(run_single_task(task, initial_states, model, processor, cfg, video_dir, model_device))
    results["duration"] = time.time() - start_time

    output_dir = local_log_dir / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print(f"Task {cfg.EVALUATION.task_id} completed: {results['successes']}/{num_trials} successes")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
