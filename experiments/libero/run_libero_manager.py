import hashlib
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def load_sample_task_file(sample_path: Path, output_file: Path, task_suite_names: list[str]) -> Path:
    """Validate a frozen ``suite,task_id`` list and write it to ``output_file``."""
    if not sample_path.exists():
        raise FileNotFoundError(f"MULTIRUN.sample_file not found: {sample_path}")
    known = set(task_suite_names)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with sample_path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2 or not parts[0] or not parts[1].strip().isdigit():
                raise ValueError(f"{sample_path}:{i} expected suite,task_id: {raw!r}")
            suite = parts[0].strip()
            if suite not in known:
                raise ValueError(
                    f"{sample_path}:{i} suite '{suite}' not in {sorted(known)}"
                )
            lines.append(f"{suite},{int(parts[1])}")
    if not lines:
        raise ValueError(f"Empty sample file: {sample_path}")
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSample task list loaded: {sample_path}")
    print(f"Task list written: {output_file}")
    print(f"Total sampled tasks: {len(lines)}")
    return output_file


def _redirect_to_sample_dir(output_dir: Path) -> Path:
    """Sampled runs go to ``libero_plus_sample`` instead of ``libero_plus`` (if that path segment exists)."""
    parts = list(output_dir.parts)
    if "libero_plus" in parts:
        parts[parts.index("libero_plus")] = "libero_plus_sample"
        return Path(*parts)
    return output_dir


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",  # the scheduler passes the resolved (maybe redirected) one
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def default_session_name(output_dir: Path) -> str:
    """Unique tmux session/socket name derived from output_dir (same dir -> same name)."""
    raw = "_".join(output_dir.parts[-2:]) if len(output_dir.parts) >= 2 else output_dir.name
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[-60:]
    digest = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"libero_{slug}_{digest}"


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
) -> None:
    script_path = Path("experiments/libero/run_libero_parallel_test.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = os.getcwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_name = os.environ.get("SESSION_NAME") or default_session_name(output_dir)
    state_dir = os.environ.get("STATE_DIR") or str(output_dir / "_sched")

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CONFIG_NAME": HydraConfig.get().job.config_name,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "PYTHON_BIN": sys.executable,  # workers use this interpreter, not the tmux default
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
            # Concurrency isolation: session, socket and scheduler state dir are all derived
            # from output_dir, so several ckpts can be evaluated on one machine in parallel.
            "SESSION_NAME": session_name,
            "TMUX_SOCKET": os.environ.get("TMUX_SOCKET") or session_name,
            "STATE_DIR": state_dir,
        }
    )

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Output directory: {output_dir}")
    print(f"tmux session/socket: {session_name}")
    print(f"Scheduler state dir: {state_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    try:
        subprocess.run(
            ["bash", str(script_path), str(task_file)],
            env=env,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding='utf-8'))
        raise


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval_libero_dits.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()
    manager = cfg.MULTIRUN

    sample_file_cfg = manager.get("sample_file")

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    if sample_file_cfg:
        output_dir = _redirect_to_sample_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_file_cfg = manager.get("task_file")
    if task_file_cfg:
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:
        task_file = output_dir / "tasks.txt"

    if sample_file_cfg:
        sample_path = Path(os.path.expanduser(os.path.expandvars(str(sample_file_cfg))))
        task_file = load_sample_task_file(sample_path, task_file, list(manager.task_suite_names))
    else:
        task_file = create_task_file(task_file, list(manager.task_suite_names))

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    if bool(manager.get("create_only", False)):
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),
    )


if __name__ == "__main__":
    main()
