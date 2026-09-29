# RoboActualizer

Official codebase for **One from Infinity: Actualizing Futures from Pretrained World Models into Robot Actions**.

<!-- TODO: add the arXiv id and project page once public -->
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](#bibtex)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/db12312607/RoboActualizer)

A pretrained video world model already lays out the plausible futures of a scene. RoboActualizer keeps it frozen and
learns only a tiny **actualizer** that selects the task-conditioned future and reads out the actions that realize it.
This repository contains the training and evaluation code on LIBERO, LIBERO-Plus and RoboTwin 2.0.

## Highlights

- 🪶 **Tiny trainable model.** A frozen V-JEPA 2.1 ViT-L encoder plus a 60M-parameter actualizer: two DiT experts
  (future latent / action) in a Mixture-of-Transformers, trained jointly by flow matching. Up to 100x fewer trainable
  parameters than existing WAMs and VLAs.
- 💻 **Single-GPU training.** The actualizer trains on cached V-JEPA latents with 32 GB peak memory; LIBERO takes about
  15 hours on one GPU, with no embodied pretraining.
- ⚡ **39 ms inference.** The encoder runs once per observation; only the actualizer denoises, with KV, AdaLN and
  cross-attention caches and CUDA graphs (on by default). Over 25 Hz on an RTX A6000.
- 🌍 **Robust generalization.** 63.1% on LIBERO-Plus with LIBERO training data only (Fast-WAM, 6B: 51.5%).

## Results

| Benchmark | Checkpoint | Inference | Success (%) | Eval config |
|---|---|---|---|---|
| LIBERO (4 suites x 10 tasks x 50) | DiT-S, step 173580 | 4 steps, no CFG | 98.00 | `eval_libero_dits` |
| LIBERO-Plus (10,030 tasks x 1) | DiT-S, step 173580 | 4 steps, action text-CFG 1.5 | 63.1 | `eval_libero_plus_dits` |
| LIBERO-Plus (10,030 tasks x 1) | DiT-B, step 130185 | 4 steps, action text-CFG 1.5 | 65.0 | `eval_libero_plus_ditb` |
| RoboTwin 2.0 (50 tasks x clean/randomized x 25) | DiT-S fp32, step 176540 | 10 steps, no CFG, fp32 | 58.84 | `eval_robotwin_dits` |

## Index

- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Preparation](#dataset-preparation)
- [Inference with Released Checkpoints](#inference-with-released-checkpoints)
- [Training](#training)
- [Inference with Your Trained Checkpoints](#inference-with-your-trained-checkpoints)
- [Switches](#switches)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

```text
Roboactualizer/
├── configs/
│   ├── data/                  # dataset configs (LIBERO, RoboTwin)
│   ├── model/                 # actualizer + V-JEPA backbone configs
│   ├── task/                  # training configs: libero_dits, libero_ditb, robotwin_dits
│   ├── sim_*.yaml             # shared simulator eval settings
│   └── eval_*.yaml            # one eval config per released checkpoint
├── scripts/
│   ├── train.py
│   ├── precompute_text_embeds.py   # umT5 text embedding cache
│   ├── precompute_rgb_latents.py   # V-JEPA latent cache
│   └── compute_latent_norm_stats.py
├── experiments/
│   ├── libero/                # LIBERO / LIBERO-Plus managers, single-task eval, summarizers
│   └── robotwin/              # RoboTwin manager, single-task eval, policy hooks
├── src/Roboactualizer/
│   ├── models/roboactualizer/ # V-JEPA backbone, MoT DiT experts, flow-matching policy
│   ├── datasets/              # LeRobot datasets with cached latents
│   └── trainer_jointfm.py
├── envs/                      # conda environment files
├── third_party/RoboTwin/      # vendored RoboTwin 2.0 eval code and task_config
├── checkpoints/               # V-JEPA, umT5 and released checkpoints
├── data/                      # datasets and latent caches
├── runs/                      # training outputs
└── evaluate_results/          # evaluation outputs
```

## Environment Setup

Two conda environments: `roboact_libero` runs LIBERO and RoboTwin, `roboact_plus` runs LIBERO-Plus. Both use
Python 3.10, torch 2.7.1 + CUDA 12.8 and `mujoco==3.3.2` (matching the LIBERO data). All commands run from the
repository root.

### roboact_libero (LIBERO + RoboTwin 2.0)

```bash
conda env create -f envs/roboact_libero.yml
conda activate roboact_libero
pip install -e . --no-deps

# LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
pip install -e ~/LIBERO --no-deps
echo N | python -c "import libero.libero"        # writes ~/.libero/config.yaml

# RoboTwin 2.0 (code and task_config/ are vendored in third_party/RoboTwin)
cd third_party/RoboTwin
bash script/_download_assets.sh                   # downloads assets/ from HuggingFace
cd envs && git clone https://github.com/NVlabs/curobo.git && cd curobo
git checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
# warp-lang 1.x moved device_from_torch to the top level
sed -i 's/wp.torch.device_from_torch(/wp.device_from_torch(/' src/curobo/geom/sdf/world_mesh.py
pip install -e . --no-build-isolation
cd ../../../..

# register the policy with RoboTwin
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Do not run RoboTwin's `script/_install.sh`: the environment file already has its packages, and its sapien/mplib
source patches and pytorch3d were not part of the environment behind the reported RoboTwin result.

### roboact_plus (LIBERO-Plus)

```bash
conda env create -f envs/roboact_plus.yml         # includes ImageMagick for wand
conda activate roboact_plus
pip install -e . --no-deps

git clone https://github.com/sylvestf/LIBERO-plus.git ~/LIBERO-plus
pip install -e ~/LIBERO-plus --no-deps
echo N | LIBERO_CONFIG_PATH=~/.libero_plus python -c "import libero.libero"   # writes ~/.libero_plus/config.yaml
```

`experiments/libero/run_libero_plus.sh` finds this env and sets `LIBERO_CONFIG_PATH=~/.libero_plus`, `MAGICK_HOME`
and `LD_LIBRARY_PATH` itself.

## Model Preparation

Required for both training and inference.

**V-JEPA 2.1 encoder (frozen).** Put it under `checkpoints/vjepa2/` with the file names below, and cache the V-JEPA 2
source once:

```bash
mkdir -p checkpoints/vjepa2
# 300M (ViT-L/16 384, embed dim 1024): used by all released checkpoints
wget -O checkpoints/vjepa2/vjepa2_1_vit_large_384.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
# 80M (ViT-B/16 384, embed dim 768)
wget -O checkpoints/vjepa2/vjepa2_1_vit_base_384.pt https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
python -c "import torch; torch.hub.list('facebookresearch/vjepa2', trust_repo=True)"
```

The model configs use the 300M encoder (`model.backbone.model_name=vjepa2_1_vit_large_384`,
`model.backbone.ckpt_path=checkpoints/vjepa2/vjepa2_1_vit_large_384.pt`). To train with the 80M encoder, pass
`model.backbone.model_name=vjepa2_1_vit_base_384 model.backbone.ckpt_path=checkpoints/vjepa2/vjepa2_1_vit_base_384.pt`
to the latent precompute and to training; its latents are 768-dimensional, so it needs its own latent cache and
latent norm stats. `model.backbone.download_if_missing=true` downloads the file to `ckpt_path` automatically.

**Text encoder.** The Wan umT5 text encoder and tokenizer are downloaded on first use into
`checkpoints/DiffSynth-Studio/` and `checkpoints/Wan-AI/`.

## Dataset Preparation

### LIBERO

The four `*_no_noops_lerobot` datasets (LeRobot 2.1) from
[yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam):

```bash
huggingface-cli download yuanty/LIBERO-fastwam --repo-type dataset --include "*_no_noops_lerobot.tar.gz" \
    --local-dir data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2 && for f in *.tar.gz; do tar -xzf "$f"; done && cd ../..
```

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin 2.0

We use the same RoboTwin 2.0 data as flex-pi: [flex-pi/robotwin_3d](https://huggingface.co/datasets/flex-pi/robotwin_3d)
(LeRobot 2.1, 50 tasks, 27,500 episodes). Training uses a 50-demo-per-task subset (seed 42, 2,500 episodes) and
RGB only. The script downloads the metadata, `dataset_stats.json` and the subset to `data/robotwin_3d/`:

```bash
python scripts/download_robotwin_3d.py --subset-only
python scripts/verify_robotwin_3d.py
```

To store the data elsewhere, set `ROBOTWIN_3D_DIR` for both scripts and point the `./data/robotwin_3d` paths in
`configs/data/robotwin_jpfm.yaml` to it.

### Latent and text caches

The actualizer trains on cached text embeddings and V-JEPA latents (`roboact_libero`):

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_dits
python scripts/precompute_rgb_latents.py task=libero_dits model.latent_norm_stats_path=null \
    +rgb_latent_cache_dir=./data/int2_clip4/libero
python scripts/compute_latent_norm_stats.py ./data/int2_clip4/libero

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_dits
python scripts/precompute_rgb_latents.py task=robotwin_dits model.latent_norm_stats_path=null \
    +rgb_latent_cache_dir=./data/int4_clip4/robotwin
python scripts/compute_latent_norm_stats.py ./data/int4_clip4/robotwin
```

## Inference with Released Checkpoints

The checkpoints are on [Hugging Face](https://huggingface.co/db12312607/RoboActualizer). Download them into
`checkpoints/roboactualizer/`:

```bash
huggingface-cli download db12312607/RoboActualizer --exclude "data/*" --local-dir checkpoints/roboactualizer
cd checkpoints/roboactualizer && sha256sum -c SHA256SUMS && cd ../..
```

```text
checkpoints/roboactualizer/
├── libero_dits/          step_173580.pt  dataset_stats.json  config.yaml   # LIBERO, LIBERO-Plus (DiT-S)
├── libero_ditb/          step_130185.pt  dataset_stats.json  config.yaml   # LIBERO-Plus (DiT-B)
└── robotwin_dits_fp32/   step_176540.pt  dataset_stats.json  config.yaml   # RoboTwin 2.0
```

Keep each `dataset_stats.json` next to its checkpoint: evaluation loads it from there and checks it against the hash
stored in the checkpoint. `config.yaml` is the training config, for reference only.

Each checkpoint has one eval config; shared settings are in `sim_libero.yaml`, `sim_libero_plus.yaml` and
`sim_robotwin.yaml`.

**LIBERO**, DiT-S (`roboact_libero`):

```bash
conda activate roboact_libero
export LIBERO_CONFIG_PATH=~/.libero
CUDA_VISIBLE_DEVICES=0,1 python experiments/libero/run_libero_manager.py --config-name eval_libero_dits \
    ckpt=checkpoints/roboactualizer/libero_dits/step_173580.pt MULTIRUN.max_tasks_per_gpu=2
python experiments/libero/summarize_results.py --output_dir ./evaluate_results/libero/libero_dits/<timestamp>
```

**LIBERO-Plus**, all 10,030 tasks, one trial each (the launcher uses `roboact_plus`):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash experiments/libero/run_libero_plus.sh --config-name=eval_libero_plus_dits \
    ckpt=checkpoints/roboactualizer/libero_dits/step_173580.pt MULTIRUN.max_tasks_per_gpu=2
CUDA_VISIBLE_DEVICES=0,1 bash experiments/libero/run_libero_plus.sh --config-name=eval_libero_plus_ditb \
    ckpt=checkpoints/roboactualizer/libero_ditb/step_130185.pt MULTIRUN.max_tasks_per_gpu=2
conda activate roboact_plus
python experiments/libero/summarize_results_plus.py \
    --output_dir ./evaluate_results/libero_plus/<libero_dits|libero_ditb>/<timestamp>
```

**RoboTwin 2.0**, all 50 tasks, demo_clean then demo_randomized, unseen instructions (`roboact_libero`):

```bash
conda activate roboact_libero
python experiments/robotwin/run_robotwin_manager.py --config-name eval_robotwin_dits \
    ckpt=checkpoints/roboactualizer/robotwin_dits_fp32/step_176540.pt MULTIRUN.gpu_ids=[0,1] MULTIRUN.max_tasks_per_gpu=2
# results: ./evaluate_results/robotwin/<ckpt_tag>/<timestamp>/summary.csv
# resume:  add MULTIRUN.resume_dir=./evaluate_results/robotwin/<ckpt_tag>/<timestamp>
# <ckpt_tag>: <dir above checkpoints/>_step_176540, e.g. Roboactualizer_step_176540
```

**Single task / single phase:**

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py --config-name eval_libero_dits \
    ckpt=checkpoints/roboactualizer/libero_dits/step_173580.pt EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 EVALUATION.output_dir=./evaluate_results/libero/debug
python experiments/robotwin/eval_robotwin_single.py --config-name eval_robotwin_dits \
    ckpt=checkpoints/roboactualizer/robotwin_dits_fp32/step_176540.pt EVALUATION.task_name=adjust_bottle EVALUATION.task_config=demo_clean gpu_id=0
```

RoboTwin workers pick GPUs with `MULTIRUN.gpu_ids` (physical ids); LIBERO uses `CUDA_VISIBLE_DEVICES`. A worker needs
about 13 GB (LIBERO) or 18 GB (RoboTwin, fp32) of GPU memory and peaks at about 37 GB of RAM while loading.

## Training

Single GPU, `roboact_libero`:

```bash
conda activate roboact_libero
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=libero_dits      # DiT-S, bf16, 20 epochs (173,580 steps)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=libero_ditb      # DiT-B, bf16, 15 epochs (130,185 steps)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py task=robotwin_dits    # DiT-S, fp32 master weights, 10 epochs (176,540 steps)
```

Checkpoints go to `output_dir` of the task config (`./runs/<run_name>/` for LIBERO). Logging uses wandb; add
`wandb.enabled=false` to turn it off. Change `run_name` together with any hyperparameter so runs do not share a
directory.

## Inference with Your Trained Checkpoints

Use the commands above with `ckpt=<run>/checkpoints/weights/step_N.pt`. The run's `dataset_stats.json` in `<run>/` is
found automatically; pass `EVALUATION.dataset_stats_path=...` if it lives elsewhere. Checkpoints trained with
`fp32_master_weights=true` must be evaluated with `mixed_precision=no`.

## Switches

| Key | Meaning |
|---|---|
| `model.dit_size` | `s` (384 wide, 6 heads) or `b` (768 wide, 12 heads) |
| `fp32_master_weights` | train with fp32 parameters |
| `mixed_precision` | inference precision: `bf16` (LIBERO) or `no` (fp32, RoboTwin) |
| `model.eval_encode_fp32` | encode the observation clip in fp32 (matches the cached training latents) |
| `EVALUATION.text_cfg_scale` | action text-CFG scale (default 1.5; 1.0 disables it) |
| `EVALUATION.num_inference_steps` | denoising steps (10) |
| `model.use_kv_cache`, `use_cuda_graph`, `use_adaln_cache`, `use_cross_kv_cache` | inference speedups, on by default; outputs are bit-identical with them off |

## Acknowledgements

The data pipeline, training launch scripts, text-encoder wrapper and LIBERO/RoboTwin evaluation code are adapted from
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM). The actualizer model, V-JEPA latent pipeline and inference
acceleration are new.
We use [V-JEPA 2](https://github.com/facebookresearch/vjepa2) as the world model and the Wan umT5 text encoder. The
vendored RoboTwin code comes from [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin); we also thank the
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)
teams.

## BibTeX

```bibtex
@article{du2026roboactualizer,
  title={One from Infinity: Actualizing Futures from Pretrained World Models into Robot Actions},
  author={Bang Du and Yichen Xie and Shuqi Zhao and Yuxin Chen and Menglin Wu and Masayoshi Tomizuka},
  journal={arXiv preprint},
  year={2026}
}
```
