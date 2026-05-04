# Installation

```bash
git clone <anonymous-repo-url>

cd Evo_depth
conda create -n evo_depth python=3.10 -y
conda activate evo_depth
pip install -r requirements.txt
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL

```

# Simulation Benchmark

All simulation clients and `Evo_depth_server.py` are aligned on **websocket port 9000**. Set `EVO_DEPTH_SERVER_PORT` if you need another port, and pass the same URL from each client (`--server_url`, shell scripts, or YAML).

## LIBERO Benchmark

### 1. Prepare the environment for LIBERO

```bash
conda create -n libero python=3.8.13 -y

conda activate libero

cd LIBERO-evaluation/

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

cd LIBERO

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .

pip install websockets

pip install huggingface_hub
```

### 2. Model Preparation

#### 2.1 Download Model Weight

```bash
hf download xxx --local-dir /path/to/save/checkpoint/
```

#### 2.2 Server checkpoint and port

Point the server at a directory that contains `config.json`, `norm_stats.json`, and `mp_rank_00_model_states.pt`:

```bash
export EVO_DEPTH_CKPT_DIR=/path/to/your/checkpoint
# optional (default 9000):
export EVO_DEPTH_SERVER_PORT=9000
```

Or edit the defaults in `Evo_depth/Evo_depth/scripts/Evo_depth_server.py` under `if __name__ == "__main__":`.

### 3. Run LIBERO Evaluation

```bash
# Terminal 1
conda activate evo_depth
cd Evo_depth
python scripts/Evo_depth_server.py

```

```bash
cd LIBERO-evaluation

# Usage: ./test_libero.sh [log_path] [task] [server_url]
# example:
./test_libero.sh ./logs libero_spatial ws://127.0.0.1:9000
```

## LIBERO PLUS Benchmark

### 1. Prepare the environment for LIBERO-PLUS

```bash
# Clone our repository
git clone https://github.com/LinqingZhong/LIBERO-plus.git
cd LIBERO-plus

# Install the new LIBERO package
pip install -e .

# New dependencies installed on top of LIBERO
sudo apt install libexpat1
sudo apt install libfontconfig1-dev
sudo apt install libpython3-stdlib
sudo apt-get install libmagickwand-dev
apt install -y cmake build-essential
sudo apt install -y libgl1-mesa-dev libglib2.0-0


pip install websockets
pip install PyYAML
pip install torch==2.5.1

pip install -r requirements.txt
pip install numpy==1.26.4
pip install -r extra_requirements.txt


sudo apt update
sudo apt install -y libgl1-mesa-dev libglib2.0-0
```

### 2. Run LIBERO-PLUS Evaluation

EvoDepth provides a small wrapper client and script in `LIBERO-PLUS-evaluation/`. Run the client from your **LIBERO-plus** clone root (or pass `--filter_json_path` to `libero/libero/benchmark/task_classification.json` there). The client defaults to `ws://127.0.0.1:9000`, same as the EvoDepth server.

#### 2.1 Start EvoDepth server

This is the same server used for LIBERO:

```bash
# Terminal 1
conda activate evo_depth
cd Evo_depth
python scripts/Evo_depth_server.py
```

#### 2.2 Run LIBERO-PLUS client

In another terminal:

```bash
cd LIBERO-PLUS-evaluation

# Usage:
# ./test_libero_plus.sh <log_path> [task]
#   log_path: directory to save log.txt and videos
#   task    : libero_spatial | libero_goal | libero_object | libero_10
#             (default: libero_spatial)

# Example:
./test_libero_plus.sh /path/to/logs libero_goal
```

This script calls:

```bash
python libero_plus_client.py \
  --task_suites <task> \
  --log_file <log_path>/log.txt \
  --video_log_dir <log_path>/videos
```

where horizon and max steps are automatically chosen per task using the same  
`HORIZON_BY_TASK` and `steps` settings as the LIBERO benchmark.

## MetaWorld Benchmark (MT50)

### 1. Environment (MetaWorld + client)

Create a dedicated conda env and install MuJoCo, Meta-World, and client deps (Gymnasium is pulled in by `metaworld`):

```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco
pip install metaworld
pip install websockets
pip install opencv-python
pip install packaging
pip install huggingface_hub
pip install pyyaml
```

Equivalent one-liner from `Metaworld-evaluation/`:

```bash
pip install -r requirements-metaworld.txt
```

On headless machines, `MUJOCO_GL=egl` is set from `metaworld_eval.yaml` (see `mujoco_gl`); use `glfw` locally if you enable a display window.

### 2. Evaluation layout


| File                                       | Role                                                                        |
| ------------------------------------------ | --------------------------------------------------------------------------- |
| `Metaworld-evaluation/metaworld_eval.yaml` | Server URL, logging, MT50 camera, horizons, video/debug flags               |
| `Metaworld-evaluation/mt50_order.json`     | Task order and difficulty groups (`easy` / `medium` / …)                    |
| `Metaworld-evaluation/tasks.jsonl`         | Language prompts; lines may use `task_index` or `idx` to match MT50 indices |


Relative paths in the YAML are resolved against the YAML file’s directory.

### 3. Model / server (same as LIBERO)

Use the same `EVO_DEPTH_CKPT_DIR` / `EVO_DEPTH_SERVER_PORT` as in **LIBERO → Server checkpoint and port** above. In `Metaworld-evaluation/metaworld_eval.yaml`, set `server_url` to match the server (default `ws://127.0.0.1:9000`).

### 4. Run evaluation

Terminal 1 — EvoDepth server:

```bash
conda activate evo_depth
cd Evo_depth
python scripts/Evo_depth_server.py
```

Terminal 2 — MT50 client (from repo root):

```bash
cd Metaworld-evaluation

# Defaults: read metaworld_eval.yaml (edit server_url / log_dir there), then:
./test_metaworld.sh

# Or pass log base directory and websocket URL:
./test_metaworld.sh ./metaworld_logs ws://127.0.0.1:9000

# CLI overrides (see metaworld_client.py --help):
python metaworld_client.py --config metaworld_eval.yaml --server_url ws://127.0.0.1:9000 \
  --log_dir ./logs --horizon 17 --episodes 10 --target_level all
```

Each run creates `<log_dir>/<run_name>/eval.txt` and `<log_dir>/<run_name>/videos/` (default `run_name` is a timestamp).

## VLA-Arena Benchmark
### 1. Prepare the environment for VLA-Arena
``` bash
cd VLA_Arena_Evaluation
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena
conda create -n vla_arena python=3.11
pip install .
pip install websockets==15.0.1 draccus
cd ..
```

### 2. Run VLA-Arena Evaluation
#### 2.1 Start EvoDepth server
This is the same server used for LIBERO:
``` bash
cd Evo_depth
python scripts/Evo1_server.py
```
#### 2.2 Run VLA-Arena client
In the other terminal, you can run the evaluation scripts.
``` bash
cd VLA-Arena-evaluation
python vla_arena/vla_arena_client.py  \
--execution_horizon 10     \
--seed 10    \
--num_episodes_per_task 10  \  
--server_url ws://127.0.0.1:9000  \  
--log_out_dir ./logs/exp_h10_s27  
--save_video_mode all \
--max_episode_steps 300 
# or you can use the test_vla_arena.sh
```



## Training on your own dataset

Training follows the same **LeRobot v2.1-style** : each dataset root should contain `meta/` (e.g. `tasks.jsonl`, `episodes.jsonl`, stats) and `data/` / `videos/` as consumed by `lerobot_dataset_pretrain_mp.py`. This repo uses a **three-stage** schedule (see `[train.sh](Evo_depth/train.sh)`): (1) action head only, (2) DA3 + action head, (3) VLM + DA3 + action head full fine-tuning.

### 1. Prepare data

Download or place a LeRobot-format dataset, then point to it from `dataset/config.yaml` (or a copy such as `dataset/config_libero.yaml`).

```bash
mkdir -p Evo_depth_training_dataset && cd Evo_depth_training_dataset

# Example: shallow clone then pull LFS when ready
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/<ORG>/<YOUR_DATASET>
cd <YOUR_DATASET>
git lfs pull
```

Use any HF or local path; replace `<ORG>/<YOUR_DATASET>` with your dataset id.

### 2. Modify config

#### 2.1 `dataset/config.yaml`

Edit `[Evo_depth/dataset/config.yaml](Evo_depth/dataset/config.yaml)`: under `data_groups`, set each dataset `path` and `view_map` (camera folder names under `videos/<chunk>/`). Keep `max_action_dim`, `max_state_dim`, and `max_views` consistent with your training script (`--per_action_dim`, `--state_dim`, and the model’s expected number of views).

#### 2.2 Parquet cache directory

Preprocessed windows are cached as `.pkl` files.

- **Default directory** (when `LeRobotDataset(..., cache_dir=None)`): set in `[Evo_depth/dataset/lerobot_dataset_pretrain_mp.py](Evo_depth/dataset/lerobot_dataset_pretrain_mp.py)` **lines 173–176** — if `cache_dir` is `None`, `self.cache_dir` becomes `Path("./cache/lerobot_pretrain")` (relative to the process working directory, usually the inner `Evo_depth/` package).
- **To override without editing that file**: pass `cache_dir=...` into `LeRobotDataset` inside `[prepare_dataset](Evo_depth/scripts/train.py)` (**lines 155–162**); the constructor call currently omits `cache_dir`, so the default in `lerobot_dataset_pretrain_mp.py` applies.

### 3. Start training

#### 3.1 Accelerate and DeepSpeed

From the inner package directory `Evo_depth/` (where `ds_config.json` and `scripts/train.py` live):

```bash
conda activate evo_depth
cd Evo_depth
accelerate config   # once per machine; multi-GPU: set num_processes accordingly
```

Use `[ds_config.json](Evo_depth/ds_config.json)` with `accelerate launch` as in `train.sh`.

#### 3.2 Stage 1 — action head only

Train `**--finetune_action_head**` only. With `**--use_da3**`, the Depth Anything 3 branch is in the forward path but stays **frozen** until you add `**--finetune_da3`** in stage 2. VLM stays frozen without `**--finetune_vlm**`.

```bash
cd Evo_depth

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port 29519 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage1 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 5000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --use_da3 \
  --save_dir /path/to/checkpoints/stage1/
```

For multi-GPU, set `--num_processes` to match `accelerate config`.

#### 3.3 Stage 2 — DA3 + action head

Add `**--finetune_da3**` (keep `**--finetune_action_head**`). Resume from the last step of stage 1 (`--resume --resume_pretrain --resume_path .../step_<N>`; `N` must match the checkpoint tag on disk).

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port 29519 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage2 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 10000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --finetune_da3 \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --use_da3 \
  --save_dir /path/to/checkpoints/stage2/ \
  --resume \
  --resume_pretrain \
  --resume_path /path/to/checkpoints/stage1/step_5000
```

#### 3.4 Stage 3 — full model (VLM + DA3 + action head)

Add `**--finetune_vlm**` together with `**--finetune_da3**` and `**--finetune_action_head**`. Resume from stage 2’s final step (again align `--resume_path` with the real `step_*` folder).

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  --main_process_port 29519 \
  scripts/train.py \
  --run_name Evo_depth_3stages_stage3 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --use_da3 \
  --finetune_vlm \
  --finetune_action_head \
  --finetune_da3 \
  --disable_wandb \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /path/to/checkpoints/stage3/ \
  --resume \
  --resume_pretrain \
  --resume_path /path/to/checkpoints/stage2/step_10000
```

#### 3.5 (Optional) Resume mid-stage

Keep the same finetune flags as the stage you are in; set `**--resume**` and `**--resume_path**` to the checkpoint directory (e.g. `.../step_20000`). See `scripts/train.py` for all flags.

The canonical copy-paste layout is in `[train.sh](Evo_depth/train.sh)`; edit paths, `max_steps`, and `step_*` to match your machine.
