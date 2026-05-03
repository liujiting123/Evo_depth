# Evo1 LeRobot Version

This directory stores the LeRobot-based Evo1 implementation copied from Evo-RL.

It is intended as a reference copy for future integration back into the Evo1 repository. It is not the original Evo1 training entry.

## Environment

Two environment variants are kept here:

- [requirements-a800-cu118-flashattn.txt](/Users/javadcc/code/Evo-1/evo1_lerobot/requirements-a800-cu118-flashattn.txt)
- [requirements-3090-cu128.txt](/Users/javadcc/code/Evo-1/evo1_lerobot/requirements-3090-cu128.txt)

Recommended setup:

- Prefer the **A800 + CUDA 11.8 + FlashAttention** environment.
- This is closer to the original Evo1 validated setup.
- For Evo1, `flash-attn` is still the recommended configuration.
- The 3090 / CUDA 12.8 environment is kept because it was validated to launch and train this LeRobot version, but it is a fallback path rather than the preferred setup.

Manual installation example:

```bash
conda create -y -n evo1-lerobot python=3.10 pip
conda activate evo1-lerobot
python -m pip install --upgrade pip
python -m pip install -r requirements-a800-cu118-flashattn.txt
```

If you specifically need the 3090 path:

```bash
conda create -y -n evo1-lerobot-3090 python=3.10 pip
conda activate evo1-lerobot-3090
python -m pip install --upgrade pip
python -m pip install -r requirements-3090-cu128.txt
```

## Training Stages

The LeRobot Evo1 version keeps the same two-stage semantics:

- `training_stage=stage1`
  - freeze VLM, train action head
- `training_stage=stage2`
  - finetune VLM and action head

## Resume Semantics

Two resume modes must be distinguished:

### Stage handoff: stage1 -> stage2

Use:

- `--policy.path=<stage1_checkpoint_dir>/pretrained_model`
- `--resume_pretrain=true`

Meaning:

- load weights only
- do **not** restore optimizer state
- do **not** restore scheduler state
- do **not** restore global step

This is the LeRobot equivalent of the original Evo1 stage handoff.

### Same-stage resume

Normal training resume should use the training framework checkpoint restore path, not `resume_pretrain=true`.

## Dataset Version

This LeRobot version trains through the LeRobot dataset pipeline.

Practical notes:

- the original Evo1 dataset loader is not the default training path here
- the experiments used LeRobot-style datasets
- v3.0-style data was used in the training comparisons and compatibility experiments

## Checkpoint Format

This LeRobot Evo1 version saves weights in the LeRobot policy format:

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_postprocessor.json`

For stage handoff, `--policy.path` should point to:

- `.../checkpoints/<step>/pretrained_model`

## Reference Commands

These commands are for the **standalone `evo1_lerobot` directory**.

### Stage 1

The parameter values below are aligned with the original Evo1 README where possible:

- `dropout=0.2`
- `optimizer_lr=1e-5`
- `optimizer_weight_decay=1e-3`
- `scheduler_warmup_steps=1000`
- `num_layers=8`
- `chunk_size=50`
- `n_action_steps=50`
- `max_action_dim=24`
- `max_state_dim=24`
- `steps=5000`

```bash
cd /path/to/evo1_lerobot
PYTHONPATH=. accelerate launch \
  --config_file lerobot/policies/evo1/accelerate_four_gpu_bf16.yaml \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<your_dataset_repo> \
  --dataset.root=<your_dataset_root> \
  --dataset.image_transforms.enable=false \
  --policy.type=evo1 \
  --policy.training_stage=stage1 \
  --policy.vlm_model_name=<path_or_hf_id_to_InternVL3-1B> \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dropout=0.2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-3 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.num_layers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_action_dim=24 \
  --policy.max_state_dim=24 \
  --batch_size=16 \
  --num_workers=4 \
  --steps=5000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=2500 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=<output_dir_stage1>
```

### Stage 2

```bash
cd /path/to/evo1_lerobot
PYTHONPATH=. accelerate launch \
  --config_file lerobot/policies/evo1/accelerate_four_gpu_bf16.yaml \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<your_dataset_repo> \
  --dataset.root=<your_dataset_root> \
  --dataset.image_transforms.enable=false \
  --policy.path=<output_dir_stage1>/checkpoints/005000/pretrained_model \
  --policy.training_stage=stage2 \
  --resume_pretrain=true \
  --policy.vlm_model_name=<path_or_hf_id_to_InternVL3-1B> \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.dropout=0.2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-3 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.num_layers=8 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_action_dim=24 \
  --policy.max_state_dim=24 \
  --batch_size=16 \
  --num_workers=4 \
  --steps=80000 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir=<output_dir_stage2>
```
