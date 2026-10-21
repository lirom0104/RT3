# DA3 Giant thermal camera tokens + LoRA

This implements SEAR's adaptation mechanism on standalone `da3-giant` (40 blocks,
1536 channels). Load the original Giant checkpoint first, freeze it, clone its two
camera tokens into `thermal_camera_token`, and train those tokens plus LoRA.
RGB and thermal views share the LoRA layers. RGB predictions can therefore change.

[Experiment B: thermal-only LoRA](THERMAL_IR_ONLY.md) adds per-token modality
routing while keeping global RGB/thermal attention and the original training
recipe. The default adapter scope remains shared.

The default adapter covers `blocks[13:40]`, including `attn.qkv`, `attn.proj`,
`mlp.w12`, and `mlp.w3`. Rank 64, alpha 128 and dropout 0.1 give **42,470,400**
trainable parameters. This maps to DA3's aggregation phase; it is not a layer-for-layer
copy of VGGT. Existing SEAR adapter weights are not compatible.

Giant's fused xFormers SwiGLU bypasses Linear.forward. Adapted blocks explicitly
call the same pretrained `w12` and `w3` modules, so FFN LoRA participates in both
forward and backward. Camera tokens are selected at block index 13. Modality masks
follow reference-view reordering. With provided camera conditioning, the learned
thermal-token offset relative to the original camera tokens is added on thermal
views, preserving calibrated pose tokens. The no-camera training path is unchanged.
Non-reentrant activation checkpointing is used
for trainable backbone blocks.

## Data and losses

`configs/thermal_giant_sear.json` points to the local Giant 1.1 checkpoint and SEAR
dataset. The loader discovers processed paired `transforms.json` files, resolves
their relative paths, and uses the original SEAR scene-level split. Spaces in scene
names are normalized for matching. In the supplied dataset this gives:

- 39 training scenes, 4,646 paired frame entries.
- 12 validation scenes, each evaluated in RGB, mixed, and thermal modes.
- 6 difficult scenes held out entirely from training and model selection.
- 3 excluded scenes (`house`, `messy-living-room`, `red-container`) with missing
  depth/calibration fields. Exact reasons are written to `data_manifest.json`.

The labels are VGGT-derived **pseudo labels**, not independent measured ground truth.
Modality is taken from the `rgb`/`thermal` dictionary key because some source `type`
fields are incorrect. Each modality uses its own depth and calibration. Images,
depths and intrinsics undergo the same aspect-preserving resize and center crop;
invalid depth holes remain masked. Input images receive ImageNet normalization once.

Training samples eight views with overlap, random order and a varying thermal ratio.
It uses mild brightness/gamma augmentation, 392x518 pixels, batch size 1 on GPU 0,
and two accumulation steps (16 views per update). Validation uses fixed
views and no augmentation.

The training loss is adapted to DA3: robust log-depth plus multi-scale depth-gradient
loss, and 5x camera loss (rotation matrices, translation, and field of view).
One detached median depth scale per sequence is applied to both predicted depth and
translation. Per-frame scale alignment is deliberately not used. Both prediction
and target poses are OpenCV w2c matrices referenced to the first sampled view;
DA3's internal c2w pose encoding is not compared directly to VGGT's w2c encoding.
Confidence outputs are exported but are not explicitly supervised by this loss.
This is SEAR's adapter mechanism with a DA3-compatible loss, not an exact reproduction
of SEAR's complete training recipe. Gaussian and ray-pose inference are not trained.

## Training

From the repository root, using the existing `da3` environment:

```bash
TZ=Asia/Shanghai CUDA_VISIBLE_DEVICES=0 /home/lf/miniconda3/envs/da3/bin/torchrun \
  --standalone --nproc_per_node=1 scripts/train_thermal_giant.py
```

The same launch command is available as `bash scripts/run_thermal_giant.sh`.
The launcher defaults to GPU 0 and derives its process count from `CUDA_VISIBLE_DEVICES`.
The current local run uses tmux session `da3-sear-giant` and writes console output
to `artifacts/thermal_giant_sear/train.log`. Monitor it with:

```bash
tail -f artifacts/thermal_giant_sear/train.log
```

Default: 100 epochs maximum, AdamW at 5e-5, 10% linear warmup then constant LR,
weight decay 0.01, BF16 autocast, gradient clipping at 1.0. Stop after 20 epochs
without improvement in average mixed/thermal validation loss. A baseline evaluation
is recorded before training. RGB metrics are tracked separately. The baseline may
remain the best checkpoint if no trained checkpoint improves validation.

Artifacts in `artifacts/thermal_giant_sear/`:

- `data_manifest.json`, `run_config.json`: exact split and training settings.
- `metrics.jsonl`, `baseline_eval.json`, `latest_eval.json`: training and validation.
- `first_step_gradients.json`: per-adapter-tensor gradient norms.

Weights in `checkpoints/thermal_giant_sear/`:

- `last.pt`: optimizer, scheduler, adapter, RNG and batch position for resume.
- `adapter_last.pt`, `adapter_best.pt`: adapter-only exports for inference.

The trainer stores weights under `<repository>/checkpoints/<output directory name>`
by default, including when `--output` selects a smoke-test run. Use
`--checkpoint-dir` to choose another weight directory. Logs, metrics and inference
results stay in `artifacts/`.

Checkpoint writes are atomic. Save every 50 optimizer steps and every epoch.
Resume with the same resolution, views and epoch budget:

```bash
TZ=Asia/Shanghai CUDA_VISIBLE_DEVICES=0 /home/lf/miniconda3/envs/da3/bin/torchrun \
  --standalone --nproc_per_node=1 scripts/train_thermal_giant.py \
  --resume checkpoints/thermal_giant_sear/last.pt
```

Resume uses the current config's `patience` and restores the checkpoint's
consecutive non-improving epoch count. Raising patience from 10 to 20 therefore
allows 10 more epochs if validation never improves; an improvement resets the count.

The current run was switched from two GPUs to GPU 0 at checkpoint step 2950.
For a checkpoint saved with more GPUs, add `--allow-world-size-change` and increase
accumulation to preserve `world_size * accumulation`. The resume code maps the
consumed sampler prefix and requires unchanged optimizer steps per epoch, preserving
the LR schedule. It restores model, optimizer and scheduler states; dropout RNG
behavior is not bitwise identical after changing GPU count. The original two-GPU
checkpoint is retained as `checkpoints/thermal_giant_sear/last_before_gpu0_step_2950.pt`.

For a short full-Giant integration check, use a new output directory and
`--max-steps 2 --skip-initial-eval`. This verifies training execution, not convergence.
Do not use the public `DepthAnything3.forward()` for training: it disables gradients.
This trainer calls the underlying `DepthAnything3Net` directly.

## Inference

Create `input.json`, preserving the desired view order:

```json
[
  {"image": "/path/to/rgb.png", "thermal": false},
  {"image": "/path/to/thermal.png", "thermal": true}
]
```

```bash
/home/lf/miniconda3/envs/da3/bin/python scripts/infer_thermal_giant.py \
  --adapter checkpoints/thermal_giant_sear/adapter_best.pt \
  --input input.json --output artifacts/prediction.npz
```

Output contains depth, confidence, OpenCV w2c extrinsics, intrinsics at processed
resolution, modality mask and original image paths. Giant depth has arbitrary scale.
Use `--base-checkpoint` if pretrained weights were moved. In Python, `load_giant`,
`load_adapter` and `model(images, thermal_mask=mask)` provide the same path.

## Checks

```bash
PYTHONPATH=src /home/lf/miniconda3/envs/da3/bin/python -m unittest discover \
  -s tests -p 'test_thermal_adapter.py' -v
```

Tests cover initialization equivalence, adapter save/load, reference reordering,
LoRA and camera-token gradients with checkpointing, invalid masks/conditioning,
and consistent depth/pose scale and coordinate handling.
