# Experiment B: thermal-only LoRA

This experiment starts from the same frozen DA3-GIANT-1.1 weights as the shared
LoRA baseline. It initializes new LoRA matrices (zero B matrices) and copies the
original camera tokens into trainable thermal camera tokens. It does not load a
previously trained shared adapter.

`AdapterConfig.lora_scope="thermal_only"` applies the low-rank residual only to
tokens belonging to thermal views, including their camera tokens. The scope
covers qkv, attention output projection, and both SwiGLU projections in blocks
13 through 39. RGB and thermal views still participate in the same global
attention. RGB predictions in mixed inputs may therefore change; RGB-only
inference follows the frozen RGB path. Both RGB and thermal geometry losses
remain active during joint training.

Routing masks follow reference-view reordering and local/global attention
reshaping. Each mask is passed explicitly into activation checkpointing for
backward recomputation. Shared adapters and older checkpoints without
`lora_scope` retain the original shared behavior. A resume cannot switch scopes.

## Configuration and launch

`configs/thermal_giant_sear_ir_only_20261015.json` keeps the original
`thermal_giant_sear.json` data split and training hyperparameters. The only
differences are the LoRA scope and experiment output/checkpoint paths:

- SEAR dataset and original scene split; no new label repair or detail losses.
- Eight views at 392 by 518 pixels, rank 64, alpha 128, dropout 0.1.
- Learning rate 5e-5, AdamW, 10% warmup, at most 100 epochs, patience 20.
- One GPU and accumulation two: 16 input views per optimizer step, matching the
  original two-GPU baseline with accumulation one. The sampling and dropout RNG
  streams are not bitwise identical across those GPU layouts.
- The checkpoint selection score remains the average mixed/thermal geometry
  loss. RGB validation is also reported.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4 bash scripts/run_thermal_giant.sh \
  --config configs/thermal_giant_sear_ir_only_20261015.json
```

Weights are saved under `checkpoints/thermal_giant_sear_ir_only_20261015/`.
Configuration, scene manifest, baseline evaluation, metrics and launch logs are
under `artifacts/thermal_giant_sear_ir_only_20261015/`. `adapter_best.pt` may
remain the untrained baseline if training does not improve validation.

To resume, append:

```bash
--resume checkpoints/thermal_giant_sear_ir_only_20261015/last.pt
```

Existing inference scripts load the scope from the adapter checkpoint. They
must supply an accurate `thermal_mask` for thermal views; an omitted mask means
all views are RGB.

## Verification

```bash
PYTHONPATH=src /home/lf/miniconda3/envs/da3/bin/python -m unittest discover \
  -s tests -p 'test_*.py'
```

The routing tests exercise RGB-only preservation with nonzero trained updates,
reordered local/global masks at all four projections, checkpointed gradients
across overlapping forwards with different masks, RGB-loss gradients reaching
thermal adapters through attention, and adapter round-trip/legacy loading.

A two-step full-Giant smoke run uses separate output and checkpoint directories
named `thermal_giant_sear_ir_only_smoke_20261015`, with `--max-steps 2` and
`--skip-initial-eval`. This checks execution and finite gradients, not model
quality. Compare completed models on RGB-only, thermal-only, well-lit mixed,
and low-light mixed inputs using identical preprocessing and view selection;
report RGB boundaries, depth, poses and cross-modal geometric consistency.
