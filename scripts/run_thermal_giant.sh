#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TZ=Asia/Shanghai
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a training_gpus <<< "$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec /home/lf/miniconda3/envs/da3/bin/torchrun \
  --standalone --nproc_per_node="${#training_gpus[@]}" scripts/train_thermal_giant.py "$@"
