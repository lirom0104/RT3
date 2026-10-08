#!/usr/bin/env bash
# Pin all CUDA work to physical GPU 1. Within Python it is logical cuda:0.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rgbt_gpu_uuid="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)"
if [[ "$rgbt_gpu_uuid" != GPU-* || "$rgbt_gpu_uuid" == *$'\n'* ]]; then
    echo "Could not identify physical GPU 1; refusing to start." >&2
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$rgbt_gpu_uuid"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
echo "RGBT: physical GPU 1 ($rgbt_gpu_uuid), logical cuda:0" >&2
exec bash "$script_dir/rgbt_python.sh" "$@"
