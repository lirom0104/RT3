#!/usr/bin/env bash
# Use the installed da3 environment, including its newer libstdc++ for gsplat.
set -euo pipefail
RGBT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RGBT_ENV="${RGBT_ENV:-/home/lf/miniconda3/envs/da3}"
export PYTHONPATH="$RGBT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RGBT_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DA3_LOG_LEVEL="${DA3_LOG_LEVEL:-WARN}"
exec "$RGBT_ENV/bin/python" "$@"
