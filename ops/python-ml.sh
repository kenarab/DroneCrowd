#!/usr/bin/env bash
set -euo pipefail
export TMPDIR=/home/alephbet/home.codex/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export npm_config_cache="$TMPDIR/npm-cache"
export PIP_CACHE_DIR="$TMPDIR/pip-cache" UV_CACHE_DIR="$TMPDIR/uv-cache"
export TORCH_HOME="$TMPDIR/dronecrowd-torch-cache"
export CUDA_CACHE_PATH="$TMPDIR/dronecrowd-cuda-cache"
export XDG_CACHE_HOME="$TMPDIR/dronecrowd-xdg-cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${DRONECROWD_PYTHON:-/home/alephbet/home.agents/venvs/ml/bin/python}" "$@"
