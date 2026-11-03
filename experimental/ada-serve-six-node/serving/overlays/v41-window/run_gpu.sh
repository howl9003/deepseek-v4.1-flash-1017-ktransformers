#!/bin/bash
# V41 window KV GPU stages under the bench lock, on the GPU's node. argv: stages. No model load.
T=/scratch/aster/bench/v41-window-r1
PY=/scratch/aster/bench/phase-g-g0-environment-20261030-r3/venv/bin/python
cd "$T" || exit 2
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "GPU busy"; exit 3; }
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH
export PYTHONPATH=$T/src/python SGLANG_JIT_CACHE_DIR=$T/jit TRITON_CACHE_DIR=$T/.triton
# A private copy of the served files: nothing here may import from, or write under, a served trial.
export BASE_SGLANG=/scratch/aster/bench/fp4-kv-r1/base-served PYTHONDONTWRITEBYTECODE=1
export REF_KERNEL=/scratch/aster/models/DeepSeek-V4.1-Flash-1017/inference/kernel.py
export OMP_NUM_THREADS=4
exec flock -n /scratch/aster/bench/.bench.lock numactl --cpunodebind=6 --membind=6 "$PY" -B test_v41_kv_gpu.py "$@"
