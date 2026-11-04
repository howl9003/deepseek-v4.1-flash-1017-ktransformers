#!/bin/bash
# test_draft_forward.py on fea01: two spare cores, nice 19, idle IO, 16 GiB cap, no swap, no GPU.
# Refuses to start while a timing run (ada8c2 / adacb2 controller) is running.
set -euo pipefail
R=/scratch/aster/bench/six-node-performance-r1/kernel-steer-r1/route3-mma-r1/dspark-cpu-r1
W=/scratch/aster/bench/dspark-cpu-r1
PY=/scratch/aster/bench/phase-g-g0-environment-20261030-r3/venv/bin/python
STAGE=${1:-0}
KT=${2:-over}        # kt_kernel tree: over (served native) or tol (apply_tolerant.py build)
TOL=${3:-}           # KT_CONTRACT_V2_TOLERANT_FROM, empty = unset
TAG=${4:-}
for u in phase-g-g1v-ada8c2-controller-r1 phase-g-g1v-adacb2-controller-r1; do
  [ "$(systemctl --user show $u -p SubState --value)" != running ] || { echo "timing run $u is running: not starting"; exit 2; }
done
mkdir -p $W/out/lib $W/oracle
[ -e $W/out/lib/cpu_pin_shim.so ] || /usr/bin/nice -n 19 gcc -O1 -shared -fPIC -o $W/out/lib/cpu_pin_shim.so $R/cpu_pin_shim.c -lnuma
[ -e $W/out/lib/librepack_avx2.so ] || cp /scratch/aster/bench/05yuki-baseline-3991/recipe_loader_repack_v2/librepack_avx2.so $W/out/lib/
exec systemd-run --user --collect --wait --pipe -p MemoryMax=16G -p MemorySwapMax=0 \
  /usr/bin/env -i HOME=$HOME PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  LD_PRELOAD=$W/out/lib/cpu_pin_shim.so \
  PYTHONPATH=$W/$KT:$W/over:/scratch/aster/bench/math-main-r8/runtime-deps ${TOL:+KT_CONTRACT_V2_TOLERANT_FROM=$TOL} \
  KT_MXFP4_BACKEND=contract_v2 KT_CONTRACT_V2=1 KT_CONTRACT_V2_MERGE_POOLS=all KT_CONTRACT_V2_LAZY_ROUTE_SCRATCH=1 \
  KT_CONTRACT_V2_QUANT_AVX2=1 KT_MXFP4_BATCH_INVARIANT=1 KT_MXFP4_SCALE_CODES=1 KT_MXFP4_DEC_VCONV=1 KT_MXFP4_DEC_PFS=1024 \
  KT_GGUF_FAST_HEADER=1 KT_SWIGLU_CLAMP=1 KT_TASKQUEUE_SPIN_US=2000 \
  KT_GGUF_REPACK_AVX2_LIBRARY=$W/out/lib/librepack_avx2.so \
  KT_GGUF_REPACK_AVX2_SHA256=079211bdc048127426fc829ee51f9f18d70881e0e76b908b4051ff9e0aa19658 \
  KT_MXFP4_GGUF_MANIFEST=/scratch/aster/bench/kt-native-pilot/evidence/expert-inventory/expert-ranges.json \
  KT_MXFP4_GGUF_MANIFEST_SHA256=9246e760c554a746662f3537cb2aadb626a90d827d2edbdda2e31438c7f30d12 \
  KT_DSPARK_CPU_DRAFT=1 KT_DSPARK_DRAFT_GGUF=/scratch/aster/models/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf \
  KT_DSPARK_DRAFT_MANIFEST=/scratch/aster/bench/kt-native-pilot/evidence/dspark-expert-inventory-v1/expert-ranges.json \
  KT_DSPARK_DRAFT_MANIFEST_SHA256=b75712873df442ab6047ee4eefad8b0c2b79e0ab3048f444d456fce77bc0fb7d \
  /usr/bin/taskset -c 39,47 /usr/bin/nice -n 19 /usr/bin/ionice -c3 \
  $PY -B $R/test_draft_forward.py $W/out/DRAFT_FORWARD_stage$STAGE$TAG.json $W/oracle $STAGE
