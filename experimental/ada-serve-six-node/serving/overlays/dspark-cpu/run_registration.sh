#!/bin/bash
# Seven runs of test_registration.py (served copy / overlay, selector off / on, target alone / target + draft; selector
# on with the apply_tolerant.py kt-kernel and KT_CONTRACT_V2_TOLERANT_FROM=40, and once with the served kt-kernel, which
# must refuse), then compare_registration.py. CPU only, two spare cores, only the private copies are imported.
set -euo pipefail
R=/scratch/aster/bench/six-node-performance-r1/kernel-steer-r1/route3-mma-r1/dspark-cpu-r1
W=/scratch/aster/bench/dspark-cpu-r1
PY=/scratch/aster/bench/phase-g-g0-environment-20261030-r3/venv/bin/python
OUT=$W/out/registration
mkdir -p $OUT
run() {  # tree selector scenario name [kt_kernel tree] [extra env]
  env -i ${6:-} HOME=$HOME PATH=/usr/bin:/bin CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 KT_CONTRACT_V2=1 \
    KT_DSPARK_CPU_DRAFT=$2 KT_NUMA_NODES=0,1,2,4,5,6 \
    KT_DSPARK_DRAFT_GGUF=/scratch/aster/models/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf \
    KT_DSPARK_DRAFT_MANIFEST=/scratch/aster/bench/kt-native-pilot/evidence/dspark-expert-inventory-v1/expert-ranges.json \
    KT_DSPARK_DRAFT_MANIFEST_SHA256=b75712873df442ab6047ee4eefad8b0c2b79e0ab3048f444d456fce77bc0fb7d \
    PYTHONPATH=$W/${5:-$1}:$W/$1:/scratch/aster/bench/math-main-r8/runtime-deps \
    timeout 900 /usr/bin/taskset -c 39,47 /usr/bin/nice -n 19 $PY -B $R/test_registration.py $OUT/$4.json $3 \
    > $OUT/$4.log 2>&1 || true
  tail -1 $OUT/$4.log
}
run base 0 target A_served_target
run base 0 target+draft B_served_target_draft
run over 0 target C_overlay_off_target
run over 0 target+draft D_overlay_off_target_draft
run over 1 target E_overlay_on_target tol KT_CONTRACT_V2_TOLERANT_FROM=40
run over 1 target+draft F_overlay_on_target_draft tol KT_CONTRACT_V2_TOLERANT_FROM=40
# fail closed: selector on with the served kt-kernel build (no tolerant mode) must refuse at draft construction
run over 1 target+draft G_overlay_on_served_native over
if grep -q "KT_DSPARK_CPU_DRAFT needs the apply_tolerant.py kt-kernel build" $OUT/G_overlay_on_served_native.log; then
  echo "PASS selector on with the served kt-kernel: the draft refuses at construction" | tee $OUT/G_RESULT.txt
else
  echo "FAIL selector on with the served kt-kernel did not refuse" | tee $OUT/G_RESULT.txt
fi
$PY -B -I $R/compare_registration.py $OUT | tee $OUT/COMPARE.txt
