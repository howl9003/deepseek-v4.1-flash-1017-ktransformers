#!/bin/bash
# Path B poller detector and pairing fix, host-only test (tests/test_memop_detect.cpp): build with the host's g++ against
# this directory's cpu_backend (memop_detect.h, task_queue.h, task_queue.cpp) and run it. CPU only, about 1 min, three
# threads: plain / detect / forced (the detector's runs, unchanged) plus the KT_MEMOP_PAIRED runs on graph and ring slots.
#   bash tests/run_test.sh <out dir> [<module .so>]   (writes <out>/test_memop_detect.json; exit 0 only on PASS)
# With a module: its sha256 goes to <out>/module.sha256, naming the build these sources went into (the gate checks both).
set -euo pipefail
OUT=$1
MOD=${2:-}
SRCDIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
sha256sum "$SRCDIR"/cpu_backend/memop_detect.h "$SRCDIR"/cpu_backend/task_queue.h "$SRCDIR"/cpu_backend/task_queue.cpp \
  "$SRCDIR"/tests/test_memop_detect.cpp "$SRCDIR"/cpu_backend/cpuinfer.h > "$OUT/inputs.sha256"
if [ -n "$MOD" ]; then sha256sum "$MOD" > "$OUT/module.sha256"; fi
g++ --version | head -1 > "$OUT/compiler.txt"
g++ -O2 -std=c++17 -pthread -Wall -I"$SRCDIR/cpu_backend" "$SRCDIR/tests/test_memop_detect.cpp" "$SRCDIR/cpu_backend/task_queue.cpp" \
  -o "$OUT/test_memop_detect" 2> "$OUT/compile.log" || { cat "$OUT/compile.log"; exit 2; }
set +e
"$OUT/test_memop_detect" > "$OUT/test_memop_detect.json"
rc=$?
set -e
cat "$OUT/test_memop_detect.json"
exit $rc
