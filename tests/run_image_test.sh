#!/bin/bash
# pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3): host-only test of operators/avx2/kt_image.hpp (the image
# reader of load_weights_image). Makes a synthetic source and image and the expected arena bytes with the given vector
# maker (concept-r1/trials/queue/staged-ns2/mk_image_vectors.py, which builds the image with staged-ns1/img_build.py),
# compiles tests/test_kt_image.cpp with the host's g++ and runs it. CPU only, about a minute.
#   bash tests/run_image_test.sh <out dir> <python> <mk_image_vectors.py>   (writes <out>/test_kt_image.json; exit 0 on PASS)
set -euo pipefail
OUT=$1
PYB=$2
MK=$3
SRCDIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
sha256sum "$SRCDIR"/operators/avx2/kt_image.hpp "$SRCDIR"/tests/test_kt_image.cpp "$MK" > "$OUT/inputs.sha256"
"$PYB" -B "$MK" "$OUT/vec" > "$OUT/vectors.log" 2>&1 || { tail -20 "$OUT/vectors.log"; exit 3; }
g++ --version | head -1 > "$OUT/compiler.txt"
g++ -O2 -std=c++20 -pthread -Wall -Wextra "$SRCDIR/tests/test_kt_image.cpp" -o "$OUT/test_kt_image" 2> "$OUT/compile.log" \
  || { cat "$OUT/compile.log"; exit 2; }
set +e
"$OUT/test_kt_image" "$OUT/vec" > "$OUT/test_kt_image.json"
rc=$?
set -e
cat "$OUT/test_kt_image.json"
exit $rc
