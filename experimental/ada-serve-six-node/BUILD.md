# Rebuilding C2

This is the source recipe for the served build, not a binary qualification: a rebuilt or relocated binary needs its own identity run against C2. Use an isolated environment.

## Sources

1. Check out `05yuki/deepseek-v4.1-flash-1017-ktransformers` at `5d53f72741dac6cbabd975a823f8ab3ab40dc0df` and let its tier 2 setup build SGLang `5d8e90a0f96e6ba35603f9b27f2f1039268c810e` and KTransformers `95009ea6856c0799e517e93cb12be5e8494bc7ce`. The recipe's two optional patches are not applied: the overlay already carries them.
2. Check, then apply:

```sh
python apply_overlay.py --sglang /path/to/sglang --ktransformers /path/to/ktransformers
python apply_overlay.py --sglang /path/to/sglang --ktransformers /path/to/ktransformers --apply
```

Every changed input and overlay file is checked against `SOURCE_STACK.json` before any write. After `--apply` the script compares all 6,176 files of the SGLang tree with `SERVED_SGLANG_INVENTORY.json`, the served tree. The recipe-patched base has 6,131 files, 6,094 equal the served tree, and the 82 overlay rows cover the other 37 and the 45 new files. The rows were taken from the served trial's own inventory, every overlay file hash-checked against it; they have not been re-applied to a fresh checkout.

3. Preserve the installed KT Python layer from `runtime/kt_kernel`, the Engram adapter from `runtime/engram-adapter`, and the wrappers in `runtime/`.

## Native builds

`receipts/builds/` holds the build receipts. `SERVED_NATIVE_INVENTORY.json` has the hash of every served package file. The compiled files themselves are not in this snapshot; `EXCLUSIONS.json` lists them.

KT, from the overlaid tree. The r16 build's cache records `CMAKE_CUDA_ARCHITECTURES=89;120a`, `KT_CUDA_STRICT_MATH=ON`, `KT_ENABLE_CONTRACT_V2=ON`, `CPUINFER_ENABLE_LTO=OFF`, `LLAMA_NATIVE=OFF`, `LLAMA_AVX512=OFF`:

```sh
cmake -S "$KT/kt-kernel" -B "$BUILD/kt" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DKTRANSFORMERS_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES='89;120a' -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCPUINFER_ENABLE_LTO=OFF -DKTRANSFORMERS_ENABLE_CPPTRACE=OFF \
  -DLLAMA_NATIVE=OFF -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_FMA=ON \
  -DLLAMA_F16C=ON -DLLAMA_AVX512=OFF -DLLAMA_AVX512_VNNI=OFF \
  -DLLAMA_AVX512_BF16=OFF -DLLAMA_AVX512_VBMI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKT_ENABLE_CONTRACT_V2=ON -DKT_CUDA_STRICT_MATH=ON
cmake --build "$BUILD/kt" --target kt_kernel_ext
```

The served extension is SHA256 `fd5072c3832a15abec609cbc238cfb127754131a257c0b7fcaca2900098e6751`.

Path A kernels, from `native/decode-gemv`, `native/expert-static-r3`, `native/prefill-imma`; the exact commands are in `receipts/builds/NATIVE_RESULT-g0.json`:

```sh
nvcc -c patha_expert_static.cu -o kernel.o -O3 -std=c++20 \
  -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_120a,code=sm_120a \
  -fmad=false --expt-relaxed-constexpr --compiler-options=-fPIC ...
g++ -shared kernel.o -ltorch ... -lcudart -o patha_expert_static_sm89_e0_s3_b8_t0.so
```

`wo_a`: `native/wo-a-cuda/build_wo_a.py`, sm_89 only, `-O3 -fmad=false`. Route 3: `native/route3/route3_decode.py` builds `route3_decode.cu` with torch's extension loader, `-O3 -gencode=arch=compute_89,code=sm_89 -fmad=false --expt-relaxed-constexpr`, host `-O2`, `ROUTE3_EXT_NAME=route3_phase2_l2p0_r1`, `ROUTE3_L2_POLICY=0`. `runtime/route3_decode.py` only loads the built extension and requires SHA256 `51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649`; another build has to be qualified as a new candidate.

GGUF repacker: `native/repack/`, the tested source behind `${REPACK_LIBRARY}` in the profiles.

Row store, as the recipe does it: `g++ -O2 -std=c++17 -shared -fPIC -pthread runtime/engram-adapter/row_store.cpp -o librow_store.so`.

## Environment

Python 3.12, torch 2.14.1 with CUDA 13 wheels, CUDA toolkit 13.3. `dependencies/observed-pip-freeze.txt` is the frozen list of the interpreter that served these loads.

## Launch

`profiles/*-as-run.json` hold the exact argument lists and environments; `*-template.json` replace host paths with the variables in `*-substitutions.json`. Set those and point the native paths at your rebuilt files. Environment and argument order are part of the record. `serving/` has the wrapper that serves the build on fea01 behind llama-swap.

## Bit identity

Identity holds within one recorded build. Our identity check compares 11,496 scored positions, two replays, a 4K and two 35K cases and a rotation against the C2 archive on fea01. Check a rebuild against your own accepted baseline the same way, and run `validation/kv_checks.py` unchanged through `validation/feature_client.py`.
