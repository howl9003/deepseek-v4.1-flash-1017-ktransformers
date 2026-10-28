# Rebuilding the recorded candidate

This is the source recipe for the tested build, not a fresh binary qualification. Packaging performs no compilation or model run. Use an isolated environment and bounded build jobs; retain the original resource guards for subsequent validation.

The top-level recipe's tier2 setup provides the upstream dependency and base patch stack. Apply both optional policy patches, then the checked overlay as described in README. `dependencies/observed-pip-freeze.txt` records the tested environment; the source-patched SGLang and KT packages take precedence over stock wheels. The private renderer dependency was `deepseek-recipe==0.2.0`; its metadata and licenses are retained. Never install arbitrary newer versions and claim the same tested environment.

The native build uses the following CMake options, preserved verbatim apart from external paths:

```sh
cmake -S "$KT/kt-kernel" -B "$BUILD/kt" \
  -DCMAKE_BUILD_TYPE=Release -DKTRANSFORMERS_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCPUINFER_ENABLE_LTO=OFF -DKTRANSFORMERS_ENABLE_CPPTRACE=OFF \
  -DLLAMA_NATIVE=OFF -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_FMA=ON \
  -DLLAMA_F16C=ON -DLLAMA_AVX512=OFF -DLLAMA_AVX512_VNNI=OFF \
  -DLLAMA_AVX512_BF16=OFF -DLLAMA_AVX512_VBMI=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DKT_ENABLE_CONTRACT_V2=ON
cmake --build "$BUILD/kt" --target kt_kernel_ext -j 1
```

The CMake overlay retains `strict_activation.cpp` with `-fno-fast-math`, `-ffp-contract=off`, and `-fno-reciprocal-math`, plus the canonical-math and contract source changes. Do not replace these with global fast-math flags. The tested native extension hash was `271595dcc2f5b39fead6be72747af0b5daa248994dc934c5eee6eb91f63ea062`.

Route 3's component loader is `route3-mma-r1/phase2-cuda/route3_decode.py`. Its build uses `-O3`, `-gencode=arch=compute_89,code=sm_89`, `-fmad=false`, `--expt-relaxed-constexpr`, and host `-O2`. Set `ROUTE3_EXT_NAME=route3_phase2_l2p0_r1` and leave `ROUTE3_L2_POLICY=0` for the default candidate. CUDA's cuSPARSE headers came from the tested environment's `nvidia/cu13/include`; include that directory when the toolkit installation lacks those headers. The exact historical build script and receipt are retained under `validation/historical/` and `receipts/builds/`.

The model's `runtime/route3_decode.py` directly loads the already-built extension; it never invokes a compiler. It requires binary SHA256 `51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649` and records origins. This guard is deliberately preserved. A different rebuild must be independently qualified and admitted as a new candidate, not silently substituted. Default build11 SASS is included, SHA256 `c336cdda06154131fc2159f8259525331179ba5b48c36b0b6420dcb21bb41901`.

For the optional L2 component experiment, `ROUTE3_L2_POLICY=1` adds the cache-policy macro. It remains off in the two model profiles. Component bit identity was validated separately; this is not a model performance win. The archive retains the original r1 CUDA source and the tested r2 source with the switch.

The FP8 and SiLU binary tables in the contract overlay are deterministic numerical lookup tables, not model weights or checkpoint-derived tensors. Their hashes are checked by the NumPy oracle in `contract-reference/contract_v2.py`. The adapter's row-store library can be rebuilt from its retained `row_store.cpp`; the base recipe supplies kv-vmm, and `auxiliary/repack/` supplies the tested GGUF repacker source. Keep the immutable library hashes and origin checks until replacement builds have passed the relevant checks.

Numerical profile: four running/queued requests, optional admission and prefix-cache policy active, SWA prefill lending off. Fast profile: historical single-request/radix-disabled configuration, SWA prefill lending on. Both retain six compute nodes, 42 workers, fixed hot routing (`KT_HOT_DYNAMIC=0`), contract DP4A/AVX2, Route 3 policy0, and full decode graphs. Consult the full profile JSON rather than copying only these highlights.
