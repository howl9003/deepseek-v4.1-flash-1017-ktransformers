# Recipe book

## Assemble the software

Start from Linux x86-64 with Python 3.12, a CUDA-enabled PyTorch installation and a filesystem supporting direct I/O. The measured environment used PyTorch 2.14.1+cu130. The public recipe's setup provides the model loader, Engram adapter, patched kTransformers base, placement file and KV virtual-memory library. The kt release branch is a source overlay for that base.

Use fresh paths. Set `RECIPE`, `SGLANG` and `KT` to the three local release checkouts, `C4_ROOT` to a new installation directory, and `C4_INPUT` to the directory for qualified inputs. The installation uses 05yuki's public recipe at `27ee782` for GGUF setup, the matching dependencies and the patched kTransformers sources. The release checkout supplies the complete C4 SGLang tree after setup. Set `C4_BASE_RECIPE` to a local mirror if needed. Supply the 1017 model view with its Engram checkpoint, matching tokenizer and MXFP4 GGUF. Use the base's matching prebuilt archive or unpacked folder for its MMQ kernel and shared libraries. Set `BASE_PREBUILT` to this input and `WHEEL` to the qualified kt wheel. The build step installs the GGUF fallback reader alongside the direct image reader.

```sh
mkdir -p "$C4_ROOT" "$C4_INPUT"
git clone --no-checkout "${C4_BASE_RECIPE:-https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers.git}" "$C4_ROOT/tree"
git -C "$C4_ROOT/tree" checkout --detach 27ee782c15a9ff7807dc3d52d6f0108dc9fb445e
(cd "$C4_ROOT/tree" && PREBUILT="$BASE_PREBUILT" TIER=2 MODEL="$MODEL" GGUF="$GGUF_DIR" bash setup.sh)
if [ "$(basename "$MODEL")" != checkpoint ]; then
  ln -s "engram-$(basename "$MODEL")" "$C4_ROOT/tree/state/engram-checkpoint"
fi
test ! -e "$C4_ROOT/tree/third_party/sglang-base" &&
  mv "$C4_ROOT/tree/third_party/sglang" "$C4_ROOT/tree/third_party/sglang-base"
git clone --local --branch main "$SGLANG" "$C4_ROOT/tree/third_party/sglang"
MMQ=python/sglang/srt/layers/moe/kt_gguf_moe_mmq/llama_mmq
cp -R "$C4_ROOT/tree/third_party/sglang-base/$MMQ/ggml" "$C4_ROOT/tree/third_party/sglang/$MMQ/"
PY="$C4_ROOT/tree/.venv/bin/python"
"$PY" -m pip install --no-deps --no-build-isolation -e "$C4_ROOT/tree/third_party/sglang/python"
cd "$RECIPE"
python3 -I -B preflight.py "$C4_ROOT/tree/third_party/sglang" "$KT"
```

### Build and qualify the native inputs

Use CUDA 13.4 for the SM120 shared libraries. The native kt builder uses the AVX2 variant, strict FP options, the base's patched kTransformers `95009ea`, and the 0.7.0 CPython 3.12 Linux wheel whose SHA-256 begins `5b5141c8`. Set `WHEEL` to that file. Supply the pinned 40-layer GGUF range inventory as `$C4_INPUT/expert-ranges.json` (SHA-256 `9246e760c554a746662f3537cb2aadb626a90d827d2edbdda2e31438c7f30d12`) and the qualified AVX2 repacker as `$C4_INPUT/librepack_avx2.so` (digest in `profile.json`). The commands below regenerate both from the target GGUF and the included AVX2 source. Install the Python gguf reader and NumPy in the build environment.

```sh
export C4_INPUT IMG_GGUF
# IMG_GGUF is the full GGUF file. The inventory output directory must be new.
"$PY" -B native/inventory.py --source "$IMG_GGUF" --output "$C4_ROOT/inventory"
cp "$C4_ROOT/inventory/expert-ranges.json" "$C4_INPUT/"
g++ -std=c++17 -O3 -mavx2 -fPIC -shared -Wall -Wextra -Werror -fno-exceptions -fno-rtti \
    native/repack_avx2.cpp -o "$C4_INPUT/librepack_avx2.so"
cp inputs/bl_table.json "$C4_INPUT/"
"$PY" -B native/inputs.py kernels "$C4_INPUT"
"$PY" -I -B native/build_kt.py "$KT" "$C4_ROOT" "$WHEEL" "$C4_INPUT" --cuda "$CUDA_HOME"
```

The kernel step reconstructs all 65,536 FP32 SiLU entries from BF16 input bit patterns on CUDA and checks the measured digest. The kt step installs the recipe's host adapter with the single rebuilt extension and both image hooks. Build pins include a disassembly check and the exact dense binary. If the local toolchain produces a different dense binary, the measured `libv3dense.so` is an external input.

Set `IMG_GGUF` to the GGUF **file** and `C4_TRACES` to the reference activation fixtures for layers 3, 20 and 39. The component reader accepts `cpu-trace-136-64`, `cpu-trace-152-61-fp32` and `cpu-trace-157-47` as safetensors with their routing JSON sidecars. Use `nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv` to identify `GPU0` and `GPU1`.

```sh
export IMG_GGUF C4_TRACES
export PYTHONPATH="$C4_ROOT/tree/third_party/sglang/python:$C4_ROOT/tree/engram-adapter:$C4_ROOT/kt-pkg"
for device in "$GPU0" "$GPU1"; do
  CUDA_VISIBLE_DEVICES="$device" "$PY" -B native/review/qualify_gpu.py "$C4_INPUT" "$C4_INPUT/libv3gpu_r3i.so" || exit 1
done
"$PY" -I -B native/install_pins.py "$C4_ROOT" "$C4_INPUT" "$GPU0" "$GPU1"
```

The review compares each TP half, the whole-expert sum and final BF16 outputs against the CPU oracle at multiple row counts. The resulting receipts are device-specific. Keep the input directory and the installed model tree together when reproducing a result.

### Prepare the expert image

The image writer verifies source ranges against the inventory as it reads the GGUF. Set `IMAGE` to a fresh directory with about 269 GiB for the 40 layers plus at least 32 GiB of free reserve. The final command reconstructs every original expert range from the image and checks it against the pinned digest before writing the local format proof. This gives a new installation a proof bound to its own manifest digest.

```sh
export IMG_RANGES="$C4_INPUT/expert-ranges.json"
mkdir -p "$IMAGE" "$C4_ROOT/image-report"
"$PY" -B native/img_build.py space "$IMAGE"
"$PY" -B native/img_build.py build "$C4_ROOT/image-report" "$IMAGE" all 21600
"$PY" -B native/img_build.py finish "$IMAGE"
"$PY" -B native/img_build.py verify "$C4_ROOT/image-report" "$IMAGE"
"$PY" -B native/inputs.py image-proof "$C4_INPUT" "$IMAGE" "$C4_ROOT/native-build"
```

Use a fresh runtime directory for this profile. The extension prebuild and live process must use the same installation paths, toolkit, visible devices and compiler environment. The prebuild runs the tree's own extension builders twice, checks that Ninja has no work on the second pass, and records the build files, binaries and cache identity. The launch reads that receipt. Run `verify` after the server has loaded to check the actual extension state. The default HTTP port is 18209. The TP broadcast queue starts at 18210.

The model, GGUF, base prebuilt artifacts, reference wheel, device traces, panel tensors, reference probability parts and paired capture binaries are separate inputs. Only file digests, numeric reference values, scripts and code are included here.

## Refit the hardware constants

| Constant on the test server | Measurement on a new host | How the model uses it |
|---|---|---|
| 2 × RTX 5070 Ti, 16 GB per card, PCIe 4.0 ×16 on different sockets, NUMA nodes 1 and 6 | Use `nvidia-smi topo -m`, `lspci -vv` and `numactl --hardware`. Trace simultaneous pinned host-to-device transfers on both ranks. The reference sustained 27.6 GB/s per card. | Divide the actual staged bytes by concurrent bandwidth to obtain the prefill upload floor. |
| 2 × EPYC 7543, 64 physical cores total, eight NUMA parts, 16 × 32 GB DDR4 RDIMMs | Record CPU, NUMA node and memory-channel mapping. Measure CPU queue wait and run time separately for 1, 2 and 4 rows under serving traffic. | Size the expert split by the slowest part. Apply the table factor only to exposed CPU work. |
| 40 layers, six routes per row, 9,400,320 bytes per TP half-slot | Record free memory after capture, actual lent bytes and number of slots per layer. Count routed hits after admission. | Fit GPU share at each slot cap and turn unsatisfied routes into CPU time. |
| 130 ms upload floor per full sweep, 152 ms device sweep at 12,288 rows | Timestamp the first and last group on the GPU. Record planner width, backlog, late-tail eligibility and selected CPU share on both ranks. | Price the actual chunks, then measure first-token time for solo, pair and four-way traffic. |
| P2 bank upper critical threshold 85 °C on this machine | Collect DIMM sensors and assertion/deassertion events through the full soak at matched prompt length and concurrency. | Use table B as the temporary thermal setting. Recheck the equal split after changes to cooling. |

Keep throughput, cold latency and quality measurements separate. Collect per-request output counts and actual cache hits, and use enough successive decode reports to exclude the first token from a decode-only fit. Table B's 74.47 tok/s four-session soak and table A's 48.42 tok/s single-session soak have different workloads. Use the 1/2/4-session measurements only at their documented concurrency and temperature.
