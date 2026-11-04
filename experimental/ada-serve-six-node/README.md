# Ada serving build: C2

The build served on fea01 since 4 November 2026: DeepSeek-V4.1-Flash-1017 on 05yuki's recipe `5d53f727` (SGLang `5d8e90a0`, KTransformers `95009ea`) with our overlay, on one modded 32 GB RTX 4080 SUPER at 320 W, 2 x EPYC 7543 and 512 GB DDR4-3200.

## What it is

- **Routed experts, integer-exact:** CPU and GPU compute each routed expert in the same integers on the reference's activation grid, with one rounding at the end, combined in logical expert order as the reference sums them. The bits do not depend on device, NUMA part count, batch shape or placement, so KV caching, sessions, hot-slot swaps and host reload replay exactly.
- **Dense, fixed sequence:** one fixed sequence of FP8 tensor-core instructions with the same K splits at decode and prefill, so decode and prefill share bits on Ada.
- **KV and logits on the reference's grid:** window KV in the V41 layout, compressed KV kept as FP4, FP32 logits.
- **Speed on top, bit-exact:** the selectors in `SERVED_SELECTORS.md`.
- **Vision:** the release's vision tower, loaded on demand.

`PRECISION_LEDGER.md` lists where C2 still differs from DeepSeek's reference. "Exact vs C2" means equal bit for bit to this build's archived outputs.

## Results

Six NUMA nodes and 42 workers unless stated. Breadth is the median of the six `tools/bench.py --other-text` workloads: Chinese prose, English prose, English technical prose, math, JSON and Python code.

| Measure | Value | Receipt |
|---|---|---|
| Breadth, bench profile | 33.3 tok/s (33.35, 33.25) | `receipts/performance/` |
| Breadth, all eight nodes | 35.05 tok/s in both rounds | `receipts/performance-eight/` |
| Breadth, served profile with vision | 33.7, 34.15 tok/s | `receipts/serving/` |
| One, two and four sessions | 32.1, 42.3, 52.0 tok/s in total | `receipts/serving/` |
| 35K prefill, fresh load then second round | 69.80 / 55.76 s, then 52.36 / 51.15 s | `receipts/performance/` |
| Identity | A second load equals the first bit for bit: 11,496 scores, two replays, the 4K and two 35K cases, rotation | `receipts/identity/`, `receipts/identity-repeat/` |
| Vision | Text exact vs C2 except prompts holding token 129264; the image test names both colours | `receipts/identity-vision/` |
| Serving gate | KV checks pass, host reload equal, disk tier bit-exact from 1K to 131K | `receipts/gate/` |
| Public panel against DeepSeek's code on this card | KL 0.01019 (the reference's own noise floor 0.00996), top-1 96.98% | `evaluation/panel/` |

## Contents

- `overlays/`, `SOURCE_STACK.json`, `apply_overlay.py`: 82 SGLang and 16 KTransformers files over the recipe base. `SERVED_SGLANG_INVENTORY.json` and `SERVED_NATIVE_INVENTORY.json` hash every served file.
- `runtime/`: the KT Python layer, the Engram adapter with batched reads, and the kernel wrappers.
- `native/`: kernel sources: integer-exact decode, expert and prefill kernels, `wo_a`, Route 3, and the GGUF repacker.
- `contract-reference/`: the integer contract's NumPy oracle and lookup tables.
- `profiles/`: the exact launches (`*-as-run.json`) of the bench, identity, serving-gate and served loads, host-free templates, expert placement, layer quota and the vision view.
- `receipts/`: results and resolved profiles of those loads, and the native build receipts.
- `validation/`: the frozen feature client and KV checker, and the FP4 and V41 KV kernel tests.
- `evaluation/panel/`: panel and reference scoring tools, with sudoingX's panel tools.
- `serving/`: the fea01 serve wrapper, model switch entry and reload watchdog.
- `candidates/`: tested, exact vs C2, not served.
- `BUILD.md`: how to rebuild. `licenses/`, `dependencies/`.

## Attribution

Reference: DeepSeek's official DeepSeek-V4.1-Flash-1017 inference code. This build stands on 05yuki's integration, including the KV lend of commit `7fef854e`, on KTransformers and on SGLang; see the branch's `CREDITS.md` and `licenses/`. `evaluation/panel/third_party` is sudoingX's `dgx-spark-deepseek-v41`, Apache-2.0, unmodified, with its licence file beside it.
