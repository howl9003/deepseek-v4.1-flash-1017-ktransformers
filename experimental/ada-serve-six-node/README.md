# Ada serving build: six-node tested snapshot

This optional source snapshot records the build served now: **C2 with vision, served from 4 November 2026.** C2 is the 1 November build below plus three numeric corrections towards DeepSeek's reference: FP4 compressed KV, V41 window KV and FP32 logits. Vision changes only the launch. **Breadth median 33.3 tok/s on the performance profile and 33.65 on the serving profile, on one RTX 4080 SUPER at 320 W, six NUMA nodes, 42 workers; 35.05 on all eight nodes.** The snapshot was first packaged for C1 on 2 November. C1's results follow below, and its profiles and receipts are in `profiles/c1-20261101/` and `receipts/c1-20261101/`.

It sits beside `experimental/contract-route3-six-node` and replaces none of it. The top-level recipe on this branch is still upstream `7e199f2` and is not the base of this snapshot: see [Base](#base).

## C2 with vision

- **Identity:** C2 was captured (adac2a, `ada-identity-c2-20261103-r1`) and repeated bit for bit on a second load (adac2b): 11,496 scores, both replays, the 4K and two 35K cases with top 8, and rotation. Against C1: paired NLL +0.0025 (SE 0.0036) on the 24 pieces, frozen quality screen PASS. `receipts/identity/`, `receipts/identity-repeat/`.
- **Vision:** text exact vs C2 on all 11,496 scores, the replays, the 4K and second 35K cases, rotation and a replay after an image (adac2w, `receipts/identity-vision/`). The first 35K case is skipped because its random prompt holds the image token 129264. The image test names both colours. `serving/results/VISION_SERVING_RESULTS_r1.md`.
- **Timing:** performance profile breadth 33.3 (33.35, 33.25), Japanese 35.3 and 36.1, 35K prefill 69.80 / 55.76 s then 52.36 / 51.15 s, `receipts/performance/`. Serving profile with vision 33.7 and 34.15; one, two and four sessions 32.1, 42.3 and 52.0 tok/s in total, `receipts/serving/`.
- **Gate:** KV checks pass, host reload equal with the frozen checker unchanged, disk tier bit-exact at 1K to 131K, `receipts/gate/`.
- **Panel:** KL 0.01019 against the reference's noise floor of 0.00996, top-1 96.98%, `serving/results/V41_WINDOW_RESULTS_r1.md`.

## C1 results, 1 November 2026

Two rounds, one load, six-workload breadth median as the rate.

| Workload | Round 0 tok/s | Round 1 tok/s |
|---|---:|---:|
| Chinese prose | 32.0 | 31.8 |
| English prose | 33.7 | 34.1 |
| English technical prose | 32.0 | 31.8 |
| math | 34.3 | 34.0 |
| JSON | 33.5 | 33.4 |
| Python code | 32.6 | 32.4 |
| **Breadth median** | **33.05** | **32.90** |
| Japanese prompt | 35.1 | 35.8 |
| Two 35K prefills, seconds | 69.50 / 55.46 | 51.93 / 50.80 |

- Identity: all 11,496 selected scores `EXACT` vs C1, plus the 4K and 35K long cases and the replays. `ada-identity-engram-20261101-r1`, `receipts/identity/`.
- Serving gate: frozen feature client, `kv_status` PASS, host-reload gate unchanged, SHA256 `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`. `ada-gate-engram-20261101-r1`, `receipts/gate/`.
- These are single matched campaigns, not a confidence interval.

**What the rate was measured on.** 32.98 comes from the performance profile: single request, radix off. The serving gate runs the same selectors with four running and queued requests, radix and host HiCache on, and proves behaviour, not speed. The serving profile was timed separately: breadth 33.15 on one session, 51.1 tok/s total on four. See `serving/`.

## What exact means here

Identity runs compare against our accepted build, C2 since 3 November and C1 before, never against DeepSeek's reference. Experts sit on the reference grid; dense FP8 is Route 3 with Ada truncation, bit-identical only to itself. Measured distance from DeepSeek's own inference code running on Ada, 24 pieces, 11,496 positions: the served build NLL +0.0034 (SE 0.0044), mean absolute change 0.130; the reference against itself with only MoE batching changed +0.0036 (SE 0.0041), 0.121. `results/reference-panel-r1/REFERENCE_PIECES_RESULTS_r1.md`. On sudoingX's public panel C1 scores KL 0.0111 and top-1 96.74%, 1.11x the reference's own noise floor: `serving/results/SERVED_PANEL_RESULTS_r1.md`. `PRECISION_LEDGER.md` lists every non-bit-identical step.

## Contents

- `SERVED_SELECTORS.md`: each selector, its value, what it does and what it gained.
- `overlays/`, `SOURCE_STACK.json`, `apply_overlay.py`: 82 SGLang files and 16 KTransformers files over the base. `SERVED_SGLANG_INVENTORY.json` holds the hash of all 6,176 files of the served tree, C2 with vision.
- `runtime/`: installed KT Python layer, Engram adapter with the batched `io_submit` row store, Route 3 and `wo_a` wrappers, path A expert and prefill wrappers.
- `native/`: sources of path A decode, expert and prefill kernels, and the `wo_a` CUDA kernel. Route 3's kernel is unchanged: `../contract-route3-six-node/route3-mma-r1/phase2-cuda`.
- `profiles/`: `*-as-run.json` holds the exact launch arguments and environment of C2's performance, identity, gate and vision loads, C1's are in `profiles/c1-20261101/`; `*-template.json` replaces host paths with the variables in `*-substitutions.json`.
- `tools/`: the chain that staged and ran these loads, including the panel capture chain. They are records of what ran on fea01, with its paths and guards in them.
- `results/`: write-ups, each selector's apply script and test, and the reference-panel tools. `validation/`: the frozen feature client and `kv_checks.py`.
- `receipts/`: run receipts of the three loads and native build results.

`EXCLUSIONS.json` lists what is left out: weights, captures, traces, logprob payloads, compiled binaries and the 35 MB gate report. `FORK_MANIFEST.json` hashes every file added on this branch.

## Base

Upstream SGLang `5d8e90a0f96e6ba35603f9b27f2f1039268c810e` with the tier 0, 1 and 2 patches of recipe commit `5d53f72741dac6cbabd975a823f8ab3ab40dc0df` in `05yuki/deepseek-v4.1-flash-1017-ktransformers`; KTransformers `95009ea6856c0799e517e93cb12be5e8494bc7ce` with that commit's kt-kernel patches; llama.cpp `a94e6ff8`, pybind11 `bb05e081`. The overlay is cumulative, so the two optional patches from the earlier snapshot are already inside it. `BUILD.md` has the steps.

## Tried and not served

- **Rotated slot copies** (`KT_KV_HOT_DYN_COPY_CHUNK=262144`): 26.4 tok/s, 20% slower on the server, although a six-part bench had shown +2% against +19%. Swaps and Engram timing did not change. The cause is not settled. Prefill did not move. Code is not included.
- **Frozen slot swaps** (`KT_KV_HOT_DYN_FREEZE_AFTER`): 31.53 tok/s, a diagnostic. Code is not included.
- **Path A dense and the single route** are off (`SGLANG_FP8_PATHA_DENSE`, `SGLANG_SINGLE_ROUTE`); the owner has parked them.

## Resource and publication constraints

Compute uses NUMA nodes 0, 1, 2, 4, 5, 6 and 42 workers. Nodes 3 and 7 are reserved for FEA CPU work; their RAM is shared, with a 64 GiB FEA reserve and a 416 GiB no-swap model cap. Loads require 480 GiB available, 51 GiB free or clean on every node, no other GPU compute and 320 W.

No upstream PR follows from this branch. PR6 stays held, and nothing here changes defaults in the top-level recipe.

## Attribution

Reference: DeepSeek's official DeepSeek-V4.1-Flash-1017 inference code and released arithmetic contract. This build stands on 05yuki's integration, including the KV lend of commit `7fef854e`, on KTransformers and on SGLang; see the branch's `CREDITS.md` and `licenses/` in the earlier snapshot. `results/reference-panel-r1/third_party` is sudoingX's `dgx-spark-deepseek-v41`, Apache-2.0, unmodified, with its licence file beside it.
