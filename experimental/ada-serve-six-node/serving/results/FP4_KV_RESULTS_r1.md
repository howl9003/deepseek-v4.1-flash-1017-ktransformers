# FP4 compressed KV on Ada: results, 2 Nov 2026

**The model run passes: paired NLL inside one standard error, decode flat, serving gate and disk tier pass. Speed and memory don't improve, but it is closer to the reference, so the owner chose to move the served build to it.** Score moved -0.00056 nats (SE 0.00084), breadth 33.05 against 32.98, prefill +0.8%. The memory gain is 0.71 GiB, and it does not reach the experts: the KV lend was already borrowing those pages. Selector `SGLANG_SM89_FP4_COMPRESSED_KV`, default off until the cutover. Scope: `FP4_KV_COMPONENT_SCOPE_r1.md`. Plan: `FP4_BASELINE_MOVE_PLAN_r1.md`.

## Model run

Six nodes, 320 W, FEA job 1716 on its own nodes, one load per stage, admission 448 GiB. Trials `ada-panel-fp4kv-20261102-r2` (panel), `-r3` (decode check), `ada-perf-fp4kv-20261102-r2` (timing), `ada-gate-fp4kv-20261102-r2` (gate). Baseline is the 32.98 build. Chains adafp1 to adafp4.

| Panel, 254,846 positions | KL | KL p99 | Top-1 | NLL |
|---|---:|---:|---:|---:|
| Reference on Ada | | | | 0.47077 |
| Reference, MoE batch 2,048 (noise floor) | 0.00996 | 0.123 | 97.01% | 0.47105 |
| Served 32.98 build | 0.01106 | 0.137 | 96.74% | 0.47150 |
| **FP4 compressed KV** | **0.01098** | **0.137** | **96.73%** | **0.47116** |

Paired against the served capture by conversation, 197 conversations: **-0.00056 nats, SE 0.00084, within one SE**. 214,827 positions changed, 96,318 of them by more than 0.01 nats.

| Domain, KL / top-1 | Served | FP4 |
|---|---:|---:|
| code | 0.0077 / 97.15% | 0.0077 / 97.16% |
| agentic | 0.0137 / 96.43% | 0.0131 / 96.37% |
| chat | 0.0246 / 94.95% | 0.0242 / 94.93% |
| knowledge | 0.0181 / 95.30% | 0.0192 / 95.38% |
| long | 0.0158 / 96.78% | 0.0155 / 96.67% |
| math | 0.0087 / 96.74% | 0.0083 / 96.52% |
| multilingual | 0.0167 / 96.68% | 0.0166 / 96.53% |
| writing | 0.0247 / 94.74% | 0.0236 / 94.88% |

| Timing | 32.98 build | FP4 | Change |
|---|---:|---:|---:|
| Breadth, two rounds | 32.98 (33.05, 32.90) | 33.05 (33.0, 33.1) | Inside noise |
| Japanese | 35.1, 35.8 | 35.3, 35.9 | Inside noise |
| 35K prefill, four prompts | 69.50, 55.46, 51.93, 50.80 s | 69.84, 55.76, 52.57, 51.29 s | +0.5, +0.5, +1.2, +1.0%; +0.8% in total |
| Cold prefill in the gate, 8K | 23.26 s | 23.42 s | +0.7% |
| Cold prefill in the gate, 41K | 73.69 s | 73.95 s | +0.4% |
| Cold prefill in the gate, 131K | 231.07 s | 232.55 s | +0.6% |

Prefill is within the +1% rule overall, and at the limit on the two steady 35K prompts. The estimate was +0.6%.

**Gate:** KV checks pass, all replay ids equal, host reload equal with the frozen checker's hash unchanged. Disk tier bit-exact (ids, logprobs, top lists) at 1K, 8K, 41K and 131K, 1.9 to 4.5 s against 17 to 233 s cold, all within 0.1 s of the served build's disk reads. A restart from disk was not run. KV lend ran on the new page size in all four loads with no error.

**Decode against prefill:** eight conversations at 256 tokens and eight at 1,024, 32 tokens each. All 512 positions agree within 4.3e-7 nats, about one float32 step of the logprob, and 261 are bit-equal. So the panel's prefill score applies to decode here.

**Memory:** after the pool is built, free VRAM goes from 17.46 to 18.17 GB, so 0.71 GB at the 1,048,576-token cap these runs use. Node minima did not move. The first lend then finds 1.45 + 16.37 = 17.82 GiB on the 32.98 build and 0.78 + 17.08 = 17.86 GiB here: 1,006 slots against 1,008. The empty pool's pages were already going to experts through the lend, so freeing them adds nothing while the pool is mostly empty. The 2 GB estimate below was for an uncapped pool and assumed nothing was borrowing it. Uncapped capacity does rise, 8.73M to 16.01M tokens.

**The large changes are not FP4 faults.** Two positions moved a long way from both the reference and the served build: conversation 76 token 101 (-0.02 to -11.4) and conversation 136 token 64 (0 to -8.7). Both came out context-free: the top tokens are `<`, newline, `#` and `##` in nearly the same proportions wherever it happens. The reference has 3 such rows on the panel, the served build 3 and FP4 5, all between token positions 47 and 104, and two of the rows are common to all three. Repeating seven conversations, these included, matched the saved capture bit for bit. So these rows are deterministic and a property of the model at early positions. Five against three is too few to call a difference. `blank_rows.py`, `kl_outliers.py` and `outlier_panel.py` in `reference-panel-r1/`.

## What moving costs

I first left this off, for no measured gain. The owner's point stands: it is closer to the reference by design, and fidelity is one of the project's criteria.

- Score, decode rate and expert slots do not measurably move, and prefill loses 0.8%.
- It ends exact vs C1, which every gate and identity run compares against, so it needs a new baseline.
- Disk KV written by the V4 build cannot be read by it, so the store starts empty.

It removes a rounding we added ourselves in #3 and the Triton fake-quant's two RoPE flips in 151 million values.

**Not the only KV difference.** The window KV also sits off the reference's grid on all 40 layers. That fix is in `../v41-window-r1/` and implies this one.

## Kernel tests

Tree `/scratch/aster/bench/fp4-kv-r1/src`, a copy of the served tree with this overlay. GPU runs were on fea01's 4080 SUPER with the server unloaded.

**Correction:** the off-arm test imported the served files directly, and Python left three `.pyc` files in `ada-serve-20261102-r4` at 17:15. The provenance guard then refused the reload. The other session removed them and the tree matches its pins again. No source file changed. `run_gpu.sh` now runs with no bytecode and compares against a private copy of those files.

| Test | Result |
|---|---:|
| A. Addresses at P = 1, 2, 16, 64, 128, selector on and off, SM120 refused | Pass |
| A. Every E2M1 x E4M3 product exact in BF16 | 0 inexact |
| B. Real CUDA store against the pure-torch step | 0 of 151,011,328 values |
| C. Loader against the `V41_FP4` dequant | 0 of 4,521,984 values |
| D. FP4 attention against a twin reading the same values as BF16 rows, 1 to 256 rows, with NaN inputs | 0 of 3,000 calls |
| E. Selector-off kernels against the served files | 0 of 600 calls |
| F. Spills, staged and unsplit kernels | 0. Unsplit 166-172 registers, staged 37-113 |

**Test B found one thing worth knowing.** The store and the existing Triton fake-quant differ on 2 of 151 million values, both in the RoPE tail. The store equals the pure-torch step on all of them, so the Triton fake-quant is the one that is off. So this path also removes those rare flips. They also differ in the sign of zero on 3,584 values; the numeric value is the same.

**One bug the GPU found.** Module-level kernel constants must be `tl.constexpr`. The Triton interpreter did not enforce that, so the CPU tests missed it.

## Price

All on the same 4080 SUPER, `ada`-style graph replay for decode, 40 calls with random indices, median of 30 replays. Eight layers of 40 use this cache.

| Decode, main + extra top-k | Rows | V4, us | FP4, us | Eight layers, ms a token |
|---|---:|---:|---:|---:|
| 512 + 64 | 1 | 20.5 | 20.2 | -0.003 |
| 768 + 512 | 1 | 37.5 | 31.6 | -0.047 |
| 1024 + 512 | 1 | 45.1 | 39.4 | -0.045 |
| 768 + 512 | 4 | 57.7 | 51.4 | -0.051 |
| 768 + 512 | 8 | 106.0 | 89.0 | -0.136 |
| 512 + 64 | 8 | 47.6 | 51.0 | +0.027 |

Decode is flat or slightly faster at the real 512 extra top-k. The worst case, small extra at 8 rows, is +0.035 ms.

| Write, one layer | FP4 | V4 path (fake-quant, then store) |
|---|---:|---:|
| 1 row | 20.7 us | 67.9 us |
| 35,840 rows | 83.8 us | 219.3 us |

That is one launch instead of two, measured outside a graph with a sync per call. So it is an upper bound on the saving, not a token figure.

**Prefill:** the attention call itself is slower, 1.28 to 1.55 ms at 256 rows and 4.85 to 5.86 ms at 1,024 rows, about +20%. At 35K that is about +0.28 s on eight layers, +0.6% of 50.8 s [estimate, assumes every row reads the full extra top-k]. Within the +1% rule, but the kernel is 20% slower, so a model run should confirm it.

## Keep rule

| Rule | Limit | Result |
|---|---:|---:|
| Tests A to F | 0 mismatches, no spills | Pass |
| Decode | +0.3 ms a token | Pass, breadth flat on the model |
| Prefill at 35K | +1% | +0.8% in total on the model, +1.2% and +1.0% on the two steady prompts |
| Paired NLL | Within one SE | Pass, -0.00056 (SE 0.00084) |

## Memory estimate before the run

3.98 to 1.96 GB at a full pool, 2.02 GB freed, 6,815,770 tokens x 296 B [estimate, assumes the extra pools hold the same token counts as the index capacities]. Whether that became about 91 more expert slots depended on the KV lend. It became 2: see Memory above.

## Not tested

- Restart from disk, and the 4K prefill case.
- Serving profile with four sessions. The gate covers the disk tier and KV checks, not the dsh sessions.
- Host KV under a real long session mix.

## Precision ledger

Entered in `PRECISION_LEDGER.md` as tested, not served.

Logs: `run3.log` to `run6.log` in `/scratch/aster/bench/fp4-kv-r1`. Overlay: `apply_fp4_kv.py`, `sm89_fp4_tile.py`.
