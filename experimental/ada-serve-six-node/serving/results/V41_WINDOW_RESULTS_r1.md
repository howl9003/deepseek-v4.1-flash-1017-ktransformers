# V41 window KV on the served route: results, 3 Nov 2026

**With the FP4 compressed KV and FP32 logits, the served build now sits within 0.0002 KL of the reference's own noise floor: KL 0.01019 against 0.01106 served and a floor of 0.00996. Top-1 is 96.98% against the floor's 97.01%.** About 79% of the gap above the floor is gone. It is also slightly faster, breadth 33.3 against 32.98, and the serving gate and disk tier pass. Prefill costs +0.6%. Selector `SGLANG_SM89_V41_KV` plus `--enable-fp32-lm-head`, default off until the C2 cutover; the selector implies the FP4 compressed caches.

## Timing and gate

Trials `ada-perf-v41kv-20261103-r2` (adavr2) and `ada-gate-v41kv-20261103-r2` (adavr3), six nodes, 320 W, one load each.

| Measure | 32.98 build | FP4 only | V41 + FP4 + FP32 head |
|---|---:|---:|---:|
| Breadth, two rounds | 32.98 (33.05, 32.90) | 33.05 (33.0, 33.1) | **33.3 (33.35, 33.25)** |
| Japanese | 35.1, 35.8 | 35.3, 35.9 | 35.3, 36.1 |
| 35K prefill, four prompts | 69.50, 55.46, 51.93, 50.80 s | 69.84, 55.76, 52.57, 51.29 s | 69.80, 55.76, 52.36, 51.15 s |
| 35K prefill in total | | +0.8% | **+0.6%** |
| Cold prefill in the gate, 8K / 41K / 131K | 23.26 / 73.69 / 231.07 s | 23.42 / 73.95 / 232.55 s | 23.46 / 73.75 / 232.53 s |

Breadth is up about 1%, which fits the head's 0.24 ms and the window reader's few hundredths. The window writer's saving pulls prefill back a little from FP4's +0.8%. The two steady 35K prompts are +0.8% and +0.7%, inside the +1% rule.

**Gate:** KV checks pass, all replay ids equal, host reload equal, frozen checker unchanged. Disk tier bit-exact (ids, logprobs, top lists) at 1K, 8K, 41K and 131K, 1.9 to 4.4 s against 17 to 233 s cold. Lowest node in the three runs 9.5 GiB (node 7), above the 4 GiB guard. A restart from disk and four dsh sessions were not run; they belong to the serving-profile step of the C2 plan.

## C2: the new baseline, 3 Nov 2026

**C2 is this build, and it reproduces.** A first load recorded it, and a second load matched it bit for bit: all 11,496 scored positions, both replays, the 4K and two 35K cases with their top 8, and the rotation check. Chains adac2a and adac2b, trials `ada-identity-c2-20261103-r1` and `-r2`.

| Check against C1 | Result |
|---|---|
| Replays | same tokens and text; logprobs differ, as expected |
| 24 pieces, paired NLL | +0.0025 (SE 0.0036), within one SE: passes the rule |
| Frozen quality screen, 12,276 positions | PASS, NLL 0.887, limit +0.02 |

**Serving profile on C2** (adasfp, trial `ada-serve-c2-20261103-r1`, new disk store `ada-kv-disk-c2-20261103-r1`):

| Measure | Served build (adasrv) | C2 |
|---|---:|---:|
| Breadth, two rounds | 33.15 (32.65, 33.65) | **33.65 (33.2, 34.1)** |
| Sessions 1 / 2 / 4, total tok/s | 31.0 / 41.7 / 51.1 | 31.8 / 41.7 / 51.5 |
| Cold 35K prefill, first and second prompt | 78.03, 61.06 s | 75.99, **74.47 s** |

The second cold 35K prompt took 22% longer, with nothing cached in either run. The performance profile shows no such cost (+0.6%), so it is the serving profile's. My guess is the new disk store writing while the Engram tables read from the same NVMe, but that is one sample and not measured.


Six nodes, 320 W, one load, admission 448 GiB. Trial `ada-panel-v41kv-20261103-r2`, chain adavr1. The head-only arm is the engine audit session's `adahd1`: the 32.98 build plus `--enable-fp32-lm-head`, trial `ada-panel-fp32head-20261103-r1`.

| Panel, 254,846 positions | KL | KL p99 | Top-1 | NLL |
|---|---:|---:|---:|---:|
| Reference on Ada | | | | 0.47077 |
| Reference, MoE batch 2,048 (noise floor) | 0.00996 | 0.123 | 97.01% | 0.47105 |
| Served 32.98 build | 0.01106 | 0.137 | 96.74% | 0.47150 |
| FP4 compressed KV only | 0.01098 | 0.137 | 96.73% | 0.47116 |
| FP32 head only | 0.01072 | 0.136 | 96.91% | 0.47124 |
| **V41 window + FP4 + FP32 head** | **0.01019** | **0.129** | **96.98%** | **0.47126** |

- **KL above the floor:** 0.00110 served, 0.00076 with the FP32 head, 0.00023 with everything. The head is worth about a third of the gap and the window KV with FP4 most of the rest. The audit had estimated 0.00015 for the head; the measured value is 0.00034.
- **Paired NLL against the served capture,** by conversation: -0.00034, SE 0.00069, within one SE. Head only: -0.00027, SE 0.00007, an improvement well outside its error bar.
- **Ties:** the head-only arm has no exact top-two ties, against 5,214 on the served build.
- **Decode against prefill:** eight conversations at 256 tokens and eight at 1,024, 32 tokens each. All 512 positions agree within 4.0e-7 nats.
- **Context-free rows:** 3, as in the reference: 157,47 and 64,69, which every arm and exact arithmetic give, and 136,64, which FP4 also gives. Row 41,104, which the served build got wrong where the reference and exact sums are fine, is right on this build. The head-only arm still has it, so the window KV is what moved it.

| Domain, KL / top-1 | Served | This build |
|---|---:|---:|
| code | 0.0077 / 97.15% | 0.0071 / 97.38% |
| agentic | 0.0137 / 96.43% | 0.0116 / 96.77% |
| chat | 0.0246 / 94.95% | 0.0230 / 95.04% |
| knowledge | 0.0181 / 95.30% | 0.0179 / 95.48% |
| long | 0.0158 / 96.78% | 0.0143 / 97.08% |
| math | 0.0087 / 96.74% | 0.0081 / 96.87% |
| multilingual | 0.0167 / 96.68% | 0.0154 / 97.06% |
| writing | 0.0247 / 94.74% | 0.0230 / 95.18% |

**Head time,** measured by the audit session on the card: BF16 2.22 ms, FP32 1.98 ms at one row, 2.10 ms at eight. The FP32 head is faster.

## Kernel results, 2 Nov 2026

**The fix passes every kernel test: the window store now matches DeepSeek's `act_quant` on all 151 million test values, decode and prefill kernels agree, and decode is slightly faster.**

**What it fixes.** The served V4 layout holds 448 window values in E4M3 with one scale per 64 and leaves the 64 RoPE values in BF16, on all 40 layers. The 1017 reference quantises all 512 with one scale per 32 (`model.py` `_window_kv`). The V41 writer branch also makes the reference's two BF16 rounds, after the norm and after RoPE, which the V4 branch skipped.

## Tests

Tree `/scratch/aster/bench/v41-window-r1/src`: the FP4 overlay plus `apply_v41_window.py`. GPU runs were on the 4080 SUPER with the server unloaded.

| Test | Result |
|---|---:|
| B. V41 store against DeepSeek's `act_quant`, block 32, UE8M0, fused quant + dequant | 0 of 151,011,328, sign of zero included |
| B2. FP4 store against DeepSeek's `fp4_act_quant` after RoPE | 0 of 151,009,280, sign of zero included |
| C. Loader against the V41 dequant | 0 of 4,521,984 |
| D. Attention against a twin reading the same values as BF16 rows, 1 to 256 rows, with NaN inputs | 0 of 3,000 calls |
| D2. Staged against unsplit on real V41 and FP4 caches, 1 to 8 rows | 0 of 1,000 calls |
| E. Selector-off kernels against the served files | 0 of 600 calls |
| F. Spills, staged and unsplit | 0. Unsplit 128 registers, staged 37-64 |

**Test B found a bug upstream.** SGLang's V41 store floored the scale after dividing by 448. DeepSeek floors `amax` at 1e-4 first, so the scale floor is 2^-22, not 2^-13. Before the fix, 4,238 of 151 million values differed, all in tiny blocks in the test's edge rows. It only reaches blocks whose largest value is under 0.0448, so the real effect should be small. The overlay fixes it in `store_row_fp8`. SGLang's own pure-torch reference has the same error, so its old test could not catch it. 05yuki's SM120 path uses this store, so this is worth raising with him.

B compares the store against the reference's quantiser, not the whole norm + RoPE + store path. That path is covered by the model run.

## Price

Same 4080 SUPER, graph replay for decode, 40 calls with random indices, median of 30 replays. One call is as a layer makes it: 128 window rows plus the compressed top-k, next to FP4 compressed KV. All on 40 layers.

| Decode, compressed top-k | Rows | V4, us | V41, us | 40 layers, ms a token |
|---|---:|---:|---:|---:|
| 64 | 1 | 12.1 | 11.0 | -0.045 |
| 256 | 1 | 13.7 | 13.0 | -0.031 |
| 512 | 1 | 16.9 | 16.5 | -0.014 |
| 512 | 4 | 24.8 | 23.5 | -0.053 |
| 512 | 8 | 42.8 | 40.5 | -0.094 |

| Prefill, 128 + 512 | V4 | V41 | Change |
|---|---:|---:|---:|
| 256 rows | 1.02 ms | 1.01 ms | -1.0% |
| 1,024 rows | 3.56 ms | 3.71 ms | +4.1% |

| Write | V41 | V4 |
|---|---:|---:|
| 1 row | 11.7 us | 12.2 us |
| 35,840 rows | 28.3 us | 50.7 us |

Decode is flat or slightly faster. Prefill reads may cost up to +4%, on top of FP4's +0.8%, and the write saves some of it back. A model run has to settle that: the V4 main cache in these tests held random bytes, not stored values.

Pages shrink 584 to 528 B a token on the window pool, about 10%.

## Not tested yet

- The combined norm + RoPE + store path against the reference, kernel by kernel. The panel covers it at model level.
- Serving profile on this build: four dsh sessions, 16K follow-ups, restart from disk. These are step 3 of the C2 plan.
- The reload hold has a few seconds' gap. On the 3 Nov reload, its own short chat request ran in the 3 s between "fired up" and the pause. That did no harm, but a long request landing in that gap could still trip the settle.

Overlay `apply_v41_window.py`, `sm89_v41_tile.py`. Tests `test_v41_kv_gpu.py`, `run_gpu.sh`.
