# Ada wo_a CUDA kernel, 1 Nov 2026

Moving `_parts` to a CUDA kernel took breadth to **27.65 tok/s** from 25.95, +6.6%, and +40% on the 19.80 we started from on 31 Oct. Numerics are exact, the serving gate passes, and prefill did not move. The selector stays default off and the server is unloaded.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, tree r6, arm 0 Route 3, prefill path A on.

| Build | Breadth tok/s (rounds) | Workload range | Trial |
|---|---|---|---|
| Fused glue | 25.95 (26.0, 25.9) | 25.1-27.1 | `ada-perf-glue-20261031-r1` |
| **+ wo_a CUDA kernel** | **27.65 (27.75, 27.55)** | **26.6-29.0** | `ada-perf-woa-20261101-r1` |

That is 2.43 ms a token off the whole request, 39.46 to 37.03 ms. Rows, round 0 then round 1: Chinese 26.7, 26.6; English 29.0, 28.7; English technical 27.3, 27.1; math 28.8, 28.8; JSON 28.2, 27.8; Python 27.3, 27.3. Japanese medians 29.8, 29.8 against 27.4, 28.0. Prefill: 35K [70.07, 55.73], [52.02, 50.99] s; 4K [23.62, 16.96], [18.89, 16.70] s, against [69.20, 56.00], [52.31, 51.36] and [23.48, 16.90], [18.77, 16.63].

## What changed

`SGLANG_DSV41_FIXED_WO_A_CUDA=1` (`wo-a-cuda-r1/`, one branch in `fixed_wo_a_fp8._rows`). `_parts` is the grouped FP8 `wo_a` projection: 8 groups of 1024 x 4096, 33.6 MB of cold weights a call. In Triton it ran at 346 GB/s.

`wo_a_parts.cu` is `route3_decode.cu` with a group axis and the epilogue of `_parts`. It keeps the per-element sequence read from the kernel's own PTX: one zero-accumulator `m16n8k32` e4m3 MMA per K block, then `mul.rn`, `mul.rn`, `add.rn`, no `.ftz`, four fixed K partitions, and the unchanged Triton `_merge`. It covers decode rows only, up to 16 and only this shape. Longer rows stay on Triton, and the bits are the same either way. The binary is built for sm_89 and pinned by SHA in `wo_a_parts.py`, with no compiler in the server.

## Pricing

Repeated calls on one weight tensor flatter everything, because 33.6 MB sits in the 64 MB L2: Triton reads 42.4 us there against 97 us in the graph. Forty distinct weights in a chain, one call each, reproduce the graph.

| Kernel, 40 cold weights | us a call |
|---|---|
| Triton `_parts` | 96.9 (graph: 97.0) |
| BPC 8, 3 stages | 51.5 |
| **BPC 4, 3 stages, kept** | **50.4** |
| BPC 4, 4 stages | 52.1 |

That prices 1.86 ms a token. The timed run gave 2.43 ms; the other 0.57 ms is unexplained, as no timeline was taken on this build. My earlier estimate of about 1 ms was low: it scaled from Route 3's per-call fit, not from a cold chain.

## Evidence

- Kernel test on the card, no model (`wo-a-cuda-r1/test_wo_a_parts.py`, WO_A_IDENTITY_PASS): 240 cases, 50,135,040 partials bitwise equal to the Triton `_parts`, at 1 to 16 rows. Scales covered the served range, products into and past the subnormals, overflow to infinity, and non-powers of two, so the literal `mul`, `mul`, `add` is tested and not assumed.
- The whole wrapper through the quantiser and `_merge`: BF16 output equal at 1, 2, 4 and 8 rows.
- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-woa-20261101-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the fused glue build's on all three prompts.
- Serving gate (`ada-gate-woa-20261101-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged, cleanup empty.

## Memory

Node 2's low point fell on two of the three loads: 7.0, 8.2 and 6.4 GiB, against 8.6, 8.4 and 8.6 on the fused glue build. No guard came near; the reserve guard sits at 4 GiB. The fall is in clean page cache (4.1-6.1 GiB against 6.3-6.6), while free memory held at 2.0-2.3 GiB. The new binary is 423 KB, so it cannot be the cause by size, and node 2 is known to swing load to load. I did not establish why these loads ran lower. It should be watched if this build is served.

## Precision ledger

Non-bit-identical steps added: none. Same instructions in the same order on the same operands, measured bitwise and EXACT end to end.

The kernel has no in-kernel scale check, as `_parts` has none. Route 3's check exists because its fused epilogue assumes powers of two; this epilogue assumes nothing.

## Remaining levers

1 ms a token is now worth about 2.7% of breadth. Gains are estimates.

| Lever | Size now | Likely gain | Needs |
|---|---|---|---|
| Fused top-k on the eight indexer layers | about 2.03 ms serial | about 1 ms | Kernel work; check what draft #7 already covers |
| `_fused_gather_attn` | 1.53 ms serial | not priced | Cold-chain pricing first |
| Faster merge | 1.2-1.5 ms of the CPU window | up to about 0.7 ms | Vectorising the per-column shift |
| Per-layer slot split | layer 0 served 42%, layer 25 68% | about 0.3 ms | Small change in `_plan` |
| Engram through Route 3 | 1.34 ms serial | 0.86 ms, measured on its overlay | Not bit-identical until a paired numerical run; your call |
| More slots | 950 now | about +8 points of share replayed | A KV trim that changes the pinned capacity; your call |

The LM head is at its floor: 2.14 ms is 618 GB/s on 1.3 GB of BF16 weights.

Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.

Next build: `ADA_TOPK_RESULTS_r1.md`, breadth 29.68. It also corrects the token figures above: 39.46 and 37.03 are seconds for a 1024-token request, so this build saved 2.37 ms a token, with 0.51 unexplained.
