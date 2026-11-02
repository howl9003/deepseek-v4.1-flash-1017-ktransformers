# Ada decode timeline and slot rule, 31 Oct 2026

Superseded on rate by [ADA_GLUE_RESULTS_r1.md](ADA_GLUE_RESULTS_r1.md): breadth 25.95, exact, with the glue fused.

The Nsight timeline on the kept build found two more gains, both bit-neutral. Breadth is now **24.98 tok/s** against 23.78, +5.0%, and +26% on the 19.80 we started the day with. Numerics are exact, the serving gate passes, and prefill did not move. Selectors stay default off and the server is unloaded.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, tree r6, arm 0 Route 3, prefill path A on.

| Build | Breadth tok/s (rounds) | Workload range | Trial |
|---|---|---|---|
| Kept build (stagger + round worker) | 23.78 (23.90, 23.65) | 23.1-25.2 | `ada-perf-stagger-worker-20261031-r1` |
| + slot rule, half-life 64, hysteresis 1.5 | 24.35 (24.25, 24.45) | 23.9-25.4 | `ada-perf-rule64-20261031-r1` |
| **+ route reference** | **24.98 (24.85, 25.10)** | **24.3-26.2** | `ada-perf-rule64-ref-20261031-r1` |

Rows on the final build, round 0 then round 1: Chinese 24.5, 24.3; English 26.2, 25.6; English technical 24.5, 24.8; math 25.7, 25.8; JSON 25.1, 25.4; Python 24.6, 24.8. Prefill: 35K [69.26, 56.05], [52.38, 51.47] s; 4K [23.59, 16.82], [18.81, 16.68] s, against the kept build's [70.06, 55.88], [52.58, 51.53] and [23.60, 16.89], [18.86, 16.63].

## Where a token goes

Three captures of 64 tokens from token 512 of a 1,024-token request, on the kept build. Trial `ada-nsys-stagger-worker-20261031-r1`; data in `nsys-r1/`. Median tokens inside the range match the tokens around it, so node tracing did not disturb decode.

| ms per token | English technical | Python | Japanese |
|---|---|---|---|
| Serial GPU work | 19.97 | 19.83 | 19.55 |
| GPU part of the expert window | 8.88 | 8.63 | 10.82 |
| GPU idle, waiting for the CPU | 7.02 | 8.91 | 1.67 |
| LM head and tail | 2.19 | 2.19 | 2.19 |
| **Total** | **38.06** | **39.57** | **34.23** |
| Median token period | 38.31 | 39.35 | 34.29 |
| Windows where the GPU finished last | 34% | 28% | 77% |

Serial runs from the end of one handoff copy to the end of the next `_pack_kernel`, when the CPU can start. Nothing runs beside it, so every microsecond there is on the token. In the window, GPU experts and CPU routes run together and the slower one sets the length. Graphs follow each other 0.11 ms apart; the GPU is never idle between tokens.

Serial content, ms per token: Route 3 6.23, `_parts` 3.93, attention gather 1.53, Engram matmuls 1.34, small torch kernels 2.06 (1,506 of them), and about 2.03 extra on the eight indexer layers. Branches overlap, so these do not add to the wall time.

## What changed

**Slot half-life.** The benchmark's rate sat below the capture's, about 3 ms a token on English technical, and the gap was inside decode: GPU share. Benchmark rows ran 34% in their first half and 50% in their second, against 69-76% on Japanese. The slots spent the whole request adapting. My earlier note that the swap budget was not the limit was true but stopped short: rounds used about 12 swaps of 320, and the limit was the score's half-life of 256 tokens, which had never been tuned.

Every routed expert is counted on the device whoever serves it, so one traced run gave a replay of any rule against real routes (`dyn-trace-r1/`, `tools/replay_dyn_slots_r1.py`, `_r2.py`). The served rule reproduces the measured share within 0.3%, worst stretch 0.7%. With copies landing 66 ms + 2.1 ms a copy after issue, half-life 64 with hysteresis 1.5 is best: 56% share against 50% on the breadth rows, CPU routes 109 against 124. Shorter half-lives lose to their own copies. Timed, it matched: English technical's first half rose from 34% to 46%, with about 24 swaps a round and about 20 of 2,560 rounds deferred.

**Route reference.** `KT_CONTRACT_V2_ROUTE_REF=1` (`tools/build_ada_ref_r1.py`, native `ada-native-ref-20261031-r1`, kt_ext c1fb79ed). Each part copied 57 KB of int64 partials per route on its main thread after the workers had finished, and the merge then read the copy: the serial gap before merge in the worker traces. The merge now reads the same integers in place. Worth +2.6%.

## Evidence on the final build

- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-final-20261031-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the kept build's on all three prompts.
- Serving gate (`ada-gate-final-20261031-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged.
- Trace tests `dyn-trace-r1/test_dyn_trace.py` and the round worker test pass on the traced file; the trace only adds lines.
- Memory: lowest node 8.3 GiB, no guard near, none changed.

## Precision ledger

Non-bit-identical steps added: none. The half-life and hysteresis decide which device holds an expert; the route reference decides which buffer holds the same integers. Both are measured EXACT.

## Remaining levers, ranked

1 ms a token is worth about 2.4% of breadth. Gains are estimates unless marked measured.

| Lever | Size now | Likely gain | Needs |
|---|---|---|---|
| Fuse the glue kernels: route end to `_pack_kernel` 0.87 ms, handoff copy to combine 0.93 ms, 1,240 kernels | 1.8 ms serial, measured | about 1.2 ms | Fused kernels; end-of-step check detection, already accepted 20 Oct. Bit-neutral |
| `_parts` as a CUDA kernel | 3.93 ms serial, measured | about 1 ms, earlier estimate | Kernel work. Bit-neutral |
| Engram through Route 3 | 1.34 ms serial, measured | 0.86 ms, measured on its overlay | Not bit-identical until a paired numerical run; a ledger entry, your call |
| Fused top-k on the eight indexer layers | about 2.03 ms serial | about 1 ms | Kernel work; check what draft #7 already covers |
| Unequal slots per layer | layer 0 served 42%, layer 25 68% | +1.9 points of share replayed, about 0.3 ms | Small change in `_plan` |
| Faster merge | 1.2-1.5 ms of the CPU window | up to about 0.7 ms | Vectorising the per-column shift. Bit-neutral |
| More slots | 950 now | ×1.5 about +8 points replayed without copy latency | A KV trim, which changes the 1,048,576-token capacity the profile pins; your call |
| Engram host wait | 0.4-0.9 ms serial | not known | Row-store lookup in a host callback; cause not measured |

All of the bit-neutral rows together would take a token from about 39 ms to about 35, breadth about 28. Separately, the first request after any long prefill pays about 2.5 s to re-lend slots, which is why the Chinese row and p0 sit low; that is one request, not a token rate.

## Corrections and open items

- The earlier split put GPU-only time at 18.6-19.2 ms. The timeline makes it about 22 ms.
- The two end-of-run `FileNotFoundError` statuses on the timestamp trials are explained: the path A origin receipt exists only after a long prompt. The capture sent a 4K prompt and its after-client gate passed.
- Rule candidates at 16-32 tokens were replayed only. `KT_KV_HOT_DYN_EVERY` below 8 cannot be replayed from 8-token counts and is untested at this half-life.
- Isolated stalls of 50-150 ms appear in the streams, inside and outside the capture. They cost about 0.5 ms a token on average, and the cause is not measured.
- Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
