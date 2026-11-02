# Ada live top-k at the served capacity, 1 Nov 2026

Turning on the live top-k kernels at the served capacity took breadth to **29.68 tok/s** from 27.65, +7.3%, and +50% on the 19.80 we started from on 31 Oct. Numerics are exact, the serving gate passes, and prefill did not move. The selector stays default off and the server is unloaded.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, tree r6, arm 0 Route 3, prefill path A on.

| Build | Breadth tok/s (rounds) | Workload range | Trial |
|---|---|---|---|
| wo_a CUDA kernel | 27.65 (27.75, 27.55) | 26.6-29.0 | `ada-perf-woa-20261101-r1` |
| **+ live top-k** | **29.68 (30.05, 29.30)** | **28.6-31.2** | `ada-perf-topk-20261101-r1` |

That is 2.47 ms a token, 36.17 to 33.70 ms. Rows, round 0 then round 1: Chinese 29.8, 28.9; English 31.2, 30.5; English technical 28.9, 28.6; math 30.6, 30.5; JSON 30.3, 29.6; Python 29.0, 29.0. Japanese medians 31.65, 31.5 against 29.8, 29.75. Prefill 35K: [69.87, 55.79], [52.22, 51.36] s against [70.07, 55.73], [52.02, 50.99]. The 4K prefill rows are not in this run's logs, so they are not compared here.

The rounds sit 0.75 tok/s apart, against 0.2 on the wo_a build. Every workload rose in both rounds, so the gain is not in doubt, but the second decimal is.

## What changed

`KT_IDX_LIVE_TOPK=1` was already in the C1 profile and did nothing. `live_topk_enabled` rejects `lmax > KT_IDX_LIVE_TOPK_MAX_L`, the default is 65,536, and the served capacity is 524,290 at ratio 2 (layers 2, 8, 14) and 1,048,580 at ratio 1 (20 source, 24, 28, 32, 36 consumers). So every earlier rate ran torch's top-k on all eight indexer layers.

The change is `KT_IDX_LIVE_TOPK_MAX_L=1048580` in the profile plus one fix in `live_topk.py` (`topk-live-r1/`, overlay d6d61621 from tree r6's 9ca99b15, `apply_live_blocks.py`). Among blocks, torch drops a live logit that is itself `-inf` and the kernel kept it. That was the only mismatch the fuzz found: 78 `blocks/neginf` cases before the fix, none after.

`select_candidate_block_ids` stays on torch, because it needs torch's sorted tie order.

## Pricing

Captured-graph replay with synchronize, about 9 us of replay overhead in each figure, n 700 / 1500.

| Layer | Torch us | Live us |
|---|---|---|
| Ratio 2, full | 137 | 18 |
| Ratio 1, full | 162 | 21 |
| Consumer | 172 | 22 |
| Source slots, logits, publish | 89 | 89, unchanged |

That prices about 1.18 ms a token and gave an estimate of 28.5. The timed run gave 2.47 ms. The later timeline, `ADA_TIMELINE_TOPK_RESULTS_r1.md`, measures the direct saving at 1.44 ms and refutes the cleaner-L2 reading this paragraph first offered: kernels I did not touch run at the same speed. About 0.9 ms stays unattributed.

Correction to the wo_a write-up: its "39.46 to 37.03 ms" is seconds for a 1024-token request. In ms a token that build saved 2.37, not 2.43, so its unexplained part is 0.51, not 0.57.

## Long contexts

The live kernels use one CTA per row, so their cost grows with the live length while torch's barely does. Priced, us:

| n | 35,000 | 100,000 | Capacity |
|---|---|---|---|
| Ratio 2, torch / live | 147 / 64 | 149 / 145 | 236 / 692 |
| Ratio 1, torch / live | 172 / 66 | 185 / 149 | 389 / 1394 |
| Consumer, torch / live | 206 / 65 | 222 / 108 | 430 / 753 |

Ratio 2 layers cross at about a 200K-token context. Summed over the layers, interpolating between these points, the whole token crosses at about 250K tokens and costs up to about 3.7 ms a token more at the full 1M. None of that is timed; the longest timed prompt here is 35K. Serving contexts that long would want a length-based fallback to torch, which does not exist.

## Evidence

- Fuzz on the card with the overlay (`topk-live-r1/test_live_capacity.py`): 1,200 slots, 1,200 full and 800 among-blocks cases equal torch, covering ties, +-0, NaN, +-inf, up to capacity, 1 to 8 rows.
- Captured-graph replay with device-side lengths: LIVE_REPLAY_PASS 480, both ratios, full and blocks.
- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-topk-20261101-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the wo_a build's on all three prompts.
- Serving gate (`ada-gate-topk-20261101-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged, cleanup empty.

Still untested, as on earlier builds: a raised flag on a live server, and an 8-row capture in the gate.

## Memory

Node 2's low point was 8.8, 8.1 and 8.5 GiB on the three loads, against 7.0, 8.2 and 6.4 on the wo_a build and 8.6, 8.4 and 8.6 on the fused glue build. This build's kernels add no host memory, so the wo_a dip reads as load-to-load swing, though its cause was never established. No guard came near; the reserve guard sits at 4 GiB.

## Precision ledger

Non-bit-identical steps added: none. The selection equals torch's in the fuzz and replay, and EXACT end to end.

## Remaining levers

Superseded by the ranked table in `ADA_TIMELINE_TOPK_RESULTS_r1.md`, which puts the CPU window first.

Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
