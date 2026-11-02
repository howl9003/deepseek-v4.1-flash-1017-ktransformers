# Ada Engram through Route 3, 1 Nov 2026

Sending the Engram `wkv` projection through the Route 3 kernel lifts breadth to **32.98 tok/s (33.05, 32.90)**, up 3.0% from 32.0. Numerics are exact against C1 and prefill did not move. Selector `SGLANG_ROUTE3_ENGRAM=1`, default off; no change to the binary.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, on top of the staged gather build.

| | Breadth tok/s (rounds) | Workload range | Mean request | Trial |
|---|---|---|---|---|
| Before | 32.0 (32.05, 31.95) | 30.9-33.2 | 32.01 s | `ada-perf-gather-20261101-r1` |
| Engram through Route 3 | **32.98 (33.05, 32.90)** | 31.8-34.3 | 31.08 s | `ada-perf-engram-20261101-r1` |

Rows, round 0 then round 1: Chinese 32.0, 31.8; English 33.7, 34.1; English technical 32.0, 31.8; math 34.3, 34.0; JSON 33.5, 33.4; Python 32.6, 32.4. Japanese medians 35.1, 35.8 against 34.0, 34.6. Prefill 35K: [69.50, 55.46], [51.93, 50.80] s against [69.65, 55.43], [51.93, 50.83].

## What changed

`wkv` on the two Engram layers, 1 and 14, is 157.9 MB of E4M3 weights read at twice a token. It ran through the general `_w8a8_block_fp8_matmul_hopper` on grid (200, 4) at 670 us a call. The Route 3 kernel was already generic in N and K, so the change is one row in `route3_decode.SHAPES` behind a selector, `engram-route3-r1/evidence/route3_decode.py`, b9f9f91e to 40ed2446. Component price 0.243 ms a call, 2.76x, or 0.86 ms a token. The served gain on the breadth median is 0.90 ms.

## Evidence

- Component: 90 bitwise comparisons on real weights, 29 Oct (`engram-route3-r1/`).
- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-engram-20261101-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the gather build's on all three prompts.
- Scale check: exact, 2,710 swaps logged.
- Serving gate (`ada-gate-engram-20261101-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged, cleanup empty.

## Memory

Run minimum, GiB by node 0-7: identity 9.8, 11.5, 9.0, 11.2, 14.0, 11.9, 12.0, 11.3; gate 8.4, 11.3, 9.1, 10.0, 13.6, 10.9, 13.1, 10.6. No guard stop. Node 0's 8.4 is the lowest on this stack, against 11.9-12.6 on the final CPU-window build and 12.2 on the gather gate. Engram did not change memory use, so read this as spread between loads, and a reason to watch node 0 if this build is served.

## Precision ledger

Non-bit-identical steps added: none. Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
