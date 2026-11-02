# Ada staged gather, 1 Nov 2026

Splitting the SM89 attention gather into three stages lifts breadth to **32.0 tok/s (32.05, 31.95)**, up 2.4% from 31.25. Numerics are exact and prefill did not move. Selector `SGLANG_SM89_ATTN_STAGED=1`, default off.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, on top of the final CPU-window build (`ADA_CPU_WINDOW_RESULTS_r1.md`).

| | Breadth tok/s (rounds) | Workload range | Mean request | Trial |
|---|---|---|---|---|
| Before | 31.25 (31.35, 31.15) | 30.3-32.2 | 32.82 s | `ada-perf-r16-20261101-r1` |
| Staged gather | **32.0 (32.05, 31.95)** | 30.9-33.2 | 32.01 s | `ada-perf-gather-20261101-r1` |

Rows, round 0 then round 1: Chinese 31.1, 31.0; English 32.8, 33.2; English technical 31.1, 30.9; math 33.2, 33.0; JSON 32.5, 32.4; Python 31.6, 31.5. Japanese medians 34.0, 34.6 against 33.1, 33.8. Prefill 35K: [69.65, 55.43], [51.93, 50.83] s against [69.10, 55.24], [51.86, 50.81].

## What changed

`_fused_gather_attn` took 1.53 ms of the serial zone. It was one Triton kernel that ran scores, online softmax and output in series inside each block. In an online softmax only the running max and the FP32 folds are order-bearing, not the MMAs, so scores and output tiles can run in parallel.

Three kernels replace it, in `kernels/ops/attention/sm89_gather_staged.py` (`gather-r1/gather_staged.served.py`, f7b9cb17): `_scores_kernel` and `_probabilities_kernel` on grid (hb, rows, blocks), `_output_kernel` on grid (hb, rows, 8). It applies at rows <= 8, h_q <= 64, split_k == 0; anything else stays on the old path. `sm89_dual_scope.py` overlay ad723c64 to 75d14867.

Priced on 40 distinct calls in a captured graph, including about 9 us of overhead a call: 39.2 to 20.5, 55.5 to 26.4 and 19.6 to 13.3 us, about 0.93 ms a token. The served gain on the breadth median is 0.75 ms.

## Getting it exact

Each of these changed bits until it was pinned. They are worth keeping for the next Triton kernel on this path.

- FP32 `tl.sum` order follows the layout, `#mma` against `#blocked`. The probabilities are anchored on `#mma` through a zero from a dummy `tl.dot`, so the sum and the BF16 cast run on the same layout as the old kernel.
- `tl.dot` takes its kWidth from the narrowest type it can trace backward. A branch in `_output_kernel` hid the uint8 loads, gave kWidth 2 instead of 4 and split K differently across the MMAs. This was the last mismatch. The tile now comes from a `tl.where` with no branch.
- `tl.maximum` drops NaN. `tl.where` keeps the bits.

I named the BF16 cast as the cause partway through and was wrong: the two cast instructions were not it, the kWidth was.

## Evidence

- Kernel fuzz: 3,000 calls, 1,500 with NaN and infinity, 1,500 finite, 0 mismatched (`gather-r1/test_gather_staged.py`).
- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-gather-20261101-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the final build's on all three prompts.
- Scale check: exact, 2,710 swaps logged.
- Memory run minimum, GiB by node 0-7: 12.0, 11.8, 7.9, 11.0, 13.9, 12.2, 12.3, 10.7. No guard stop.
- Serving gate (`ada-gate-gather-20261101-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged, cleanup empty. Run minimum 12.2, 12.3, 7.1, 10.2, 14.2, 12.1, 13.5, 10.6. Node 2's 7.1 is the lowest yet on this stack (8.0 on the final build's gate); the guard did not act.

## Precision ledger

Non-bit-identical steps added: none.

Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
