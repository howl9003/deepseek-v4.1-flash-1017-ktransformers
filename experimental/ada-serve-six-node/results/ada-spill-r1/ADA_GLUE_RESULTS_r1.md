# Ada fused route glue, 31 Oct 2026

Superseded on rate by [ADA_WO_A_RESULTS_r1.md](ADA_WO_A_RESULTS_r1.md): breadth 27.65, exact, with `_parts` on a CUDA kernel.

Fusing the glue took breadth to **25.95 tok/s** from 24.98, +3.9%, and +31% on the 19.80 we started the day with. Numerics are exact, the serving gate passes, and prefill did not move. The selector stays default off and the server is unloaded.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, tree r6, arm 0 Route 3, prefill path A on.

| Build | Breadth tok/s (rounds) | Workload range | Trial |
|---|---|---|---|
| Final build before this | 24.98 (24.85, 25.10) | 24.3-26.2 | `ada-perf-rule64-ref-20261031-r1` |
| **+ fused route glue** | **25.95 (26.0, 25.9)** | **25.1-27.1** | `ada-perf-glue-20261031-r1` |

That is 1.54 ms a token off the whole request, 41.00 to 39.46 ms. Rows, round 0 then round 1: Chinese 25.2, 25.1; English 26.5, 27.1; English technical 25.7, 25.4; math 26.9, 26.6; JSON 26.1, 26.1; Python 25.9, 25.7. Japanese medians 27.4, 28.0 against 26.0, 26.6. Prefill: 35K [69.20, 56.00], [52.31, 51.36] s; 4K [23.48, 16.90], [18.77, 16.63] s, against the final build's [69.26, 56.05], [52.38, 51.47] and [23.59, 16.82], [18.81, 16.68].

## What changed

`KT_CONTRACT_V2_GLUE_FUSED=1` (`glue-fused-r1/glue_fused.py`, one selector branch in `contract_v2/adapter.py`). It applies to captured decode only; every eager call takes the adapter's own path and its immediate asserts.

The timeline's 1.8 ms covered only the two serial spans. Mapped to source, all of the glue sits in `adapter.apply`, and there was more of it inside the expert window, so the fusion removes 51 launches a layer rather than 29, by source count:

| Where | Launches a layer | What they were |
|---|---|---|
| Route end to `_pack_kernel` | 15 | A resident mask overwritten straight after, the CPU and resident ownership masks, the owner sum, the zero fill of `routes` |
| Handoff copy to combine | 14 | The CPU/GPU `where`, the expected-owner mask, two `all` reductions, two `_assert_async`, the finite check |
| Expert window | 21 | The slots' owner sum and both passes' active ids |
| Expert window | 1 | A second quantisation of the same `x` into the same buffer |

- **`_route_glue`** is `_route_kernel` with the resident table, plus the two passes' active ids and a one-byte owner per route, in one launch.
- **`_combine_owned`** is `_combine_ordered`, except that each route's term comes from the half of one buffer that owns it: GPU half, CPU half, or zero. The handoff copy lands in the CPU half, so the select and the zero fill both go.
- **Checks** move to the static expert path's sticky flag word as bits 64 (ownership) and 128 (non-finite weight). The worker clones that word after every forward, and the result processor reads it before any output leaves. This is the graph-mode rule accepted 20 Oct.

## Evidence

- Kernel test on the card, no model (`glue-fused-r1/test_glue_fused.py`, GLUE_FUSED_ALL_PASS): 10,800 random routes against `_route_kernel` plus the adapter's torch ops, including duplicate ids, id -1, zero weights and both tables hitting one id; counts equal; 400 combines bitwise against `where` plus `fast_combine.combine`, with junk planted in every unowned slot; 5 two-pass cases on synthetic experts, 1 to 8 rows, against `expert_routes_patha` twice.
- Flags: a slot beyond the shadow and a row beyond the residents set bit 64; NaN and both infinities set bit 128; clean calls leave the word at 0.
- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-glue-20261031-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the final build's on all three prompts.
- Serving gate (`ada-gate-glue-20261031-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged, cleanup empty. Its server captured 4, 2 and 1 rows on the fused path.
- Each run logged its capture on the fused path, and no server log holds a flag error.
- Memory: lowest node 8.4 GiB, no guard near, none changed.

## Precision ledger

Non-bit-identical steps added: none. The same integers land in another buffer, the same FP32 adds run in the same order, and the dropped quantisation rewrote identical codes.

One behaviour changes, and it is not numeric: in captured decode, a failed ownership or weight check is now found at the end of the step instead of in the layer. The output is still discarded before it leaves. Eager calls keep the immediate asserts.

## Not tested

- The flag bits are proven at kernel level on the card. No live server has seen a raised bit. `check_result` is existing code, but in these runs it has only read a clear word.
- 8-row capture did not run in the gate, which captures 4, 2 and 1. The kernel test covers 1 to 8 rows.
- No timeline was taken on this build, so the 51 launches is a source count and the 1.54 ms is the only measured figure.

## Remaining levers

The table in [ADA_TIMELINE_RESULTS_r1.md](ADA_TIMELINE_RESULTS_r1.md) stands without its glue row. By size: `_parts` as a CUDA kernel about 1 ms, fused indexer top-k about 1 ms, faster merge up to about 0.7 ms, per-layer slot split about 0.3 ms, all estimates. Those four together would give breadth about 28. Engram through Route 3 and the KV trim for more slots remain the owner's calls.

The empty resident pass still costs five launches a layer when no resident row is routed. Skipping it means changing the pinned native kernel, so it was left.

Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
