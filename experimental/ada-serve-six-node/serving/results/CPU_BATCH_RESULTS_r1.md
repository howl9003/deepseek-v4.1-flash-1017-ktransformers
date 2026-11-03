# CPU window batch on C2, 3 Nov 2026

**Exact vs C2. Serving-profile breadth 34.15 (33.75, 34.55) against C2's 33.65 (33.2, 34.1): about +1.5%, all 12 workload pairs up. One session +2.4%, two and four sessions flat. Cold 35K prompts unchanged. Kept default off.**

Three selectors on one native (`ada-native-cpu-batch-20261103-r1`, .so 73866edc54f1804505fc0d81aa1dcdf59153bf70d91fa96d82aa489ee49c9458), all default off:

- `KT_CONTRACT_V2_SHARED_X=1`: at one token, one input quantisation serves every route in a part (`../shared-x-r1/`).
- `KT_CONTRACT_V2_FUSED_PHASES=1`: at qlen ≤ 8, gate/up, activation, the second quantisation and down run as one pool job, each task waiting only on its own expert's previous phase. The pool deals tasks in increasing order from one counter, so the earliest unfinished task can always run.
- `KT_CONTRACT_V2_MERGE_IN_PARTS=1`: each part merges its share of the column chunks inside its own NUMA job after an all-parts barrier, instead of in a second job.

All three are scheduling only: same calls, same splits, same per-column merge. Patch `apply_cpu_batch.py` on the shared-x header (7ab65054… → 321a9349…), build `build_ada_cpu_batch_r1.py` (90 s).

## Identity: adacb1, `ada-identity-cb-20261103-r1`

`ADA_NUMERICAL_EXACT_VS_C2` against the C2 archive (`ada-identity-c2-20261103-r1`, adac2a):

- 11,496 scores exact.
- Replays 0 and 1 exact; rotation exact, with swaps logged (1,666 and 413).
- 4K, 35K first and 35K second exact, top-logprob ids and values included.
- Lowest node during the run 8.3 GiB (node 4); the guard stops below 4 GiB.

## Timing: adacb2, `ada-serve-cb-20261103-r1`, serving profile, against C2's adasfp

| Measure | C2 (adasfp) | CPU batch (adacb2) |
|---|---:|---:|
| Breadth, round 0 | 33.2 | 33.75 |
| Breadth, round 1 | 34.1 | 34.55 |
| Japanese | 35.35, 36.15 | 35.65, 36.55 |
| 1 session | 31.79 tok/s | 32.54 tok/s |
| 2 sessions | 41.65 tok/s | 41.63 tok/s |
| 4 sessions | 51.53 tok/s | 51.14 tok/s |
| Cold 35K prompts (round 0) | 75.99, 74.47 s | 75.75, 74.35 s |
| 16K first turn, first token | 27.5-62.6 s | 27.3-63.0 s |
| 16K follow-up, first token | 18.3-21.7 s | 18.3-21.6 s |
| Lowest node | 9.5 GiB | 9.4 GiB |

Every workload is 0.3-0.7 tok/s faster in both rounds, about 0.45 ms a token. At two and four sessions the window is no longer the limit, so nothing moves there.

The serving profile's cold 35K prompts are about 75 s, against 61 s on the performance profile; the CPU batch leaves that unchanged, so the gap is not in the CPU window. Still open.

## Verdict

Exact, a small single-session gain, no cost seen. Worth serving only together with something larger, since a C2 reload is needed to switch it on.
