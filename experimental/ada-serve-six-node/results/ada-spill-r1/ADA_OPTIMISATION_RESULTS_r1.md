# Ada path optimisation on the 4080 SUPER, 31 Oct 2026

Superseded on rate by [ADA_TIMELINE_RESULTS_r1.md](ADA_TIMELINE_RESULTS_r1.md): breadth 24.98, exact, on the same layout.

The kept build reads 23.78 tok/s breadth against 19.8 on the same tree, +20%, with bit-exact numerics and the accepted memory layout per node. It is default-off and not served; the server stays unloaded.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, tree r6, arm 0 Route 3, prefill path A on.

| Build | Breadth tok/s (rounds) | Japanese | Workload range | Trial |
|---|---|---|---|---|
| Served layout | 19.80 (19.75, 19.85) | 22.6 | 17.0-22.0 | `ada-perf-20261031-r1` |
| Cold placement | 19.93 (20.10, 19.75) | 22.85 | 18.8-21.7 | `ada-perf-cold4-20261031-r1` |
| Cold placement + round worker | 20.80 (20.90, 20.70) | 23.9 | 19.8-22.7 | `ada-perf-cold4w-20261031-r1` |
| One-sixth spill, not servable | 21.75 (21.90, 21.60) | 24.65 | 21.0-23.5 | `ada-perf-spill6-20261031-r1` |
| Stagger | 22.78 (22.85, 22.70) | 24.8 | 21.9-23.8 | `ada-perf-stagger-20261031-r1` |
| **Stagger + round worker** | **23.78 (23.90, 23.65)** | **25.9** | **23.1-25.2** | `ada-perf-stagger-worker-20261031-r1` |

Prefill on the kept build: 35K [70.06, 55.88], [52.58, 51.53] s and 4K [23.60, 16.89], [18.86, 16.63] s, against served [69.42, 55.78], [52.23, 51.27] and [24.54, 16.68], [18.59, 16.36]. The stagger costs about 0.3 s on the 4K and 35K second reads.

## What changed

**`KT_EXPERT_ARENA_SPILL_STAGGER=1`** (`build_ada_stagger_r1.py`, native `ada-native-stagger-20261031-r1`). Every part used to spill at the same stripe offset, half to node 3 and half to node 7, so one spilled id sent all 42 workers to a single node, half of them across the socket. That is 40 GB/s against 218, the measured 4-6x route cost. Now each part spills only to its own socket's spill node, at its own offset. A stripe position has at most one part on each spill node, and one or two parts remote instead of six. Bytes per node are the same on full stripes; partial stripes move about 6 MiB an arena from node 3 to node 7. Which node holds a page cannot change a bit.

**`KT_KV_HOT_DYN_WORKER=1`** (`hot-round-worker-r1/`). A swap round's copies are enqueued on a worker thread and enter the tables at the first forward after they are done, about 85 ms from issue. Worth +4.4%, with no round deferred. This restores in a new form what draft #8 meant to do; the `DEFER`, `ASYNC`, `BYTES_MB`, `DWELL`, `BATCH_LOAD`, `REFILL_FAST` and `ROUTE_FOLD` settings in the accepted environment are read by no code in C1 or r6.

## Evidence

- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-stagger-20261031-r1`); 4K and both 35K prompts have their layer flags clear.
- Long cases: text, ids and every logprob equal the served layout's on all three prompts, and equal the prefill-off arm's on all three.
- Serving gate (`ada-gate-stagger-20261031-r1`): all seven cache checks pass, every replay equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged.
- Layout test (`stagger-r1/test_arena_stagger.cpp`): selector off matches the accepted header at every 2 MiB; selector on keeps each node's bytes and never puts two parts on a spill node at one position. ALL_PASS.
- Worker test (`hot-round-worker-r1/test_round_worker.py`): no table entry ever ahead of its copy. ALL_PASS.
- Memory: node 2 low point 9.0-9.5 GiB across all stagger loads, nodes 3 and 7 at 10.7-14.8 GiB. No guard came near, and none was changed.

## Where a token goes now

Primed 65-token captures after 1,024 tokens, stagger and worker on (`ada-workers-stagger-20261031-r1`). Times in ms per token.

| Workload | Token | GPU-only gaps | GPU-bound forward floor | CPU past that floor | Other | CPU routes of 240 |
|---|---|---|---|---|---|---|
| English technical | 39.8 | 18.6 | about 11 | about 7.5 | about 2.7 | 122 |
| Python | 40.9 | 18.8 | about 11 | about 9 | about 2.2 | 153 |
| Japanese | 36.7 | 19.2 | about 11 | about 4 | about 2 | 70 |

A layer's window is the slower of its CPU routes and its GPU routes: the poller publishes only once the stream reaches its sync. With no CPU route on a layer the window is still about 280 us, so about 11 ms a token is the GPU's own expert path, nearly flat with the number of GPU routes. About 30 of 40 ms is now GPU-side.

The 1,024-token curves are flat after the first 128 tokens (about 43 ms there, 37-41 after). The 25-33 ms step every 8th token is gone. A soft bump of about 4 ms across two or three tokens remains on English technical and Python, as means by position modulo 8; Japanese is flat within 2 ms.

The swap budget does not limit the hit rate. Rounds use about 9 swaps of a budget of 320, so raising `KT_KV_HOT_DYN_SWAPS` or lowering `EVERY` will not move it. GPU share sits at 34-38% on the hard workloads because 950 slots hold about 6.7% of the ids.

## Refuted on the way

- Cold placement. The 6.4% was weighted on all routes, but the GPU takes the hot ids, so the CPU sees the tail. Replaying real CPU routes gives remote bytes of 15.7% to 17.9% on English technical. Never rank placement on all-route counts.
- One-sixth spill as a serving layout: node 2 low point 6.3, 4.6 and 3.5 GiB across three loads, under the 4 GiB guard once.
- Shmem huge pages, burst gaps, fold fast path, node hardware, worker pinning, many-node memory limit.

## Precision ledger

Non-bit-identical steps added: none. The stagger moves pages between nodes and the worker moves when a slot enters the tables; both are bit-neutral by the logical combine order and are measured EXACT.

## Prefill-off arm

`ada-identity-stagger-prefilloff-20261031-r3`: the kept build with `KT_GPU_STREAM_PATHA=0`. 11,496 EXACT vs C1. On the 4K and both 35K prompts, text, ids and every logprob equal the path A arm. First-token latency on and off: 4K 22.48 and 27.05 s, 35K 69.68 and 181.18 s, second 35K 62.91 and 169.19 s. The r1 and r2 namespaces stopped on assertions in my own chain that path A was on, before any request ran; they hold no results.

## Open items

- The diagnostic trial `ada-workers-stagger-20261031-r1` ends with RUN status FAIL on a FileNotFoundError at the end of the run, as the earlier timestamp trial did. All six captures passed the client's validation and the curves are complete. The cause is not verified; the likely one is the prefill origin check, since that trial sends no long prompts.
- Selectors stay default off. Nothing is pushed, PR6 stays held, and the accepted server profile is unchanged.
- About 30 of 40 ms a token is GPU-side, and the kernels inside it are unmeasured on this build. The next step is a timeline capture, which needs the owner's OK.
