# Rotated copies and frozen swaps on the server, 2 Nov 2026

**Rotating slot copies across nodes cost 20% on the server, so it is dropped. Freezing swaps costs 4.5%, so swaps earn their copies.** Both runs are on the 32.98 build: six nodes, 42 workers, 320 W, two rounds, one load each. FEA idle. Neither went to identity, since neither is kept.

| Run | Breadth, rounds | Breadth | Japanese | 35K prefill, s |
|---|---|---|---|---|
| Engram build (baseline) | 33.05, 32.90 | **32.98** | 35.1, 35.8 | 69.50, 55.46, 51.93, 50.80 |
| Rotated copies, 256 KB (`adacperf`) | 26.1, 26.7 | **26.4** | 23.8, 24.2 | 69.66, 55.42, 51.99, 50.80 |
| Swaps frozen (`adavperf`) | 31.75, 31.3 | **31.5** | 34.0, 34.7 | 69.19, 55.20, 51.94, 50.84 |

The frozen run also works as a control: same day, same machine state and the same page-cache eviction as the rotated run, and it stays within 5% of the baseline. So the rotated drop is the selector's, not the machine's.

## Why rotation lost

The bench said +2% on the window at 256 KB against +19% in served order, and an estimated gain of 0.7-1.5 ms. The server lost about 7.6 ms a token instead.

- Swap counts and GPU route share did not move: 663 against 662 swaps in the same 64 rounds, 74.0% against 74.1%. Engram lookup and miss I/O did not move either.
- Copies reached the tables later, 80-104 ms after issue against 64-91 ms.
- Rounds took longer: 64 rounds in about 10 s against about 7.7 s.
- The Python loop in `_load_hot_rotated` turns about 18 copy calls an expert into about 100, on the lender thread, while the main thread runs decode. The bench had no competing thread. GIL or CUDA driver-lock contention would explain it. **That is an inference**: no timeline was taken on this run.

The bench measured worker windows, not the cost of issuing the copies, so it could not see this. A native rotation might still work, but it is only worth building if a timeline shows copies are worth 1 ms or more.

## What the frozen run says

With no swaps after warm-up, the GPU keeps the routes it has, so the CPU takes more. Copies stop, and breadth still falls 1.5 tok/s. So the copy cost is smaller than what swapping earns in GPU share. That caps what any better copy scheme can win: rotation's estimated 0.7-1.5 ms stays an upper bound, and it is unmeasured.

Selectors stay default off and out of the serving profile: `KT_KV_HOT_DYN_COPY_CHUNK`, `KT_KV_HOT_DYN_FREEZE_AFTER`.
