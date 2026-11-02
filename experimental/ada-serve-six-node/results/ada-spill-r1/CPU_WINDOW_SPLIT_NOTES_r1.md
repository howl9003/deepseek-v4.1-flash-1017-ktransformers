# CPU window split, offline, 1 Nov 2026

No contact with fea01. Local files only. Nothing under the project was changed.

## Data used

- Worker traces: `adastts-r1/worker-traces/*` (31 Oct 10:14-10:15), the newest. Build `ada-native-ts-stagger-20261031-r1`: stagger on, node 0 rebalance 1, merge pools all, no route ref, half-life 256, rounds every 8. 64 tokens each, 14,612 calls with routes. `adats-r1` (02:32) is older and was not used.
- They are the first 65 tokens of a request, not steady state. English and Python traces carry 123 and 154 CPU routes a token, Japanese 72. Today is about 85, so per-call fits are the numbers to trust, and ms/token is quoted at 85 routes (40 a + 85 b).
- Nsight captures, token 512 on: `ada-spill-r1/nsys-topk-r1` (1 Nov, route ref, half-life 64) and `nsys-r1` (31 Oct, same build as the traces).
- Source: `kt-ada-src/operators/avx2/contract_v2-moe.hpp`, `contract_v2_kernels.inc`, `expert_arena_spill.hpp`, `moe_base.hpp`; `worker-timestamps-r1/source-r3/native/cpu_backend/worker_pool.cpp`. Model: hidden 5120, moe_intermediate 2304, 384 experts, part slice 384.
- Scripts in the scratchpad: `cpu_split1.py` to `cpu_split10.py`, `nsys_swap.py`.

Phase map: 1 quant x, 2 gate/up plus finish to BF16, 3 activation, 4 quant h, 5 down, 100 merge. Scatter copy sits between 5 and 100 on this build, on each part's main thread.

## 1. Stage table

Fast state, pools 1-5 so pool 0's old arena is out. Critical path is the last of the five. Six traces pooled, 11,178 calls. Measured.

| Stage | Runs | Mean pool us | Last of five us | ms/token at 85 routes (last) |
|---|---|---|---|---|
| Queue start to first task | Serial: dispatch, then each part's main thread (bookkeeping, input copy) | 7.2 + 1.07 n | 7.7 + 0.98 n | 0.39 |
| Quant x | Parallel in a part, one task per route | 7.9 + 1.21 n | 8.0 + 1.21 n | 0.42 |
| Gate/up | Parallel, seven workers a part | 5.3 + 51.7 n | 4.9 + 57.6 n | 5.09 |
| Step: activation, quant h, three dispatches | Parallel, tiny tasks | 3.7 + 0.83 n | 3.8 + 0.87 n | 0.23 |
| Down | Parallel, seven workers a part | 6.7 + 26.9 n | 7.2 + 28.9 n | 2.75 |
| Last part done | | | 32.1 + 89.7 n | 8.91 |
| Gap to merge | Serial on each part's main thread, then join and redispatch | | 5.1 + 7.4 n | 0.84 traced |
| Merge | Parallel, 20 workers, one 256-column task each | | 4.8 + 6.8 n | 0.77 |
| Merge end to queue end | Serial | | 0.7 | 0.03 |

Then queue end to publish about 2 us when the GPU is already waiting. t0 to queue start is 3-6 us.

Route ref removed the scatter, so the gap's n term is gone; today's gap is about 5 us, 0.2 ms (inferred from the intercept).

Inside a part all seven workers start within 0.3 us. Cost of a phase ending: mean idle at the end of gate/up 11.7 us, of down 6.2 us. Dispatch gaps total about 3 us a call.

### Reconcile with 65 + 125 n

All calls: 99.6 + 109 n primed, 97.1 + 109 n cold. At n = 2.1 that is 328 us, and 65 + 125 n gives 328, so the formula agrees at the operating point. But it blends two states:

- Fast, all pools: 36.9 + 111 n. Less scatter (7.4 n) and pool 0 (about 7 n) gives about 40 + 97 n on today's build. Inferred, no trace exists on it.
- Slow: 94 + 173 n.

## 2. The two states (largest finding)

Per-call merge time is bimodal: median 6.9 us a route, or 18-19 us. The slow calls come in unbroken runs of 2-8 tokens across all layers, sharp edges, about 8 tokens apart on Japanese. Slow share by trace: English 29%, Python 5%, Japanese 49%, cold 13%, 7%, 43%.

At equal routes, slow against fast, last of five, ms/token at 85 routes: start 0.39 to 0.68, gate/up 5.09 to 6.59, down 2.75 to 3.49, gap 0.84 to 2.13, merge 0.77 to 2.10. Whole window 10.9 to 18.5, so +7.5 ms while it lasts. Measured.

Latency probes move too: first task 1.5-1.65x, skew across pools 1.6-4x, activation 1.13-1.39x. Quant x does not move (1.00-1.03x), so the core clock is fine. That pattern means the cost is in memory and fabric, not in the core. Per task, gate/up: pools 1-2 +3-5%, pools 3-4 +11-24%, pool 0 +43-62%. Merge on every pool +2.7x. Inferred from the pattern.

Averaged over the traces: English 3.0, Python 0.6, Japanese 3.0, cold 2.0, 0.9, 2.8 ms/token.

### Matching it to slot copies

Nsight, H2D copies with no graph node, 983,040 and 245,760 B (one part's nibbles and scales). Rounds every 280 ms, about 8 tokens, 400-640 MB each, about 90 ms of copying, 13.6 GB/s. They cover 25% of decode time on the 1 Nov build and 7-10% on 31 Oct.

| Capture | Median window, no copy | In a copy | Tokens with copies, wait | Without |
|---|---|---|---|---|
| 1 Nov English | 334 us | 423 | 14.8 ms | 10.2 |
| 1 Nov Python | 323 | 437 | 13.5 | 8.0 |
| 1 Nov Japanese | 265 | 317 | 2.9 | 1.6 |
| 31 Oct English | 349 | 450 | 8.0 | 6.9 |
| 31 Oct Python | 394 | 458 | 10.3 | 8.8 |
| 31 Oct Japanese | 291 | 350 | 2.2 | 1.6 |

Averaged over tokens on the 1 Nov build: by median window 0.9, 1.0, 0.5 ms/token; by token wait 1.7, 1.8, 0.45.

**What is not proven.** The link is correlation. Trace slow share (Japanese 49%) exceeds the copy time in a steady capture of the same build (7%), and Python's 5% does not fit a simple copy-time rule. Traces are early tokens, when slots churn hardest, but copy volume there is not recorded, so another trigger is possible. The mechanism (GPU DMA against partials and weights in the fabric) is a guess. The 1 Nov rule added more swaps, and rounds every 4 came after this capture, so the share may be larger now. No timeline exists on that build.

Decisive test when fea01 is free: one worker-trace run with copy timestamps, or one with swaps frozen after warm-up. No code, bit-neutral.

Not covered by the earlier refutation: "burst gaps" there meant spacing in worker memory bursts. "Do not retune SWAPS" was about GPU share. This is copy timing, not swap count.

Also seen, outside the window: on the 1 Nov capture the breadth rows have tokens of 57-77 ms against 33, most at or next to a burst, none on Japanese. Mean period 36.6 against median 33.5 ms on English. Worth a look, as the top-k write-up put stalls at 0-0.3 ms.

## 3. Split of the parts and the merge

- Gate/up: by output rows. Each part owns 384 of 2304 rows and every one of 5120 input columns. Output rows are disjoint, so no merge. `load_weights` lines 606-608.
- Down: by input columns, K. Each part owns 384 of 2304 K columns and writes all 5120 outputs. Output rows are **not** disjoint, so the sum is required. Lines 13-15, 613-616, 640-663.
- So gate/up is already what the question asks of down, and down is the only K-partitioned step.

Bytes a route: 6 x 40,960 = 245,760 B int64 plus 6 x 5,120 = 30,720 B EW. Remote: 204,800 + 25,600 = 230,400 B. Across the socket: 138,240 B. Output 10,240 B to one buffer. At 85 routes, 19.6 MB a token. Who: `KT_CONTRACT_V2_MERGE_POOLS=all`, 20 chunks of 256 columns on 20 workers, 3-4 a pool, one task each, so span is about one task.

### What the merge cost is

- Median 6.9 us a route on every workload, repeat or not.
- Warm routes (same layer, same expert within 1-4 tokens) against cold: 6.3-7.8 against 6.9-8.7 on English and Python, no clear gap on Japanese. Cold EW and BufferB headers are at most about 2 us, not the driver.
- 256 columns at the bench's 15.6 ns is 4.0 us, so arithmetic could be over half of the fast state's 6.9 us if the costs add. That does not sit well with "the cost is pulling partials". Most of the mean's excess is the slow state. Not re-proposing AVX2: 3 us a route on 85 routes is 0.26 ms, about 0.24 tok/s, close to what two rounds can see, so the no-gain result is weaker evidence than it reads. The cause of no-gain stays open.

### Row split of down

Part holds [853, 2304] instead of [5120, 384].

- Arena: same nibbles and scale codes, EW and EWmin shrink from 2 x 5120 to 2 x 853 bytes. EW becomes the whole-row maximum directly, so no EWmax across parts.
- GPU: this is the expensive piece. `load_hot` and `_arena_load_parts` copy w2 as K-slices (the 983,040 B pieces), `contract_expert_arena_infos` exports the same block, and `kt_stream_prefill` segments and the prefill path all use that shape. A second CPU-only copy would need a third more weight memory, which is out.
- Scratch per route: BufferA down plus BufferC down, 44,416 to 27,560 B, -38%.
- Traffic: 230,400 B a route falls to 23,040 B of BF16 h, 10x less, but all-to-all and before down. Shipping digits instead is 92,160 B, so ship BF16 and quantise 2304 values on every part.
- Numerics: this is the single-part contract literally (`v2_linear`): EA over 72 groups, EW over the row, exact int64 at base EA + EW - MARGIN, one RNE, exact 2^B, BF16. Group boundaries stay at multiples of 32, so digits are equal. The K-merge equals it whenever no term truncates, and `forward_routes_binding` throws otherwise. So it should stay bit-identical on all accepted data. Needs a paired run.
- Two edges. Fold budget falls from 12 (G = 12) to 10 (G = 72); observed SA 2-6 plus SW 3-4 can reach 10, past that the general kernel takes over, exact but slower. Folding per K-slice through the kernel's g0/g1 keeps today's speed and is exact. And in the general kernel a term below MARGIN with zero low bits passes the K-merge's lost-bit count but fires the hook on a whole-row base, so a row split can throw where today it runs. No such case on recorded data, but it has to be checked.
- Barrier: every part needs all of h, so the free run between gate/up and down ends. Replay of max(pre) + max(down) against max(pre + down): +5.5 us a call fast, +8.5 slow. Measured by replay, 0.22-0.34 ms.
- Sum (estimates): removes gap 5 and merge 5 + 6.8 n, 24.5 us at n = 2.1. Adds barrier 5.5, gather 1.4, part sync 3, quant of 2304 about 5, finish 2. Net -7.5 us a call, **0.3 ms** in the fast state. About 1.8 ms while slow, so about 0.75 ms blended. If the copy state is fixed first, only the 0.3 ms is left.

## 4. Last part behind the mean

Pooled, pools 1-5, fast state: 30.4 us a call, about 1.2 ms a token. By trace: English 23.9, Python 31.9, Japanese 17.0 us. Slow state 77.7 us. With pool 0 discounted by 1.32 and kept in as a sixth pool: 1.9-2.1 ms on English and Python, at their higher route counts. Measured.

Split of last minus mean, fast, five pools, us a call: English start 0.2 + service 22.3 + end-of-phase tail 1.1 + other 0.3; Python 0.2 + 30.2 + 1.1 + 0.4; Japanese 0.3 + 15.4 + 0.9 + 0.4. So 90-95% is service time. All seven workers run, start within 0.3 us, and the tasks are the same size, so it is memory and placement, not scheduling. That reading is an inference: traces cannot see inside a task.

Placement regression, us per route:

- Pool 0, route in the moved extent: +66-71 gate/up, +34-37 down, so about +100. This is the old 32%. It is per route, not per worker: pool 0's median task equals the others', with p90 of 45-126 us. r16 halves that extent and keeps it on the socket.
- Pool 3 (node 4), spilled route: +6-10 gate/up, +5-7 down, so +12-17. Plus a standing 4% per task.
- Other pools: spill costs nothing or reads slightly negative, -8 to +2.

Work stealing would cost 2-3x on a stolen task, so the 1.2 ms cannot all be recovered.

## 5. Ranking (gains are estimates)

| Rank | Piece | Size | Likely gain | Work |
|---|---|---|---|---|
| 1 | Pace or time slot copies; diagnostic first | 0.9-1.8 ms steady, 0.6-3 early | 0.5-1.2 ms | Small, Python round worker. Bit-neutral. Check share with the replay tools, since slower landing costs share |
| 2 | One x digit set per part at qlen 1 | 1.2 us a route of quant, plus digit misses | 0.05-0.25 ms | About 10 lines. All routes share x, so digits are equal by construction |
| 3 | Merge run from the parts' own job; lock-free wake | Gap 5 us, skew 3-4 us, start 8 us | 0.1-0.3 ms | About 25 lines, worker pool |
| 4 | No phase barrier inside a part, per-expert task chain | 11.7 us idle, (1 - 1/n) of it recoverable, plus 2 us of dispatch | 0.2-0.3 ms | About 60 lines, order only |
| 5 | Row split of down | Merge plus gap 1.0 ms | 0.3 ms alone, about 0.75 if rank 1 fails | Large: arena, loader, GPU copies, prefill, identity |
| 6 | Stealing across parts | 1.2 ms | Up to 0.5 ms | Invasive; remote tasks 2-3x |
| - | EW normalised at the part, 79 MB a node | 11% of merge bytes | 0.1 ms | Moderate; node 0 is tight |

Rank 2's digit gain has support only from the pre-read test (shared digits; weights pre-read gave nothing, a full repeat 25%), which does not separate digits from codes and EW. A one-route first-task test was tried and is confounded by selection: the 30.6 against 20.8 us reading is explained by slow first tasks taking one task. Test it on the pre-read bench with per-expert digits, before building.

Gate/up and down at 7.8 ms are compute plus memory on ruled-out levers, so nothing is proposed there.

## Corrections to existing notes

- "57 KB a route" in `ADA_TIMELINE_RESULTS_r1.md` and `tools/build_ada_ref_r1.py`: a part's partial is 5120 x 8 = 40,960 B.
- Merge "1.2-1.5 ms" is a blend: 0.77 fast, 2.10 slow at 85 routes.
- Node 0 "all seven workers 32% slower" reads as uniform. It was a per-route placement cost.
- `pool_path.py` prints the eighth field as a wall/cpu ratio; that field is `sched_getcpu()`.
