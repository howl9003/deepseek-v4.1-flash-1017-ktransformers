# Ada decode timeline on the live top-k build, 1 Nov 2026

Superseded on rate by `ADA_CPU_WINDOW_RESULTS_r1.md`: breadth 31.25, exact. Its corrections apply here: the More slots row has no KV trim behind it, and the merge gave nothing.

The GPU serial zone is now 15.3 ms, down from 20, and every bit of that fall is the kernels we replaced. The largest cost left is the CPU: on breadth rows the GPU sits idle 8 ms or more a token waiting for CPU routes, and the CPU finishes last in 75-83% of windows. So the next gains are on the CPU side, not in GPU kernels.

Most of the unexplained top-k gain did not turn up. The timeline lifts the direct saving from 1.18 to 1.44 ms and refutes my L2 guess, but about 0.9 ms stays unattributed.

Three captures of 64 tokens from token 512 of a 1,024-token request, same workloads as 31 Oct, on the build timed at 29.68. Trial `ada-nsys-topk-20261101-r1`, data in `nsys-topk-r1/`. Diagnostic only: no acceptance, selectors default off, server unloaded.

## Where a token goes

31 Oct kept build in brackets.

| ms per token | English technical | Python | Japanese |
|---|---|---|---|
| Serial GPU work | 15.32 (19.97) | 15.35 (19.83) | 15.72 (19.55) |
| GPU part of the expert window | 7.26 (8.88) | 7.68 (8.63) | 9.58 (10.82) |
| GPU idle, waiting for the CPU | 8.56 (7.02) | 8.06 (8.91) | 1.97 (1.67) |
| LM head and tail | 2.17 (2.19) | 2.17 (2.19) | 2.17 (2.19) |
| **Total** | **33.32** (38.06) | **33.27** (39.57) | **29.45** (34.23) |
| Median token period | 33.53 (38.31) | 33.45 (39.35) | 30.01 (34.29) |
| Windows where the GPU finished last | 17% (34%) | 25% (28%) | 60% (77%) |

Serial content, ms per token: Route 3 6.17, `wo_a_parts_kernel` 2.07, attention gather 1.53, Engram matmuls 1.35, hc mix and sinkhorn 0.90, `_partials` 0.38, `_pack_kernel` 0.35, small torch kernels 0.25 (down from 2.06 and 1,506 kernels), and 0.57 extra on the eight indexer layers. Layer 1 also carries a host wait of about 0.5-0.9 ms beyond its matmul, varying by prompt. A plain layer's serial is 323 us, down from 397.

The tail is at its floor. `Kernel2`, the LM head, is 2.14 ms.

## The unexplained gain

**Direct saving.** The eight indexer layers cost 2.01 ms over plain layers on 31 Oct and 0.57 ms now, in all three captures: ratio 2 layers 212 to 64 us, the source 346 to 171, consumers 258 to 53. So top-k saved 1.44 ms directly, not the 1.18 I priced in isolation.

**What the timed run saved.** 2.47 ms on the breadth median. Averaging seconds across the five rows that exclude the Chinese row gives 2.34 ms, so the median is not flattering it. That leaves about 0.9 ms, or 0.9 s a request.

**Ruled out:**

- *Cleaner L2, my guess in the top-k write-up.* Wrong. Kernels I did not touch run at the same speed: Route 3 6.23 to 6.17 ms, gather 1.53 to 1.53, LM head within 2%.
- *Stalls.* Neither build has many; they cost 0-0.3 ms a token in the stamps, and no less on this build.
- *CPU workers sleeping through longer serial zones.* Windows after indexer layers matched plain layers in both captures, for example 386 and 397 us on 31 Oct.
- *Graph launch.* The API fell from 13 to 6 ms with the node count, but launches queue about 30 ms ahead of their first node, so nothing waits on them.

The whole serial fall is accounted for directly: wo_a 1.86, top-k 1.44, glue 1.08, about 4.4 of the 4.5-4.65 ms on breadth rows. So the missing 0.9 ms is not in steady GPU serial work. It may be in the windows or the request's early tokens, or partly noise, since the top-k rounds sat 0.75 tok/s apart. A capture of the wo_a build could isolate it, but the gain is banked and that run would change nothing, so I am leaving it.

## The CPU window

A window lasts as long as the slower side. On breadth rows that side is the CPU: about 395 us a layer against 182-192 us of GPU work. Japanese, with far better slot share, runs 289 us and the GPU is last in most windows.

The capture's replay recomputes each measured window. Estimates, ms per token:

| Change | English technical | Python | Japanese |
|---|---|---|---|
| Measured | 33.31 | 33.25 | 29.41 |
| 1 ms off the serial zone | 32.31 | 32.25 | 28.41 |
| 50 us off each window's GPU part | 32.97 | 32.75 | 28.21 |
| CPU lead down 20% | 31.69 | 31.74 | 29.11 |
| CPU lead halved | 29.26 | 29.46 | 28.65 |
| CPU never last | 25.22 | 25.67 | 27.89 |

On breadth rows a millisecond off the CPU's lead is worth about a millisecond of token, for 8 ms. The last row is the bound, about 39 tok/s, not a target.

The GPU has room. It is idle for over half the window on those rows, so more experts on the card would shorten the CPU's side and use time it is already wasting. Slots are the limit, at 950.

I have not split the CPU window itself. Route compute, imbalance across the 42 workers and merge (1.2-1.5 ms, measured earlier) have not been timed on this build.

## Early tokens

Stamps from these requests, median ms per token:

| Tokens | English technical | Python | Japanese |
|---|---|---|---|
| 2-32 | 43.2 | 39.4 | 40.6 |
| 32-128 | 36.7 | 35.1 | 31.0 |
| 128-256 | 33.2 | 32.0 | 29.4 |
| 256-505 | 32.0 | 32.9 | 30.1 |

The climb to the steady rate costs 0.4-0.8 ms averaged over a 1,024-token request. The second row is slots adapting. The first row is 10 ms high on every prompt, including Japanese, so it is not only share, and its cause is not measured. It matters more on short replies than breadth shows.

## Caveats

- Tokens inside the range run about 1 ms above their neighbours on the two breadth rows (33.3 against 32.0-32.9), up from 0.4 ms on 31 Oct. Japanese does not. The stream's steady rate is about 32.3 ms, so the CPU wait above may read slightly high.
- The 31 Oct build also had the old slot rule, so changes in the window rows mix several builds. Only the serial rows separate cleanly.
- No timeline exists for the glue or wo_a builds.

## Remaining levers, ranked

1 ms a token is about 3% of breadth. Gains are estimates.

| Lever | Size now | Likely gain | Needs |
|---|---|---|---|
| Split the CPU window | 8 ms idle on breadth rows | none itself; sizes the next three | A worker timestamp run on this build |
| Faster merge | 1.2-1.5 ms of the CPU window | up to about 0.7 ms, near 1:1 | Vectorising the per-column shift |
| More slots | 950 now | about +8 points of share replayed, perhaps 1.5 ms now the CPU is last | A KV trim that changes the pinned capacity; your call |
| Per-layer slot split | layer 0 served 42%, layer 25 68% | about 0.3-0.4 ms | Small change in `_plan` |
| Engram host wait, layer 1 | 0.5-0.9 ms serial | not known | Find the wait in the row-store lookup |
| Engram through Route 3 | 1.35 ms serial | 0.86 ms, measured on its overlay | Not bit-identical until a paired numerical run; your call |
| `_fused_gather_attn` | 1.53 ms serial | not priced | Cold-chain pricing first |
| First 32 tokens | 10 ms a token high | about 0.3 ms on a 1,024-token request, more on short replies | Cause not measured |
| Live `select_candidate_block_ids` | about 100 us | about 68 us | Torch's sorted tie order |
| Skip the `-inf` tail store | | about 35 us | Small kernel change |

Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
