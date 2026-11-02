# Ada CPU window, 1 Nov 2026

The timeline said the CPU was the limit, so this round went after the CPU window. Breadth is now **31.25 tok/s**, up 5.3% from 29.68 and 58% from the 19.80 we started with on 31 Oct. Numerics are exact, the serving gate passes, and prefill did not move. Selectors stay default off and the server is unloaded.

All timing is two rounds on fea01, six nodes, 42 workers, 320 W, tree r6, arm 0 Route 3, prefill path A on.

| Step | Breadth tok/s (rounds) | Workload range | Verdict | Trial |
|---|---|---|---|---|
| Live top-k | 29.68 (30.05, 29.30) | 28.6-31.2 | Base | `ada-perf-topk-20261101-r1` |
| Node 0 rebalance off | 30.60 (30.55, 30.65) | 29.2-31.6 | Fast, but see Memory | `ada-perf-norebal-20261101-r1` |
| + KV lend on every tensor | 30.68 (30.60, 30.75) | 29.6-32.3 | Kept, 0.25 s a request | `ada-perf-kvlend-20261101-r1` |
| + AVX2 merge, chunk 128 | 30.73 (30.60, 30.85) | 29.3-32.0 | **Dropped, no gain** | `ada-perf-merge-20261101-r1` |
| + per-layer quota, rounds every 4 | 30.93 (31.00, 30.85) | 29.4-32.0 | Kept, near the noise floor | `ada-perf-quota4-20261101-r1` |
| + Engram batched reads | stopped by the memory guard | | Lookups fixed; node 0 had to change | `ada-perf-rowaio-20261101-r1` |
| **Final: node 0 sixteenth, all the above but the merge** | **31.25 (31.35, 31.15)** | **30.3-32.2** | **Exact, gate passed** | `ada-perf-r16-20261101-r1` |

That is 1.7 ms a token on the breadth median, 33.70 to 32.00, and 1.63 s on the mean 1,024-token request. Rows, round 0 then round 1: Chinese 30.4, 30.3; English 31.8, 32.2; English technical 30.5, 30.3; math 32.1, 32.1; JSON 31.8, 31.5; Python 30.9, 30.8. Japanese medians 33.1, 33.8 against 31.65, 31.5. Prefill 35K: [69.10, 55.24], [51.86, 50.81] s against [69.87, 55.79], [52.22, 51.36].

Steps are stacked, so each row's gain depends on the rows above it, and small differences sit inside the noise. Averaged over the six rows, a request took 34.45 s at the base, then 33.58, 33.33, 33.18 and 32.82 s across the kept steps.

## What the worker traces showed

Existing traces from the kept build were enough to split the window; no new load was needed.

- All seven workers on node 0 ran 32% slower per task: 38 us against 29 on gate/up, 16.5 against 12.5 on down. `KT_EXPERT_ARENA_REBALANCE_NODE0=1` had moved an eighth of that part's weights to the other compute nodes, three of the five across the socket. That part finished last in 44% of calls.
- The fastest pools run at about 33-40 GB/s a node. The earlier sweep found kernel prefetch and smaller tasks gave nothing, and the pre-read test below puts the memory share at no more than about 30% of a task.

## What changed

**Node 0 sixteenth.** `KT_EXPERT_ARENA_REBALANCE_NODE0=3`, native `ada-native-r16-20261101-r1`, kt_ext fd5072c3. Node 0 now moves a sixteenth of its stripes, all to node 3, its own socket's spill node. Layout test `node0-sixteenth-r1/test_arena_r16.cpp`: ALL_PASS, value 1 equals the accepted header at every 2 MiB, value 3 changes only node 0's part.

**Engram batched reads.** `DSV41_ENGRAM_AIO=1`, `engram-aio-r1/`, `librow_store.so` 45850492. The capture showed the GPU waiting for the host at layer 1 on 34-56% of tokens, 0.6-2 ms each time. Layer 1 lookups took about 1.1 ms against 0.35 ms for the same 24 rows at layer 14: they start with the token, when the service cores are busiest, so waking 16 I/O threads was slow. A call's misses now go down in one `io_submit` and the callback thread polls them, so nothing has to wake. Lookups are 0.25 ms on both layers. Prefill-sized lookups stay on the thread pool. Test: 107,370 rows byte-equal to the thread pool on the real table, 2,250 of them across page edges, no errors.

**KV lend on every tensor.** `KT_PREFILL_KV_LEND_ALL_TENSORS=1`, `kv-lend-all-r1/`, overlay 281ce314. Our port of 05yuki's 7fef854e behind a selector. Eight KV tensors lend instead of one, 1.45 GiB instead of 0.53, 1,005 slots instead of 952.

**Per-layer quota.** `KT_KV_HOT_LAYER_QUOTA`, `layer-quota-r1/`, overlay c96c434a. Slots follow each layer's demand, 15-34 a layer instead of 25-26. Unset, `_plan` equals the accepted function on 400 cases. Replay gave 3.5% fewer CPU routes on prompts the quota never saw, but the measured GPU share barely moved (64.8% against 64.5%), so treat the gain as small.

**Rounds every 4 tokens.** `KT_KV_HOT_DYN_EVERY=4`, no code. It ran together with the quota, so the two are not separated.

## Dropped

**AVX2 merge.** Four times faster on one thread (15.6 to 3.9 ns a column) and bit-exact on 1.87 million adversarial columns, but it changed nothing on the served rows. So the merge's cost is pulling six parts' partials across nodes, not arithmetic. The code stays in `merge-avx2-r1/`, out of the build, and the final native does not carry it.

**Pre-reading routes into L3.** Tested before building (`l3-preread-r1/test_l3_preread.cpp`: the real kernel, seven pinned workers, one node, served shape). A cold matrix takes 30 us; after the workers have read it first, 27-36 us; a straight repeat, 21-25 us. Work stealing puts tasks on other cores, so a pre-read mostly does not land where it is used, and even the best case is only about 30% faster. Closed.

## Memory

Turning the rebalance off outright was a mistake I had to back out. It passed four loads with node 0 at 10.9-11.6 GiB, then on the fifth a 35K prefill dropped node 0 from 10.6 to 4.2 GiB of usable memory in seconds and the per-node reserve stopped the run, as it should. Kernel reclaim had moved 6.5 GiB of file pages to the active list, where the guard does not count them. The rebalance was added on 27 Oct for this same condition. Guards were not touched, and the run was not retried.

| Node | Rebalance on | Off | Sixteenth (perf, identity, gate) |
|---|---|---|---|
| 0 | 14.5-15.9 | 10.9-11.6, one stop at 4.2 | 11.9, 12.6, 12.3 |
| 2 | 8.1-8.8 | 7.2-9.5 | 9.8, 9.2, 8.0 |
| 3 | 12.9-14.5 | 13.0-13.7 | 11.4, 11.1, 10.9 |

A sixteenth did not win back as much node 0 headroom as hoped. Three clean loads do not prove the margin: a repeat of that 6.5 GiB swing from 11.9 would land near 5.4 GiB, above the 4 GiB guard by little. This should be watched if the build is served.

## Evidence on the final build

- Numerics: 11,496 scored positions EXACT vs C1 (`ada-identity-final2-20261101-r1`); 4K and both 35K prompts have their 40 layer-flag checks clear.
- Long cases: text, ids and every logprob equal the live top-k build's on all three prompts.
- Scale check, new in `client_ada_identity_r2.py`: a 2,048-token streamed prefill on the fresh server, then 640 tokens of English, `/flush_cache`, and the same prefill. Text, ids, selected and top logprobs all exact, 2,710 swaps logged. The log counts all swaps, not resident rows alone.
- Serving gate (`ada-gate-final2-20261101-r1`): all seven cache checks pass, replays equal, cold, device and host paths exact with logprob delta 0.0, host-reload gate SHA unchanged, cleanup empty.

Still untested, as on earlier builds: a raised flag on a live server, an 8-row capture in the gate, and anything past a 35K prompt. There is no timeline on this build.

## 05yuki issue 9

The issue is a qualification report; his comment there points at 23dd4ea.

| His commit | Here |
|---|---|
| 23dd4ea, resident rows keep the wrong scales after a swap | Same bug, found 27 Oct and fixed differently: `KT_PREFILL_LEND_EXPERT_SCALES=1` restores scales from the mirror. On, and the scale check above confirms it on this build |
| dc21bdb3, pool clear crashes while slots hold memory | Same fix, `KT_PREFILL_SWA_CLEAR_RELEASE_HOT=1`. On, and the check's flushes ran clean |
| 7fef854e, KV lend reaches every tensor | Missing until today, now ported |

## Precision ledger

Non-bit-identical steps added: none. Pages, slots, swap timing and how rows reach memory cannot change a bit, and all of it is measured EXACT.

## Corrections

- The timeline write-up listed more slots through a KV trim as a decision for the owner. There is no such lever: the KV pool is about 1.9 GB and the lend already reaches 1.45 GiB of it. Weights take 11.75 GB, and 1.5x slots would need 8 GiB more than the card has.
- I estimated the merge at up to 1 ms. It gave nothing.

## Remaining levers

1 ms a token is about 3% of breadth. Gains are estimates.

| Lever | Size now | Likely gain | Needs |
|---|---|---|---|
| Work stealing between parts | last part 2.5 ms behind the mean on the old trace, less now | up to about 1 ms | Thread-pool change; invasive |
| Engram through Route 3 | 1.35 ms serial | done: see `ADA_ENGRAM_RESULTS_r1.md` | |
| `_fused_gather_attn` | 1.53 ms serial | done: see `ADA_GATHER_RESULTS_r1.md` | |
| First 32 tokens | about 10 ms a token high | about 0.3 ms on a 1,024-token request, more on short replies | Cause not measured |
| Lower the 1 GiB spare floor | about 27 slots per 512 MB | about 0.1 ms | Risk of running short of VRAM |
| Live `select_candidate_block_ids`, skip the `-inf` tail store | | about 0.1 ms | Torch's sorted tie order |

Selectors stay default off and the accepted server profile is unchanged. Nothing is pushed and PR6 stays held.
