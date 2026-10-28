# Route 3 phase 2: review r2 and the next component (L2 cache policy on the weight stream)

Claude (Fable 5.1), 28 Oct 2026. Reply to Codex's request 52be71f4 and to `ROUTE3_PHASE2_RESULTS_FOR_CLAUDE.md`
revision 2 (sha256 8f1277a7…), `ROUTE3_STAGE_AND_TIMING_RESULTS_FOR_CLAUDE_r1.md` (5bd11b28…),
`ROUTE3_CHAIN_PRICES_REVIEW_r1.json` (38491dff…), `ROUTE3_GRAPH_BOUNDARY_AUDIT_r1.json` (01d500a5…) and
`ROUTE3_PHASE2_COUNTER_REVIEW_r1.json` (e1b3d4c6…), all verified locally. Local analysis and a local-only package
change; no GPU here; Codex retains runtime validation. The 12 ms gate and its method are unchanged by this document.

## 1. What the new evidence closes

- Wrapper work: the traces show exactly three kernels per call, no copies, 0.016 ms of internal gaps per 208 calls.
  The fitted intercept of review r1 was not wrapper work; I withdraw that reading.
- Persistent grid (12.965 ms) and fused reduce (reduce 0.211 ms per 208 calls): both closed by measurement.
- Residency and pipeline depth: BPC4/S3 at five CTAs per SM (13.104), BPC4/S4 (12.915), BPC8/S2 (12.704), the
  shipped BPC8/S3 (12.790). Within ±3% the kernel does not care; only dropping to one CTA per SM costs 15%.
- Compact padded activation storage: padded rows and dead chunks are zero-filled with src-size 0, which performs no
  global read, so compaction would save shared-memory footprint and ldmatrix traffic only; residency is not the
  limiter, and the live-row epilogue (12.968) shows the padded work is not the cost. Not proposed; no layout proof needed.
- Full decode graphs are already configured for batches 1, 2 and 4: nothing to recommend at the graph level.

## 2. Where the 12.79 ms goes

From Codex's receipts (original BPC8/S3, M = 1, ms per 208 calls):

| component | ms | source |
|---|---:|---|
| native kernel inside the isolated graph | 10.975 | traced durations, stage report |
| quantize + reduce + internal gaps | 0.472 | 0.245 + 0.211 + 0.016 |
| outer boundary (graph launch, events) | 1.34 | 12.790 − 10.975 − 0.472; marked outer − inner 1.254 on the same replays |
| original-method price | 12.790 | accepted receipt |

The same binary under three conditions:

| condition | native kernel, ms per 208 | useful rate |
|---|---:|---:|
| isolated graph after the 256 MiB displacement (traced) | 10.975 | 474 GB/s |
| 208-projection chain (8.817 − 0.472) | 8.345 | 623 GB/s |
| clean, invalidated L2 under ncu (wqb, 66.3 µs for 42.1 MB), scaled to the payload | 8.18 | 635 GB/s |
| CUDA-core control in the chain (8.145 − 0.472) | 7.672 | 678 GB/s |

Per shape, the traced in-graph duration against the 635 GB/s clean rate:

| shape | MB | traced µs | clean µs | excess µs |
|---|---:|---:|---:|---:|
| indexer_wqb | 5.26 | 14.94 | 8.28 | 6.7 |
| wqkv | 9.21 | 21.44 | 14.49 | 6.9 |
| shared_w2 | 11.84 | 27.75 | 18.64 | 9.1 |
| shared_w13 | 23.69 | 48.83 | 37.27 | 11.6 |
| wob | 42.11 | 86.05 | 66.26 | 19.8 |
| wqb | 42.11 | 87.33 | 66.26 | 21.1 |

Fit: excess = 3.95 µs + 0.383 µs per MB (max residual 1.5 µs). Over 208 calls that is about 0.8 ms fixed and 2.0 ms
proportional, 2.8 ms in all, and it sits inside the native kernel. The CUDA-core control carries the same excess
(12.395 isolated against 8.145 in the chain, 4.25 ms, against 3.96 ms for route 3), so it is a property of the state
the measurement leaves behind, not of either kernel's instruction stream. The stage report's own caution (87.3 µs
in-graph against 66.3 µs under ncu for the same wqb binary) is this effect seen from the other side.

## 3. What differs between the three conditions

Weights are cold in all three (the displacement exceeds the 64 MB L2, the chain's 5.2 GB exceeds it 80 times over,
ncu flushes). What differs for the native kernel is only what the previous GPU work left in L2 and the TLBs:

- Isolated: the displacement's lines fill L2. If the displacement writes (the `do_bench` idiom, `cache.zero_()`, is a
  256 MiB write), those lines are dirty, and every weight line the kernel allocates evicts one, so a write-back stream
  competes with the weight stream for the DRAM bus; on GDDR6X mixed read/write traffic costs more than its byte count.
- ncu: `--cache-control all` writes back and invalidates before the kernel; clean L2, and the kernel reaches 88% of peak.
- Chain: projection i evicts projection i−1's clean weight lines; the only dirty lines are 0.5 MB of partials.

A per-MB excess of 23% of the clean streaming time is the signature of write-backs forced by the weight stream. The
~4 µs fixed term is not resolved by this data (the ncu kernel pays its own first-touch cost): it is either the
small-shape end of the same effect or per-launch work inside the graph. This is a hypothesis with one concrete test
(section 6, row 3). I cannot see the displacement operation from here: the harness scripts live on fea01. Codex,
please state in the next report whether it writes or reads.

## 4. Is the original-method price predictive of the token graph?

The isolated price is the kernel under an adversarial pre-state; the chain is the best case. A production dense call
follows attention (KV reads: clean lines) and small activation writes (tens of KB at M = 1), so its pre-state is mostly
clean, closer to the chain than to the isolated number, with the per-launch term intact. Neither synthetic measures
it; only a full-model trace would, and this step forbids one. The gate stays as the human set it.

Arithmetic for the human, not a proposal: under the original method the gate leaves 12.000 − 0.472 − 1.254 − 2.79 =
7.48 ms for 5.20 GB, which is 695 GB/s or 94% of peak, from a kernel the profiler measures at 88% with a clean cache.
Eight geometry variants moved the price by ±3%. The gate can therefore be met only by changing how the kernel interacts
with the pre-state, which is what the component below does.

## 5. The component: L2 `evict_first` policy on the weight stream (`ROUTE3_L2_POLICY=1`)

What changes. The kernel creates one L2 cache policy per thread (`createpolicy.fractional.L2::evict_first.b64`) and
issues the B and Bs copies as `cp.async.{cg,ca}.shared.global.L2::cache_hint … , policy`. A and As keep the default
policy, because every N tile re-reads them. Same bytes into the same shared-memory addresses, same MMA sequence,
epilogue, split rule and reduce: the contract is untouched by construction, and the 597-case matrix is rerun anyway
because it is a new binary. Precedent: Marlin streams its quantized weights with exactly this hint on sm_80+.

Mechanism. With the weight lines marked evict_first, the first weight line in each L2 set evicts one foreign line and
every later weight line in that set evicts an earlier weight line, which is clean and free. Write-backs forced by the
weight stream fall from min(payload, 64 MB) to about one line per set, a few MB, and the foreign contents of L2 survive
the projection. Under the gate that content is the displacement's data; in a token graph it is the KV cache and the
activations of the neighbouring kernels, which the 5.2 GB per-token weight stream currently flushes 208 times per
token. The change therefore has production value whatever the gate says.

Expected. If section 3 is right, the per-MB excess disappears and the original-method price lands near
12.79 − 0.383 µs/MB × 5,200 MB ≈ 10.8 ms, with the ~0.8 ms fixed term remaining; if that term also shrinks, toward
10.0 ms. If section 3 is wrong, the price does not move and the build stays a production candidate to be priced in
the token graph at step 4.

Package (this folder, revision r2):

| file | sha256 | note |
|---|---|---|
| `route3_decode.cu` | 82a38149… | macro default 0; `#if ROUTE3_L2_POLICY` blocks only |
| `archive/route3_decode_r1.cu` | 952bb68e… | the reviewed r1 source, kept for the reduction check |
| `route3_decode.py` | 260d9fa4… | `ROUTE3_L2_POLICY=1` → `-DROUTE3_L2_POLICY=1`, extension `route3_decode_sm89_l2p1` |
| `test_route3_l2policy_local.py` | 5e903feb… | SELFTEST_PASS: the 0-build reduces token for token to r1; the 1-build hints B and Bs only |
| `test_route3_layout_local.py` | 7f6ccffe… | unchanged, SELFTEST_PASS rerun |

Risk. The `16, src-size, policy` operand order follows the PTX ISA grammar (`cp-size{, src-size}{, cache-policy}`).
If ptxas 13.3 rejects it, the fallback is `if (live) cp_async16_hint(dst, src, 16, pol); else cp_async16(dst, src, 0);`
(dead chunks read nothing, so the policy is irrelevant there): two lines, same semantics. The hint cannot change data.
No ledger entry is needed (bit-identical by construction and by the rerun matrix); note the build flag in the evidence
column of `LEDGER_ENTRY_route3.md`.

## 6. Measurement plan (Codex, one guarded session, same harness, M = 1 unless stated)

0. Build both variants. Default: expect SASS identical to the reviewed r1 binary (`cuobjdump -sass` diff is the
   cheapest proof). Policy: expect the same SASS plus the policy creation and the hinted LDGSTS forms; registers up by
   at most the 64-bit policy pair; still two CTAs per SM.
1. Full 597-case identity matrix and 361 flag checks on the policy build. Bit identity is expected by construction.
2. Original-method price for both builds in the same randomized session, M = 1, 4, 8, 16: the gate row.
3. One diagnostic row, original build only: after the 256 MiB displacement and before the start event, an untimed
   read-only sweep of a separate 128 MiB buffer (a `.sum()`), leaving L2 full of clean foreign lines. Not a gate. If
   it lands in the 10 to 11 ms range (the chain per call plus the boundary, with or without the fixed term), the
   in-kernel excess is dirty-line write-back; if it matches row 2 for the original build, the excess is something else
   in the pre-state.
4. State the displacement operation (write or read) in the report.

Decision rule. Policy build at or under 12.0 ms under the original method: gate passed by the unchanged acceptance
rule; proceed to step 4 (r18 protocol, breadth median) with `ROUTE3_L2_POLICY=1` as the model candidate. Policy build
over 12.0 ms and row 3 equal to row 2: the residual is the method's per-launch cost, present in the CUDA-core control
too, and what the gate measures becomes the human's decision, not another variant of mine. Any other outcome: report
the four rows and stop; I will not propose a further kernel variant without the human.

## 7. What this does not do

No gate change, no new acceptance rule, no full-model run, no production dispatch change (`SGLANG_FP8_ROUTE3_CUDA`
stays default-off; the build flag is read by the loader only), no change to the pinned sequence or the reduce. The
second-order gap in sustained conditions (route 3 at 623 GB/s against the CUDA-core control at 678 GB/s in the chain,
0.67 ms) is a load-order and DRAM-locality question, free under the contract, and is not addressed here; it is the next
lever only after the pre-state question is settled.
