# Precision ledger (KT + SGLang engine, all models)

Record each numerical change, its reference, acceptance evidence and limitations here.

## 28 October: dense FP8 projections on the Ada tensor core (route 3 preparation; no runtime change yet)

Scope: the five dense FP8 projections per layer (`wq_a`+`wkv` fused 1792x5120, `wq_b` 32768x1280, `wo_b` 5120x8192,
shared `w1`+`w3` 4608x5120, shared `w2` 5120x2304) plus the indexer projection, E4M3 activations with per-32 UE8M0
scales, E4M3 weights with 32x32 UE8M0 block scales, on fea01's RTX 4080 SUPER (AD103, SM89).

| Step | Knob | Precision | Evidence |
|---|---|---|---|
| Fixed split-K FP8 matmul, every M (prefill today; decode in the accepted r18 and contract r5 runs) | `SGLANG_FP8_FIXED_SPLIT_K=4`, `SGLANG_FP8_FIXED_DECODE_TRANSPOSE=1` | **not FP32 accumulation.** Each 32-wide K block is one `mma.sync.m16n8k32.e4m3` with a zero accumulator; on Ada that is the QMMA unit: products exact, two fused 16-product blocks, significands aligned to the largest exponent with 2 integer and 13 fractional bits, shifted-out bits truncated (no sticky bit), 20-bit adder, result truncated to fp32. Between blocks: FP32 multiply by the activation and weight scales and FP32 add in ascending K, four split partials reduced in a fixed order, one BF16 cast. Batch, tile and launch invariant on this machine (per-row bits equal for M = 1..768); bits are specific to the Ada block structure. | Khattak and Mikaitis 2025 (bit-accurate model validated on L40S and RTX 1000; AD103 probe pending), SageAttention2's RTX 4090 measurement (accumulator keeps 13 mantissa bits); PTX ISA: accumulation "with at least single precision", order and rounding unspecified. Identity receipts: `fixed-fp8-r1`, `clean-fp8-modes-r1`, `FP8_DECODE_PREPARATION.json`. |
| Dense CUDA-core fixed-order family (contract K3 runtime) | `SGLANG_FP8_FIXED_CUDA_CORE=1` | IEEE FP32 FMA chain in one fixed per-row order; exact products; more precise than the tensor-core path; different bits from it and from the reference | `KERNEL_CANDIDATE_PROGRESS_r3: selected_dense` (438 + 176 checks); K3 numerical review (26 gates) |
| Route 3 (proposed): decode kernel sharing the fixed split-K sequence | new default-off geometry selector, to be named | **bit-identical** to the fixed split-K path above by construction (same source and per-row sequence; identity test required); reverts the dense bits of K3 to the r18/r5 arithmetic | this draft: `six-node-performance-r1/kernel-steer-r1/route3-mma-r1/` |

Not bit-identical to each other: the Ada tensor-core path, the CUDA-core FP32 path, the integer-exact contract, and
the reference's own TileLang kernel (Hopper: 32-product fused blocks with 13 fractional bits; Blackwell: 25).
"The reference" for the dense projections is therefore a semantic contract (E4M3 x E4M3, scaled, FP32-class
accumulation), not a bit standard, and every one of these paths is a distinct cold context under the gates.
Cross-architecture bit identity exists only for integer-exact arithmetic (INT8 MMA with 32-bit accumulation, dp4a,
AVX integer dots).

## 23 October (morning): isolated indexer memory and PR-wrapper checks

Evidence: `investigations/progress-review-20261023/experiments-0845/RESULTS.md`.
No live serving source or launch setting was changed; Qwen restored after GPU tests.

- Indexer: existing upstream in-place ReLU/weighting plus experimental row-tile budgets 256/128 MiB versus current
  out-of-place/1024 MiB. All 24 GPU cases match score tensors bit-for-bit and top-512 sets exactly, on generated BF16
  operands with checkpoint H=32/d=128. Six CPU configurations also match with ragged causal masks. This is fixture-specific
  evidence, not universal or end-to-end model identity. FP32 weights would change type-promotion semantics and are outside
  this test. Changing chunk shape can select different GEMM algorithms; model/logit A/B remains required.
- SM89 PR #3: exact final wrapper passes all 399 GPU fixture checks, including concurrent independent graphs and eager/
  replay equality. Underlying kernel source matches the new pin; installed runtime remains da7e3f1c for this isolated test.
  Numerical differences from portable attention remain. Auto split is not bitwise batch-invariant; forced split 0 passes
  that check. Full rebased-model validation remains pending; the PR stays draft.
- Prefill route seeding: replay harness sanity checks only, no policy deployed and no real-route gain established.

## 22 October (evening): speedup ports on the HEAD tree (`head_rebase/ports/`, campaigns 8-13)

Every step below is on the isolated HEAD tree only. "Bit-identical" means proven by a kernel-level
bit test on fea01 and, where stated, by identical greedy completions end to end; see
`head_rebase/RESULTS.md` for the measurements.

| Step | Knob (default) | Precision | Evidence |
|---|---|---|---|
| One-pass multi-row CPU MXFP4 expert kernel (2..8 rows per expert from one weight decode) | `KT_MXFP4_BATCH=fast` (unset = old operator) | **bit-identical** to the old operator at every m and row: same per-row FMA chain, sum assignment, group order and hsum; only interleaving across rows and the slab parking differ | `bench/batch/kt_rows/out/test.txt`: 0 differing values, m = 1..16, 8 shapes, group 32 and 16, codes and fp32 scales, built with the real `-O3 -ffast-math` flags; bench bits check identical at every m |
| Same kernel, batch-invariant mode | `KT_MXFP4_BATCH=exact` (off) | reorder only: rows that the old operator put in 4-row blocks (m >= 4) now use the single-row even/odd fp32 chains and one final add, so every row equals a plain decode of that token (removes the "M4 vs remainder" boundary of 21 Oct); ~1e-7 relative | same test: 0 differing values against the single-row path at every m |
| Hot-slot swap policy and mechanics (hysteresis 1.25, dwell 2, byte cap, deferred tables, worker-thread copies, batched part loads, fast refill, route fold) | `KT_KV_HOT_DYN_HYST` default 1.0 -> 1.25, `KT_KV_HOT_DYN_DWELL` 0 -> 2; the rest 0 (`launch-env-hot-slots.sh` turns them on) | no new arithmetic; route fold and batched loads bit-identical (CPU tests, same int64 ids / fp32 weights / slot bytes); the policy knobs change *when* experts move, so outputs stay history-dependent exactly as the dynamic-slot entry above | `ports/hot_slots/REPORT.md`, `test_hot_slots.py` 17 checks ok |
| GPU glue: fused LSE merge + attention sink kernel | `KT_SM89_FUSED_MERGE=1` (0) | **bit-identical** to the torch chain (libdevice math, no FMA contraction, no FTZ, same rounding points) | GPU bit test PASS (`evidence/gpu_glue/test_merge_sink.gpu.log`) |
| GPU glue: shared expert forked after the CPU submit | `KT_SHARED_AFTER_SUBMIT=1` (0) | scheduling only, same kernels and join point | end-to-end identity below |
| GPU glue: one-kernel pinned-memory handoff pack / zero-copy read | `KT_HANDOFF_PACK=1` (0), `KT_HANDOFF_ZC_OUT` (0) | bytes only | GPU test PASS (`test_handoff_zc.gpu.log`) |
| GPU glue: task-queue spin, poller hint and pinning (C++) | `KT_TASKQUEUE_SPIN_US=2000`, `KT_MEMOP_POLL_HINT=1`, `KT_MEMOP_POLL_CPU`/`KT_TASKQUEUE_CPU` | no arithmetic | - |
| GPU glue: indexer top-k on the live length | `KT_IDX_LIVE_TOPK=1` (0) | designed bit-identical (same selected set incl. torch's tie rule, same output order, same slots); CPU interpreter test PASS; **GPU test pending** (campaign 13) after a Triton constexpr fix | `ports/gpu_glue/REPORT.md` section 1 |
| End-to-end check of the glue knobs | frozen hot set (`KT_KV_HOT_DYNAMIC_SLOTS=0 KT_KV_HOT_DYNAMIC_RESIDENT=0`) | the six greedy completions of the adaptation probe are **identical character for character** with every passing glue knob on against all off | `evidence/adapt-gg-frozen{,-base}/report.json` (campaign 11) |
| Single-launch dual-scope sparse attention on SM89 (SGLang's Triton kernel; `ports/single_list_attn`) | `KT_SM89_DUAL_SCOPE=1` (0), `_SPLITK` auto, `_CFG` 16,64,8 | **LOSSY at rounding, not algorithm**: one online softmax over window + compressed rows instead of two partials + fp32 LSE merge; QK as exact bf16 products accumulated fp32 by tensor-core mma; P rounded to bf16 for P.V (l sums the fp32 P); one final bf16 rounding instead of three; sink in the final denominator; split-K fixed-order fp32 combine; `auto` makes a row's result depend on the batch row count (`SPLITK=0` batch-invariant). GPU measurement vs fp64 (random real-shape inputs): mean abs error 3.4e-4 vs today's 4.6e-4 (std 1), 1.24e-3 vs 1.59e-3 (std 3); ulp p50/p99 0.30/8.1 vs 0.36/6.8; vs today's output 59-70% of elements identical, 27-33% one bf16 ulp off, <1% more than 16; max abs diff 3.1e-2 = 2.3% of the row RMS. 49/49 quality checks pass; greedy outputs can change at near-ties. | `evidence/single_list_attn/test_dual_scope.gpu.log`; campaign 15: +5/+4/+10% decode at 1/2/4 streams, prefill 379 -> 713-755 tok/s |
| Launch diet: fused act-quant + FP8 GEMV + split-K reduce, SwiGLU in the shared expert's down prologue, ticketed mHC seam (`ports/launch_diet`) | `KT_MULTI_GEMV=1`, `KT_FUSED_SWIGLU=1`, `KT_FUSED_HC_SEAM=1` (all 0) | **bit-identical**: the fused kernels replay the original kernels' operations (ue8m0 quant constants, GEMV layouts, split-reduce order, TileLang post fma contraction, RNE bf16 roundings); NaN payloads differ only with NaN inputs | GPU bit tests PASS (`evidence/launch_diet/test_multi_gemv.gpu.log`, `test_hc_seam.gpu2.log`); frozen-hot-set greedy texts identical 6 of 6 vs the all8 base (campaign 18) and vs the split6 base (campaign 22c); no throughput gain measured (campaigns 18, 22c) |
| FlashInfer native DSV4 sparse-MLA decode kernel, community SM89 retarget (`ports/flashinfer_sm89`, yhfgyyf/flashinfer 09259459, vendored as `kt_flashinfer_sm89`) | `KT_SM89_FLASHINFER=1` (0; **never applied to the tree**, tried and rejected) | **LOSSY by design, well beyond the dual-scope kernel**: Q NoPE rounded to FP8 e4m3 per (row, head, 64-dim block) with a ue8m0 scale, QK as e4m3 x e4m3 mma + fmaf scale multiply, softmax exp2 fast-math, P rounded to bf16 for the RoPE P.V and to FP8 e4m3 (per 64-dim V block, per chunk) for the NoPE P.V against the raw FP8 V bytes, bf16 split outputs merged in fp32 in split order; row-count dependent with the chunk heuristic. GPU measurement vs fp64 on real-shape fixtures: mean abs err 5.9e-3 vs dual 2.6e-4 (std 1), 4.0e-2 vs 1.2e-3 (std 3); ulp p50/p99 6.6/352 vs 0.32/9.1; max abs err / row RMS 0.28-0.74 vs 0.011-0.013; vs the dual-scope output 4.6-8.6% of elements identical, 24-30% more than 16 ulp off | `evidence/flashinfer_sm89/test_flashinfer_sm89.gpu.log` (campaign 25b): RESULT: FAIL (37 emulation-match bounds at 1-2.9% beyond 2 ulp), 521 structural checks ok; slower than dual-scope at every row count above one; not adopted |
| Mixed-chunk scheduling (antirez's prefill yield) | `--enable-mixed-chunk` (off; `launch-env-mixed-chunk.sh`) | not bit-identical for the ~20 decoder tokens computed inside a prefill forward of >= 512 rows: their routed experts run on the GPU streamer (16-bit weights, fp32 accumulate, bf16 combine) and their attention on the extend path instead of the decode graph and the CPU/hot-slot path; prefill rows' GEMMs run at M = chunk + decoders | `ports/serving_ports/REPORT.md` section 1; campaign 8 |
| DSpark speculative decoding, greedy | `--speculative-algorithm DSPARK`, block 2/3/5 | accepted tokens are the argmax of the multi-row verify forward; the CPU window batches the union of the rows' experts (per-part BF16 partial order can differ from a 1-row decode, the existing per-part class); the quality client refuses speculative servers, so only acceptance statistics and the degeneracy check were run | campaigns 6, 7, 10, 12 |
| DSpark compact (ragged) verify with the verify-cost scheduler (`ports/spec_scheduler`: per-request verify lengths, batch gate, skip windows, draft skipping, one CUDA graph per token tier) | `SGLANG_RAGGED_VERIFY_MODE=compact`, `SGLANG_DSPARK_COST_SCHED=1`, `SGLANG_DSPARK_RAGGED_FINE_BUCKETS=1`, `SGLANG_DSPARK_RAGGED_ROW_MAPS=1` (all off / static) | lossless in distribution (greedy commits the target verify argmax at every position; a skipped draft verifies only the anchor), **not bit-identical** to static DSpark or to plain decode: verify rows per request and per step change, so the CPU operator's rows-per-expert count and the graph tier change and near-tie argmaxes flip; integer row maps (Engram hash, token->request) are bit-identical for equal blocks | campaign 23b/23c: `test_engram_ragged.py --gpu` ALL PASS; greedy texts vs static DSpark 0 of 6 identical (first diffs at chars 114-268, one at 5); degeneracy check passed on every completion; not adopted (speculation loses to plain decode at every batch size on this hybrid) |
| DSpark opportunistic acceptance (ds4 default mode) | `SGLANG_DSPARK_OPPORTUNISTIC=1` (0; `apply_dspark_opportunistic.py`, **not applied**) | LOSSY, not distribution-preserving for sampled requests: accept while the draft equals the fp32 argmax of the verify logits, then sample from the full target at the first mismatch; argmax tokens over-represented; greedy unchanged | `ports/serving_ports/REPORT.md` section 3 (full wording); never run |
| Degeneracy check in the bench harness | `bench_matched.py`, `adapt_probe.py`, `bench_yuki.py`, `--fail-on-degenerate` | measurement only (repeated 4-gram ratio, distinct ratio, longest run, short reply) | every campaign 8-13 completion passed |

Refused or deferred lossy items seen in the engine survey (`ports/engine_survey/REPORT.md`): deferred CPU
experts (next layer's attention before the CPU experts land), int8 activations for the dense GEMV
(llama.cpp / Maya dp4a), a single online softmax over window + compressed rows (one-launch attention; the
fallback only if the fused merge had failed its bit test, which it did not), FP4/INT4 dense weights.

## 22 October: boundaries against the released reference (`checkpoint/inference/model.py`)

Read from the checkpoint's own reference code (`Linear.forward`, `Expert.forward`, the KV and
indexer paths) and the installed trees. No new quantization format is introduced anywhere; the
differences are where activations are quantized and where fp32 sums are rounded.

| Stage | Reference | Pinned bench / HEAD tree on fea01 | Direction |
|---|---|---|---|
| Routed experts, input | `act_quant` to FP8 (block 32, ue8m0) then `fp4_gemm` (W4A8) | CPU AVX2: BF16 input x MXFP4, fp32 FMA (W4A16); routed-input FP8 QDQ off; GPU slots: BF16 x MXFP4 fp32 | we keep more input precision; not identical |
| Routed experts, internals | gate/up to fp32, SwiGLU with clamps in fp32, `act_quant` again before w2 | CPU: gate/up rounded to BF16, SwiGLU (clamp kept), down input BF16; down partial per NUMA part rounded to BF16, fp32 merge, BF16 out | extra BF16 roundings per NUMA part (6 or 8); pool count is part of the numerics |
| Dense projections | `act_quant` FP8 block 32, tilelang `fp8_gemm`, fp32 accumulate | FP8 block GEMV / split-K, fp32 accumulate in a different order (05yuki `fp8-decode`) | reorder only |
| wo_a | convert.py dequantizes to BF16; BF16 einsum after `act_quant(o)` | o QDQ bit-exact; wo_a kept in FP8 blocks, fp32 w*s GEMV (M=1), multi-row kernel (M=2..4), dequant+einsum otherwise | we skip the BF16 weight rounding; not identical |
| Main KV cache | `act_quant(kv)` FP8 + BF16 RoPE part | fp8_e4m3 KV, V4 layout on SM89 | same format |
| Compressed KV and indexer | `fp4_act_quant(latent, 16, e4m3 scales)`; indexer q and k `fp4_act_quant(..., 32)` | SM89: V4.1 FP4 layouts unavailable (SM100 only); torch/Triton indexer with the FP8/torch logits path, `SGLANG_OPT_USE_TOPK_V2=0` | different selection arithmetic: top-k candidates can differ |
| Sparse attention | tilelang `sparse_attn` | SM89: Triton sparse decode with fp32 merge and three BF16 rounding points (also for prefill rows) | reorder; prefill through the decode kernel |
| Router | BF16 gemm then top-k | cuBLAS gemvx BF16->FP32 on SM89 (SM90+: `tiny_gemm`) | reduction order at the routing boundary |
| Engram | rows dequantized on lookup | same (file-backed adapter, `wkv` FP8 GEMV) | reorder only |
| hc mixing, norms | tilelang | torch/Triton paths on SM89 (`TILELANG_MHC_PRE=0`, `DEEPGEMM_HC_PRENORM=0`) | reorder only |
| Compiler | tilelang kernels | kt-kernel built -O3 -ffast-math (recipe default; FTZ/DAZ checked for the merge) | contraction/reassociation allowed in the CPU kernel |
| Expert placement | all experts, one path | 21 fixed (pinned) or dynamic hot slots (HEAD): which experts take the GPU path depends on routing history | not bit-identical, history-dependent |

Refused cuts, all off: deferred experts (`KT_DEFERRED_EXPERTS=0`), int8 activations (`KT_CPU_A4_INT8`,
`KT_MXFP4_G32_INT8`), the old Strata Q8_0 activation path, any requantized weights. Bit-identical
speedups: MXFP4 scale codes, PSHUFB activation conversion, sparse-valid-bound, NUMA batch merge
(order-preserving), fused route kernel.

## 22 October: isolated 05yuki HEAD tree on fea01 (not the pinned bench; no quality gate yet)

Tree `/scratch/aster/bench/05yuki-head-646326c` (SGLang main da7e3f1c + 05yuki tier 0/1
patches, KT 95009ea + identical kt-kernel-tree.patch). Campaign under
`investigations/yuki-diff-20261022/head_rebase/`. Numerically relevant steps, all
unqualified on this tree (no 49-check suite, answer-ID or logprob comparison has run):

- Dynamic hot slots and resident rows (`KT_KV_HOT_DYNAMIC_SLOTS=1`,
  `KT_KV_HOT_DYNAMIC_RESIDENT=1`, `KT_KV_HOT_EXTRA=auto`): 23-24 experts a layer move
  from the AVX2 MXFP4 path to the GPU MXFP4/FP32 Triton path, chosen by decayed routing
  counts every 16-32 forwards. Not bit-identical to the pinned bench and history-dependent:
  which routes run on the GPU depends on the preceding text and on free memory. 05yuki
  reports each eager call's slot output within 1.6x of the CPU's own bf16 rounding and a
  whole-call difference of 4.1e-3 median / 5.0e-3 max vs all-CPU (one bf16 rounding), on
  their machine. Our measurement: throughput only (RESULTS.md).
- Sparse-valid-bound in `flash_mla_sm120_triton.py` (loop to `valid_topk`): bit-identical
  on the pinned base (recipe_results/sparse-valid-bound-v1, 21 cases + graph vectors); the
  same one-line change on da7e3f1c, not re-run through that fixture.
- `KT_WOA_MULTIROW=1` on HEAD's wo_a dispatch (ports/wo_a_multirow): M=2..4 decode rows use
  the direct FP8 kernel (fp32 w*s) instead of dequant + bf16 einsum; M=1 unchanged. On the
  pinned base: 502 dispatch checks passed, 35/35 answer sequences matched, batched
  selected-logprob max delta 0.002454. On HEAD: ported by hand, component fixture not
  re-run (torch 2.14.1 / new Triton).
- `KT_NUMA_BATCH_MERGE=1` in the rebuilt extension: qlen 2-4 order-preserving six-way merge,
  bit-identical by construction and by fixture on the pinned base (2016 cases FTZ/DAZ on and
  off); same sources (hashes verified) rebuilt against the identical KT tree.
- Base change 1aa0e962 -> da7e3f1c: upstream kernels not audited beyond the patch-delta
  review (FP8 GEMV, wo_a GEMV, O-QDQ and engram prefetch unchanged; mHC/indexer/attention
  internals not compared).
- Unchanged: FP8 block GEMV, wo_a native FP8 GEMV at M=1, O-QDQ block 32, MXFP4 scale codes,
  routed-input QDQ off, no deferred experts, DSpark off, SM89 Triton sparse decode kernel.

Measured on r4 (all ports on, streaming sessions on): the 49-check quality suite passed
49/49; the session protocol passed all lifecycle checks, but retained-replay versus
cold-replay selected IDs differed in 3 of 8 comparisons and, where equal, the maximum
selected-logprob difference was 0.14-0.90 (pinned bench: all equal, max 0.0025). This is the
history dependence of the hot slots, not a port defect. Still owed: a matched answer-ID and
logprob comparison against v13 on the same prompts as a separate pass.

## 21 October: existing MXFP4 row variants reproduced in installed extension

mxfp4-row-order-v1 completed CPU-only using the donor's actual CPUInfer/MoE API,
unchanged extension/header and synthetic released-format weights. M1/2/4/6/8
identical input/routes,8seeds,6repeatcalls each, finite/poison/same-variant checks.
M6grouped4 and remainder2 differ for2seeds (9 and1 BF16coordinates); all repeated
calls stable. Grouped4 always equals M4, tail2 always equals M2; M1=M2 and M4=M8.
This demonstrates a deterministic arithmetic-order mechanism, not a data race in
these cases. It does not prove causality for the real captured MLP or quality
equivalence. No numerical/kernel changes were made and no scalar contract added.
Small hot-set timing illustrates existing weight reuse (M8~260.5us vsM6~267.9us);
not a production or DRAM benchmark. Evidence recipe_results/mxfp4-row-order-v1.

## 21 October: completed sparse-bound model window and first MLP divergence

donor-sparse-model-v1 completed with source restored, all owned cleanup passed.
The existing attention candidate retained checkpoint/cache formats; full-model
long streams and four-session protocol checks completed. This is functional
and performance evidence, not a released-model quality qualification.

Actual N2 capture (28rows, duplicate14IDs/positions, unique slots) now localizes
the first duplicate-row difference to layer1.mlp.output:3/71680BF16values,
maxabs0.00048828125, relativeRMS2.4373e-5. All earlier captured stages match.
Separate N1 comparison first differs at layer0MLP1element,maxabs0.0009765625.
Differences amplify later; the layer bracket alone does not assign their cause.
Source inspection identifies a testable arithmetic-order difference: default
KT M4 uses one per-row group accumulator; remainder rows use even/odd accumulators
combined at the end. Expert occurrence grouping can send duplicate rows through
different variants. Test this with the existing donor component harness before
claiming a state bug or changing reduction order. No serial-scalar production
contract and no new quantization is implied. Complete evidence and interpretation:
recipe_results/donor-sparse-model-v1/RESULT.md and layer-analysis/report.json.

## 21 October: speed-driven arithmetic policy and completed head observation

User accepts numerical differences when their computing-speed benefit can be
explained and validated. Preserve original checkpoint payloads; compare actual
quality and numerical effects instead of requiring a scalar reference summation
order. Parallel reduction/FMA is permitted. Additional lossy formats require
explicit evidence and approval; no new lossy format is enabled by this update.

`donor-prefill-head-v1-retry2` completed, exact source restoration passed. Five
saved witnesses show repeatable N1/N2 outcomes. In each N2 selected rows13/27,
5083/5120 BF16 hidden coordinates already differ BEFORE the LM head, maxabs
0.303924560546875, RMSdelta0.0803606138 versus row0RMS0.2130593061. Logit RMSdelta
1.0419839621. These compare two donor executions, not an accuracy oracle.
The head/sampler is not the sole cause; layer localization reuses the donor dumper
in the current bounded daily window. Do not call this a proved correctness defect
or a harmless precision effect before explaining the path.

New upstream6d7231 changes SGLang base, not KT CPU arithmetic. Reviewed FP8 GEMV,
O-QDQ and GPU MXFP4 FP32 helper ASTs match the enabled3991 baseline. New mHC state
management/post kernel requires qualification; no duplicate-row fix established.

Candidate `recipe_sparse_valid_bound_v1` omits only trailing all-masked attention
tiles. It does not change cache formats, valid tile reductions or read lengths
back to the host. Component comparison fixes the same tile configuration, checks
output/LSE words and dynamic graph lengths. Natural autotuner selection remains
a separate test because a changed tile can change reduction association. At this
entry the fixed-config component has passed on fea01:21 cases and3 dynamic graph
length vectors retain bit-identical BF16 outputs and FP32 LSE, graph versus eager
also exact. N4short/cap4096 event median3.696608->0.243712ms; full512 remains
0.552960->0.555008ms. Evidence recipe_results/sparse-valid-bound-v1; runtime patch
is not installed. Normal autotuner companion also PASS (9cases,3graphvectors):
both selected configurations retain exact cross-source output/LSE, natural
results match their selected fixed configuration, and graph/eager match. Original
selected32/8warps versus candidate16/4warps in the declared warm-cache history;
cross-config outputs differ in a few BF16 coordinates and pass precommitted donor
decode regression tolerances. This is not a new independent quality oracle.
Evidence recipe_results/sparse-valid-autotune-v1. This establishes
only the tested finite fixtures and selected tile, not full-model numerical parity.

## 21 October: ordinary donor CPU tail probe completed

`recipe_results/cpu-tail-v1` records54 actual donor calls with the unchanged
AVX2 extension, original saved MXFP4 expert weights and ordinary BF16 inputs.
Shared-route and odd-tail duplicate28 outputs match solo14 exactly, including
selected rows13/27. Isolated row1 calls differ by one BF16 word out of71680 per
pattern (max6.103515625e-5, RMS2.2797176769e-7). Same-shape repeatability,
changed-neighbor isolation, input immutability and recovery passed.

Source review identifies different legal FP32 accumulation associations in M4
and M1-tail paths. This synthetic fixture does not reproduce the full-model
divergence and supplies no independent numerical error oracle. It supports
neither a reduction rewrite nor a blanket correctness claim. Preserve the donor
baseline and validate parallel reductions independently, without universal
scalar-summation or cross-batch bitidentity requirements.

The first full pre/post-head observation was interrupted by external Qwen
residency during loading. Its later retry completed; see the newer entry above.

## 21 October: all8 baseline and unresolved packed-prefill discrepancy

The completed all8/56worker/21resident donor run reached19.1757tokens/s mean solo.
All five visible solo texts agree within the run but differ from the split6 run;
complete SSEoutput IDs are absent. No runtime arithmetic or weight-format change
was introduced. Operational batching/cleanup and actual shared decode passed.

Proper-chat duplicate requests diverge beforeEOS. The fixed14-ID one-token
witness yields token334 versusEOS1 within each actual two-request prefill, with
opposing logprob margins0.375/1.0. This is an eager first-output effect, not solely
decodegraphs, speculation, ignoredEOStails or a near tie. Logged256/512tokens
are rounded scheduling budgets; actual packed lengths are14/28. Last-logit rows
are13/27. Source review has not identified a cause. The subsequent diagnostic
observed selected pre-head hidden rows and corresponding logits; no tolerance relaxation
or assumption of released-model parity follows from protocol success.

Evidence: recipe_results/donor-resident21-all8-v2-retry2/RESULT.md and raw
chat-diagnostic/; validated35-request analysis in
recipe_results/resident21-source-audit/live-all8-chat-summary.json.

## 21 October: measured 05yuki baseline and separate format candidates

Current engine is isolated recipe3991/KT95009/SGLang1aa with the small SM89
attention dispatch and NUMA placement adaptations. The user explicitly chose
donor arithmetic as the working baseline. Earlier native-engine contracts below
remain comparison evidence, not requirements to rewrite the donor before use.

- Original MXFP4/E8M0 expert and FP8 dense checkpoint payloads are retained.
  Additional A4/INT8 quantization and deferred-expert dropping are disabled.
  Donor CPU/GPU activation, intermediate rounding and routing-weight schedules
  are not claimed identical to the released reference pipeline.
- SM89 uses the donor's existing portable Triton attention with its mixed
  584-byte cache representation. This is not the separate native cache layout.
  Eight component values exceeded the original bound; diagnostic witnesses
  isolated two BF16 midpoint/cancellation cases, with FP64 favoring each
  implementation in a different case. No tolerance was relaxed and no broad
  numerical PASS is claimed. See recipe_results/sm89-witness-v6-run1.
- Target O-output block32 FP8 QDQ and native FP8 Oa storage are already enabled
  by the donor launcher. Do not duplicate that quantizer.
- Optional routed-input E4M3/E8M0 QDQ is defaultOFF and currently uninstalled.
  CPU component checks passed with approximately3–6% small-fixture overhead;
  shared CUDA dispatch boundary passed372 checks and30 graph replays using
  recording consumers. Neither establishes full-model quality or speed. It
  stays separate from the first DSpark comparison.
- Original DSpark payloads were exported unchanged. Actual GPU constructor
  checked all2398 sources (one authenticated static confidence-head skip) and
  packed expert contents. Donor draft Oa is explicitly expanded from original
  FP8 to BF16; retain and disclose this existing representation. Initial DSpark
  trial sets Markov-W2 BF16 conversion and TP sharding OFF, retaining original
  FP32 Markov weights. Target arithmetic remains the measured donor baseline.

The first target-only HTTP smoke reached17.453/17.575tokens/s end-to-end with
coherent identical text on one short prompt; steady decode logs20.40/20.32.
This is usability evidence, not broad quality or released-reference parity.
DSpark fullservice subsequently passed graphs/streams/cleanup. Last fullywarm
request14.052t/s; samecandidate text across3requests, but latephrase differs
frombaseline ("profilers in most IDEs" versus "profiler in your IDE"). TokenIDs
were not exposed. This establishes neither numericalparity nor qualityloss.
Speculation remainsOFF for next controlled configuration. Multi-session
correctness and broaderquality remain separate. Current state is in HANDOFF.md.

## 20 October: opt-in SM89 FP8 split4 pilot

21:44 follow-on: fresh flag-off v26 matches candidate v25 tokens/logit hashes
for all nine cases; median off/on speed ratio1.00050. Candidate capture launch
counters prove dispatch selected; no meaningful full-model speed gain established.
Default stays off. CPU diagnostic v27 preserved every original arithmetic token
and passed full graph/eager and cross-run logits; accepted untimed binary restored
afterward. Timings are process-lifetime overlapping wall spans and include
instrumentation overhead, not a new numerical contract or performance result.

Before any full-model measurement, precommit the following numerical regression
screens: mean forward/reverse full-vocabulary KL each <=1e-4 nats; maximum each
<=1e-3; absolute mean fixed-target NLL delta <=.001 nats/token; p99 absolute
target log-probability delta <=.01; zero top-1 flips when the baseline top-1/top-2
margin exceeds .1 nats. Require finite results, same-backend graph/eager identity,
state isolation and cache release. These are conservative project pilot screens,
not DeepInfer thresholds or broad quality qualification.

Candidate reuses the existing SGLang Hopper-named Triton kernel on SM89:
SWAP_AB=True, SPLIT_K=4, M16/N64/K32, four warps, three stages, six measured
projection shapes, M1..4. Released E4M3/E8M0 input/weight formats, block32 scales,
quantization boundaries and BF16 outputs remain. FP32 reduction association
and scale multiplication order change; historical logit bit identity is not
required across backends. Engram and wqb stay excluded.

Original component probe passed252 cases; installed-dispatch follow-on passed48,
including M3 and excluded Engram controls. The upstream relative-MAE criterion
<.001 passes against an independent CPU block reference. Large ULP distances
exist in small outputs; full-model evidence is still required. Follow-on guard
37.46s,482MiB, no production mutations, empty residents before/after.

Default-off module installed with backup at pilot/evidence/fp8-swap4-backup:
before2639985c5ea7980b2f07daace3bf2205d13381666e983d9be57f0d9032aaa69d,
after5cbd73eb02665f615ebe8d1eff6cd48872e70c1aa619e312f3e5963707f3675c.
Flag SGLANG_DSV41_SM89_FP8_SWAP4=1; all8-v25 is candidate-only.
Comparison adapter track_a_graph/fp8_model_compare retains the original nine
cases plus three authored fixed-trajectory cases. Review caught and corrected
teacher input selection before execution; baseline uses supplied teacher tokens,
not the candidate's unrelated predictions, on teacher-forced cases.

Full-model result T205649: PASS. All12 baseline/candidate comparisons have
max_abs=0 and zero measured KL/NLL differences; all9 candidate graph/eager
comparisons bit-identical. Actual split capture and baseline normal dispatch
were witnessed; source/destination coverage, native experts, neighbor isolation,
slot0 exclusion, pool release,279 graph replay/verification checks and independent
report/census checkers PASS. Pilot screen qualification only, not a broad quality
corpus or a universal bit-identity guarantee. Remote full logits retained beside
report; local result-1.json/window-1.json under track_a_graph/fp8_model_compare.
Window581.32s, GPU13166MiB, RSS353299677184bytes, cleanup PASS,left unloaded.
Solo9.61–9.85tokens/s,N2~15.54aggregate,N4~20.15aggregate. No controlled graph
on/off speed conclusion: old graph timing is from another run; same-process
eager comparison showed no gain. Candidate remains an explicit opt-in.

Latest user direction (2026-10-20): inspect DeepInfer's actual precision policy so we do not
build everything exact. Preserve released formats and model semantics; qualify optimized
arithmetic without making historical eager bit identity a universal production requirement.
Earlier narrow definitions of "lossless" below are historical, not a reason to reject an
otherwise qualified upstream kernel.

## Current acceptance direction: official DeepInfer source audit

Separate source lead: deepseek-recipe commit
935817c66f5d6e64838f419692545f06787e485f (0.2.0,20October) was inspected locally.
Its Rust/Python code converts API requests, renders/tokenizes conversations and
parses outputs. No FP8/BF16/MXFP4 arithmetic tolerance or kernel acceptance policy
was found; it delegates model inference to the caller. Sampling defaults and
prompt/tokenization semantics are behavior policies, not numerical precision
budgets. No package build/tests or inference benchmark run for this source.

Inspected source revision `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`,
local `completion_work/deepinfer-review/upstream`. Source inspection only; no DeepInfer
kernel build, SM89 measurement or quality qualification has been performed.

DeepInfer does **not** publish one universal acceptable precision threshold in the inspected
tests. Its policy is operation-specific:

- `deepinfer/testing.py:4-23`: the shared arithmetic comparator gates
  `1 - 2<x,y>/(||x||²+||y||²) < eps`, default `eps=1e-4`. For nonzero finite
  inputs this is normalized squared error, not elementwise relative error or a ULP limit.
  RMSE, max absolute error and a log-ratio-derived statistic called `kl` are printed but
  not independently gated. NaNs fail; infinities are zeroed before comparison. Do not
  inherit that infinity handling for new ports: require finite results where expected.
- `tilelang_ops/norm.py:365-367` uses this default tolerance.
  `tilelang_ops/compress.py:338,415` uses the tighter `eps=1e-8` for compressed
  output and exact equality for the compressor state cache.
- `tilelang_ops/fused_transform.py:1210-1288` requires exact outputs versus its reference
  for fused norm/RoPE/Hadamard/cast/cache-store cases, including FP8 storage bytes and
  untouched cache guard bytes. These tests must not be described as tolerance-based.
- `deepinfer/environ.py` defaults `DEEPINFER_BATCH_INVARIANT` to false.
  `worker/runtime_config.py:67-69` opts into deterministic library algorithms for that
  mode; `eval/run_eval.py:74` labels it slower. Universal solo/batch bit identity is
  therefore an optional operating mode, not the default production requirement.
- `tests/models/test_batch_invariant.py` explicitly enables that mode and requires exact
  token/logprob results. `test_kl.py` also enables it and defaults the compared statistic
  to zero. `test_offline_inference.py:144-164` requires exact PPL goldens tied to each
  platform and parallel configuration. These are strict regression tests, not a stated
  cross-hardware error budget. Do not claim DeepInfer's model tests generally allow drift.
- `models/v4_modules/mhc_ops.py` uses DeepGEMM's `tf32_hc_prenorm_gemm` with 16 CUDA
  split-K partials followed by fused reduction/RMSNorm. This supports reuse of optimized
  parallel implementations, not serial scalar emulation. The function name alone does
  not establish its effective mantissa precision or authorize arbitrary TF32 truncation;
  inspect the pinned dependency before substituting another implementation.
- `eval/README.md` supplies AIME/HLE/GPQA Pass@1 evaluation workflows, but the inspected
  material does not define a universal permissible quality regression percentage.

Practical policy for the current KT/SM89 work:

1. Keep released checkpoint formats/scales, activation Q/DQ locations, routing semantics
   and intentional rounding boundaries. Preserve exact indexing, state ownership,
   cache write ranges and queue/cancellation behavior.
2. Reuse each donor kernel's own reference and acceptance test. Permit parallel
   reductions, FMA, fusion and tensor-core arithmetic when that implementation passes
   applicable component and model checks. Do not substitute `eps=1e-4` for every test,
   weaken a failed test retrospectively, or use a passing global metric to hide local
   nonfinite values or routing/state defects.
3. Keep old bit-hash runs as valuable regression evidence. A new optimized backend
   can establish a new qualified baseline: fixed-token logprob/logit drift, NLL/PPL and
   representative task checks first, then 1/2/4-session isolation and performance.
   Changed autoregressive tokens alone are not proof of quality loss; identical tokens
   on a small smoke test alone are not proof of quality preservation.
4. Choose and record numerical/quality limits before each candidate is judged, using
   the relevant upstream test and existing fixtures. No general new numerical limits
   or quality-regression allowance have been established by this source audit.
5. Defer universal batch invariance, exhaustive arithmetic tables and cross-device
   bit emulation. Do not build another validation framework; extend the existing
   harness only as needed for the specific optimized path being measured.

Primary source: https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01

### Active DeepSeek CPU implementation audit (20 October, after P5)

The native1017 gate/up/down path already calls KT's packed MXFP4 AVX2 FMA GEMMs.
It does not use the separate proposed GLM exact-dot implementation or FP64 scalar
dot emulation. Activation Q/DQ is also AVX2 by default. Do not attribute current
DeepSeek speed to those unrelated exact-emulation proposals.

Narrow existing-donor candidates, not yet qualified or enabled:

- `opt_kt_qdq/patch/after/operators/avx2/native1017_phases.hpp:215` computes hidden
  SiLU with scalar `std::exp`. KT's existing `avx2::act_fn` could replace that
  arithmetic, with its polynomial approximation and range clamp explicitly tested.
  Preserve route multiplication before hidden BF16 and E4M3/E8M0 Q/DQ; generic
  `apply_activation` stores BF16 too early for this adapter.
- `mxfp4-moe.hpp:962` has `KT_MXFP4_G32_FAST=1`, folding scales and shortening FMA
  chains. It requires four grouped occurrences of an expert and can be bypassed
  by small per-NUMA K; no ordinary solo gain follows from its existence.
- `mxfp4-moe.hpp:2055` has `KT_NATIVE1017_MERGE=parallel`. Auto already parallelizes
  at >=12 occurrences. Forcing it for N1 changes channel ownership, not arithmetic;
  an added NUMA epoch may cost more than it saves.

These are source-inspection leads. Rank them against the graph profile, then
measure selected changes; no speed or quality result is established here.

## A. Precision inherent to the released checkpoints (not ours; listed for completeness)

| Model | What | Notes |
|---|---|---|
| DeepSeek-V4.1-Flash-1017 | Routed experts MXFP4 (E2M1 + E8M0 per 32); dense/attention linears E4M3 block-32 weights with E8M0 scales; BF16 Oa; activations Q/DQ to E4M3/E8M0 per 32 at the released points (input, hidden, O) | Engine reproduces the released reference rounding points (native1017 phase contract, native SM89 attention) |
| GLM-5.3-Flash | FP8 block-128 weights; serving path quantizes activations to E4M3 per 128-group (W8A8) | Contract choice D3 pending |
| Qwen3.8-Flash-Next | BF16 weights | Reference schedule choice U4 pending |

## B. Installed changes qualified against the post-router/Oa DeepSeek baseline

Later changes were qualified bit-identical to their explicitly recorded baseline (tokens and per-step FP32 logit row SHA,
9 labels, `compare_runs.py`): threaded/header-cached reader, check batching v1/v2, ratio-0 admission,
shared-expert overlap v2, KT vector QDQ + exact merge,
inverse-RoPE gather, tuned FP8 block-GEMM tile configs, Track A P0 harness, P2 async ABI (off/strict),
P3 KT host transport (off/on). See HANDOFF.md for each run.

The router/Oa fixed-row changes altered results and established a new baseline at T114809.
They must not be included in a blanket claim of bit identity to the earlier engine or official implementation.
Subsequent exact regression passes establish preservation of that baseline, not independent whole-model quality.

## C. Validated reduction-order changes (accepted class; not bit-identical)

**Measured but not qualified: existing TileLang mHC POST (20 October evening).**
`mhc_post_upstream_probe/result-1.json` preserves the failed component run. Six
regular cases exceed the pre-existing `.002/.002` post criterion by at most one
BF16 ULP; an adversarial ±2^24 cancellation witness returns zeros versus the
released expression's .25/1.0. This is FP32 accumulation association/FMA, not a
moved BF16 boundary or changed storage format. Finite, input immutability and
same-backend graph equality checks pass. The donor's test uses `mhc_post` itself
as a reference for a larger fusion, so it is not independent error qualification.
No thresholds were relaxed. Only 6.3–7.3 microseconds/call event-span improvement
was measured; this candidate is parked, with its runtime flag unchanged/off.
The cancellation witness alone does not prove model-quality loss, but the
candidate has not established acceptance or enough benefit to prioritize.

| # | Change | Status | What differs | What stays identical | Evidence / acceptance |
|---|---|---|---|---|---|
| C1 | Track B / B-core tiled sparse-MLA attention (SM89) | stage 1 tested, not integrated | FP32 accumulation order of QK and PV (mma.sync tiles); online-softmax running maximum taken per **32-slot tile** (current kernel: 64-slot chunks), so exp(s - m) is evaluated against different intermediate maxima and rescaled | cache dequant, score scaling, expf calls, BF16 P from unrounded exponent, FP32 denominator, sink handling, IEEE division, BF16 output rounding, status bits | stage-1 `evidence/trackb-s1-*`: 0 FP64-budget violations, error vs FP64 <= current kernel in every case, 99.96-100% bit-equal to current. **User accepted the 32-slot max points as a validated reduction-order change on 2026-10-20.** Integration needs a full-model logit-drift qualification. |
| C2 | Fixed-row router and grouped Oa launches | installed; changed baseline, incomplete general batch invariance | Different cuBLAS path/reduction at singleton rows; downstream quantization can change codes | Stored weight/activation formats and declared quantization locations | HANDOFF router/Oa runs T112545 and T114809. Router synthetic FP64 error was 8.9e-6 for M1 versus 6.9e-5 for the padded path. Component and model isolation checks pass, but arbitrary solo/shared prefill identity does not. Record this separately from later bit-exact optimizations; independent quality qualification remains required. |

## D. Semantics changes that are not precision changes

| Change | Status | Detail |
|---|---|---|
| Deferred / graph-mode check detection (Track A P1/P2/P5) | user-accepted 2026-10-20 for graph/deferred mode; eager default stays strict | A failed value check is detected at the end of the step instead of mid-step; the cohort is discarded and nothing reaches logits. Outputs on success are bit-identical. |
| Batch invariance (open item) | known | Solo vs shared runs differ from step 0 because prefill GEMM kernel choice depends on M (cuBLAS/Triton M>=24). Each configuration is deterministic and bit-reproducible; decode-side router/Oa invariance is fixed. Full M-independence is a separate project. |

## D2. Measured hardware facts that constrain references

| Fact | Evidence | Consequence |
|---|---|---|
| The tested SM89 FP8 kernels differ from the probe's exact-dot reference and four candidate scale/accumulation forms; reported exact block-dot agreement was 4.9e-5 on the synthetic corpus | glm_oracle_v2 probe, `evidence/kt-cmd-window-20261020T181804` (`glm-oracle-v2-probe-1/probe/probe_forms.json`); five shapes with uniformly sampled finite E4M3 codes | This rules out those tested exact emulations, not every CPU implementation or FP8 tensor-core path. Qualify representative real-layer error and model behavior under the already allowed parallel-arithmetic contract. Exact-dot arithmetic is useful as an oracle, not a mandatory production implementation. |

## E. Reference decisions (user, 2026-10-20)

- **GLM-5.3:** experts W8A8 (FP8 activation Q/DQ per 128 exactly as SGLang serving; no dropped or moved quantization point); KV cache BF16 latent (FP8 KV excluded as a precision drop); reference = torch text-path oracle with explicit rounding points following SGLang's boundaries, validated against Maya's FP64 oracle, then frozen.
- **Qwen3.8 SiLU table qualified (2026-10-20):** table sha256 `8da55714cc0200356753a576a34be2ee722cdf037b22f1338965437d7cdff031` (manifest `c5db8380…`), measured from the installed JIT CUDA `silu_and_mul` (fast-math, FTZ after) on SM89; all 2^32 (gate, up) BF16 pairs bit-identical between GPU and the CPU model. Bit-identical to GPU serving by construction (deliberately not exact expf).
- **Qwen3.8:** reference = SGLang's GPU fused-MoE schedule; CPU SiLU reproduces the SM89 GPU fast-math SiLU via a 65,536-entry table (bit-identical to GPU serving, deliberately not exact expf). HC-mix: deterministic path to be pinned (atomic fused kernel excluded as non-deterministic).
# Resident21 cross-session diagnostic, 21 October 2026

The donor arithmetic is unchanged in resident21 v2; original routed-input QDQ
candidate and DSpark remain off. Solo chat outputs repeat exactly as visible
text, with 18.101 tokens/s mean on the selected short workload; they differ
from the five-resident output and are not a released-model parity result.
The separate native-ID test used untemplated completion prompts. Solo repeats
and the post-batch replay match exactly, but all N2/N4 comparisons differ and
an identical-input N2 pair diverges at index2 after common IDs6029,16.
Neither harmless rounding nor a session-state defect has been established.
Next gates use actual chat-encoded fixed IDs and first-divergence logprob
margins, plus a real ratio-2 compression-state component witness. Preserve
the original evidence; do not mark four-session correctness as passed.

