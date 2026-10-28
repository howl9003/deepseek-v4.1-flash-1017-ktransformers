# Route 3: one MMA sequence for the dense FP8 projections, decode and prefill (r1)

Claude (Fable 5.1), 28 Oct 2026, local draft for Codex. Nothing here ran on fea01. Tags: **[S]** read in source,
**[M]** measured on fea01 by Codex (receipt named), **[H]** estimate. Constraints unchanged: default-off selectors,
frozen gates, 416 GiB no-swap, 64 GiB FEA reserve, six nodes, 42 workers, 320 W, PR6 held.

## 0. What route 3 is, in one paragraph

Batch invariance for the dense FP8 projections comes from one exactly specified per-row sequence that every kernel
implements, whatever its tile. The sequence already exists: it is what `_w8a8_block_fp8_matmul_hopper` with
`SGLANG_FP8_FIXED_SPLIT_K=4` computes, and Codex proved its per-row bits identical for M = 1..768 across the
16-row and 64-row tiles [M, `fixed-fp8-r1`, `clean-fp8-modes-r1`]. The accepted r18 and contract r5 runs used it at
every M. Its one-row speed is the problem (22.2 ms per token for 208 calls against a 7.0 ms bandwidth floor
[M, `KERNEL_PRICE_SUMMARY_r2`]), and that is a launch-geometry problem, not an arithmetic one (section 2). So:
keep the sequence, keep the prefill kernel, and give decode a kernel with the right geometry. Phase 1 does this
inside the same Triton source, which makes identity automatic; phase 2 is a CUDA C++ kernel only if Triton cannot
reach the bandwidth floor. The dense CUDA-core family (r3) is retired for decode and prefill by this route, which
also removes its 63 s cost at 35K [M, `KERNEL_CANDIDATE_PROGRESS_r3: selected_dense`].

## 1. The shared sequence, pinned [S: `fp8-decode-source/fp8_kernel.py` 148-258, 260-272, 811-945]

For output row m and column n, with A the E4M3 activation codes `[M, K]`, `As` the per-32-column activation scales
`[M, K/32]` (FP32, power-of-two values), B the E4M3 weight codes `[N, K]`, `Bs` the per-32x32-block weight scales
`[N/32, K/32]` (FP32, power-of-two), block_k = 32:

1. K is cut into T = K/32 blocks t = 0..T-1 in ascending order. With SPLIT_K = 4 the blocks are partitioned into
   four contiguous ranges, split s owning blocks `[s*ceil(T/4), min((s+1)*ceil(T/4), T))` (the kernel's
   `tiles_per_split = cdiv(cdiv(K, 32), 4)`). The ranges are fixed by K alone, never by M, N or the tile.
2. Per block t: `D_t = mma.sync.m16n8k32.f32.e4m3.e4m3.f32(A[m, 32t:32t+32], B[n, 32t:32t+32], C = 0)`. One
   instruction per block, accumulator input zero (`tl.dot(a, b)` with BLOCK_SIZE_K = 32 lowers to exactly one
   k32 MMA from a zero accumulator). This is the Ada QMMA unit: two fused 16-product blocks, exact products,
   alignment to the largest exponent with 2 integer and 13 fractional bits, truncation, 20-bit adder, result
   truncated to fp32 [S: Khattak and Mikaitis 2025, Fig. 4 and the fp8 row of their parameter table; see
   `LEDGER_ENTRY_route3.md`].
3. Per block t, in FP32: `acc_s = acc_s (+) (D_t (x) As[m, t] (x) Bs[n/32, t])`. Because both scales are powers of
   two (UE8M0 by construction: the weight scales are stored as E8M0, the activation scales come from the
   `ue8m0` quantiser), the two multiplies are exact and the only rounding per block is the add. The two forms
   Triton can emit (two `mul.rn.f32` then `add.rn.f32`, or `mul.rn.f32` then `fma.rn.f32`) therefore give the
   same bits (`route3_sequence_reference.py`, check 4), and the contraction choice is not part of the arithmetic
   as long as the scales stay powers of two **and no product underflows**: a power-of-two multiply is exact
   only while the result is a normal float32; a subnormal result is rounded (or flushed, if the kernel was compiled
   with `.ftz`), and then the two forms differ. For this model the margin is large (section 1a), but the sequence is
   retained exactly, not argued away: read the accepted kernel's PTX once (`triton.compile(...).asm["ptx"]` or
   `TRITON_KERNEL_DUMP=1`), record the epilogue instructions including any `.ftz` modifier in the preparation
   receipt, keep them unchanged in phase 1 (same source) and copy them literally in phase 2 with `.rn` intrinsics
   and the same denormal mode. The selector additionally asserts that `As` and `Bs` are exact powers of two.

1a. Underflow margin [H, from the value ranges]: the smallest nonzero block result is one product of two
   E4M3 subnormals, 2^-18; the activation scale is at least 2^ceil(log2(1e-4 / 448)) = 2^-22 because the quantiser
   floors `amax` at 1e-4; so the scaled term is at least 2^-40 times the weight block scale, and float32 normals
   extend down to 2^-126. A weight block scale below 2^-86 (E8M0 code below 41) would be needed for any rounding
   to occur. Codex can close this with one read of the minimum E8M0 code over the dense scale tensors in the
   checkpoint headers; the sequence is kept regardless.
4. Per split: the partial `P_s[m, n]` is stored to a `[4, M, N]` FP32 workspace. The partials are reduced by
   `_reduce_block_fp8_split_k` (`tl.sum` over the split axis). **Reuse that kernel unchanged** in every path; its
   internal order then never has to be known.
5. The reduced FP32 value is cast once to the output dtype (BF16) by the reduce kernel's store.

What is explicitly not part of the sequence: tile sizes, warps, stages, the SWAP_AB transpose (Codex showed it
preserves the per-row partials: "N tiling and transpose only, identical FP32 split partials" [M,
`FP8_DECODE_PREPARATION.json`]), the order in which CTAs run, and how many rows share a tile. Rows 1..15 of a
16-row tile at M = 1 are zero padding and do not touch row 0's result.

## 2. Why the current kernel is slow at one row [S]

With `fixed_split`, the decode config is `BLOCK_SIZE_M 16, BLOCK_SIZE_N 128 (32 with the transpose option),
BLOCK_SIZE_K 32, num_warps 4, num_stages 3, SPLIT_K 4`, grid `(N/BN, 4)`:

| shape (N x K) | CTAs at BN = 128 | CTAs at BN = 32 | bytes per CTA per K step |
|---|---|---|---|
| 1792 x 5120 (wqkv) | 56 | 224 | 4 KB / 1 KB |
| 4608 x 5120 (shared w13) | 144 | 576 | |
| 5120 x 2304 (shared w2) | 160 | 640 | |
| 5120 x 8192 (wo_b) | 160 | 640 | |
| 32768 x 1280 (wq_b) | 1024 | 4096 | |

The card has 80 SMs. At BN = 128 three of the five shapes launch fewer than two CTAs per SM, each CTA streams a
4 KB weight slab per step with three stages in flight, and every step pays an address, scale-load and barrier
cost for 16 x 128 outputs of which one row is live. The transpose option's gain (12 to 33 percent on cold weights
[M, `fp8_tiling_research`]) came from the CTA count, not from arithmetic. Marlin reaches near-peak bandwidth at
batch 1 to 16 with the same 16-row padding by keeping many warps streaming with deep asynchronous pipelines; the
MMA's wasted rows are free when the kernel is bandwidth-bound.

## 3. Phase 1: the same Triton kernel, decode geometry [H for the speed]

Add a decode configuration to the `fixed_split` branch, selected for M <= 16 (the decode graph batches are 1, 2,
4), default-off behind a new selector (suggested `SGLANG_FP8_FIXED_DECODE_GEOMETRY=1`):

- `BLOCK_SIZE_M = 16`, `BLOCK_SIZE_K = 32` (required by the sequence), `SPLIT_K = 4` (same partition), `GROUP_SIZE_M = 1`.
- `BLOCK_SIZE_N = 16` (Triton's minimum for `tl.dot`; two n8 MMAs per step), `num_warps = 1`, so one warp owns
  16 columns and streams 512 B per K step as one 16-byte load per lane.
- `num_stages = 6 to 8` so each warp keeps 3 to 4 KB in flight; with grid `(N/16, 4)` that is 448 CTAs for the
  smallest shape and 1280 for N = 5120, 16 or more resident warps per SM, 30 to 60 KB in flight per SM, which is
  what Little's law needs for 736 GB/s at this card's latency [H].
- Sweep `BLOCK_SIZE_N` in {16, 32}, `num_warps` in {1, 2}, `num_stages` in {4, 6, 8}, with and without SWAP_AB.
  Keep the configuration that is fastest at M = 1 and still within 10 percent of it at M = 4 and 8.
- The workspace and reduce launch are unchanged; the 512 MiB fixed workspace logic does not trigger at M <= 16.

Identity is by construction (same source, same per-row sequence), but it is still tested (section 5). Expected
price: 208 calls between 8 and 10 ms per token against 22.2 today and the 7.0 ms floor [H]. If the best Triton
configuration stays above 12 ms, go to phase 2.

## 4. Phase 2: a CUDA C++ Marlin-style decode kernel reproducing the sequence

Only if phase 1 stalls. Structure:

- One CTA = 4 warps, each warp owns 16 columns of one K split; grid `(N/64, 4)`; persistent variant optional.
- Weights: `cp.async.cg` 16-byte chunks into a 4-stage shared-memory ring of `[64 columns][32 K]` tiles (2 KB per
  stage per CTA), padded to avoid bank conflicts; B fragments built with `ldmatrix` the way CUTLASS's SM89 FP8
  iterators do, or by direct 32-bit register loads following the m16n8k32 fragment map in
  `route3_sequence_reference.py` (verify the map against the PTX ISA figures before coding).
- Activations: the M <= 16 rows' E4M3 codes for the current K block are held in registers in the A-fragment
  layout; rows beyond M are zero. `As` per row and `Bs` per 32-column block are loaded once per K block.
- Per K block: one `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` per n8 tile with `C = 0` (literal zero
  registers), then the pinned FP32 sequence from section 1 step 3 written with explicit `__fmul_rn` /
  `__fadd_rn` / `__fmaf_rn` so the compiler cannot re-contract it.
- Epilogue: store the split partial to the same `[4, M, N]` workspace layout; launch the unchanged
  `_reduce_block_fp8_split_k`.
- Everything else (scale indexing by block, masking for K not divisible by 32 which does not occur in these
  shapes, output dtype) copied from the Triton kernel.

## 5. Exactness plan (component, Codex's harnesses)

- Bit identity between the decode configuration (phase 1) or kernel (phase 2) and the unchanged prefill kernel on
  the five checkpoint shapes plus the indexer shape, real weights and scales, M in {1, 2, 4, 8, 13, 16, 26, 32,
  128, 538, 768}: every row of the decode result equals the same row computed by the prefill kernel in a batch of
  any other size. Reuse `replay-audit-r1/test_fp8_dispatch.py` and the `fixed-fp8-r1` / `clean-fp8-modes-r1`
  cases (438 arithmetic plus 176 dispatcher checks were the bar for dense r3).
- Changed-input CUDA graph replay at batch 1, 2, 4; row permutation and partition cases; the K = 1280 shape
  (40 blocks, 10 per split) and K = 8192 (256 blocks, 64 per split) as the split-boundary edge cases.
- Raw partial check: dump the `[4, M, N]` partials from both kernels and compare bytes before the reduce, so a
  mismatch is located to a split.
- One probe for the ledger, not for acceptance: Khattak and Mikaitis's fp8 test vectors on the 4080 SUPER
  (`p1 = 1, p2 = p3 = 2^-23-n_eab` with decreasing n_eab, and the accumulator-input test from SageAttention2),
  to confirm 13 fractional bits and truncation on AD103 as on L40S. Half a day; if it disagrees, the ledger entry
  changes, the route does not.

## 6. Price plan

Same guarded component session as the earlier six-family pricing (8 GiB no-swap scope, one service CPU, GPU UUID,
320 W, shared lock, no model resident), cold-weight CUDA-graph timings including quantisation and the reduce:

| row | kernel | M |
|---|---|---|
| 1 | current fixed path (BN 128 / BN 32 transpose) | 1, 4, 8 |
| 2 | phase 1 best configuration | 1, 4, 8 |
| 3 | original dispatch (GEMV, skinny), for reference only | 1, 4, 8 |
| 4 | prefill kernel, unchanged | 512, 6144, 36864 |

Report per-call medians, per-token ms (208 calls), and bytes per microsecond against 736 GB/s. Prefill rows are
there to prove nothing moved: the bounded 35K component must stay at 6.2 s [M, `selected_dense`].

## 7. Acceptance

The dense arithmetic returns to the fixed split-K bits that r18 and contract r5 accepted; the decode kernel adds
no new rounding. The full-model candidate is K3 with dense r3 replaced by this route, so its bits differ from K3
and it is a new cold context under the r18 protocol: cold repeat, eight retained splits, rotation, quality screen,
paired NLL within one standard error of accepted K3, feature and host-reload checks exact, resource and settle
receipts unchanged. Prefill of 35K tokens should return from 265 s toward the 194 s of the pre-K3 contract run,
less whatever the PRMT route saves [H].

## 8. Where else this sequence exists, and what it is on other hardware

The design (one shared MMA sequence per machine) ports; the bits do not. Per architecture, FP8 x FP8 with FP32
accumulation [S: Khattak and Mikaitis 2025 unless marked]:

| architecture | instruction the two kernels would share | fused block | fractional bits | note |
|---|---|---|---|---|
| Volta SM70 (V100) | none for FP8 or INT8; convert E4M3 to FP16 (exact) and use `mma.sync.m8n8k4.f16` with f32 accumulate | 4 products + c | 23, truncated, no extra alignment bits | a 32-wide block is 8 chained instructions whose accumulator passes through the 5-term adder each time, so the per-block value also depends on chaining; no INT8 MMA, so the integer route would be dp4a-only (CUDA cores) and prefill-slow there |
| Turing SM75 | FP16 HMMA after exact conversion; INT8 MMA `m8n8k16.s8` available | fp16 path as Volta class | | first generation with exact INT8 MMA (the integer route's prefill becomes possible) |
| Ampere SM80/86 | FP16/BF16 HMMA `m16n8k16` after exact conversion | 8 products | 24, one extra alignment bit, truncated | more precise than Ada FP8; different bits |
| Ada SM89 (fea01) | `mma.sync.m16n8k32.e4m3` to QMMA | 16 products | 13, truncated | this document; SageAttention2 measured the same on the RTX 4090 |
| Hopper SM90 | `wgmma.m64nNk32.e4m3` to QGMMA | 32 products | 13, truncated | `mma.sync` FP8 on Hopper converts to FP16 and runs HMMA with interleaved pairs and a late RNE add of c; the two Hopper paths differ from each other |
| Blackwell SM100 | `tcgen05.mma kind::f8f6f4` | 32 products | 25, truncated | `kind::mxf8f6f4` applies UE8M0 block scales inside the unit, a different contract from our FP32 scaling outside |
| Blackwell SM120 (RTX 50, RTX PRO 6000) | `mma.sync.m16n8k32.e4m3` to QMMA with B200 numerics | 32 products | 25 | block-scaled `mma.sync` listed for sm_120a in CUTLASS |
| AMD CDNA3 (MI300X) | FP8 MFMA, FNUZ encoding (max 240) | not characterised here | | OCP E4M3 codes above 240 are not representable: lossy; CDNA4 adds OCP FP8 and MX formats |
| CPU AVX2 (fea01) | FP32 FMA, IEEE | 1 | 24 (RNE) | matches a CUDA-core fixed-order kernel bit for bit, never a tensor-core path |

Cross-architecture bit identity exists only for exact arithmetic: INT8 MMA with 32-bit integer accumulation
(Turing onward, AMD MFMA, AMX) and dp4a or AVX integer dots on CUDA cores and CPUs. That is contract v2's property
and the integer-plane route's; route 3 is per-machine.

## 9. Risks

- Triton's compile may lower the FP32 epilogue differently for the one-warp layout (vectorisation does not change
  contraction, but the identity test, not the argument, decides). If it does, phase 2 with explicit `.rn` ops.
- The reduce kernel is reused unchanged on purpose; changing its block size or `tl.sum` lowering would change bits.
- The QMMA probe may show AD103 differs from L40S in some edge case; that only changes the ledger text.
- K-split boundaries are fixed by K; a future change to `SPLIT_K` or `BLOCK_SIZE_K` is an arithmetic change and
  needs re-acceptance, so the selector must freeze both.

## 10. Division of labour

Codex: PTX pin of the epilogue (section 1.3), the phase 1 sweep and identity tests, the price table, the QMMA probe,
and the guarded full-model candidate under the r18 protocol. Claude: this specification, the fragment-layout and
sequence reference (`route3_sequence_reference.py`), the ledger entry, and the phase 2 kernel text if phase 1 stalls.
Neither side touches the other's lock; no runtime installation before the component gates pass.
