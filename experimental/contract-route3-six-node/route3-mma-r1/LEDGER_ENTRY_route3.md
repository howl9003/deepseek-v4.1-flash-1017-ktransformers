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
