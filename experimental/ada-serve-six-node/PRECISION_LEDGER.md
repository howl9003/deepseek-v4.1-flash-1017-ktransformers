# Precision ledger: C2 against DeepSeek's reference

Identity runs compare against C2, this build's own archived outputs, never against DeepSeek's kernels, which do not run on this card. This page lists where C2's arithmetic differs from the 1017 reference (`inference/model.py`), and every step in our build that changed bits.

## Where C2 differs from the reference

| Stage | Reference | C2 | Why |
|---|---|---|---|
| Routed experts | FP8 activations per 32 (UE8M0), `fp4_gemm` (W4A8), FP32 accumulation | Same activation grid (E4M3 per 32, `amax` floor 1e-4); exact integer sums, one round to nearest even per route, FP32 combine in ascending logical expert id | The same bits on CPU and GPU, at any NUMA part count, batch shape or placement |
| Dense FP8 | One FP32 chain over K, one MMA per 32-wide block | Same activations, weights and scales; K cut into four fixed ranges with a running sum each, reduced in fixed order; Ada's 16-product block truncation | Decode and prefill share these bits; one-row decode gets enough parallel work. The truncation belongs to the card |
| `wo_a` | Weights dequantized to BF16, BF16 einsum | FP8 blocks, E4M3 MMA per 32 with FP32 scales, four K splits, ordered merge | Speed, and one weight rounding fewer |
| Index scores | BF16 einsum, relu, weight, sum | Same BF16 rounding points in one Triton kernel; even and odd halves through BF16 MMA, FP32 sum across heads | Speed. The sum order differs, so near-ties can pick different positions |
| Sparse attention | TileLang `sparse_attn` | SM89 Triton sparse kernels with an FP32 merge and three BF16 rounding points; prefill rows go through the same kernels | Reorder only |
| Router | FP32 `F.linear` gate | BF16 MMA into FP32, eight K splits, fused gate with `log1p` | Fixed sequence |
| Logits | FP32 head weight, `F.linear(x.float(), weight)` | BF16 x BF16 MMA into FP32, no BF16 cast | Closer to the reference than BF16 logits, not equal |
| Activation floor | `amax` floored at 1e-4 | 1e-10 on dense projections and Engram `wkv`; 1e-4 on routed experts and `o` | Upstream default; changes only values under 3.7e-9 in blocks under 1e-4 |
| hc mixes | FP32 `F.linear`, `torch.rsqrt` | K-sliced `tf32x3` dots, `1/sqrt`, TileLang Sinkhorn | Batch invariance |

**The same as the reference:** compressed KV (FP4 with E4M3 scales per 16), window KV (all 512 values E4M3 with one UE8M0 scale per 32, with the BF16 rounds after the norm and after RoPE; the store equals DeepSeek's `act_quant` on 151M test values), index keys and queries (`fp4_act_quant` values, stored packed, 68 B a position).

**Upstream store fix:** SGLang's V41 store floors the scale after dividing by 448 (2^-13); the reference floors `amax` at 1e-4 first (2^-22). The overlay uses the reference's order in `store_row_fp8`. It changes only blocks whose largest value is under 0.0448.

## Steps that changed bits

Every other selector in `SERVED_SELECTORS.md` is bit-exact.

| Step | Selector | Effect |
|---|---|---|
| Logical combine order | `KT_CONTRACT_V2_COMBINE_ORDER=logical` | 1,427 of 11,496 scores moved; NLL -0.0006 (SE 0.0009); bits no longer depend on expert placement |
| Compressed KV kept as FP4 | `SGLANG_SM89_FP4_COMPRESSED_KV=1` | Removes a second rounding into the V4 layout. Panel KL 0.01106 to 0.01098, paired NLL -0.00056 (SE 0.00084) |
| Window KV on the reference's grid | `SGLANG_SM89_V41_KV=1` | Replaces the V4 layout (448 values per 64-scale, RoPE values in BF16). With the other two, panel KL 0.01019 |
| FP32 logits | `--enable-fp32-lm-head` | On its own KL 0.01106 to 0.01072, paired NLL -0.00027 (SE 0.00007); removes exact top-two ties on 2.05% of positions |

The last three together are C2's change from the build before it: paired NLL +0.0025 (SE 0.0036) on the 24 pieces of `tools/nll.py`, frozen quality screen PASS.

## Measured distance

- **Public panel** (sudoingX's, 254,846 positions, against DeepSeek's code on this card): KL 0.01019, top-1 96.98%; the reference against itself with only MoE batching changed: KL 0.00996, top-1 97.01%.
- **24 pieces of `tools/nll.py`:** +0.0034 (SE 0.0044) before the three KV and logits changes, +0.0025 (SE 0.0036) from them; the reference against itself +0.0036 (SE 0.0041).
