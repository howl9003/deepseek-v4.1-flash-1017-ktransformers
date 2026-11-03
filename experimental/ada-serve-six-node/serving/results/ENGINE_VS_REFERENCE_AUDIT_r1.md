# Served engine against the 1017 reference: independent audit, 2 Nov 2026

**No wrong logic found: no wrong algorithm, off-by-one, wrong position or wrong layer in any stage on the served path. One new grid gap: the head rounds logits to BF16 where the reference keeps FP32. The ledger has no row for it.** Turning on the existing flag, measured on the card on 3 Nov, takes KL from 0.01106 to 0.01072: 31% of our 0.0011 excess over the floor. Paired NLL improves by 0.00027, 3.8 standard errors, and the head gets 0.24 ms faster. It also removes the exact ties between our top two tokens, which were on 2.05% of positions.

I read the reference first and wrote my own stage list, then traced the served tree `ada-serve-20261102-r4` against its real launch. The method passes the test you set: it finds the V4 window layout on all 40 layers without the ledger's help.

"Read" below means read in code. "Measured" means run against the captured panel on CPU, except the head-only arm, which ran on the card. Scripts and results: `reference-panel-r1/audit-r1/`, and on fea01 `/scratch/aster/bench/engine-audit-r1/`.

## What is wrong, ranked by likely effect

Our KL is 0.01106, the floor 0.00996, so the excess is 0.0011.

| # | Stage | What differs | Kind | Size | Status |
|---|---|---|---|---|---|
| 1 | Window KV, all 40 layers | V4 layout: 448 values E4M3 with one scale per 64, 64 RoPE values BF16. Reference: all 512 E4M3, one scale per 32 | Different grid | Not measured, could be the largest | Known, being fixed |
| 2 | **Head** | **Logits rounded to BF16. Reference FP32** | Different grid, avoidable | **Measured: KL -0.00034, 31% of the excess; NLL -0.00027, SE 0.00007** | **New. Not in the ledger. Flag exists** |
| 3 | Compressed KV, layers 2-39 | FP4 step, then requantised into the V4 layout: a second rounding | Different grid | KL 0.01106 to 0.01098, NLL -0.00056, SE 0.00084. Measured earlier | Known, being fixed |
| 4 | Window KV source | Kernel quantises and rotates FP32 registers. Reference rounds to BF16 after the norm and again after RoPE, then quantises | Rounding points | Part of #1, not separable yet | **New. Goes away if #1 uses the kernel's V41 branch** |
| 5 | Activation floor, dense projections | `amax` floored at 1e-10. Reference 1e-4 | Wrong constant, harmless | Changes only values under 3.7e-9, in blocks whose largest value is under 1e-4 | New. Not in the ledger |

Everything else is either the same or a known reduction-order difference. Those are in the stage table.

## The head (#2)

**Reference.** `model.py:1008` keeps the head weight in FP32 "so the logits come out in fp32 directly", and `model.py:1014` runs `F.linear(x.float(), self.weight)`.

**Served.** `SGLANG_DSV41_FIXED_LM_HEAD=1` selects `fixed_lm_head` at `srt/layers/logits_processor.py:1175`. `--enable-fp32-lm-head` is off, and no `enable_lm_head_output_fp32` is set on the config, so `fp32_output` is False. That takes `padded_bf16_head` in `kernels/ops/gemm/fixed_lm_head.py`, a BF16 x BF16 `torch.mm` with BF16 output.

**Measured on the 254,846 panel positions of the served capture:**

| Check | Served | Reference on Ada |
|---|---|---|
| Top 8 logprob differences all on a 2^-5 grid | 100% of positions | 0% |
| Top logit's grid step | 0.125 on 71.5%, 0.0625 on 24.3%, 0.25 on 2.8% | none |
| Top two tokens exactly tied | 5,214 positions, 2.05% | |

- **Top-1.** 2,683 of our 8,305 top-1 disagreements sit on an exact tie. On 2,471 of them the reference's top token is our second.
- **Greedy.** Temperature 0 picks between exact ties with no signal, so a tied position can change the output token.

**Why it was missed.** Every identity check compares logprobs against C1, which has the same head. No ledger row lists the head at all.

**The fix.** `--enable-fp32-lm-head` reaches `_fixed_fp32_head`, which uses `fixed_projection.linear`: BF16 x BF16 into FP32, no final BF16 cast. It changes only the logits, so KV and caches are untouched.

### Head-only arm, measured 3 Nov

Arm `adahd1`, trial `ada-panel-fp32head-20261103-r1`: the 32.98 panel chain with only that flag, no KV selectors, one load, admission 448 GiB. The profile guard admits that one argument and nothing else. It ran between the other session's `adavr1` panel and its timing run.

| Panel, 254,846 positions | KL | KL p99 | Top-1 | NLL |
|---|---:|---:|---:|---:|
| Reference, MoE batch 2,048 (noise floor) | 0.00996 | 0.123 | 97.01% | 0.47105 |
| Served 32.98 build, BF16 head | 0.01106 | 0.137 | 96.74% | 0.47150 |
| **Same build, FP32 head** | **0.01072** | **0.136** | **96.91%** | **0.47124** |

- **Paired against the served capture**, by conversation over 197 conversations: NLL **-0.00027, SE 0.00007**, so 3.8 SE better. 194,231 positions changed, the largest by 0.61 nats.
- **KL falls in all eight domains**, by 0.0003 to 0.0005 each. The excess over the floor drops from 0.0011 to 0.00076.
- **Ties:** none of the top two are tied now. Top-1 disagreements drop from 8,305 to 7,886.
- **Decode against prefill:** sixteen conversations, 512 positions, all within 4.2e-7 nats. So the score applies to decode.
- **Head time on the card**, real head weight, eager calls with a sync each, median of 32. The FP32 route is faster at every decode row count: the BF16 route always pads to 64 rows.

| Rows | BF16 head, ms | FP32 head, ms |
|---:|---:|---:|
| 1 | 2.22 | 1.98 |
| 2 | 2.22 | 2.01 |
| 4 | 2.22 | 2.04 |
| 8 | 2.22 | 2.10 |

**My estimate was low.** Rounding the reference's logits to BF16 on CPU predicted 0.000145 of KL. The measured fall is 2.3 times that. The simulation guessed each logit's level and left the tail bucket alone; the measured number is the one to use.

**Costs.** Bits change against C1, so this is a ledger step and needs a new baseline. Prefill holds FP32 logits for every returned row, twice the BF16 size. The panel ran with no memory trouble. The other session's `adavr` chain already carries the flag together with the V41 and FP4 KV changes.

Evidence: `audit-r1/adahd1-score-r1.json`, `adahd1-paired-r1.json`, `head-ties-adahd1-r1.json`, `head-time-r1.json`, `run-adahd1.log`. The chain is `derive_adahd1.py` and `run_adahd1.py`.

## Other new items

**#4, window KV source.** `main_norm_rope.cuh:416-438` is the V4 branch: `data = x * norm * w` in FP32, no BF16 round, RoPE on the FP32 values, E4M3 from FP32. The reference goes through BF16 twice (`model.py:293`, `model.py:405`). The V41 branch at `main_norm_rope.cuh:394-410` already does it the reference's way, and its comment says so. If the window work in `v41-window-r1` ports a store of its own, it should keep those two rounds.

**#5, activation floor.** `fp8_utils.py:1827` passes `eps=1e-10` unless Path A is on, and `per_token_group_quant_8bit_v2.cuh` bakes in `1e-10f`. This covers `wq_a` + `wkv`, `wq_b`, indexer `wq_b`, `wo_b`, shared `w1` + `w3`, shared `w2` and Engram `wkv`. Routed experts use 1e-4 (`contract_v2/kernels.py:32`), and `o_fp8_qdq` uses 1e-4.

With power-of-two scales the floor only moves the scale, not the 3-bit mantissa grid. A value changes only if its block's largest value is under 1e-4 and the value itself is under 2^-28, in E4M3's subnormal range. I expect no visible effect. Cheap test: count blocks under 1e-4 per projection on a few conversations, in the CPU reference harness.

The same JIT builds with `--use_fast_math`. The Path A build of that kernel already has the right floor and strict flags.

**Smaller constants, no effect expected:**
- Index K store: `store_fp4_index_k_cache` repacks with `quantize_fp4_indexer_tensor`, ties to even, floor 1e-4 on `amax / 6` (`fp4_indexer_quant.py:97-100`). The reference's floor is 6 x 2^-126. Values already on the grid stay on it unless a block's `amax` is under 6e-4; k is RMS-normed, so that should not happen.
- Router renormalisation: `moe_fused_gate` is called with epsilon 0, so it guards `sum > 0`. Reference adds 1e-20. Same result for any sum above about 1e-13.
- Attention: the reference starts the running max at -1e30 and the served kernel at -inf. Same unless a row has no valid index, and the current token is always valid.

## Blank rows

Nothing here explains the context-free rows. They occur in the reference itself, and no stage differs in logic. BF16 logits round them but cannot create them. Worth knowing for that work: those rows respond to the reference's MoE batch size alone, which is pure rounding, so they sit on a numerical edge somewhere both engines share.

## Full stage table

Paths without a prefix are under `third_party/sglang/python/sglang`. All rows are read from code unless marked. Confidence is high unless marked. "Rounding" means the same arithmetic with a different reduction order or a kernel's own float steps.

| Stage | Reference | Served (selector) | Result |
|---|---|---|---|
| Tokenizer | `generate.py:156`, `encoding.py:810` | `deepseek_recipe` wheel 0.2.0 via `srt/parser/deepseek_recipe_renderer.py`, picked by `--tool-call-parser deepseekv41` | **Same. Measured:** 10 of 10 prompts token-equal, chat and thinking modes, against `encoding.py`. Tool calls not tested |
| Panel input | ids from `panel.safetensors` | same ids through `/generate` | Same by construction: chat template plays no part in the panel |
| Embedding | `model.py:174`, BF16 | `embed_tokens`, BF16 | Same. **Measured** checkpoint dtype BF16 |
| hc copies | `model.py:1261` | `hc_broadcast`, `mhc.py:1881` | Same |
| Engram hash | `engram.py:163-184`, token map, odd multipliers, primes, offsets | `srt/layers/engram.py:125-247`, kernel `engram_hash.py:98` | Same algorithm. Compressed vocab asserted at 99,092. History row replaces the `cache` buffer |
| Engram lookup | `model.py:317-321` | `engram_gather`, rows from the checkpoint's own `original-engram-*.safetensors` through the NVMe adapter | Same, one BF16 round |
| Engram `wkv` | fp8 Linear, block 32 | Route 3 fixed split-K (`SGLANG_ROUTE3_ENGRAM=1`) | Rounding, plus floor #5 |
| Engram gate | `model.py:353-365`, clamp 1e-6, eps 1e-20, signed sqrt | `engram_gate.py:33-45` | Same formula. Layers 1 and 14, before the block, `pre` kept (`deepseek_v4.py:4497-4517`) |
| hc mixes | `model.py:950-957` FP32 | `hc_mix_stats` (`mhc.py:2342`): K slices, `tf32x3` dots, `1/sqrt` | Rounding. `tf32x3` not in the ledger |
| Sinkhorn | `kernel.py:407-474`, 20 iterations, eps 1e-6 | `hc_split_sinkhorn_kernel` (`mhc.py:242`), TileLang | Same kernel body |
| `hc_pre` | `model.py:959-962` | `mhc_combine` (`mhc.py:2743`), then RMSNorm | Same. Lag kept: `HcState.pre` |
| `hc_post` | `model.py:964-968`, sum over i of `comb[i,j]` | `mhc_post_tilelang` (`mhc.py:1272`): `a[i_hci, i_hco]` | Same direction, one BF16 round |
| RMSNorm, all | `model.py:281-293`, eps 1e-20 | `forward_cuda`, `rmsnorm.cuh`; `dsv41_sparse.py:50` for index and compressor. Eps from `rms_norm_eps` 1e-20 | Rounding |
| Final collapse | `model.py:1271-1272`: last `ffn_pre`, norm | `deepseek_v4.py:4907-4918`. V4's `hc_head` not used: `hc_pre_from_prev_sublayer` True | Same |
| RoPE table | `model.py:369-389` | `dsv4/rope.py:27-68`, FP32 on the CPU, 1,048,576 rows | Same algorithm. Medium: ulps may differ from a GPU-built table |
| RoPE theta | `model.py:681-697`: layers 0-1 10,000 no YaRN; layers 2-39 160,000 with YaRN, factor 16 | `deepseek_v4.py:1397-1417` | Same |
| RoPE rotate | `model.py:392-406` | `rope.cuh:69-80` for q and o; `fp4_rope_fake_quant.py` for index and compressed | Same sign and conjugate |
| Latent position | `model.py:538-543`, first token of group | `group_pos`: `deepseek_v4_backend.py:3829`, `c2_decode_pool.py` | Same |
| Q | `model.py:771-773`, no head norm | `_compute_q_b` (`deepseek_v4.py:2009`); `q_head_norm` False on V4.1 | Same. No quant on main Q, same as reference |
| Window KV | `model.py:701-708`, `act_quant(…, 32)` | `fused_k_norm_rope_flashmla`, V4 branch. `SGLANG_DSV4_KV_LAYOUT` defaults to `v4` (`environ.py:1801`) | **Different: #1, #4** |
| Window ring | `model.py:410-426`, 128 slots, self included | `build_causal_swa_page_indices` (`attn_metadata_kernels.py:533`): `pos - k`, k < 128, length `min(seq, 128)` | Same set, newest first. Order only affects FP32 rounding |
| Compressor, ratio 2 | `model.py:464-485`, FP32, softmax over the pair | `project()` with `SGLANG_DSV41_FIXED_PROJECTIONS=1`; `c2_decode_pool`; `use_fused_compress` False on SM89 | Same algorithm. BF16 MMA into FP32 against IEEE FP32: rounding |
| Compressor, ratio 1 | `model.py:461-462` | `project()` then `finish()` | Same |
| Compressed KV quant | `kernel.py:161-163`, `model.py:761`: FP4, E4M3 scale per 16, floor 2^-9 | `fp4_rope_fake_quant.py`, same floor and clamp | Same, then **#3** |
| Shared KV | `kv_source_layers` 2, 8, 14, 20 | `source_layer_of` (`deepseek_v4_memory_pool.py:1895`) | Same |
| Shared index | `index_source_layers` 2-36 | one `sparse_page_indices` per ratio, overwritten by each source | Same |
| Index K | `model.py:544-547`, FP4 per 32, UE8M0 | `index_keys`, packed E2M1 + exponents | Same values, see the store floor above |
| Index Q | `model.py:551-553` | `queries`, `_rope_fq4` | Same |
| Index weights | `model.py:556`, scale exactly 1/64 | `head_weights` | Same |
| Index score | `model.py:557-558`, BF16 einsum, relu, weight, BF16 sum | Decode `fp4_indexer.py:423-430`, FP32 sum then BF16. Prefill `scores()`, the reference's own form | Rounding. **Medium:** near-ties can differ, already in the ledger. Rounding points checked; effect not sized |
| Reach | `model.py:565-568`: `(pos + 1) // ratio` | `scoring.py`, same | Same |
| Candidates | `model.py:584-611`, 2,048 blocks of 8, newest pinned | `select_candidate_block_ids` (`candidate_blocks.py`) | Same algorithm. Layer 20 publishes, 24-36 consume, 2-14 full top-k |
| Top-k, ties | `model.py:579-581`, torch topk | `live_topk_full`, `live_topk_among_blocks`, torch on prefill. `KT_IDX_LIVE_TOPK_MAX_L` covers 1M | Same set given equal scores. Tie rule follows torch by its docstring |
| Capacity | `index_topk` 512, `max_seq_len` 64K in interactive use | 1,048,576, candidates and top-k uncapped below that | Same rule, longer table. Above 64K is beyond the reference's own run |
| Sparse attention | `kernel.py:311-389`, scale 512^-0.5, BF16 probabilities, sink per head | Rows up to 8 `sm89_gather_staged`; above that `_fused_gather_attn_dsv4_dual_scope_kernel`, slices of 1,024 rows | Same formula on staged. Medium: unsplit kernel not read line by line; the staged kernel states bit equality |
| `attn_sink` | F32, not scaled | same | Same |
| Inverse RoPE | `model.py:782` | `fused_rope_inplace(…, inverse=True)` | Same |
| `o` QDQ | `model.py:787`, block 32, floor 1e-4 | `o_fp8_qdq_` (`SGLANG_DSV4_O_FP8_QDQ_BLOCK=32`) | Same |
| `wo_a` | `model.py:788-789`, BF16 einsum on dequantised weight | `fixed_wo_a_fp8.grouped`, E4M3 MMA per 32, 4 K splits (`SGLANG_DSV41_FIXED_WO_A`, `_CUDA`) | Accepted arithmetic. Ledger text is stale |
| Dense FP8 | `kernel.py:208-307`, activation per 32 UE8M0, 32x32 weight scales | Route 3 up to 16 rows, fixed split-K above. `act_scale_ue8m0` True | Accepted arithmetic, plus floor #5 |
| Gate | `model.py:813-828`, FP32 linear, sqrt softplus, bias picks only, sum + 1e-20, then 1.5 | `MoEGate.forward` with `fixed_projection.linear`, `moe_fused_gate` with log1p, lowest id wins ties | Same formula. Gate input BF16 MMA: rounding. `num_hash_layers` absent on V4.1, so no hash routing. Medium: did not confirm which of two equivalent top-k branches runs |
| Route scale | applied before `w2` | `fuse_routed_scaling_factor_in_topk` True, reached through the KT wrapper's `__getattr__` | Same order |
| Routed experts | `model.py:843-853`, `kernel.py:478-559` | contract v2: `kernels.py`, `contract_v2-moe.hpp` | Accepted: exact integer sums against truncated MMA. Quant floor 1e-4 |
| SwiGLU | gate max 10, up ±10 | table plus clamp | Same. **Measured:** table saturates at 9.99954605, silu(10), for every BF16 gate above 10, up to 448 |
| Shared expert | `model.py:889`, FP8 | `DeepseekV2MLP`, `silu_and_mul_masked_post_quant.cuh:40-48`, same clamps | Same, plus floor #5 |
| Sum order | `model.py:897-906`, ascending id FP32, shared last, one BF16 | `combine_routes`, logical ids, `shared.add_`, `finalize_output` | Same |
| Head | `model.py:1008-1014`, FP32 | BF16 | **Different: #2. Measured; flag measured** |
| Layers past 40 | `compress_ratios` 40-42 are DSpark | DSpark not served (`speculative_algorithm` None) | Not applicable |
| Vision | `bias_vl` only inside image spans | `--language-model-only` | Same on text |
| Prefill vs decode | one path | same kernels by row count; different index-sum code | **Measured earlier:** 512 positions within 4.5e-7 nats |

## Ledger corrections, proposed diff

`completion_work/kt_engine_decision/PRECISION_LEDGER.md`, for the session editing it. I changed nothing there.

Add to the 1 November table:

```diff
+| Logits | FP32 head weight, `F.linear(x.float(), weight)`, FP32 logits | BF16 x BF16 `torch.mm` with BF16 output (`SGLANG_DSV41_FIXED_LM_HEAD=1`, `--enable-fp32-lm-head` off). Every logit sits on the BF16 grid, step 0.125 on 71.5% of positions; top two exactly tied on 2.05% | Never chosen. Flag on, measured 3 Nov (`adahd1`): KL 0.01106 to 0.01072, top-1 96.74 to 96.91%, paired NLL -0.00027 (SE 0.00007), no ties, head 2.22 to 1.98 ms at one row. Not served. `reference-panel-r1/audit-r1/` |
+| Activation floor | `amax` floored at 1e-4 | 1e-10 on dense projections and Engram `wkv`; 1e-4 on routed experts and `o` | Upstream default, Path A has the right one. Effect limited to values under 3.7e-9, not measured |
+| hc mixes | FP32 `F.linear`, `torch.rsqrt` | K-sliced `tf32x3` dots, `1/sqrt`, TileLang Sinkhorn | Batch invariance |
```

Change these rows:

```diff
-| `wo_a` | Weights dequantized to BF16, BF16 einsum | FP8 blocks, FP32 weight-times-scale GEMV | Speed, and one weight rounding fewer |
+| `wo_a` | Weights dequantized to BF16, BF16 einsum | FP8 blocks, E4M3 MMA per 32 with FP32 scales, 4 K splits, ordered merge (`SGLANG_DSV41_FIXED_WO_A=1`, CUDA parts up to 16 rows) | Speed, and one weight rounding fewer |

-| Window KV, every layer | ... Coarser on the first 448, finer on the tail, size not measured | ...
+| Window KV, every layer | ... Coarser on the first 448, finer on the tail. Also quantised and rotated from FP32 registers with no BF16 round after the norm or after RoPE, where the reference rounds both. Size not measured | ...
```

Mark these rows of the 22 October table as superseded. Each one describes a path no longer served:

```diff
-| Routed experts, input | ... CPU AVX2: BF16 input x MXFP4, fp32 FMA (W4A16) ...
+| Routed experts, input | ... Superseded: contract v2, E4M3 digits per 32, floor 1e-4, exact integer sums ...

-| Routed experts, internals | ... down partial per NUMA part rounded to BF16, fp32 merge ...
+| Routed experts, internals | ... Superseded: one RNE per route, FP32 combine in logical expert id; no per-part rounding ...

-| Dense projections | ... FP8 block GEMV / split-K ...
+| Dense projections | ... Superseded: GEMV is off under `SGLANG_FP8_FIXED_SPLIT_K=4`; Route 3 up to 16 rows, fixed split-K above ...

-| wo_a | ... fp32 w*s GEMV (M=1), multi-row kernel (M=2..4), dequant+einsum otherwise ...
+| wo_a | ... Superseded: see the 1 November row ...

-| Compressed KV and indexer | ... torch/Triton indexer with the FP8/torch logits path ...
+| Compressed KV and indexer | ... Superseded: FP4 index K, `fp4_index_logits_decode`, torch on prefill ...

-| Sparse attention | ... Triton sparse decode with fp32 merge and three BF16 rounding points ...
+| Sparse attention | ... Superseded: split-K off; staged gather up to 8 rows, unsplit kernel above ...

-| Router | BF16 gemm then top-k | cuBLAS gemvx BF16->FP32 on SM89 ...
+| Router | Reference gate is FP32 `F.linear`, not BF16 | `fixed_projection.linear` BF16 MMA into FP32, 8 K splits, fused gate with log1p ...

-| Engram | ... `wkv` FP8 GEMV ...
+| Engram | ... `wkv` through Route 3 (`SGLANG_ROUTE3_ENGRAM=1`) ...

-| Compiler | ... kt-kernel built -O3 -ffast-math ...
+| Compiler | ... kt-kernel built -O3 -ffast-math; several JIT GPU kernels also build with --use_fast_math ...

-| Expert placement | ... which experts take the GPU path depends on routing history | not bit-identical, history-dependent |
+| Expert placement | ... | Superseded: bit-independent of placement since the logical combine order of 29 October |
```

The 22 October "Router" row also misstates the reference: the gate is FP32, not BF16 (`model.py:813`).

## What this does not cover

- Tool-call prompts and vision.
- The native AVX2 expert backend: I read its activation line and the contract's Python model, not the whole `.hpp`. I relied on the existing CPU and GPU identity receipts.
- The unsplit dual-scope attention kernel: read at the call site only.
- HiCache restore and radix prefix paths: logic read in part. The ledger's cold-against-hit receipts cover them.
- Whether the reference harness `dsv41q` matches DeepSeek's own kernels on Hopper or Blackwell. "Reference" here means that harness on Ada.
- On the card, only the head-only arm ran. Window KV and the activation floor still have no model-level number of their own.
