# DSpark draft experts in the KT CPU pool: build r1 (CPU-tested, no card)

Claude, 3 Nov 2026. No model load, no request to the served model, no GPU, no bench lock. All CPU work ran on
cores 39 and 47 under `systemd-run --user ... MemoryMax=16G MemorySwapMax=0`, `taskset -c 39,47`, `nice -n 19`
(plus `ionice -c3`), only while neither timing controller (ada8c2, adacb2) was running. The served trial was only read;
a check found no `.pyc` or `__pycache__` written under it.

Tags: **[measured]** run on fea01 for this report; **[code]** read in the served tree; **[record]** an earlier
measurement; **[estimate]** arithmetic; **[inferred]** reasoning not checked by running anything.

## What works and what does not

**Works on the CPU [measured]:**

- **The draft GGUF is a lossless repack.** All 1,152 expert ranges (3 stages x 128 experts x gate/up/down) decode to
  the draft checkpoint's bytes. Packed E2M1 and E8M0 scales match the per-tensor SHA256 receipts, and a seeded sample
  of 108 projections (216 tensors) is byte-equal to the safetensors files.
- **Draft scope (B1, B1b).** Behind `KT_DSPARK_CPU_DRAFT=1`, the three stages become KT layers 40, 41 and 42 in their
  own registry, `KT_DRAFT_LAYERS`. The target's `KT_LAYERS`, its hot-lend `LAYERS`, its KT configs, its kt-kernel
  constructions and its load maps are identical with and without the draft.
- **Own loader (B1c).** Each draft wrapper loads through its own reader on the draft GGUF and the draft manifest. The
  class-level loader is never created, read, replaced or released by the draft.
- **Loaded bytes.** The bytes the draft wrapper hands to kt-kernel equal the checkpoint's: 36 of 36 sampled
  projections in each of the three stages.
- **Forward.** kt-kernel's contract v2 backend (`AVX2ContractV2_MOE`, six K-parts, 128 experts, top-3) equals the
  contract v2 NumPy oracle bit for bit, at 1, 3, 4 and 8 rows, per route and summed. That holds for every call outside
  the truncation regime below: 1,474,560 route values in stages 0 and 1, 1,382,400 of them non-zero.
- **Selector off.** The overlaid tree reproduces the served tree's state exactly, both for the target alone and for
  target plus draft (the served bugs included).

**Does not work as served; fixed here with a native patch [measured]:**

- **Stage 2 reaches the contract's truncation regime on ordinary inputs.** Its down projection has per-row weight
  exponent spreads up to 11 bits (99th percentile 9). The target sample has at most 5, and stages 0 and 1 have 7 and 8.
  On Gaussian test inputs, 8 of 12 stage-2 calls floor low bits somewhere.
- **The served kt-kernel then refuses the call, and the refusal is sticky.** Its truncation counter is process-wide
  and never reset, so after one draft refusal every later target MoE call would throw too. In serving, that is a
  crash.
- **The fix is a small native patch, `apply_tolerant.py`, selector `KT_CONTRACT_V2_TOLERANT_FROM=40`, default off.**
  Layers at or above 40 keep the floor-per-part sum instead of refusing. Their events go to a separate counter.
  - Layer 39 and below still refuse, as served [measured].
  - In all 8 tolerated stage-2 calls, 0 of 645,120 BF16 values differ from the oracle [measured]. This is not
    guaranteed: the tolerant sum may differ from the oracle in the last bits [code].
  - Draft numerics never reach a committed token (greedy verify commits the target argmax) [code: feasibility read].
  - Built in a new folder in 90 s; the draft path refuses at load without it [measured].

**Not done or not testable here:**

- Anything on CUDA: the adapter's draft path with pinned and packed buffers, draft graph capture with KT submits,
  the zero-expert Marlin method, the full draft load, acceptance, identity and timing. See "What still needs the card".
- Phase 2 (payoff-based placement, idle-side scheduling, a per-batch speculation switch) is written below as a
  concrete design. Nothing of it is built.

**New code findings, beyond B1-B2 [code, plus CPU emulation where noted]:**

| # | Where | What | Status |
|---|---|---|---|
| B1d | `P/kt_kernel/experts_base.py` `KExpertsCPUBuffer.get_buffer` | Pinned handoff buffers are keyed by row count only and sized by the first caller's top-k. The target's top-6 buffers are returned to a top-3 draft at the same row count (draft T = 4 and verify T = 4 collide). `pack_to_host` then declines (numel differs), and the `copy_` fallback raises on [T,6] against [T,3]. | Fixed: the draft has its own buffers keyed by (rows, hidden, top-k); kt-kernel's buffers stay untouched [measured] |
| B1e | `S/srt/layers/moe/fused_moe_triton/layer.py:1232-1245` | Draft MoE layers load without expert-location metadata, so the KT bounds check in `_weight_loader_physical` is skipped. Expert ids 1 and up index past the draft's one GPU row. This settles the feasibility read's "wrong or refused": the served tree refuses at draft weight load. | Emulated: IndexError on experts 1, 64, 127 [measured]. Fixed: draft routed experts skip the GPU loader |
| B1f | `S/srt/layers/moe/contract_v2/placement_order.py` | With `KT_CONTRACT_V2_COMBINE_ORDER=logical`, `order_for` refuses layer ids outside 0-39. It would also apply the target's permutation to draft ids, which are logical (draft TopK has no placement remap). | Fixed: the draft combine uses order None (ascending logical id) |
| B6 | `operators/avx2/contract_v2-moe.hpp` (r16) | Truncation refusal is sticky and process-wide; stage 2 trips it (above). | Fixed by `apply_tolerant.py` |

## Files (`dspark-cpu-r1/`)

| File | What |
|---|---|
| `apply_dspark_cpu.py` | The overlay. Arguments: source sglang, destination sglang, source kt_kernel, destination kt_kernel, `kt_dspark_draft.py`. It patches four served files by exact anchors with count asserts and adds one module. |
| `kt_dspark_draft.py` | New module `srt/layers/moe/kt_dspark_draft.py`, imported only with the selector on. It holds the draft KT config, wrapper construction, load, own handoff buffers, the CPU-only adapter path and the tolerant-build check. |
| `apply_tolerant.py`, `build_dspark_tolerant_r1.py` | The native patch, and its build into `/scratch/aster/bench/ada-native-dspark-tolerant-20261103-r1` (served r16 tree + patch). The artifact `kt_kernel_ext...so` has sha256 `57e25283...` and the base is `fd5072c3...`. |
| `check_gguf_lossless.py` | The GGUF against the draft checkpoint (stdlib + NumPy only). |
| `test_registration.py`, `run_registration.sh`, `compare_registration.py` | Target and draft registration in one process, seven scenarios. |
| `test_draft_forward.py`, `run_draft_forward.sh`, `cpu_pin_shim.c` | The real kt-kernel: loader isolation, loaded bytes, forward against the oracle. The shim is test-only (see below). |
| `analyze_truncation.py` | Weight exponent spreads, and the oracle's floored terms on the failing input. |
| `results/` | JSON and logs copied from `/scratch/aster/bench/dspark-cpu-r1/out/`. |

**Overlay hunks** (selector off, each one is a module constant or a branch that runs only with the selector on):

- `srt/layers/moe/kt_ep_wrapper.py`:
  - selector constant, `KT_DRAFT_LAYERS`, `_draft_layer()`;
  - `create_kt_config_from_server_args`: inside the draft scope, the draft's config (layer id 40 + stage, 0 GPU
    experts, draft GGUF, `num_layers=None`);
  - `create_weights`: draft branch before `make_shadow` (no shadow, no `KT_LAYERS`);
  - `process_weights_after_loading`: draft branch (identity map, no split, no streamer, no page-cache drop of the
    target checkpoint; it drops the draft GGUF's page cache).
- `srt/layers/moe/contract_v2/adapter.py`: draft layers go to `kt_dspark_draft.apply` before any target logic.
- `srt/layers/moe/fused_moe_triton/layer.py`: the draft's routed experts skip the GPU weight loader.
- `packages/kt_kernel/utils/amx.py`: the `_OWN_LOADER` context variable. A wrapper built under it keeps a loader
  factory and loads through its own reader. Without it, the three loader sites run exactly as served.

Overlaid sha256 (served → overlay):

| File | Served | Overlay |
|---|---|---|
| `kt_ep_wrapper.py` | `c1eb8c4c` | `cc8e5a84` |
| `adapter.py` | `01beed62` | `00a3ca48` |
| `layer.py` | `62a3d3d4` | `be83f98d` |
| `amx.py` | `88c8282c` | `f78f970a` |
| `kt_dspark_draft.py` (new) | | `ba8e3209` |

## Test results

**1. GGUF against the checkpoint** (`results/LOSSLESS.json`, 32 s) [measured]:

- Raw range SHA256 against the manifest: 1,152 of 1,152.
- Re-packed weight and scale bytes against the receipt's per-tensor SHA256: 1,152 of 1,152 each.
- Shapes: 1,152 of 1,152.
- Sampled direct byte comparison with the safetensors files: 108 of 108 weights and 108 of 108 scales. Each sampled
  file tensor also matches its receipt hash (216 of 216).
- The decode follows ggml's MXFP4 rule independently of KT's reader. The reader's own path is covered by test 3.
- Status: **LOSSLESS**.

**2. Registration, one process, CPU** (`results/registration/`) [measured]. The real SGLang code runs:
`create_kt_config_from_server_args`, `draft_model_build_scope`, `KTEPWrapperMethod`, `kt_kv_hot_lend` (`make_shadow`,
`cpu_holds_all`, `_Layer`, and the real `split` on draft wrappers), `FusedMoE.weight_loader` and the adapter
dispatch.

Stand-ins:
- kt-kernel's `KTMoEWrapper` is a recorder (the real one is test 3);
- the GPU MoE method;
- expert-location metadata: seeded [40, 384] permutations;
- `torch.cuda.synchronize`;
- the CUDA half of `split` for target layers.

The C2 KT and hot-lend switches were set. 40 target layers and 3 draft stages were built.

All 12 checks pass:
- selector off, overlay == served: target alone, and target + draft;
- selector on, target alone == served;
- selector on, target + draft: target registry, layer ids, hot-lend list (40 entries), configs, kt-kernel
  constructions, load maps and split calls are all equal to the target alone;
- the draft registry is {40, 41, 42}:
  - each draft wrapper has 128 experts, top-3, a zero GPU mask and the draft GGUF;
  - its own loader factory is set during construction;
  - its base keys are `model.layers.{0,1,2}`, and its load map is the 128-identity;
  - it has no shadow, no hot layer and no class loader;
- the draft weight loader skips experts 0, 1, 64 and 127;
- the adapter sends layer 40 to the draft path and target layers down the served path;
- the draft's handoff ids buffer is [4, 3], and kt-kernel's buffers are untouched.

Fail-closed check: with the selector on and the served kt-kernel, draft construction refuses with
`needs the apply_tolerant.py kt-kernel build`.

The served tree with a draft, for the record (scenario B):
- `KT_LAYERS[0..2]` are draft wrappers;
- the hot lend has 43 layers, three of them draft layers with e = 128;
- the draft kt-kernel wrappers are built on the target GGUF at layer ids 0-2 with the target's 384-entry
  permutations;
- the GPU loader raises IndexError for draft experts 1, 64 and 127.

**3. Real kt-kernel on the draft** (`results/DRAFT_FORWARD_stage{0,1,2}_{served,tolnative_off,tolerant}.json`) [measured]

Setup: six subpools on NUMA 4/5, 1 thread each, `max_len` 64, the served KT environment for the contract backend.
`cpu_pin_shim.so` keeps kt-kernel's threads on cores 39 and 47; without it, libnuma's `numa_bind` widened them to
the whole node, which was checked with a probe binary. The test aborts if any thread leaves those cores. Each run
takes about 25 s, with a peak of about 8-9 GiB.

Loader isolation, every run:
- a target-like wrapper creates the class-level loader on the target GGUF and target manifest;
- the draft wrapper keeps its own factory and opens nothing at construction;
- its load uses one own reader (draft GGUF, draft manifest), which is closed afterwards;
- the class-level loader is still the same object and still open.

Loaded bytes: the loaded stage's 12 sampled experts x 3 projections are intercepted in `load_experts`; the other two
stages go through `read_projection`. All three stages: 36 of 36.

Forward: 12 calls a stage, at 1, 3, 4 and 8 rows, each top-3 with one skipped slot. There are two Gaussian input
sets and one with a 40x outlier group. Results:

| Stage | Native | Calls | Bit-equal to oracle | Oracle floors terms | Native |
|---|---|---:|---|---:|---|
| 0, 1 | served `fd5072c3` | 12 each | 12 of 12, 737,280 route values each | 0 | equal |
| 0, 1 | tolerant build, variable unset | 12 each | 12 of 12 | 0 | equal |
| 0, 1 | tolerant build, `=40` | 12 each | 12 of 12 | 0 | equal |
| 2 | served | 2, then refused | 1 of 1 before the refusal | 101 terms on the 2nd call | refused (sticky: counter 117) |
| 2 | tolerant build, variable unset | 2, then refused | same as served | 101 | refused, same call |
| 2 | tolerant build, `=40` | 12 | 4 of 4 without floored terms (92,160 values); 8 tolerated calls: 0 of 645,120 values differ | 101 to 18,891 terms a call | completed; global counter 0, tolerant counter 345,784 |
| 2 | same process, layer 39 on the same weights and input | 1 | | | refused, as served |

The summed output (`forward`, ascending-id FP32 combine) equals `oracle.combine` wherever the per-route output does.

**4. Why stage 2** (`results/TRUNCATION.json`) [measured]: per-row exponent spread (max minus min E8M0 code over K)

| | gate / up (max) | down (max, p99, mean) |
|---|---|---|
| Draft stage 0 | 3 | 7, 4, 1.8 |
| Draft stage 1 | 3 | 8, 4, 1.8 |
| Draft stage 2 | 3 | **11, 9, 3.3** |
| Target layers 0, 20, 39 (8 experts each) | 3-4 | 3-5, 2-4, 1.0-1.5 |

On the failing 4-row input, stage-2 experts 0 and 126 see down-input exponent spreads of 14-18 bits. Added to the
weight spread, some terms land 3-8 bits below the 20-bit margin. On real traffic the draft's down-input spread is
unknown [inferred], so the tolerant build is required, not optional: `kt_dspark_draft.check_tolerant` refuses at load
without it.

## Launch arguments for a staged trial

Add to C2's argv:

```
--speculative-algorithm DSPARK
--speculative-draft-model-path /scratch/aster/bench/05yuki-baseline-3991/draft-checkpoint
--speculative-dspark-block-size 3        # gamma 3; resolves --speculative-num-draft-tokens 4 (verify T = 4)
--speculative-num-draft-tokens 4         # pinned explicitly for the profile guard
--speculative-num-steps 1
--speculative-eagle-topk 1
--speculative-moe-runner-backend marlin  # C2's resolved value; draft GPU MoE holds 0 experts
--cuda-graph-bs-decode 1 2 4             # C2 captured [1, 2, 4]; under DSpark the verify graphs take bs*(gamma+1) tokens
```

`--max-running-requests 4` is already in C2. Without it, DSpark resets the value to 48 (`speculative_hook.py:850-858`).

Add to the environment:

```
SGLANG_RAGGED_VERIFY_MODE=static
KT_DSPARK_CPU_DRAFT=1
KT_DSPARK_DRAFT_GGUF=/scratch/aster/models/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
KT_DSPARK_DRAFT_MANIFEST=/scratch/aster/bench/kt-native-pilot/evidence/dspark-expert-inventory-v1/expert-ranges.json
KT_DSPARK_DRAFT_MANIFEST_SHA256=b75712873df442ab6047ee4eefad8b0c2b79e0ab3048f444d456fce77bc0fb7d
KT_CONTRACT_V2_TOLERANT_FROM=40
```

Optional: `KT_DSPARK_DRAFT_LAYER_BASE=40` (the default is the target's layer count) and
`KT_DSPARK_DRAFT_DROP_CACHE=1` (the default).

Notes:
- Static verify is the default (`environ.py:1612`); it is pinned here.
- The path the brief gave, `.../checkpoint/draft-checkpoint`, does not exist. The draft is at
  `/scratch/aster/bench/05yuki-baseline-3991/draft-checkpoint` [measured].
- Flag spellings are derived from the `field_order.py` names [code]; the profile guard must classify them.
- Process: the staged trial pins the overlay files and the kt-kernel build `57e25283`. That needs a new ORIGINS and
  the frozen client exception (B3 of the feasibility read).

## Memory arithmetic

**Host RAM is +8.3 GiB, not 6.7** [estimate from code: `BufferB::required_size` in the served r16 header]:
- contract v2 keeps FP32 group scales, E8M0 codes and row bounds next to the packed weights;
- a draft expert is therefore 22.2 MiB in RAM against 17.9 MiB on disk;
- 384 experts come to 8.33 GiB, which is 1.39 GiB a part.

Cross-check: the same formula gives 333.3 GiB for the target's 15,360 experts, consistent with about 392 GiB
settled.

**Per node** [estimate, `expert_arena_spill.hpp` with C2's `SPILL_NODES=3,7`, `STAGGER=1`, `REBALANCE_NODE0=3`]:

| Node | Change |
|---|---|
| 0 | +0.87 GiB |
| 1, 2, 4, 5, 6 | +1.08 GiB each |
| 3, 7 | +1.04 GiB each |

The handoff buffers and route scratch are KiB-sized. kt-kernel's shared activation pools are sized by the target's
top-6 and do not grow [code]. The CPUInfer is shared, so there are no new threads.

**Load transient** [estimate]: the draft loader holds one stage's tensors, about 2.2 GiB, until kt-kernel has copied
them, then frees them. The draft GGUF's page cache is dropped after each stage. The draft's safetensors expert tensors
are skipped before any copy, and the prefetcher skips `.experts.` names [code].

**Does it fit? Yes** [estimate]:

| Measure | Value |
|---|---|
| Serving scope, settled | 392 + 8.3 = 400.3 GiB, against the 416 GiB cap (15.7 GiB headroom; about 13.5 GiB at the load transient) |
| Node minima during runs | 9.5-14 GiB free become about 8.4-13, against the 4 GiB guard |
| Today, quiet | free plus inactive file pages per node 10.0-14.6 GiB [measured, `/sys/devices/system/node`] |

**VRAM:** the draft's non-expert part is about 1.1-1.3 GiB [record: feasibility table]. That leaves about 840 hot
slots instead of 903.

## Design proposals (unbuilt)

Each has its expected effect and how to measure it. The base cost figures are:
- the feasibility read's session fit, about 16 ms fixed plus 15 ms per unit of expert bytes;
- the Engram timeline per token: serial GPU 13.3 ms, window GPU 7.4 ms, CPU wait 6.9 ms, head 2.2 ms;
- about 7 distinct draft experts a stage at block 3, so about 0.4 GB, about 1.5 ms of pool time a step
  [record/estimate].

### P1. A small GPU cache for the draft's hottest experts

- **Design.** Give each draft layer a hot-lend `_Layer` (index 40-42), so its experts can take dynamic slots. The
  slot format is the target's (17.9 MiB, same geometry). This needs `make_shadow` to accept N = 0 (it returns today
  when N is 0) and two banks loaded from the draft checkpoint. Phase 2.1 below describes the shared budget. 32 slots
  is 0.56 GiB.
- **Expected effect** [estimate]: if 32 slots, about 11 a stage, take a share h of the draft's expert-calls, the step
  saves about h x 1.5 ms. Minus that, 32 fewer target slots (903 to about 871); at the tail of the target's slot curve
  that costs about 0.1-0.2 ms a step. The draft only gains if h > 0.15.
- **Measure.** `KT_ROUTING_DUMP` already counts layers 40-42 with this overlay: `_routing_count` runs before the
  contract adapter, and its table has 64 layers [code]. One card run over the six breadth workloads gives the
  draft's slot-hit curve. Then an A/B on step time at 0 and 32 slots.

### P2. Overlap each draft stage's CPU experts with GPU work

- **Exact options are few** [code + inferred]:
  - Stage s+1's attention needs stage s's MoE output.
  - The shared expert already overlaps the routed experts (`KT_SHARED_AFTER_SUBMIT`).
  - The target "pipelines" only within a layer: submit, then GPU experts and shared, then sync. Deferral is off in C2
    (`max_deferred_experts_per_token 0`).
- **Inexact option.** Draft-only deferral: stage s's CPU result joins stage s+1. That changes the draft's math, never
  a committed token.
- **Expected effect** [estimate]: hides the CPU time of stages 0 and 1, about 1.0 ms of about 1.5 ms a step.
  Acceptance may drop.
- **Measure.** Acceptance first, offline: replay the 8-prompt traces through a deferred draft (Python model on the
  card, or `cpu_ref.py`). Then step time.

### P3. Start the next draft for the full-acceptance case before verify finishes

- **Design.** The next draft needs two things:
  - the target hidden of layers 37-39 at the newly accepted positions, available when verify's layer 39 ends;
  - the bonus token, available only after the head.
  Guess the bonus as the draft's own prediction for the position after the block. Start the next draft (commit KV,
  stage attention on a side stream, stage experts on the then-idle CPU) during verify's head, sampling and commit
  tail, about 4 ms. Keep the result when every drafted token and the guessed bonus are accepted; drop it otherwise.
- **Expected effect** [estimate]: P(all 3 accepted) is about 0.79 x 0.74 x 0.68 = 0.40, from the calibration table.
  P(bonus guess right) is about 0.69. So about 0.27 x (about 3.3 ms draft) = about 0.9 ms a step, roughly 2% of a
  spec step. The cost is GPU SM contention with the head on misses.
- **Measure.** Offline from the existing traces: the joint rate of full acceptance and a correct bonus guess. Then
  Nsight of the verify tail.
- **Prerequisite** [inferred]: a draft logit for position gamma+1 (query rows = gamma+1).

### P4. Switch speculation off when more than one request runs

- **The tree cannot do it per batch** [code]:
  - `DSparkWorkerV2.forward_batch_generation` sends every decode batch through `_forward_decode` (draft, then
    verify).
  - Adaptive steps exist only for EAGLE/EAGLE3 (`adaptive_spec_params.adaptive_unsupported_reason`).
  - In static mode the block size is fixed at launch (`speculative_hook.py:812-860`).
  - `set_dspark_forced_budget_frac` only shapes compact (ragged) verify, which is not the identity-tested mode.
- **What it would take:**
  - a plain branch in `_forward_decode`: target decode at 1 token a request with layers 37-39 captured, then the
    draft commit KV for that token (`project_commit_kv` + `store_commit_kv`), then `bonus_tokens` = the sampled
    token;
  - plain decode graphs (1 token a request) captured next to the verify graphs;
  - an accepted length of 1 through the overlap scheduler's result path.
- **Expected effect** [estimate]: removes the 2-4 session loss (24 Oct: N4 41.9 against 57.5 plain [record]).
  Single-session gain unchanged.
- **Measure.** The serving profile at 1, 2 and 4 sessions, gate on against off.

## Phase 2: use the valuable or idle resource by payoff (design, unbuilt)

Selector for the phase: `KT_DSPARK_PAYOFF=1`, default off. It is scheduling and placement only; the identity
argument is in each part.

### 2.1 One GPU slot budget for target and draft experts, by payoff

**Where things live today** [code, `kt_kv_hot_lend.py`]:
- Scoring is per layer:
  - `route()` (or `_route_fused`) adds 1 per routed id into `STATE["cnt"][layer]`;
  - every `DYN_EVERY` = 4 forwards, `_dyn_round` decays the scores by 0.5^(4/64) and adds the counts;
  - within each layer, it swaps up to `DYN_SWAPS x DYN_EVERY` held experts for better-scored ones when
    new > victim x 1.5.
- Slot counts per layer are fixed at lend time by `_plan(budget)`: one slot at a time to the layer with the fewest
  for its quota.
- Quotas are a static file (`KT_KV_HOT_LAYER_QUOTA`, 40 integers 14-32, sum 932; derived offline in
  `layer-quota-r1`). Nothing moves slots between layers at run time.

**Design:**

1. *Score by payoff, not by route count.*
   - Value of holding expert e of layer L: v = h(e) x w_L x (t_cpu - t_gpu).
   - h is the EWMA of the forwards in which e is used at least once. Count per call, not per route: one CPU weight
     read is saved however many rows hit it.
   - t_cpu - t_gpu is the same for every expert, because target and draft experts have the same bytes and geometry.
   - w_L is the fraction of that layer's CPU expert time that is on the critical path:
     - target, about 6.9 / (6.9 + 7.4) = 0.48 on average [record: Engram timeline];
     - per layer, from the Nsight capture;
     - draft stages, about 1.0, because each stage waits for its experts serially [inferred].
   - The step structure is accounted for without per-mode cases: the draft runs once a step with gamma (or gamma+1)
     rows, and target layers run once a step at verify rows. Per-call counts at those rows are each layer's own
     usage per step.
   - Change: the route kernel writes a per-call flag (atomic max into a [layers, 384] scratch) and adds the flags at
     the round.
2. *Draft layers join `LAYERS`* as indices 40-42. The tables are [43, 384], using columns 0-127 for drafts. Each gets a
   shadow and two banks (`make_shadow` must accept N = 0 for draft layers). Their experts already sit in memfd arenas
   (`KT_EXPERT_SHM=1`). They need registering for the zero-copy `_arena_load`, which today registers `KT_LAYERS` only.
   That costs no new memory: the registration pins the existing pages.
3. *Cross-layer allocation.* Every R rounds (for example 64 rounds = 256 forwards), compute each layer's best unheld
   value and worst held value. Move one slot from layer A to layer B when v_best(B) > v_worst(A) x 1.5. A draft
   expert takes a slot only by beating the target expert it evicts. The move is done one of two ways:
   - remap the slot's 2 MB chunks from A's shadow range to B's (a new `kt_kv_vmm` call; chunk-aligned slot counts
     through `_cost`);
   - or, first version, re-plan with `_plan` on payoff-derived quotas at the next release/relend.
4. *Swap mechanics unchanged.* The same `_arena_load` copies from kt-kernel's arenas, on the same side stream, with
   the same table publication.

**Identity.** The GPU contract kernels are bit-identical to the CPU contract per route [record:
`contract_v2/moe.py` docstring, contract-v2-r1 SM89 validation], and the combine order is the same (ascending id). So
which side serves a route never changes the target's bits. Draft-side differences can only exist in the truncation
regime: GPU floors per term, tolerant CPU floors per part. They were 0 in all tested calls, and draft numerics never
reach a committed token.

CPU test to build with it: for random routes and slot tables, ownership is exactly once (the adapter's check). The
combined output from "slot" routes (oracle per route) and "CPU" routes (kt-kernel per route) equals the all-CPU
output bit for bit, on target-geometry and draft weights.

**Expected effect** [estimate]:
- At 1 session, block 3: the draft's about 21 distinct expert-calls a step weigh about 2x a target call (w = 1.0
  against 0.48).
- If the draft's routing is as concentrated as the target's (23 slots a layer took 64.8% of the routes [record]), a
  few dozen draft slots would take about half its calls: about -0.7 ms a step, for about +0.1-0.2 ms of target CPU
  time lost.
- At 2-4 sessions, verify rows grow and target values rise, so slots flow back to the target.

**Measure:** routing dumps for layers 40-42, the per-layer w_L from one Nsight capture, then step time and the GPU
share of the draft's calls.

### 2.2 Use whichever side is idle

| Window | Idle side | Draft work that can go there | Estimated saving |
|---|---|---|---|
| Target verify tail: head 2.2 ms + sampling + commit, about 4 ms | CPU | The next draft's stage experts, only via P3 (the bonus token is unknown until the head) | about 0.27 x 3.3 = 0.9 ms a step [estimate] |
| Target verify, each layer's expert window (CPU wait about 6.9 ms a token at 1 row; more at T = 4) | GPU | Dynamic slot copies for the draft and target (already async on the side stream, `KT_KV_HOT_DYN_ASYNC=1`). Prefetching the next step's likely draft experts into slots (P1/2.1) moves their CPU time to the GPU next step. | Part of 2.1's saving; no extra |
| Draft stage s: CPU experts | GPU | Stage s's shared expert (already overlapped). The draft commit KV for accepted positions is GPU work that must precede attention; it can run at step start, overlapping the target's layer-39 tail, if the hidden states are captured early. | about 0.1-0.2 ms [estimate] |
| Draft stages: GPU attention | CPU | Nothing exact (data dependence); P2's deferral if inexact draft numerics are accepted | about 1.0 ms [estimate], acceptance cost unknown |
| Target layers' GPU-only phases (serial GPU 13.3 ms a token) | CPU | Nothing for the draft of the same step (dependence). With P3 accepted, the CPU could pre-run the guessed draft's stage-0 experts during verify layers 37-39 of the *same* step, earlier than the tail. | adds about 0.3 ms to P3 on hits [estimate] |

Identity: all of these are placement or scheduling of draft work. The committed token is still the verify argmax of
an unchanged target computation.

Test: on the CPU, P3's keep/drop rule with recorded drafts. A kept pre-started draft must equal the draft run after
verify, bit for bit, through the oracle and kt-kernel.

### 2.3 Speculate only when it pays

- **Prediction.** Speculate on step t iff E[tokens | conf] / C_spec(bs, T) > 1 / C_plain(bs). Shorten the block to
  the T that maximises the ratio.
- **Inputs:**
  - E[tokens | conf] = 1 + sum over k of prod over j <= k of sigmoid(conf_j), from the calibrated confidence head
    (within 1-4 points [record]);
  - C(bs, T) = a + b x u(bs x T), where u is the expected distinct experts relative to one row (f(T) from
    `VERIFY_BATCH.md`), a about 16 ms and b about 15 ms (session fit), plus the draft cost (about 3.3 ms with CPU
    experts).
- **What the model says.** At bs >= 2, u is already 1.4-2.4x and the prediction is negative [record: feasibility
  read], so the rule reduces to "bs = 1 and E[tokens] above about 1.25" [estimate].
- **Tree support: launch-only** [code], as in P4. A per-batch switch needs P4's plain branch. A per-batch shorter
  block also needs:
  - verify graphs for each T, and draft graphs for each query width;
  - `query_token_num` and the verify window resolved per step instead of at launch (`speculative_hook.py:812-860`,
    `dspark_worker_v2.py:266-269`);
  - the planner's static layout per T.
  The first build should be the bs gate only (one extra graph set). Block shortening comes after the bs gate.
- **Identity.** A plain step commits the decode argmax; a spec step commits the verify argmax. Both equal plain decode
  if verify rows are row-count invariant, which is the same condition speculation itself needs. CPU test: none
  possible for the full model. On the card, the 24 identity pieces with the gate forced to alternate every 7 steps.

## What still needs the card

1. A staged-trial load with the overlay and build `57e25283`. Check:
   - construction of the zero-expert Marlin method on the draft layers;
   - the draft's GGUF load: 3 stages, about 4 s each on the CPU here [measured], plus the target's normal load;
   - host +8.3 GiB and node minima;
   - VRAM and hot-slot count (about 840 expected).
2. The adapter's draft path on CUDA:
   - `pack_to_host` into the [T, 3] buffers;
   - CPU routes through `forward_routes_task` from the draft's graph capture;
   - the logical combine.
3. Tolerant events on real traffic: log `contract_v2_tolerant_events()` per step (the binding exists). Confirm the
   global counter stays 0.
4. The feasibility read's first measurement (static block 3, bench profile, one session): verify step cost against
   the 65 ms gate, token ids against the archive, accepted length, and the draft-time split with CPU experts.
5. Routing dumps for layers 40-42 (P1 / 2.1) and one Nsight capture for w_L (2.1, 2.2).

## Provenance

- **Private workspace:** `/scratch/aster/bench/dspark-cpu-r1/`:
  - `base/` = byte copies of the served sglang and kt_kernel; `diff -rq` against the served trial is empty;
  - `over/` = overlay output; `tol/` = the overlay's kt_kernel plus the tolerant build;
  - `oracle/` = `contract-v2-r1/contract_v2.py` and its two tables (SHA256 pinned by the module);
  - `out/` = results.
- **Native build:** `/scratch/aster/bench/ada-native-dspark-tolerant-20261103-r1/RESULT.json`. Header
  `1fdad87c → e7047513`, `ext_bindings.cpp` `ee0566fc → 8cf7c3bf`, configure 3 s, build 86 s on 2 cores.
- **Test-only stand-ins:**
  - `cpu_pin_shim.so` (`LD_PRELOAD`): `numa_bind` keeps its memory binding but drops CPU binding;
    `hwloc_set_thread_cpubind` is a no-op. Validated with a probe binary: without it, a bound thread moves to
    32-39,96-103; with it, it stays on 39,47.
  - Pinned allocations become plain ones in the forward test (no CUDA in the process).
- **Rules followed:**
  - `CUDA_VISIBLE_DEVICES=` on every run; no GPU, no bench lock, no request to 18209;
  - no unload, no root, no GitHub, no Zulip;
  - heavy steps were checked against the ada8c2 / adacb2 controllers first; neither was running (adacb1 was running,
    and CPU work during it was allowed).
