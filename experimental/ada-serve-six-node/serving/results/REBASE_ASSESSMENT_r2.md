# Rebase onto 05yuki main, or stay and cherry-pick: assessment r2 (3 Nov 2026)

Claude (Opus 5.5), for the owner's question: "analyse if we should rebase or its diverged too much".
Target: 05yuki/deepseek-v4.1-flash-1017-ktransformers main at `b39e26b`. Ours: fork branch
`ada-serve-six-node-20261102` (`bd05401`) and the served engine on fea01, `ada-serve-c2-20261103-r3` (C2).

Read-only throughout. Nothing pushed, no GitHub writes, no Zulip. On fea01 only `cat`, `ls`, `grep`, `sha256sum` and
two small JSON reads; the GPU timing run, the bench lock and the server were not touched. All merging and patching was
done locally in the session scratchpad (`scratchpad\rebase-r2\`, listed at the end).

Tags: **[measured]** = I ran it and saw the result; **[code]** = read from source or commit text; **[inferred]** =
my reasoning, not checked; **[estimate]** = a guess at size or time.

## Bottom line

- **It has not diverged too far.** Our overlay merges onto his current tree almost cleanly. Of the 32 SGLang files
  we modify, 16 have an unchanged base, 15 merge three-way with no conflict, and 1 file has 4 conflict hunks. All 4
  are the same fix made on both sides. The 43 files we add do not collide with anything. The FP4 and V41 KV overlay
  anchors still match, and the KT side does not move at all. **[measured]**
- **But a rebase is a new build, not a merge.** His recipe now pins SGLang `3edea93c`, 291 commits past our
  `5d8e90a0`. Those commits change 785 files under `python/sglang`. Our "exact vs C2" claim covers only the tree we
  serve. I found no arithmetic change on our SM89, TP1 decode or prefill path **[code]**. Only the numerical set can
  show that, and the serving gate has to run again too, because scheduler and cache code changed the most.
- **Recommendation: rebase later, at a trigger.** Until then, stay on our base and cherry-pick three or four items.
  The triggers are in section 4.
  - Rebasing now costs about 2 to 3 working days, about one of them on the GPU **[estimate]**.
  - It resets the C2 anchor in the middle of the optimisation and DSpark work.
  - It brings no decode gain on the 4080 that we lack **[inferred]**.
  - His README now says the base moves every few weeks, so a rebase now would go stale soon.

## 1. What upstream changed

### 1.1 It is 58 commits, not 116

The local clone was shallow, so `git log HEAD..FETCH_HEAD` had no merge base and also counted history from before
the fork point. After unshallowing, the merge base is `7e199f2be`. Upstream is 58 commits ahead and we are 13 ahead.
GitHub's compare API gives the same figures. **[measured]**

- **23 of the 58** (`002ad60` .. `5d53f72`) were reviewed on 29 Oct. They are already inside our engine: C2's
  `SOURCE_STACK.json` records `recipe_base` = `5d53f72`, SGLang `5d8e90a0` and KT `95009ea`. **[measured]**
- **35 are new since our engine base** (`ee3a66b` .. `b39e26b`). They are classified below.

### 1.2 The 35 new commits

| class | commits | what they touch | effect on us |
|---|---|---|---|
| SGLang pin moves | `188d25c` (to `20036b7a`, 103 commits), `be6d808` (to `3edea93c`, 188 more) | `setup.sh`, every patch refreshed, `requirements.txt` (`humming-kernels` 0.1.12 to 0.1.18) | the real rebase cost (section 2) |
| tier-1 features we exclude | `f492665` (20 expert-cache, plus a new KT patch), `6dcdf64` (stream-mmq sources fetched at setup) | `kt_kv_hot_lend.py`, `kt_stream_prefill.py`, and KT `mxfp4-moe.hpp`, `moe_base.hpp`, `amx.py`, `ext_bindings.cpp` | none. Expert-cache needs stream-mmq (`requires.txt`), which is off our grid, and its KT patch touches four files of our KT overlay **[code]** |
| fixes we already carry | `7fef854` (KV lend reaches every pool tensor), `23dd4ea` (resident rows' scales after a swap), `dc21bdb` (pool clear releases hot slots first), `ee3a66b` (swap round no longer waits on the GPU) | patches 15, 16, 17 | ported as `KT_PREFILL_KV_LEND_ALL_TENSORS`; fixed our own way as `KT_PREFILL_LEND_EXPERT_SCALES` and `KT_PREFILL_SWA_CLEAR_RELEASE_HOT`. All three are on in C2 **[measured, served LAUNCH.json]**. `ee3a66b`'s waits are not on our served path: with `KT_KV_HOT_DYN_WORKER=1` our round publishes through `_tbl_write` from pinned memory, and the old pending path is not taken **[code]** |
| new recipe features | `cef35da` (21 hicache), `46da340` (`KT_STREAM_CPU_SHORT`), `77021a7` (22 prefill-warmup, plus an Engram staging change), `3e1fea9` (23 swa-checkpoint-tail) | patches 14, 21, 22, 23; `engram_backend.py` | candidates (section 4.2). **Patch 23 is being ported separately as an overlay; it applies to the served tree with offsets only [measured]** |
| launcher | `2c8ea18`, `16b61cb`, `d67800e`, `26882a0` (shipped chunk hints) | `serve.sh`, `lib/kt-lend-auto.sh`, `hints/`, `setup.sh` messages | none: we launch through our own controller and calibrate our own hints |
| toolkit and packaging | `d2468ff`, `62a2219`, `57765af` (CUDA 13.1, then 13.2 wheels), `8157e6c`, `d5d906a`, `cdf6747`, `2dedf4c`, `2924ae5` (prebuilt archive, CI) | `lib/cuda-home.sh`, `lib/cuda-wheels.txt`, `.github/` | none on fea01 (see below) |
| tools | `8d06e8a`, `c6828d9` (`tools/qualify.sh`), `b430f62`, `b68d51f`, `1f14e9a` (`qualify-cache.py`, PR #11) | `tools/` | useful as tools; no engine change |
| docs and reports | `f40734f`, `41995d1`, `7691e12`, `53d1789`, `4e63e1d`, `b39e26b` | `README.md`, `docs/`, `reports/` | none |

### 1.3 The items the brief asked about

- **`53d1789` "README: the recipe follows SGLang main".** It changes only the README and does not move the pin.
  - What it adds is a policy: each tag pins two upstream commits; the patches move to a newer SGLang main "every few
    weeks", are measured again and go out under a new tag; they are exported from branch `dsv41-1017-main` of
    `05yuki/sglang-upstream`; and a patch SGLang merges is dropped. **[code]**
  - The pin itself moved earlier: `5d8e90a0` to `20036b7a` on 30 Oct ("the patches replay unchanged but one line"),
    then to `3edea93c` on 1 Nov (three conflicts, both sides kept). **[code]**
  - KTransformers stays at `95009ea`. The KT patches (`kt-kernel-tree`, `-llama-cpp`, `-cpu-handoff`) are
    unchanged line for line. **[measured]**
  - What this means for us: whatever base we take now will move again within weeks. Rebase onto a tag, not main
    HEAD, and expect to repeat the port at each tag we want.
- **Toolkit and CUDA 13.2.** This is packaging only.
  - When a machine has no usable system toolkit, or has glibc 2.42 or newer with a CUDA crt older than 13.2's
    `rsqrt` fix, `lib/cuda-home.sh` builds a `CUDA_HOME` from NVIDIA's 13.2 pip wheels. **[code]**
  - fea01 runs glibc 2.43 with a system CUDA 13.3 (`nvcc` V13.3.73), so it never falls back to the wheels.
    **[measured]**
  - Our native builds use 13.3 under our own scripts, and our JIT compiles under our pinned venv.
  - Do not adopt the wheel toolkit. A different compiler for the JIT kernels is an identity risk with no gain.
    **[inferred]**
- **Shipped chunk hints (`26882a0`).** `hints/kt-lend/` holds prefill-lend calibration files for one and for two
  RTX 5070 Ti 16 GB: chunk sizes, bytes per row and free memory at prefill. **[code]**
  - `serve.sh` seeds them only when a launch is at his defaults: context, mem fraction, GPU experts, placement,
    SWA tails, lend chunk and TP. **[code]**
  - Our VRAM use differs: the Route 3, path A and `wo_a` buffers, 0.81 mem fraction, our slot counts. So the files
    do not fit our launcher, even on 5070 Tis. **[inferred]**
  - At most they are a reference point for the card swap. They do not touch numerics.

## 2. Recipe changes against our overlays and numeric path

### 2.1 His patches, line for line (5d53f72 to b39e26b)

I compared each patch's added and removed lines per target file, ignoring context and offsets. **[measured]**

| patches | change |
|---|---|
| tier 0, all 11 (`01`..`13`, incl. `08-sm12x-indexer`, `11-v41-kv-layout`, `13-sm89-attention`) | none; context refresh only |
| tier 1 `01`, `03`-`08`, `10`, `11`, `19`; tier 2; `sglang-optional/01` | none |
| tier 1 `02`, `09`, `12` | unused imports or locals dropped |
| tier 1 `13-prefill-lend` | `req.kv.set_row_len(P)`, following SGLang #43895 |
| tier 1 `14-stream-prefill` | `KT_STREAM_CPU_SHORT` (`46da340`), +17 lines in `kt_stream_prefill.py`, 2 in `kt_ep_wrapper.py` |
| tier 1 `15`, `16`, `17` | KV lend to every tensor; `rows_home`; host-side swap tables and `index_fill_`; SWA clear releases slots; host-KV page count |
| tier 1 `18-stream-mmq` | sources fetched at setup (excluded by us) |
| tier 1 `20`-`23`, new | expert-cache (excluded), hicache, prefill-warmup, swa-checkpoint-tail |

So nothing in his patch stack changes attention, dense GEMMs, the KV layout, the LM head or the MoE arithmetic. The
moving part is SGLang itself.

### 2.2 SGLang 5d8e90a0 to 3edea93c on our path

I diffed every changed file under `srt/` and `kernels/` that DeepSeek V4 can reach. **[code]**

| area | upstream change | on our SM89 TP1 path? |
|---|---|---|
| `models/deepseek_v4.py` (our overlay) | `wo_a` FP8 gate allows ROCm 32x32 blocks; NPU weight-name mapping | no; the CUDA 32x32 path is unchanged |
| JIT headers `deepseek_v4/*.cuh` (norm+RoPE, `store.cuh`, compressor `c128*`, `hash_topk`, `topk_v1`, `silu_and_mul...`), `elementwise/*.cuh`, `per_token_group_quant.cuh` | `kDLCUDA` to `kDLGPU` in host-side tensor matchers (#47300); a comment in `utils.cuh` | no device arithmetic change; the sources' text changes, so the JIT cache rebuilds |
| `block_amax.cuh` | a grid-stride loop over key chunks (#47149) | runs; max is order-free, so the result is exact |
| `kernels/ops/layernorm/hc_combine_norm.py` (mHC) | the row split now comes from `_parts_for(m, k)`; MXFP8 padding scales zeroed in-kernel | the non-MXFP8 split gives the same parts for k = 5120, and every CTA reduces the whole row in the same order; MXFP8 is Blackwell-only |
| `layers/engram.py` | `auto` lookup resolution; DP all-to-all and reduce-scatter paths | only at TP > 1; matters for a TP2 build later |
| `layers/moe/topk.py`, `kernels/ops/moe/dsv4.py` | AMD top-k gating; simulated routing with EPLB; mega-MoE weight scale | no; DeepSeek V4 routes through its own gate, and mega-MoE is not ours |
| `layers/logits_processor.py` (our overlay) | AMD gfx1250 LM head; typed `customized_info` | no; our FP32 head is our own `fixed_lm_head` |
| `layers/sampler.py`, new `sampling_renorm.py` | deterministic top-p and top-k renorm when TP > 1 | TP1 keeps the same kernels; TP2 later gains rank agreement |
| `layers/layernorm.py`, `quantization/fp8.py`, `attention/dsv4/fp4_indexer.py` | TP all-reduce flag; NPU and AMD MoE; SM90 indexer tiles | no |
| `mem_cache/deepseek_v4_memory_pool.py` (our overlay, FP4 and V41 anchor file), `deepseek_v4_compress_state.py`, `memory_pool_host.py` | buffer descriptions for HiCache and PD (#44990), added alongside the old code | not arithmetic, but HiCache now describes our pools this way, so the serving gate must cover it |
| `managers/scheduler.py` (+459), `batch_result_processor.py`, `model_runner.py`, `mem_cache/unified_radix_cache.py` (+256), `swa.py` | refactors (requests owner list, ReqKvInfo, Req finish), a retract-abort fix (#47521), `joint_lru` eviction (opt-in), HiCache sidecar and read-done fixes | serving and caching behaviour, not arithmetic |
| `srt/layers/attention/dsv4/` | nothing changed | — |

Overlay files on the numeric path whose base did not move at all **[measured]**:
- attention: `sm89_dual_scope.py`, `dsv41_sparse.py`, `metadata.py`, `v41_indexer/dense_blocks.py`, `scoring.py`,
  `dequant_k_cache.py`, `flash_mla_sm120*.py`;
- dense FP8: `fp8_kernel.py` (Route 3 and the path A hook), `per_token_group_quant_8bit_v2.*`, `fp8_utils.py`;
- mHC and transport: `deepseek_v4_mhc.py`, `moe_transport.py`.

### 2.3 Our overlay on the new tree, measured

**Method.** I rebuilt the patched trees the way `setup.sh` does, LF only, from fresh blob checkouts.
- **Our base** = SGLang `5d8e90a0` + recipe `5d53f72` (all features except stream-mmq and cpu-gpu-overlap) + our
  optional 01 and 02.
- **New** = SGLang `3edea93c` + recipe `b39e26b`, built two ways: the same features, and the same plus 21, 22 and 23.
  All 29 and all 32 patches apply. (A first try with a reused checkout left CRLF files and made `22` fail; this
  was redone from scratch, following the 29 Oct lesson.)

**Check of the base.** My base plus our snapshot overlay, plus `apply_fp4_kv.py`, `apply_v41_window.py` and the
serve-disk scheduler and HTTP files, against the served C2 tree's recorded inventory (`PROFILE.json`):
- 6,163 of 6,176 files are identical.
- The rest are 5 GGUF headers that are Git symlinks (a Windows artifact), `streaming_session.py` (our hand-ported
  optional-01 hunk) and 7 files outside the sparse checkout.
- So the merge below starts from what we actually serve. **[measured]**

| check | result |
|---|---|
| our 43 new SGLang files | no collision with a new upstream file |
| our 32 modified files | **16 base unchanged** (hash pin holds), **15 merge three-way with no conflict**, **1 with conflicts** |
| the conflicting file, `kt_kv_prefill_lend.py` | 4 hunks, each the same fix on both sides: our selector-gated KV-lend-all and SWA-clear release next to his unconditional versions, and his `WINDOW_START` hook next to our flag. Resolution: take his lines and retire our two selectors, which are on in C2 anyway |
| merged files with their line delta | `deepseek_v4.py` 6103→6107, `deepseek_v4_memory_pool.py` 2365→2555, `deepseek_v4_backend.py`, `logits_processor.py`, `fused_moe_triton/layer.py`, `parallel_state.py`, `scheduler.py` 6623→6966, `batch_result_processor.py`, `tp_worker.py`, `managers/utils.py`, `model_runner.py`, `deepseek_v2.py`, `kt_ep_wrapper.py`, `kt_kv_hot_lend.py` 1154→1208, `kt_stream_prefill.py` 1744→1761 |
| serving overlays (serve-disk `scheduler.py`, `http_server.py`) | merge clean |
| syntax | 69 of the 70 merged or added `.py` files compile; the 70th is the conflict file |
| FP4 compressed KV (`apply_fp4_kv.py`) and V41 window KV (`apply_v41_window.py`) | every anchor matches on the merged new tree |
| DSpark CPU draft (`apply_dspark_cpu.py`) | anchors match |
| DSpark flags (`apply_dspark_flags.py`) | **breaks**: it pins `dspark_worker_v2.py` at `a78fb549`, and upstream changed that file by 80 lines |
| our optional 01 (retained-session admission) | 2 hunks rejected: `unified_radix_cache.py` #1 and `streaming_session.py` #4 (the latter was already hand-ported on 30 Oct) |
| our optional 02 (prefix-cache policy) | 2 hunks rejected: `hybrid_pool_assembler.py` #2 and `unified_radix_cache.py` #2 |
| `shared-x`, `cpu-batch`, `tolerant` (KT header patches) | untouched; KT does not move |
| stage-script hash pins (`stage_phase_g_g1v_adasc3_r1.py`) | 4 of 9 overlay base pins break: `kt_kv_hot_lend.py`, `scheduler.py`, `http_server.py`, `kt_kv_prefill_lend.py`. The adapter, `wo_a`, live top-k and SM89 attention pins hold. The Engram adapter check ("old equals new apart from the rename") breaks on his two `engram_backend.py` changes |

**A clean text merge is not a clean behaviour merge** **[code]**. `kt_kv_hot_lend.py` merges clean, but it then
carries both his and our versions of three mechanisms:
- His `rows_home`, which reloads the resident rows at a window start, sits next to our scale mirror
  (`KT_PREFILL_LEND_EXPERT_SCALES`).
- His host-side `_pending` tables sit next to our worker-thread publish.
- His slot-shrink fix sits next to our layer quota.

On contract v2, which device holds an expert cannot change a bit. So these should cost or gain time, not exactness,
but they need a read and a timed run **[inferred]**.

### 2.4 What "exact vs C2" means after a move

C2 is defined by the served tree: SGLang `5d8e90a0`, recipe `5d53f72` patches, our overlay, FP4 and V41 KV, and the
FP32 head. After a rebase, C2's evidence (the 11,496-score numerical set, the panel KL of 0.01019, breadth 33.65 in
serving) describes a different program. The rebased build has to show all of the following before we can call it
"exact vs C2":

1. The numerical set: 11,496 scores EXACT vs C2. 4K and both 35K prompts with their 40 layer-flag checks clear.
   Text, ids and logprobs equal on the long cases. The scale check (streamed prefill, English reply, flush, the same
   prefill).
2. The replay fixtures, and the host-reload gate (`eb31f84a`) unchanged.
3. The serving gate: the seven cache checks, cold, device, host and disk paths exact, retained sessions. This is
   where 291 commits of scheduler, radix and HiCache churn would show. The numerical set does not exercise it.
4. The performance set against C2: breadth median, Japanese, 35K, 4K, and the serving profile's 33.65.

If item 1 fails anywhere, the build becomes a ledgered numerical change. It then needs its own paired NLL and panel
KL against the 0.00996 floor. It also becomes a new anchor (C3), and every result taken on C2 stays C2's.

## 3. Effort

| item | size | basis |
|---|---|---|
| fork branch (git) | trivial: a trial `git replay` of our 13 commits onto `b39e26b` is clean; `experimental/` comes out byte-identical; only the root `README.md` (+4 lines), `CREDITS.md` and `.gitattributes` are added | [measured] |
| base trees at the new pins | done in scratch; needs repeating on fea01 CPU-only with exact inventory closure | [measured] locally |
| merge 15 files, resolve 4 hunks, normalise CRLF (`parallel_state.py`) | about half a day | [estimate] |
| semantic review: hot lend, KV lend, stream prefill, scheduler, HiCache buffer info against our V41 and FP4 SM89 pool layouts | about a day | [estimate] |
| re-port optional 01 and 02 (4 hunks), the DSpark flags pin, the Engram adapter pin, 4 stage pins, `SOURCE_STACK`/`PROVENANCE`/inventories, the server-info parser (new metadata keys likely, as on 30 Oct) | about half a day | [estimate] |
| environment | the same venv plus `humming-kernels` 0.1.18; torch, Triton 3.8 and FlashInfer 0.7.2 unchanged; no native rebuild (KT, Route 3, path A and `wo_a` binaries are reused) | [code] / [measured] versions on fea01 |
| GPU qualification | identity loads until exact (the 30 Oct port to `5d8e90a0` needed three: a missing KT package, then a new server-info key), the serving gate, the performance set: about one day of GPU windows | [code] precedent, [estimate] |
| **total** | **about 2 to 3 working days, about one of them on the GPU** | [estimate] |

## 4. Recommendation

### 4.1 Rebase later, at a trigger; do not rebase now

The trigger is the first of these:

1. **The SM120 build, if the 5070 Ti swap goes ahead.** That build needs a full qualification anyway, on a new
   architecture. His tier-0 SM120 path, his 5070 Ti calibration and his TP2 fixes are tuned on his current main.
   One qualification can cover both moves.
2. **His next SGLang move under a new tag** (per `53d1789`), *and* a free GPU window of about a day. Take that tag,
   not `b39e26b`, so the port is paid once.
3. **An SGLang fix we need that does not backport cleanly.** The live candidate is #47521, `c105a91b`, which
   aborts retracted requests whose KV the recompute cannot rebuild. It matters if four sessions near 1M start
   retracting.

Why not now:
- **The gain is small.** For the 4080 build, his new decode-side changes are already in ours, or ours do the same.
  His other new work improves time to first token, and SGLang's DeepSeek changes target other hardware or TP > 1.
  **[code], [inferred]**
- **The cost is real.** A rebase resets the C2 anchor in the middle of the owner's current order: the remaining
  optimisations, then the DSpark fit, then lower quants. It would also compete for the GPU with the timing run now
  in progress.
- **The fork branch should not move on its own.** A git-only rebase would put a `3edea93c` recipe at the top of a
  branch whose snapshot is built on `5d8e90a0`. The snapshot README already says its own pins, so this is harmless
  but confusing, and it needs a push. When the engine port happens, publish it as a new branch; that is the owner's
  call.

### 4.2 Cherry-pick now, onto our base, each behind a default-off selector

| # | commit | what | why | applies? | check |
|---|---|---|---|---|---|
| 1 | `3e1fea9` (patch 23) | swa-checkpoint-tail | a repeated prompt matches to its end after an answer (prefix cache) | yes, with offsets, on the served tree [measured] | already being ported separately |
| 2 | `cef35da`, `memory_pool_host.py` part only | caps the HiCache write-back staging buffer at 32 MiB | we serve HiCache with the default `page_first` layout, and in our base that allocates the full staging buffer on the GPU [code]. His figure is 330 MiB for the SWA pool, worth 36 hot slots on 16 GB cards, decode -2 to -4% without it. For us that is VRAM back to the hot slots [inferred] | yes, cleanly [measured] | bits cannot change (a copy buffer), so check exactness, slot count and breadth; skip the patch's SWA-lend parts (we run `KT_PREFILL_SWA_LEND=0`) |
| 3 | `46da340` | `KT_STREAM_CPU_SHORT`: a prompt of up to 1,024 rows leaves k of its streamed groups to the CPU | time to first token on short prompts; his one-GPU 1K went from 9.2 s to 6.1-6.8 s | merges clean into our `kt_stream_prefill.py` and `kt_ep_wrapper.py` [measured] | on contract v2 the CPU and the path A GPU stream are meant to agree bit for bit, so the numerical set should be exact [inferred]; time a 1K prompt |
| 4 | `77021a7` (patch 22) and its `engram_backend.py` hunk | one warmup prompt per length before ready; Engram staging retired behind an event, not a stream sync | the first prompt of each length (his: 35K first-prompt time 37.4 s to 33.5 s) | `warmup.py` applies [measured]; the adapter change needs the stage pin updated | serving only; about 20 s more to ready |
| — | `tools/qualify.sh`, `tools/qualify-cache.py`, `tools/nll-ref-1017.json` | his qualification tools | step 0 of the PR plan needs his `qualify.sh` baseline anyway | tools only | none |

**Do not take:**
- `f492665` and `6dcdf64`: stream-mmq and expert-cache are off our grid, and expert-cache rewrites four files of our
  KT overlay.
- `ee3a66b`: our round worker already takes these waits off the forward.
- `7fef854`, `23dd4ea`, `dc21bdb`: we have them.
- The toolkit wheels, the prebuilt archive and the shipped hints: not for our host or launcher.
- SGLang `1f834c19` (HiCache sidecars over every source transfer): it matters only for the draft-SWA sidecar, so
  take it with DSpark serving or with the rebase **[code]**.

## 5. What each side has that the other lacks

**Upstream has, we lack (and would want):**
- **Correctness.**
  - SGLang #47521 (retract abort): edge case under KV pressure.
  - SGLang #43110 (`1f834c19`): only with a draft model and HiCache.
  - Deterministic top-p and top-k renorm at TP > 1 (#38565): for a TP2 build.
  - Patch 23 (being ported).
- **Performance.**
  - HiCache staging cap (patch 21 part): VRAM to slots.
  - `KT_STREAM_CPU_SHORT`: short-prompt time to first token.
  - Prefill warmup: first-prompt time to first token.
  - No new decode-path speed for SM89 TP1 **[inferred]**.
- **Tooling.** `qualify.sh`, `qualify-cache.py`, the qualification report ledger, and 5070 Ti calibration data for
  reference.

**We have, upstream lacks** (from `PR_PLAN_05YUKI_r1.md`; about 12.4 of the 20.2 ms we saved per token sit on code he
does not have):
- **Numerics.**
  - Contract v2 integer-exact CPU experts, path A prefill experts and Route 3 dense decode.
  - FP4 compressed KV and V41 window KV on SM89, and the FP32 head.
  - The live top-k fix (`-inf` and capacity).
- **Speed.**
  - Arena stagger, the round worker, route reference, fused glue, the `wo_a` CUDA kernel.
  - Engram batched reads, staged gather, Engram through Route 3.
  - The CPU window bundle, and the CPU-batch candidates.
- **Serving.**
  - The disk KV tier.
  - Retained-session admission (optional 01) and the prefix-cache policy (optional 02).
  - The DSpark CPU draft (in progress).

A rebase does not shrink any of this. It is the reason our overlay is large, and it stays ours either way.

## 6. Risks of a rebase

Top three:
1. **The C2 anchor.**
   - The risk: 291 SGLang commits, recompiled JIT kernels and merged hot-lend code. I found no arithmetic change
     on our path **[code]**, but if any bit moves, every "exact vs C2" result (33.65, panel KL 0.01019) stays with
     the old build.
   - The cost if it happens: a new ledgered anchor and a paired NLL.
2. **Serving and cache regressions.** The heaviest churn is where the owner's serving setup lives:
   - scheduler, unified radix cache, swa and HiCache buffer-info code, which serve the four sessions and the disk
     KV cache;
   - our optional 01 and 02, which lose 4 hunks.

   The numerical set cannot catch these; only the serving gate can.
3. **Behavioural double fixes in the lends.** His `rows_home`, his unconditional KV-lend and SWA-clear code, and
   his host-side swap tables merge silently next to our selectors and round worker. Bits should hold on contract
   v2, but timing, slot counts and breadth can move either way, and a race in the worker thread would show only
   under load.

Others:
- **A moving target.** His policy moves SGLang every few weeks, so a port now buys about two weeks of being
  current.
- **GPU time.** The port competes with the timing run now in progress and the queued DSpark fit.
- **Server-info and profile parsers.** Strict parsers refuse new metadata keys; this stopped the 30 Oct port
  once.

## Evidence and method

All files below are in
`C:\Users\howl\AppData\Local\Temp\claude\C--Users-howl-Downloads-Claude-strata-deepseek1017\b00a77c4-432d-4001-9414-9702e1d66640\scratchpad\rebase-r2\`.

| file | what it is |
|---|---|
| `patch_semantic_diff.py`, `patch_delta.txt` | the per-patch, per-file comparison of added and removed lines, 5d53f72 to b39e26b |
| `build_trees_r2.sh` | patched trees from fresh LF checkouts: `trees\tree_base`, `tree_new_same`, `tree_new_plus`, `tree_new_old_recipe` (the old patches on the new SGLang: 5 fail, so his refreshed patches are required) |
| `trees\full_*` | each tree plus our optional 01 and 02 |
| `merge_r2.py` | the overlay three-way merge; its output is `trees\merged_new_same` and `trees\merged_new_plus` |
| `trees\c2_recon` | the reconstruction of the served C2 tree |
| `trees\anch_*` | the FP4, V41 and DSpark anchor runs |
| `g0_SOURCE_STACK.json`, `served_PROFILE.json` | copies read from fea01 |
| `sglang_changed_5d8e_3edea.txt` | the SGLang file list |

Other sources:
- **Commit counts.** The fork clone `scratchpad\fork-tmp` was unshallowed, and upstream main fetched to
  `refs/remotes/upstream/main`.
- **The trial rebase.** It was done with `git replay`, which in this git version moved the local branch ref. I
  restored that ref to `bd05401`, and the trial result is kept at `refs/scratch/rebased-trial`. This was local only;
  nothing was pushed.
- **The SGLang clone** at `scratchpad\rebase-check\sglang` was extended with `3edea93c` and `20036b7a`.
