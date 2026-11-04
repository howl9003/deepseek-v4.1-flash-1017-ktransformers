# DSpark plain-step gate: build r1 (CPU-tested, no card)

Claude, 4 Nov 2026. Owner's direction (4 Nov): fit DSpark into the idle times; a rough algorithm is enough while the
engine is not final. Selector `KT_DSPARK_PLAIN_GATE=1`, default off.

Work done: on this PC only. On fea01: `cat`/`grep`/`sed`/`sha256sum` and `scp` of single source files from
`ada-identity-dsp-20261104-r3`. No imports of the tree, no units, no lock, no server, no FEA jobs, no GitHub, no Zulip.

Tags:
- **[measured]**: run here for this report;
- **[code]**: read in the r3 tree at the pinned hashes;
- **[record]**: an earlier measurement from the brief;
- **[estimate]**: arithmetic;
- **[inferred]**: reasoning that nothing here has run.

## Summary

**What works [measured]:**
- The overlay applies to byte copies of the pinned files and compiles.
  - It inserts only: removing the inserted lines gives the served worker back byte for byte.
  - Every inserted statement is either the selector constant, the `None` default, or behind
    `_PLAIN_GATE_ON` / `self._plain_gate is not None`.
- 30 CPU unit tests pass. They cover:
  - the gate: the EMA, the threshold, the probe cadence, new-request reset, forget and prune, the forced pattern,
    and the overlap lag;
  - a model of the step bookkeeping, with the commit rules copied from the tree;
  - the graph-runner mechanics, against stand-ins;
  - AST checks that every API and attribute the module uses exists in the pinned sources.

**Unproven [inferred]: everything on the card.** Specifically:
- the plain DECODE graph capture beside the verify graphs;
- bit identity against C2 when DECODE and TARGET_VERIFY steps interleave on one request;
- the draft-KV commit from a DECODE hidden;
- the timing gain.

**Expected effect [estimate]:**
- Prose at T=1: from 11-18% below C2 to about 2-3% below.
- Code, math and JSON: unchanged.
- See "Timing run".

## Design

### The gate (pure Python, `PlainGate`)

- **The EMA.** Per request, an EMA of the tokens committed per speculative step, with `alpha` 0.25.
- **Its inputs.** The scheduler's own CPU counters, `Req.spec_verify_ct` and `Req.spec_num_correct_drafts`.
  - `_resolve_spec_v2_tokens` adds 1 and `accepted - 1` to them per decode result [code].
  - A plain result adds 1 and 0.
  - The gate remembers the kind of every step it issued, so it splits the counters' advance into speculative and
    plain results. Plain results never move the EMA.
  - Nothing reads the device.
- **Lag.** Under the overlap loop, a decision at step t sees results through t-2. That only delays the EMA [code].
- **The rule.** A request takes a plain step when all of these hold:
  - it has at least `min_spec` (4) speculative samples;
  - the EMA is below `threshold` (2.2);
  - fewer than `probe_every` (16) plain steps have run since its last speculative step.

  Otherwise it speculates, so each run of 16 plain steps ends with one probe.
- **New requests.** A new request has no state and speculates. State is dropped:
  - when a request finishes (`note_request_finished`);
  - when it leaves the batch;
  - when its counters go backwards.
- **Batch size.** Only bs = 1 takes plain steps. A larger batch speculates as today. Its steps are still recorded,
  so their results feed the EMA.
- **Ineligible steps.** These always speculate:
  - grammar;
  - `return_logprob`;
  - `return_hidden_states`;
  - sampling-mask capture.

  Their result shapes are verify-specific.
- **Refusals.** At startup the controller raises instead of running unvalidated for:
  - verify mode other than static;
  - attention DP;
  - PP > 1;
  - `SGLANG_SIMULATE_ACC_LEN`;
  - a dense draft;
  - a target other than `deepseek_v41`;
  - a mamba-state target.

**Break-even [record/estimate]:** 66 ms per speculative step / 30 ms per plain token gives 2.2 tokens a step. The
plain step adds the draft-KV commit, about 0.2-0.5 ms [estimate], which moves break-even to about 2.18-2.2.

### The plain step (`PlainStepController._run_plain`)

It runs in `_forward_decode`, after the idle return and before the draft. In order:

1. **Flush.** If `SGLANG_DSPARK_DEFER_CONTEXT_KV` staged a verify commit, it is flushed first, as before a prefill
   (default off).
2. **Window.** `alloc_verify_window(width=1)` takes the slot at the committed length.
   - It is the same `plan_uniform_extend` over the same over-allocated reservation verify uses, with the same host
     bound (`nxt_kv_lens_cpu`).
   - It gives the virtual and physical write ids and the `IterationIds` plan.
3. **The decode batch.** A `copy.copy` of the ScheduleBatch becomes the batch plain decode would run
   (cf. `ScheduleBatch.prepare_for_decode`):
   - `forward_mode=DECODE`;
   - `input_ids` = the bonus token;
   - `seq_lens + 1`; `seq_lens_cpu + 1`, or None, which is what DSpark's overlap path publishes [code];
   - `out_cache_loc` = the window slot;
   - `spec_info=None`.

   It goes through `ForwardBatch.init_new(capture_hidden_mode=FULL, iteration_ids=plan)`, which gives
   positions = committed length. The scheduler's batch is not touched.
4. **Forward and sample.** `target_worker.forward_batch_generation(batch=None, forward_batch=...)`:
   - The forward is the target's normal DECODE forward. With `dspark_layers_to_capture` set, the DSV4 model packs
     the layer 37-39 aux hidden in any mode [code: `deepseek_v4.py` `capture_dspark`].
   - Sampling is `model_runner.sample`, exactly as plain decode samples: greedy argmax, or the request's sampling
     on the forward-only `sampling_info` copy the scheduler installs [code].
   - The path A snapshot is taken there and again at the end.
5. **TP sync.** The tokens are TP-synced at `DSPARK_TARGET`, as prefill does.
6. **Publish.** `on_publish(seq_lens + 1)`.
7. **Draft KV.** `TargetHiddenKvInjector.inject_target_hidden(...)` with the window's cache loc, `cache_loc_2d`
   `[bs, 1]`, positions = committed length, `commit_lens = 1`, and the SWA-ring `state_slot` = `req_pool_indices`.
   This is the call `TargetVerifyExecutor.commit_hidden` makes for committed verify positions, at width 1.
8. **Result.** `GenerationBatchResult` with:
   - `next_token_ids` = `[bs, 4]` flattened, column 0 the sampled token (the scheduler slices
     `accept_lens` tokens from each stride-4 row [code]);
   - `accept_lens` = `block_accept_lens` = 1;
   - `cap_lens=None`;
   - `next_draft_input` = (bonus = sampled token, `new_seq_lens` = `seq_lens + 1`);
   - `speculative_num_draft_tokens` = 4;
   - `patha_static_flags` (`apply_dspark_flags.py` semantics).

### State the plain step keeps consistent [code + model test]

**Target KV and its derived state** (c4/c128 compressor state, compressed and indexer KV, SWA, FP4 and V41 window
KV): written by position.
- The DECODE step at position p overwrites whatever a rejected verify row left at p.
- It reads positions below p only, which committed steps wrote.
- This is inferred from the position-indexed layouts, not run. See risks 1 and 2.

**Engram history** (`EngramHasher.history`):
- DECODE commits it inside the forward (`MODE_DECODE`, `engram_hash_ids_and_commit`), so the plain step must not
  call `update_after_verify`, and it does not.
- A verify that commits only the anchor leaves exactly the history a DECODE step does (model test).

**Draft KV and SWA ring:** after every step, every committed position has target-hidden draft KV.
- Verify commits positions p..p+c-1; a plain step commits p.
- Positions past the committed length hold stale draft-block rows. The next draft forward overwrites them with its
  own block (positions p+1..p+4) before reading [code: `DraftBlockProposer._run_forward`].
- The ring is keyed by `pos % ring`.

**Seq lens and FutureMap:**
- `new_seq_lens = seq_lens + 1` is published.
- The bonus is stashed through `next_draft_input`.
- The reservation (2 x 4 slots ahead) covers a 1-token step.
- `kv_committed_len` advances by `accept_lens` = 1.

**Draft-side state not touched:** the noise counter, the staged sampling params and the observers. They only
shape proposals; greedy verify commits the target argmax.

**Bookkeeping model test [measured]:** 25 interleavings x 3 history lengths. After every step:
- the history equals the last n-1 committed tokens;
- target and draft KV cover exactly `[0, seq_len)`;
- the bonus is the next reference token.

The emitted stream equals the reference.

## Files and hashes

New files in `dspark-cpu-r1/`:

| File | What | sha256 |
|---|---|---|
| `apply_dspark_plain_gate.py` | the overlay; argv = sglang package dir, patched in place | `cf34e08e4e16901da1fa938a93228a4a453e14de57f43e460a5b3ae1e603e462` |
| `dspark_plain_gate.py` | the new module, copied into the tree by the overlay | `5b46ae729eaf8e7b62cadf729c4f6066623f3d4a8157451e0b8a6dbd5593983a` |
| `test_dspark_plain_gate.py` | 30 unittest cases (no torch) | `257cd0b82821802cf15547d5e1e0978491d132ce34270af564b82c609c052d0c` |

**Changed in the tree** (r3 tree + this overlay) [measured on byte copies]:

| File | Before | After |
|---|---|---|
| `srt/speculative/dspark_components/dspark_worker_v2.py` | `8eef0b1af504a5fabcc4a2028b98ac5b00aeec803cb976ca53af7d98b5b9e3d6` | `2ed1e6ea397b4baed945a95fc01dfa77f783b45484635a6b4c5f641e264b595a` |
| `srt/speculative/dspark_components/dspark_plain_gate.py` | (new) | `5b46ae729eaf8e7b62cadf729c4f6066623f3d4a8157451e0b8a6dbd5593983a` |

**Worker hunks** (all insertions):
1. `_PLAIN_GATE_ON`, read from the environment once at import.
2. The class default `_plain_gate = None`. A class attribute, so the worker's `__getattr__` forwarding to the target
   is never reached.
3. In `__init__`: `PlainStepController.from_env(worker=self)` under `_PLAIN_GATE_ON`.
4. In `init_cuda_graphs`: `capture_plain_graph()` before the draft's capture.
5. In `_forward_decode`: `maybe_plain_step(batch, on_publish)`, after `record_stream` and before the draft.
   Skipped under DP coordination.
6. In `note_request_finished`: `forget(rid)`.

**Pinned and read, unchanged:** the overlay asserts all of these, and a mismatch fails it.

| File | sha256 |
|---|---|
| `dspark_verify.py` | `f794385100777b5f30f96132b815cc74786f34d0b874210b959d42b117bad123` |
| `dspark_draft.py` | `f1886af3a0598dbf48242960c8b58cce86a2876074f376036d258b65eca09f7e` |
| `dspark_kv_inject.py` | `16c303b1e2f6b4cd5e1766084a498abfe193cb8223711e184ae9e09e780e8a38` |
| `dspark_planner.py` | `f95e7cd3ceb430fcaaab77b342bf1123a09adcb35917ecc85500052ac2c8cfcc` |
| `dspark_deferred_commit.py` | `c5cd906defe806e04f6f5a67ae2867ba4681980efe6ac60d33019298c4d966a8` |
| `spec_tp_sync.py` | `224fdb1aa782044fc7d253b72962c19ed105db39b772b19a3b2b71ebf799ce3f` |
| `dflash_info_v2.py` | `c66fd2a9629c5f0f36ad4eca34f2910dc8e8af509fd8c6b2fa3fd2722f18004f` |
| `runner/decode_cuda_graph_runner.py` | `b81c6f19ad8f39269e163b2aa38a0515bcb6fec5f0e885a49c12973418e04fae` |
| `model_executor/model_runner.py` | `efb9fd670c913039806a178ac2b460819f95dd0a590ae0c3ad4eb171ba82a51c` |
| `model_executor/forward_batch_info.py` | `ec43cb98a6ad635a5044cd2de84b101ff72d600fb36d59be4302734999479879` |
| `model_executor/input_buffers.py` | `893f105c50f93e141ce7cb457f4ae9c8864609a2c58697f5a9e59a343bd55aee` |
| `mem_cache/kv/alloc/iteration_ids.py` | `12560c387d4a09221c5bfa537a20b3c765b56948f215f5d458f44af5fdaa3665` |
| `managers/schedule_batch.py` | `fc1acedc503d7dab13a3570793730320491ec8ddfbc8ac47a3f9e2aa0f26fe9d` |
| `managers/tp_worker.py` | `c4e66a1fd07cdadd8ba7cf9e69fe8ca581b6e5f823135ea9652be03836e0429d` |
| `managers/utils.py` | `7a12a5d0827e2de29c0f7cf36064edcb4fbcbb4fa60dd1ec75244fc2f14d9e2e` |
| `scheduler_components/batch_result_processor.py` | `58e1006d617c6269e0ca4f184431e9723eb98229e5d788b5b2b947dff22e1d1b` |
| `layers/attention/deepseek_v4_backend.py` | `1c6ccdd48c1a69bf8fad8e67fd482972537e47ceafbf77411dc5bbcebadd6ff6` |
| `layers/engram.py` | `d440482dd9ae49d6ea7435f639c5722a08a2a4b4fb4f9ee86f3ed54b148327f2` |
| `contract_v2/patha_static_runtime.py` | `218d3d8b0d69b141c0999f5a59fd8816fd80a5c1648b95cddf75b963dcfa9df4` |

Order: after `apply_dspark_cpu.py`, `apply_dspark_flags.py` and `apply_dspark_swa_match.py`. The worker pin is the
r3 tree, which has all three. The staged trial needs a new ORIGINS entry for the two files.

## Switches and defaults

| Variable | Default | Meaning |
|---|---|---|
| `KT_DSPARK_PLAIN_GATE` | unset (off) | `1` enables. Unset: the module is never imported and every hunk is a false test |
| `KT_DSPARK_PLAIN_GATE_THRESHOLD` | `2.2` | tokens per speculative step below which plain steps run |
| `KT_DSPARK_PLAIN_GATE_PROBE` | `16` | plain steps between two probe speculative steps |
| `KT_DSPARK_PLAIN_GATE_ALPHA` | `0.25` | EMA weight of a new speculative step |
| `KT_DSPARK_PLAIN_GATE_MIN_SPEC` | `4` | speculative steps a new request runs first |
| `KT_DSPARK_PLAIN_GATE_FORCE` | unset | cyclic per-request S/P pattern, for example `SSPPP`. It overrides the EMA rule, for identity runs; ineligible steps still speculate |
| `KT_DSPARK_PLAIN_GATE_GRAPH` | `1` | `0`: no plain graph, eager plain steps (slow; for isolating graph issues) |
| `KT_DSPARK_PLAIN_GATE_LOG` | `0` | every N decode steps, log plain (graph) and speculative step counts and the EMA |

The batch-size limit is fixed at 1 (no switch). Everything else in C2 + DSpark stays as staged in
`DSPARK_CPU_BUILD_r1.md`.

## How the plain step gets its CUDA graph

**Why a second runner is needed [code]:**
- With speculation on, the target's only decode runner is a TARGET_VERIFY runner
  (`decode_cuda_graph_runner.py`, "capture mode + tokens-per-bs"):
  - `capture_forward_mode` is forced to TARGET_VERIFY;
  - its width is `decode_num_tokens_per_req` = 4;
  - there is no switch to capture DECODE graphs as well.
- Worse, a DECODE batch would be accepted by it. `can_run_graph` skips the width check when `spec_info` is None and
  accepts bs <= max_bs. So handing a DECODE batch to `model_runner.forward` unchanged would replay a 4-token verify
  graph with 1-token inputs.
- So even an eager plain step has to remove that runner for its forward.

**The plain runner [code; the property mechanics are unit-tested]:**
- `make_plain_runner_cls(DecodeCudaGraphRunner)` subclasses the served runner. It pins these as class properties
  whose setters ignore the base `__init__`'s assignments:
  - `capture_forward_mode=DECODE`;
  - `captured_req_width=1`;
  - `capture_hidden_mode=FULL`, which captures the DSpark aux hidden;
  - `capture_bs` filtered to bs 1.

  It also overrides `get_spec_info`, which returns None: a plain decode batch carries no spec info. No served file
  is changed for this.
- It is built in `DSparkWorkerV2.init_cuda_graphs`:
  - after the target's verify graphs, before the draft's, outside the draft scopes;
  - on the target's own model runner and attention backend;
  - in the global graph memory pool.
- **The backend's state.** `DeepseekV4AttnBackend.init_cuda_graph_state` replaces its per-bucket metadata dicts. The
  verify graphs read the metadata objects in those dicts, so a second call would leave them replaying stale
  metadata.
  - So the call is shadowed by a no-op for the construction only (`kept_graph_state`). It refuses a backend
    without those dicts.
  - DECODE and TARGET_VERIFY metadata live in separate `_GraphBucket`s, so the plain graphs' metadata is separate
    from the verify graphs'.
  - The verify mask and draft-extend width stay as the verify runner set them.
- **Shared buffers.** Input buffers are coalesced by (name, numel, dtype, device) (`input_buffers.py`), so the
  1-token buffers are separate allocations. The shared logits and aux outputs are consumed within each step, as the
  verify path already relies on.
- **KT sizes.** `KTMoEWrapper.set_capture_batch_sizes` is restored to the verify runner's list. KT caches handoff
  buffers for every size anyway (`KExpertsCPUBuffer.get_buffer`).
- **Routing.** `installed_runner` swaps `model_runner.decode_cuda_graph_runner` to the plain runner for the plain
  forward only, and restores it in `finally`.
  - `_forward_raw` then replays the plain graph with C2's DECODE path: out-of-graph raw metadata and in-graph prep.
  - Everyone reading the field during that forward reads the plain runner.
  - A later recapture of the target's graphs fails the next plain step instead of replaying stale graphs.
- **Fail-closed checks.** The capture refuses:
  - when forward hooks are registered (they would be traced into the graph);
  - when less than 1 GB is free.

  With decode graphs disabled, plain steps run eagerly and say so.

**Memory [estimate]:** one bs-1 DECODE graph in the shared pool, likely tens of MB to a few hundred MB. The capture
logs its GB. Check that the draft graph is still captured and the hot-slot count.

**If the graph cannot be used, the eager cost [estimate, not measured]:**
- The captured C2 decode already saves 51 launches a layer from the glue fusion alone [record: `KT_CONTRACT_V2_GLUE_FUSED`].
- Eager decode on this build is host-bound: thousands of launches a token over 43 layers, plus Python.
- So eager decode is likely 50-100 ms a token, no cheaper than a speculative step, and the gate would not pay.
- `KT_DSPARK_PLAIN_GATE_GRAPH=0` measures it if needed.

## Tests run [measured]

Run: `python -m unittest -v test_dspark_plain_gate.py`, Windows, Python 3.12, no torch.
- `PLAIN_GATE_SGLANG` pointed at byte copies of the pinned r3 files (scp of single files; hashes equal to fea01's).
- 30 of 30 pass, in 0.7 s.

| Group | Checks |
|---|---|
| Config (5) | Defaults; parsing; invalid values raise; each refusal reason; eligibility |
| Gate (12) | Speculates until 4 samples are counted (step 5 under lag 2); on prose, runs of exactly 16 plain steps between single probes; code never goes plain; threshold switch; recovery after the draft improves (two probes with alpha 0.25); plain results never move the EMA; EMA = tokens per spec step; forget, prune and counter restart reset; ineligible steps speculate but still feed the EMA; forced patterns (`SSPPP`, `SP` with ineligible steps); untracked results count as speculative |
| Bookkeeping model (2) | The interleavings above; a verify that commits the anchor equals a DECODE commit |
| Runner mechanics (4) | Properties hold DECODE/1/FULL/bs 1 through a stand-in of the base `__init__`'s assignment order; the bs filter; `installed_runner` restores on exceptions; `kept_graph_state` shadows and restores, and refuses unknown backends and double shadowing |
| Tree (7) | AST checks against the pinned sources (list below); the overlay test (below) |

The tree checks:
- the parameters of `alloc_verify_window`, `inject_target_hidden`, `make_next_draft_input`,
  `ForwardBatch.init_new`, `forward_batch_generation` and `DeferredContextCommit.flush`;
- the `GenerationBatchResult` fields;
- every `worker.*` attribute the module reads is assigned in `DSparkWorkerV2.__init__` /
  `init_attention_backends`, or is a method;
- the runner `__init__` assigns the four overridden attributes, and `capture_prepare` calls
  `self.get_spec_info(num_tokens)`;
- `can_run_graph`'s `spec_info is not None` width check, and `_forward_raw` reading `self.decode_cuda_graph_runner`;
- the backend's bucket reset and its separate DECODE/TARGET_VERIFY buckets;
- `EngramHasher.forward` commits in MODE_DECODE;
- ScheduleBatch is a plain dataclass with the fields the view sets, and the Req counters with the result
  processor's updates.

The overlay test: it applies to a temp copy and prints the hashes; only the worker changes; insertions only; six
guarded statements; a second run refuses.

**Not testable here:** anything that imports torch or the tree, including the torch-facing body of `_run_plain` and
`capture_plain_graph`. Their API surface is AST-checked; their behaviour is for the card.

## What an identity run must check

Setup: the C2 + DSpark staged trial (`DSPARK_CPU_BUILD_r1.md` launch args + this overlay), greedy, one session,
against the C2 plain-decode archive.

1. **Load:**
   - the log shows "DSpark plain gate on", then "captured plain DECODE graphs bs=[1] ... GB";
   - the draft graph is still captured (no "Disable DSpark draft cuda graph");
   - VRAM and hot-slot count compared with the trial without the gate.
2. **`KT_DSPARK_PLAIN_GATE_FORCE=P`:** every decode step is plain, so the DSpark-loaded server runs only DECODE steps.
   - Token ids must equal C2 bit for bit on the 24 identity pieces and the 35K cases.
   - This isolates the plain step itself: DECODE under the DSpark config, with hidden capture.
3. **`FORCE=SSPPP` and `FORCE=SP`:** token ids equal C2 (and today's DSpark) on the same set.
   - `SSPPP` has every transition: S to S, S to P, P to P, P to S.
   - Its length 5, with variable commit lengths, puts plain steps at every position mod 4 and mod 128. That covers
     c4/c128 block boundaries, SWA wrap and the draft's ring on the 35K cases, the main risk.
4. **Gate defaults** (no FORCE) on the breadth panel at T=0: equal to C2.
5. **Draft state (invisible in tokens, because greedy verify commits the target argmax):**
   - In run 3 on code/math/JSON, the mean accepted tokens of the speculative steps should match pure DSpark within
     noise (3.2-3.7 [record]). A drop means wrong draft KV from plain commits.
   - Exact version, optional: after `FORCE=P` and `FORCE=S` on one prompt, the draft pool's KV rows for positions
     committed in both runs must be bit-equal.
6. **Graph use:**
   - `KT_DSPARK_PLAIN_GATE_LOG=200`: plain steps equal plain graph replays.
   - `SGLANG_LOG_DECODE_GRAPH_KEY=1` on a short run: target replays alternate `mode=DECODE key_size=1` with
     TARGET_VERIFY.
7. **Guards:**
   - path A flags 0 on every result (no "C1 ... output discarded");
   - `contract_v2_tolerant_events()` global counter still 0;
   - no exceptions from the gate's fail-closed checks.
8. **T=1, not identity:** no crash; plain fraction on prose; acceptance on code unchanged. Plain steps sample from
   the target sampler and speculative steps use rejection sampling, so outputs are exact in distribution but not
   seed-equal to C2 or to pure DSpark.

## What a timing run should measure

- **Breadth.** The six breadth workloads at one session, T=0 and T=1 (prose at T=1). Three arms: gate off (today),
  gate on (defaults), and C2 plain decode. Quote the breadth median (28 Oct rule) and per-workload tok/s.
- **Per step, from step timers or one Nsight capture:**
  - plain step ms: expected about C2's decode token (about 30 ms [record]) plus the draft-KV commit, about
    0.2-0.5 ms [estimate];
  - speculative step ms: unchanged;
  - plain-step fraction per workload, and the probes' share.
- **Expected [estimate]:**
  - Prose: 16 plain steps plus one probe give about 17.7 tokens in about 16 x 30.3 + 60 = 545 ms, so about 32.5
    tok/s, about 2.5% below C2. Today it is 11-18% below.
  - Code/math/JSON: unchanged.
  - Each new request pays 5 speculative steps first.
- **Sensitivity:** threshold 1.9 / 2.2 / 2.5, probe 8 / 16 / 32, alpha 0.25 / 0.5, on one prose and one code workload.
  - Watch code workloads for prose-like stretches (comments): three 1-token steps drop the EMA from 3.5 to about
    2.06 and buy up to 16 plain steps.
- **Memory:** VRAM after load, the plain graph's GB, and the hot-slot count.

## Open risks

1. **DECODE then TARGET_VERIFY on one request is new.** C2 mixes prefill and decode; DSpark mixes prefill and verify.
   The target state (c4/c128 compressor state, compressed and indexer KV, SWA, FP4 and V41 window KV) is read here as
   position-indexed, so it should be mode-independent. Not run. Identity runs 3 on the 35K cases test it.
2. **A second runner on one attention backend.** The `init_cuda_graph_state` suppression is from reading the DSV4
   backend and its mixins (only the bucket dicts, the verify mask and the draft-extend width). Any other per-runner
   state shared between a DECODE and a TARGET_VERIFY capture would show up as identity failures in run 2 or 3.
3. **KT, contract v2, hot lend, Route 3, path A capture-time state** keyed by rows. The 1-row graphs are captured
   after the 4/8/16-row verify graphs. C2 captures 1/2/4-row decode graphs, so the row counts are supported; the
   cross-talk is unread.
4. **VRAM.** The plain graph's pool use could lower the hot-slot count, or push the draft graph under its 1 GB check.
   The draft would then fall back to eager, a large slowdown, so check the load log.
5. **No exact CPU seq lens.** The plain DECODE batch has `seq_lens_cpu=None`, as DSpark's verify does under overlap;
   C2's decode has exact ones.
   - DSV4's decode metadata is raw (req indices, seq lens, write locs), and SM89 has no length-keyed graph variants
     [code], so no effect is expected.
   - A non-SM89 build with candidate variants would pick the fallback variant.
6. **Gate inputs are scheduler counters.** A change in their semantics (grammar truncation, retraction) can misgate a
   step. That costs speed only, never identity.
7. **Untested paths:**
   - eager plain steps (`GRAPH=0`, or graphs off);
   - the deferred-commit flush (`SGLANG_DSPARK_DEFER_CONTEXT_KV`, default off);
   - bs > 1 (always speculative here).
8. **The subclass depends on `DecodeCudaGraphRunner` internals** (attribute names, `get_spec_info`). It is pinned
   and AST-checked; a changed runner fails the overlay.

## What needs the card

1. A staged-trial load with the overlay: the plain graph capture, its memory, the draft graph and the hot slots.
2. Identity runs 2-7 above. Run 2 first, since it tests the plain step alone.
3. The timing run above.
