# DSpark plain gate r3: the confidence signal (CPU-tested, no card), and a slot give-back design

Claude, 4 Nov 2026. The owner approved "the free signal and the slot give-back first". Part 1 is built here and tested
on this PC; Part 2 is a design only.

Work done: on this PC only. From fea01, read-only:
- `scp` of two source files from `ada-identity-dsp-20261104-r3` into the local pinned-tree copy: `srt/sampling/sampling_batch_info.py` and `srt/layers/sampler.py`;
- `sha256sum` and `grep` of the same tree and of `deepseek_v4_dspark.py`.

Nothing was copied to fea01. The card, the queue, the runner and the r2 files in the parent folder are untouched (r2 module still `6d0636b3...`, apply `63dc3ea7...`). adadpb and adadpn stage against r2 as planned.

Tags:
- **[measured]**: run here;
- **[code]**: read in the pinned sources;
- **[estimate]**: arithmetic;
- **[inferred]**: reasoning that nothing here has run.

## Summary

- **Built:** a per-request confidence signal, computed on the device each decode step and handed to the CPU late with no sync.
  - A calibration log in both modes.
  - A new probe rule, `KT_DSPARK_PLAIN_GATE_SIGNAL=confidence`, off by default.
  - In the default mode (`accept`) the gate issues r2's steps exactly [measured: identical step sequences to the r2 module on five acceptance patterns, with and without confidence samples arriving, single and batched].
- **Tests:** 58 [measured].
  - Without the pinned tree: 50 pass, 8 skipped.
  - With `PLAIN_GATE_SGLANG` set: 58 pass, including the overlay test (now 7 inserted blocks) and a new check on the sources the signal reads.
- **Design fix found by the tests:** a confidence probe must replace the acceptance EMA, not blend into it. At alpha 0.25 one good probe lifts an EMA of 1.0 only to 1.75, still under 2.2, and the edge-triggered probe does not repeat, so a request whose draft recovered would stay plain for good. Now the probe's tokens become the EMA, in confidence mode only.
- **Unproven [inferred]:**
  - the signal's cost on the card (about 6-8 small eager launches and one event a step [estimate]);
  - whether the confidence EMA separates prose from code well enough. The calibration lines of one card run decide `KT_DSPARK_PLAIN_GATE_CONF`; until then it is 0.0, which disables confidence probes.
- **Part 2:** feasible on the existing `kv_vmm` machinery.
  - Of the draft's 0.76 GB, about 0.67 GB can leave the card in plain mode. That returns about 35 of the 60 slots DSpark costs today (949 → about 984; C2 1,009) [estimate].
  - It needs a partial slot release in the hot lend, which does not exist today.
  - A switch costs about 30-40 ms of PCIe copy off the critical path, plus mapping work [estimate].

## Part 1: what changed

### The signal

Per step, the target's top-1 probability at the position the step commits first, at the request's sampling temperature.

**Plain step.** `_run_plain` [code: module line 808] reads `logits_output.next_token_logits` after `target_worker.forward_batch_generation` has sampled from it. `Sampler.forward` [code: `srt/layers/sampler.py`, pinned `b9b0c965...`] changes that buffer in two ways:
- An all-greedy batch only takes an argmax, so the buffer is raw logits. The signal is `exp(max - logsumexp)` at T = 1.
- Any other batch gets `logits.div_(temperatures)` then `logits[:] = softmax(...)` in place. The buffer is then softmax(logits / T), and the signal is the row maximum.

Greedy requests carry T = 1.0 (SGLang turns temperature 0 into temperature 1.0 with top_k 1, `sampling_params.py`) [code], so both reads give the same quantity.

**Verify step.** The worker hunk is placed right after

```python
logits_output = target_verify.logits_output
can_run_cuda_graph = target_verify.can_run_cuda_graph
```

in `DSparkWorkerV2._forward_decode` [code: pinned worker `8eef0b1a...` lines 1182-1183]. It calls `note_verify_logits(batch, logits_output.next_token_logits)`, which reads row 0 of each request's request-major `bs x verify_width` rows (`anchor_stride`). The hook comes before the grammar mask, the accept (`accept_and_finalize` or the sampling-accept region graph) and any sampling. The verify target forward returns before the tp worker samples (`if is_verify: return batch_result`, `tp_worker.py`) [code], so these are raw logits, scaled by `sampling_info.temperatures`.

A shape that matches neither layout is skipped and counted.

**Reads only.** `top1_probability`, `_note_logits` and `note_verify_logits` write only new tensors. The one in-place copy is into the ring's own pinned host row. Every kernel is stream-ordered after the forward that wrote the logits, so it reads them before the next forward reuses the buffer. An AST test checks this. Plain and verify steps keep their arithmetic [measured: AST tests; code review].

### The ring (`ConfidenceRing`)

- 8 pinned host rows of 64 floats, each with one CUDA event.
- `push` copies the step's vector into a free row with `non_blocking=True` and records the row's event.
- `drain`, run at the start of `maybe_plain_step`, returns the rows whose events have completed, oldest first, and stops at the first row still in flight.
- There is no `.item()`, `synchronize` or wait anywhere in the module (tested).
- With every row in flight, or more than 64 requests in a step, the push is dropped and counted.

The gate therefore sees confidence two or more steps late, like its counters.

### The gate

- **Confidence EMA.** Per request, with the same `alpha` as the acceptance EMA (0.25). Samples for requests the gate no longer tracks are ignored.
- **`accept` mode (default).** r2's timer probe, step for step.
- **`confidence` mode.** The timer probe is gone. A request in plain mode probes once when its confidence EMA rises to `KT_DSPARK_PLAIN_GATE_CONF` after being below it since its last speculative step (edge-triggered: `armed`).
  - That probe's tokens replace the acceptance EMA.
  - `min_spec`, the 2.2 threshold, the batch rule and eligibility are unchanged.
  - With `CONF=0.0` (the default) it never probes, so in plain mode a request stays plain.
- **Calibration, both modes.** Each speculative result is counted in the bucket (tenths) of the confidence EMA it was issued with, together with the tokens it committed.

### New knobs (all read once at startup)

| knob | default | meaning |
|---|---|---|
| `KT_DSPARK_PLAIN_GATE_SIGNAL` | `accept` | probe rule in plain mode: `accept` (r2 timer) or `confidence` |
| `KT_DSPARK_PLAIN_GATE_CONF` | `0.0` | confidence EMA that triggers a probe; 0.0 = never. A placeholder, to be calibrated from the log lines below |

### Calibration log

Every `KT_DSPARK_PLAIN_GATE_LOG` decode steps, after r2's counts line, which now also prints each request's confidence EMA:

```
DSpark plain gate calibration (confidence EMA at issue: speculative steps x tokens a step): none 12x2.10, 0.3-0.4 41x1.62, 0.7-0.8 88x3.41, 0.9-1.0 130x3.85; signal 17400 pushed, 0 dropped, 0 skipped.
```

- Counts are cumulative, so the last line of a run holds everything.
- `none` means the request had no confidence sample yet (its first steps, given the lag).
- The calibrated `CONF` is the lowest bucket whose mean reaches the 2.2 threshold.

### Overlay (`apply_dspark_plain_gate.py`)

- The same five r2 hunks, plus the verify hunk (seven inserted blocks in all; the test checks every inserted statement is a selector constant, the `None` default, or behind `self._plain_gate is not None`).
- Three more pins, because the signal depends on their internals:
  - `srt/layers/sampler.py` `b9b0c965...`
  - `srt/layers/logits_processor.py` `51ce9204...`
  - `srt/sampling/sampling_batch_info.py` `5ddd691c...`

  Each matches the r3 trial tree on fea01 [measured: sha256].
- The module pin is `f3b160eb...`.
- It applies to the same pinned tree as r2 (worker `8eef0b1a...`), so a stage swaps r2 for r3 by changing the pins it checks.

### Tests [measured]

58 tests in all, 21 of them new:
- config (signal, conf, invalid values);
- helpers (`conf_bucket`, `anchor_stride`);
- the ring against a fake backend (lag, out-of-order completion, full ring drops, oversized steps);
- accept mode against the r2 module loaded from the parent folder (single request on 5 acceptance patterns × 2 eligibility patterns, with and without confidence samples; a 3-request batch sequence);
- confidence mode (no timer probes, one probe on the rise, a second probe after a dip, a good probe keeps speculating with the EMA reset to 4.0, `CONF=0` never probes, code keeps speculating);
- calibration (bucket attribution, plain results not counted, empty line);
- the reads-only AST checks;
- one tree test on the sampler, tp worker, logits output, sampling info and the hook anchor.

| run | result |
|---|---|
| `python -m unittest test_dspark_plain_gate` | 58 run, 50 pass, 8 skipped (tree tests) |
| `PLAIN_GATE_SGLANG=<scratchpad>/plain-gate/tree/sglang python -m unittest test_dspark_plain_gate` | 58 run, 58 pass |

No torch on this PC: `top1_probability` and the torch ring backend are covered by AST tests and code review, not executed.

### Card checks still to run [inferred]

1. **Identity** (FORCE pattern, as adadpi). With the signal hunk the verify and plain arithmetic must not change. Add a no-logprob token check, as adadpb does.
2. **Timing in `accept` mode with `KT_DSPARK_PLAIN_GATE_LOG` set.**
   - Per-step overhead of the signal against r2 (adadpt), expected under about 0.1 ms [estimate].
   - The calibration lines give `CONF`.
3. **Timing in `confidence` mode** with that `CONF`.

## Part 2: slot give-back (design note)

### What holds the 60 slots today

The lend gives hot slots everything the driver has free:
- C2: "17.10 GiB the driver had free, 1009 slots";
- static DSpark: 16.19 GiB, 957;
- gate: 16.05 GiB, 949 [measured in launcher logs, 18.8 MB a slot].

DSpark's share is about 1.05 GiB:

| item | size | can it leave in plain mode? |
|---|---|---|
| draft weights (`DeepseekV4ForCausalLMDSpark`, FP8) | 0.76 GB | about 0.67 GB yes; about 0.08-0.09 GB no (see below) |
| target verify graph pool | 0.17 GB | no: CUDA graph private pool, fixed |
| plain DECODE graph pool | 0.02 GB | no |
| draft KV pool and buffers, SWA match | about 0.1 GB | no: plain steps keep writing draft KV |

### Which draft weights plain mode still needs [code]

Plain steps and the verify graph write the draft's context KV through `TargetHiddenKvInjector` → `draft_model.write_target_hidden_kv` (`deepseek_v4_dspark.py` `f33532ed...`, lines 856-887). That path reads:
- `project_target_hidden`: `stages[0].main_proj`, 3 × 5120 → 5120, and `main_norm`;
- `project_commit_kv`: `stages[i].self_attn.wkv` for the three stages;
- the store's norm and rope.

When the KV pool has `set_swa_key_buffer_radix_fused_norm_rope`, the verify epilogue folds this commit into the verify graph (`folds_commit`, `dspark_verify.py`), which bakes those weights' addresses. They must stay resident and in place: about 79 MB for `main_proj` in FP8 plus a few MB of `wkv` [estimate from the shapes].

Everything else of the draft is read only by the draft's own graphs, which replay only in speculative steps: the stages' q/o projections, shared experts, gates and norms. Its routed experts are already on the CPU.

### The address constraint and the primitive that meets it [code]

The draft graphs bake the virtual addresses of every weight they read, so weights may not move. `kt_kv_hot_lend.py` already solves the same problem for the slots:
- `split()` puts each slot tensor on a `kt_kv_vmm.shadow_tensor`: a fixed virtual range that reads zeros until physical memory is mapped under it.
- The lend maps physical memory under the slots.
- The release maps the zero chunk back.

Its docstring: "k lives on the device and the slots never move, so the captured decode graphs stay valid". `kt_kv_prefill_lend.py` does the same for KV pages through `kt_kv_vmm.release` / `restore` (2 MiB chunks).

### Minimal build

1. **Re-home the draft weights.** After the draft loads its weights and before its graphs are captured (`DSparkWorkerV2.init_cuda_graphs`), move every draft parameter outside the injector set onto `kt_kv_vmm.shadow_tensor` ranges (map, copy, `param.data = shadow`, as `split()` does). Keep a pinned host copy (0.67 GB host RAM). The graphs then bake those virtual addresses.
2. **`draft_evict()`, entering plain mode.** After the last speculative step's event has completed (a host poll, not a sync), map the zero chunk under the draft ranges. Give their physical bytes to the hot lend as extra slots, filled from kt-kernel's arenas by the dynamic swaps as today.
3. **`draft_load()`, before a probe or a speculative run.**
   - Release about 36 of the coldest slots: drop them from the dynamic device table, so their routes go to the CPU at once.
   - Wait for an event, not a sync, before unmapping their pages, because `cuMemUnmap` is not stream-ordered.
   - Map the pages under the draft ranges.
   - Copy 0.67 GB from pinned host on a side stream: about 27-35 ms at 20-25 GB/s [estimate].
   - Record an event.
4. **Gate input: draft resident.** A request may vote SPEC only once the load event has completed (non-blocking query), so a probe waits 1-2 steps for the upload. A speculative step that raced the upload would only draft from zeros. Verify still commits the target's tokens, so the race costs acceptance, not correctness [inferred from DSpark's verify contract].
5. **Hysteresis.** Evict only after the batch has been plain for some M steps. Each eviction or reload costs the copy plus mapping time (the initial lend mapped 17.8 GB and copied the experts in 2.6 s [measured: log], so mapping 0.67 GB is likely tens of ms [estimate]), plus refilling 36 slots by swaps afterwards (4 swaps a layer every 4 forwards).

**What is missing today [code]:** the hot lend releases all-or-nothing. Its release "sets every k to 0, syncs, maps the zero chunk back under the slots". A partial, event-ordered release of n slots, and a lend that maps a given set of pages, are the new pieces. Step 1 needs a hook in the draft model load, a new overlay on `dspark_worker_v2.py` and `kt_kv_hot_lend.py`.

**Expected return [estimate]:**
- About 0.67 GB, so about 35-36 slots: 949 → about 984 in plain mode, against C2's 1,009. The verify graph pool, the injector weights and the draft KV remain (about 25 slots).
- What that buys: the gate run's route share was 61.6% against C2's 64.8%, but part of that gap is the speculative steps' extra experts, not the slots. The offline slot replay tools can price 949 against 984 slots on recorded routes before any card run.

### Alternatives considered

- **Draft dense weights on the CPU.** Frees all 0.76 GB always, but puts about 0.76 GB of CPU reads on the critical path of every speculative step, which is CPU-bound already. Worse for code [estimate].
- **Release and re-capture graphs per switch.** Capture takes about 15 s [measured: log]. No.

## Files

| file | sha256 |
|---|---|
| `dspark_plain_gate.py` | `f3b160eb90f595ddb05e5953fd282c88269c83690e4f0a13b01f8e454c16a805` |
| `apply_dspark_plain_gate.py` | `fa9393cc3b42a2c410bcb0a77385627d427f8b5ea3a018e5ad146c95a381dbb1` |
| `test_dspark_plain_gate.py` | `e0cb744f8399c59154febb758bc15e10338800863468924505ab077003616bb1` |
