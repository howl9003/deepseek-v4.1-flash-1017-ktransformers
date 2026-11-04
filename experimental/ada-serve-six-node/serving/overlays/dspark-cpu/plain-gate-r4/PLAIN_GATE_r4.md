# DSpark plain gate r4: the draft-only probe (CPU-tested, no card)

Claude, 4 Nov 2026. Step 2a of `shadow-lane-r1/SHADOW_LANE_DESIGN_r1.md`. Selector `KT_DSPARK_PLAIN_GATE=1` with
`KT_DSPARK_PLAIN_GATE_SIGNAL=draft`; the default signal stays `accept` (r2's timer probe) and keeps its exact step
sequence. Work done on this PC only. On fea01: `sha256sum` and `scp` of two pinned source files, read-only, from the
adadpt trial's recipe tree. Nothing copied to fea01, no card, no queue, no runner.

Tags: **[measured]** run here; **[code]** read in the pinned sources; **[estimate]** arithmetic; **[inferred]** reasoning
that nothing has run.

## Files

| file | sha256 |
|---|---|
| `dspark_plain_gate.py` | `df7d0ee67d3c64013de04930ef9ad212ce9e3957b0e6b031158817e5013028ce` |
| `apply_dspark_plain_gate.py` | `4cc6bf3b6c6d4ecacf7150edb824fe394e6b017734bf281a37be52d3095b5af0` |
| `test_dspark_plain_gate.py` | `3aa87bbfbaa9deb56fb030dcfd51945afc5ee69b6a8a54146802153d939ab5f6` |

r2 (`../dspark_plain_gate.py`, 6d0636b3) and r3 (`../plain-gate-r3/`, f3b160eb) are unchanged.

## What changed against r3

- **Config:**
  - `SIGNAL` takes `draft`.
  - `KT_DSPARK_PLAIN_GATE_DRAFT_EVERY` (8): a probe before every 8th plain step.
  - `KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA` (0.5): EMA weight of a probe sample.
  - `KT_DSPARK_PLAIN_GATE_DRAFT_MIN` (2): samples needed before a switch.
- **Gate:**
  - **Probe EMA:** `note_probe` folds settled samples into a per-request EMA. Samples for unknown or speculating requests are ignored.
  - **Switch:** a plain request votes to speculate once that EMA reaches the threshold (2.2) after `DRAFT_MIN` samples. It keeps voting so (`awaiting`) until the switch's first speculative result is absorbed; from then on the acceptance EMA decides, as in r2.
  - **Reset:** the probe EMA is consumed by the switch, and cleared again on the drop back to plain.
  - **No timer probe** in this mode.
  - **Calibration:** `probe_calibration()` buckets each switch's speculative results by the probe EMA at the switch, in 0.5-token buckets. It is logged with the step counts.
- **`probe_sample(origin, output, p, drafts)`:** returns `1 + m`, where m is the number of leading draft tokens equal to the committed tokens at positions p+1 onward. The bonus sits at p, and draft token k guesses p+k. It is decided at the first mismatch among the committed positions, or once all gamma are committed; otherwise it returns None (undecided).
- **`ProbeRing`:** pinned int64 rows. Each row holds the request's committed length on the device, then its draft tokens. Rows are pushed with a non-blocking copy and an event, and drained only when the event has completed, oldest first; nothing waits. A full ring drops the push and counts it.
- **Controller:**
  - **Before every vote:** `_collect_probes` drains the ring and settles landed probes against `Req.origin_input_ids + output_ids`. Undecided ones wait (at most 16 per request); probes of requests the gate dropped are discarded.
  - **When the batch runs plain and any request wants a probe:** `_draft_probe(batch)` runs before `_run_plain`.
  - **Worker hunks:** none new. The apply script keeps r3's seven hunks; only the module pin changes, plus two new read-only pins (below).

## Where the probe hooks in [code]

`PlainStepController.maybe_plain_step`, inside r3's gate hunk in `DSparkWorkerV2._forward_decode`, sits after the idle return and before the draft. `_draft_probe` repeats the draft half of `_forward_decode`:
1. `anchor_ids = proposer.draft_block_ids_view(...) if worker._prepare_inputs_once else None`.
2. `alloc_verify_window(...)` with the full verify width, `bonus` and `anchor_ids`, and `seq_lens_cpu` from the batch or else `nxt_kv_lens_cpu`.
3. `with worker._draft_context(): proposer.propose(...)` with the same arguments.
4. It takes `proposal.draft_block.draft_tokens`.

It skips the observers' draft segment, `_verify_inputs_ready`, the planner and the verify. A test checks each of these call strings against the pinned `_forward_decode`, so drift fails the suite. The probe is skipped, and counted, only where `DraftBlockProposer._run_forward` would raise: no host seq lens and a draft backend that needs them.

## Findings

- **The window is a pure plan [code].**
  - `RunnerIds.plan_uniform_extend` (`runner_ids.py`, 4b3c95ed) calls `assign_extend_cache_locs_uniform_func` with `req_to_token=self.req_to_token`. It writes only local dicts, then returns `self.plan(...)`, which just builds an `IterationIds`.
  - The Triton kernel `assign_extend_cache_locs_uniform` (`cache_locs.py`, 05a0bf25) stores through exactly `out_cache_ptr`, `positions_out`, `anchor_out`, `full_out` and `swa_out`. It only reads `req_to_token`.
  - `IterationIds.bind` sets five fields on the forward batch and nothing else.
  - Both files are now pinned by the overlay, and tested by AST. Their hashes are the same in the `ada-identity-dsp-20261104-r3`, `ada-perf-dsp-pg-20261105-r1` and `ada-batch-dsp-pg-20261105-r1` trees [measured].
  - The draft writes its block's KV into the reserved slots for positions p.. through that window. Each plain step's injection overwrites its own position before any later draft forward reads it, as after a rejected speculative step [code + inferred].
- **The deferred commit needs no extra flush [code].**
  - A static verify stages the draft context-KV commit (`staged_bs`).
  - `DraftBlockProposer._run_forward` calls `commit.before_draft_forward(graph_bs)`, which flushes eagerly whatever the graph would miss. It then calls `commit.after_draft_forward(graph_ran)`, which counts a pending staging as written by the graph.
  - Every write zeroes `commit_lens` (disarms), and `_run_plain` flushes before each plain step.
  - So a probe with `DRAFT_EVERY` ≥ 2 follows a plain step and finds nothing staged; its graph's head write is the disarmed no-op a warm-up does. At `DRAFT_EVERY` 1 a probe can follow a verify, and the proposer's own hooks write the staging exactly as a speculative step's draft does.
- **The draft's noise [code]:** a Philox key plus a GPU offset that each draft block advances (`DraftNoise`), not the global generator. A probe advances it, so at T=1 later drafts sample differently from a run without probes. Greedy decoding is unaffected, and committed tokens never depend on the probe.

## Tests [measured]

78 tests. Without `PLAIN_GATE_SGLANG`: 67 pass, 11 skipped. With it, on the scratchpad pinned tree (plus the two fetched files): 78 pass.

The 20 new tests cover:
- the config;
- `probe_sample` (full match, mismatch, partial availability, prompt positions) and `probe_bucket`;
- the probe ring (lagged order, drops);
- **draft mode:**
  - prose never takes a timer probe;
  - probes run on every 8th plain step;
  - the switch fires when the draft starts to pay;
  - `awaiting`, then the drop back to plain, then the 16-plain-step refill when the draft was wrong;
  - samples during speculation are ignored;
  - `DRAFT_MIN` is respected;
  - no probes in the other modes;
  - calibration;
  - a two-request batch speculates only when both want to;
- `accept` and `confidence` modes equal to r3 step for step, single and batch, with r3's calibration unchanged;
- `_collect_probes` settling, waiting and dropping;
- the call sites, the window's purity and the deferred-commit contract against the pinned sources.

## Risks and what a card run needs

- **Not run on a card [inferred]:**
  - the probe's draft forward from the plain path;
  - its graph at each batch size (bs 3 has no draft graph, so it runs eager with an eager flush);
  - the probe's real cost, about 3.3 ms each, about 0.4 ms a plain step at every 8th step [estimate].
- **r3's confidence signal still runs in draft mode**, feeding only its calibration lines, at an estimated 6-8 small launches a step.
- **At T=1, m is a proxy:** it compares sampled drafts with sampled target tokens, so expect it below a verify's acceptance. The probe calibration line is there to set `DRAFT_ALPHA` and `DRAFT_MIN`.
- **A card run needs** a chain like adadpt with this folder's overlay pins (apply 4cc6bf3b, module df7d0ee6) and `KT_DSPARK_PLAIN_GATE_SIGNAL=draft`, plus the no-logprob token check (adadpb client) for identity. None was made here.
