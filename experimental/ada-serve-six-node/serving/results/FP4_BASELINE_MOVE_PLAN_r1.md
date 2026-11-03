# Moving the served build to the C2 baseline: plan, 2-3 Nov 2026

**Status 3 Nov:** steps 0 and 0b are done. The rows are the model's own (`reference-panel-r1/BLANK_ROWS_RESULTS_r1.md`). The combined build, V41 window KV + FP4 compressed KV + FP32 head, ran panel, timing and gate: KL 0.01019 against floor 0.00996, breadth 33.3, prefill +0.6%, gate and disk pass (`../v41-window-r1/V41_WINDOW_RESULTS_r1.md`). C2 is that build. Steps 1-5 remain, each unload with the owner's OK.

**Done 3 Nov 15:11 SGT: C2 is served** (owner: "Yes, switch"). Tag adasc3, trial `ada-serve-c2-20261103-r3`, about 25 minutes down. The first reload attempt failed in a second: `serve_ada_r1.py` still named the C1 prepared file in upper case, which my search had missed; fixed. Rollback: the C1 values are in the comments beside `SERVED_TAG, SERVED_TRIAL` and `RUNS`.

**Status 3 Nov afternoon:** steps 1-3 done. C2 captured and repeated bit for bit. Paired NLL against C1 is +0.0025 (SE 0.0036), and the quality screen passes. Serving profile breadth 33.65, sessions as before. The second cold 35K prompt is +22% on the serving profile, cause unmeasured. Step 4 needs the owner's OK. The serve trial must be staged with the model down, since staging takes the bench lock. `adasc2` is spent on a refused stage, so the switch uses a new tag. `serve_ada_r1.py` now names its trial in one line, `SERVED_TAG, SERVED_TRIAL`, and `swap_entry_ada_r1.py` in `RUNS`.

Owner, 2 Nov: "Yes, plan it". This replaces the "stays default off" call in `FP4_KV_RESULTS_r1.md` once it runs. Nothing below has started. **The new baseline is called C2.** C1 stays archived and is never overwritten.

**Why:** it holds the reference's FP4 values on these layers, and the panel means lean the same way (NLL gap to the reference +0.00073 to +0.00039, inside noise). It removes a rounding we added ourselves, and SM120 already uses this layout.

**What it costs:** prefill +0.8% overall, and +1.2% and +1.0% on the two steady 35K prompts, against a +1% rule. Moving means accepting that. Decode is flat, and there is no expert slot gain.

## Steps

**Changed 2 Nov: the owner chose to fix rather than measure first.** The move now covers three things together, so the baseline only moves once: FP4 compressed KV, the V41 window KV (`../v41-window-r1/`) and FP32 logits (`--enable-fp32-lm-head`, from the audit). Step 0b below is dropped: the model run on the combined build replaces it. If that run disappoints, the reference arms in 0b are how to tell which piece did it. The index score sum stays as it is: rounding only, and the audit found nothing wrong in it.

| Step | What | Needs | Time on the GPU |
|---|---|---|---:|
| 0 | Read the context-free row report | Done 3 Nov | None |
| 0b | Fix the window KV and head, model run | Done 3 Nov: V41 + FP4 + FP32 head | None left |
| 1 | Capture C2 | Model unloaded | About 1 h, one load |
| 2 | Repeat C2 on a second load, must be bit-identical | Model unloaded | About 1 h, one load |
| 3 | Serving profile on C2: timing, four sessions, dsh, restart from disk | Model unloaded | About 1 h, one load |
| 4 | Switch the selector to the new trial | Owner's OK, about 14 min down | One load |
| 5 | Records, fork, PR session | Nothing | None |

Steps 1 to 3 need about three hours with the model down. They can run overnight or split across days. The times are estimates.

### 0. Go or stop on the rows

- **Go** if the rows are a shared property: FP4 is in the same class as the reference and the served build.
- **Stop and fix first** if an FP4 fault makes them. Row 136,64 is the one to watch: it goes bad on FP4 and in no other arm.
- If 136,64 stays unexplained but the class is shared, go, and record it as open in the ledger.

### 0b. Other numeric differences

Two arms on the reference forward, same MoE batch as `ref-panel-r2`, scored like `ref-floor-r1`. Each changes one step to what the served build does, so no kernels are needed to learn the size:

- **Window KV:** first 448 values with a scale per 64, RoPE values left as BF16.
- **Index scores:** sum across heads in FP32.

**How to read them:** any small change flips near-ties, so an arm that does nothing still reads about the noise floor, KL 0.00996. Only an arm clearly above that shows a real effect. The served build's excess over the floor is 0.0011.

- Above the floor by a good share of 0.0011: fix it in the same move. For window KV that means a V41 window reader on the served attention route, tested like the FP4 overlay. `SGLANG_SINGLE_ROUTE=1` already reads V41 on SM89, so it is a starting point.
- At the floor: leave it, and record the size in the ledger.

### 1 and 2. C2

Tags `adac2a` and `adac2b`, both free. Derive from the client that produced C1, the `B1b-logical-cold-identity-r2` chain with `client_r19_logical_identity_r1.py`, so C2 passes the same checks:

- The 11,496 scored positions, both replays, 4K and two 35K contexts.
- The quality corpus, 12,276 positions, NLL limit 0.02. This is the degradation rule C1 passed, and the only part of it FP4 has not run. The panel is the stronger test, but the rule stays unchanged.
- Rotation check from `client_ada_identity_r2.py`.

`adac2a` writes the C2 archive and differs from C1 by design, so that difference is recorded and not treated as a failure. `adac2b` must equal `adac2a` bit for bit. Placement independence is not re-tested, since this change does not touch the expert path. Frozen `kv_checks.py` stays unchanged.

### 3. Serving profile

Tag `adasfp`, trial `ada-serve-c2-20261103-r1`: `adasrr` plus the FP4 and V41 overlays after the staged gather anchor, `SGLANG_SM89_V41_KV=1` and `SGLANG_SM89_FP4_COMPRESSED_KV=1`, `--enable-fp32-lm-head` classified in the profile, admission 448. The derive in `run_ada_v41c.py`'s chain is the pattern.

**New disk store:** `/scratch/aster/bench/ada-kv-disk-fp4-20261103-r1`. V4 pages can't be read by this build, so it starts empty and the first long turns are cold.

Must pass:

- Breadth within noise of 33.15. Sessions 1, 2 and 4.
- Four 16K follow-up turns, and four dsh sessions.
- Restart from disk at 8K and 41K, bit-exact against cold.
- Lowest node above the 4 GiB guard, same memory checks.
- A request sent during settling waits and completes, using `hold_requests` in `serve_ada_r1.py`.

### 4. Cutover

Change `RUNS` in `swap_entry_ada_r1.py` and the trial, launch payload and supplemental names in `serve_ada_r1.py`. No root step, since `llama-swap.json` calls the same entry. Confirm with the owner right before the unload.

**Rollback:** revert those constants and reload. Trial `ada-serve-20261102-r4` and `ada-kv-disk-20261102-r1` stay in place, so going back costs one load and nothing else.

### 5. Records

- `PRECISION_LEDGER.md`: FP4 row moves from tested to served, with C2's evidence and the prefill cost.
- Wording changes to "exact vs C2" for anything after the move. Results before it stay "exact vs C1".
- `FP4_KV_RESULTS_r1.md`: replace "Why it stays off". Update `SERVED_PANEL_RESULTS_r1.md`.
- Push to the fork branch only. Tell the PR session, because the issue draft describes FP4 KV as in testing and treats C1 as the baseline. Nothing goes upstream without the owner.

## Stop rules

Any of these stops the move and leaves serving on r4:

- `adac2b` differs from `adac2a` anywhere.
- Quality corpus over the limit.
- Serving breadth more than noise below 33.15.
- A memory guard trips, or the gate or host reload fails.

No guard, admission line or frozen check changes for this. Each stage gets one load and no retries without reading why it failed.

## Order against the rest of the list

Optimisation runs after the move are measured against C2. DSpark fitting and the lossy evaluation should use C2 as their starting point, so this goes before them. The eight-node run does not depend on it.
