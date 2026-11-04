# Upstream picks r2: the SWA lend on the serving profile, 4 Nov 2026

This follows `PORTS_r1.md`, in the same folder and style. It covers two things:

- (a) what `KT_PREFILL_SWA_LEND=1` would need to run on the HiCache serving profile;
- (b) the lend parts of 05yuki's tier 1 patch 21 (`cef35da`) that r1 left out.

Everything was read from fea01 or tested locally. Nothing was launched.

| what | script | switch | serving arm value |
|---|---|---|---|
| SWA lend itself | none, served code | `KT_PREFILL_SWA_LEND` | `1` (serving runs `0` today) |
| (b1) load-back maps its SWA pages before the copy | `apply_swa_hicache_lend.py <sglang>` | `KT_PREFILL_SWA_LOAD_MAP` | `1`, required whenever the SWA lend runs with HiCache |
| (b2) with HiCache, the SWA trim keeps 4,096 tokens | same script | `KT_PREFILL_SWA_HOST_TRIM` | `0` first, `1` as a second arm |
| (b3) first-launch calibration without the cache | not ported | none | does not apply to our launch |

## (a) What the SWA lend does, and whether serving can use it as is

### What it does

`kt_kv_prefill_lend.py` handles both lends. The KV cache configurator calls `swa_sizing`, `swa_size` and `swa_init`;
the scheduler reaches it through `kt_lend.before_forward` (`swa_flush`, `swa_tick`) and through `kt_prefill_plan`
(`bind_tree`, `swa_terms`).

- **Sizing.** The pools are budgeted at the 2,048 sizing chunk, so the SWA pool is 7,424 tokens. It is then widened
  to hold the plan's 36,864 cap twice: 77,056 tokens, 301 pages of 256.
- **Lending.** Outside prefill, every free page beyond `KT_PREFILL_SWA_LEND_KEEP` (4,096 tokens, 17 pages) is
  unmapped through kv_vmm.
- **Allocating.** An allocation that needs lent pages gets them in the free list at once. In decode or in a window
  (`_defer_ok`), their chunks are mapped at the next forward's start (`swa_flush`), zeroed.
- **Trimming.** `swa_trim` asks the radix tree to give back SWA beyond the budget: 11,520 tokens, which is the
  7,424-token pool plus `KT_PREFILL_PLAN_SAFE` 6,144 minus 2,048.
- **Planning.** The plan counts what a chunk would map (`swa_terms`).
- **Pool clear.** `KT_PREFILL_SWA_CLEAR_RELEASE_HOT=1` makes the clear give the hot slots back before the SWA pages
  map again. It is our switch for his `dc21bdb` and is already set on serving.

### Our layouts

The V41 window KV and FP4 SM89 KV layouts already work with it. The C2 performance and identity profiles run
`KT_PREFILL_SWA_LEND=1` (`ada-perf-eight-c2-*`, `ada-identity-c2-*`, `ada-perf-v41kv-*`), with this same file
(`281ce314...`). The performance log shows:

- `SWA lend: 40 K/V tensors of DeepSeekV4SingleKVPool, 301 pages of 256 tokens, 20.6 KiB a token`.
- The SWA storage is paged, not the request ring, so the allocator wrapper sees every SWA allocation.
- No c4 compress-state buffer lends with the pages. Those buffers stay mapped, which costs memory only.

Those profiles have no HiCache. Serving does, with ratio 4, write-through and the file tier, and runs `0`.

### What it is worth on serving

- Today serving keeps its whole SWA pool mapped: 79,360 tokens x 21,120 bytes = 1.56 GiB.
- With the lend, 17 pages (92 MB) stay mapped outside prefill. That frees about 1.48 GiB, roughly 80 hot slots at
  18.8 MB an expert.
- The C2 performance profile's first lend gets 1,009 slots; serving gets 903. The SWA pool (about 80) and the
  HiCache staging buffer (about 16, r1 change 3) account for most of that gap.

### Not usable as is on serving

1. **A HiCache load-back can fault.**
   - `init_load_back` runs in `get_new_batch_prefill`. It takes the SWA tail's device rows through
     `swa_attn_allocator.alloc` (`kv/storage.py _band`), which is the lend's wrapper, so lent pages come back
     deferred.
   - Then `scheduler.py:4551` calls `ready_to_load_host_cache` -> `cache_controller.start_loading`. That starts the
     host-to-device copy before `prepare_for_extend` and the forward, so before `before_forward` maps the pages.
   - This is his "third load-back faulted both ranks". It is fixed by (b1).
   - The CPU test reproduces the order on the served code: the copy hits the two unmapped pages.
2. **The frozen KV gate refuses the profile.**
   - `client-tools/kv_checks.py` (`eb31f84a`) requires `KT_PREFILL_SWA_LEND` to be `"0"`, and so does its disk copy
     `serve-disk-r1/disk-client-r1/kv_checks_disk.py` (lines 45-47).
   - A serving arm with the lend needs a separate client copy that accepts `1`, as was done for the disk tier, with
     the frozen file left pinned. That copy is the owner's call; not made here.
3. **Upstream since our base.** I went through every 05yuki commit since 27 Oct.
   - The only commits that change SWA-lend behaviour are `cef35da` (this port) and `dc21bdb` (pool clear). We have
     `dc21bdb` as `KT_PREFILL_SWA_CLEAR_RELEASE_HOT`.
   - `be6d808` only drops an unused local in `swa_restore_all` and moves hunk offsets.
   - `7fef854` and `23dd4ea` (KV lend to every tensor, hot rows home with their scales) are in our tree.
   - `3e1fea9` (swa-checkpoint-tail) is the separate `swa-tail-r1` overlay; see its interaction with (b2) below.
   - Nothing else touches the SWA lend.
4. **Plan margin.** `KT_PREFILL_PLAN_MARGIN_MB=4096` on serving was raised because the kept-mapped SWA took window
   room the plan did not see. With the lend, the plan sees the SWA through `swa_terms`. Keep 4,096 for the first arm
   and lower it toward the performance profile's 128 only after a 41K and a 131K prompt pass.

Nothing else is needed. No change to the pools, HiCache's host pools or the write-back:

- HiCache's device pointer tables are taken from the buffers' virtual addresses, which kv_vmm never moves.
- Write-backs read only pages in use, which are always mapped.
- The lazy buffer claim ends at `swa_init`, before the host pools are made.

## (b) The port: `apply_swa_hicache_lend.py`

### File

| file (under `python/sglang/`) | sha256 before | sha256 after |
|---|---|---|
| `srt/layers/moe/kt_kv_prefill_lend.py` | `281ce3141d6075aab77a2ff6d2a98520902418f32ae1bf3aacd9e634c4f33d64` | `a86aa9e5f9155a36050dd9546b7118092b6f5299aaf12e82ef94702bd07503d1` |

- Only this file changes. Patch 21's `memory_pool_host.py` part is r1's `apply_hicache_staging_cap.py`, a different
  file, so these scripts do not stack and apply in any order.
- The anchors match his patch context line for line.
- As with r1, re-render `ORIGINS_CONFIG.json` with the new hash before a card run.

### (b1) `KT_PREFILL_SWA_LOAD_MAP=1`

- `bind_tree`, which `kt_prefill_plan.chunk_for` calls at the start of every `get_new_batch_prefill`, before any
  load-back, now wraps the tree's `cache_controller.start_loading` once. The wrapper flushes the pending SWA maps
  first, then runs the copy.
- It acts only with the SWA lend on and a cache controller present. Unset, nothing is wrapped.
- This is his code; only the switch is ours, because his has no switch. On serving with the SWA lend it is not
  optional.
- When pages are pending, the flush adds one device sync at that load-back, the same sync `before_forward` would
  have done one step later.
- It covers the cache controller path we serve. The external linker and buffer-pipeline paths are off on our profile.

### (b2) `KT_PREFILL_SWA_HOST_TRIM=1`

- With a cache controller attached, `swa_trim` works to `KT_PREFILL_SWA_LEND_KEEP` (4,096 tokens) instead of the
  11,520-token budget. His default is on; ours is off.
- **Expected gain on our tree is small.** His 92-slot loss came from a cached 35K prompt keeping its whole SWA
  (0.72 GiB). Our radix tree frees out-of-window SWA when it checkpoints a request. On the C2 performance run, after
  a 35K prompt, the window-end lend left 18 of 301 pages mapped: 17 kept free plus about 1 in use. So the most the
  trim can free here is the gap between budget and keep: 28 pages, 151 MB, about 8 slots. That gap only fills with
  several sessions' prefix tails (`--swa-prefix-tails 6`, four sessions).
- **Cost.** SWA the trim evicts that is not on host is recomputed, not loaded, on a later match.
  `KT_HICACHE_SWA_TAIL_ONLY=1` stores only a chain's last window. That includes the extra page the swa-checkpoint-tail
  overlay keeps for a repeated prompt. The result is still exact, but time to first token can rise.

### (b3) Calibration without the cache: does not apply

- His change is in `serve.sh`: `KT_PREFILL_LEND_CALIBRATING=1` launches with `--disable-radix-cache` for
  `lib/kt-lend-auto.sh`'s one-prompt calibration.
- Our stage never runs a calibration launch. The plan's constants are fixed in the environment
  (`KT_PREFILL_PLAN_COST`, `_COST_PRIOR`, `_FREE_MB`, `_ROW_KB`, `_SAFE`, `_MARGIN_MB`).
- `KT_PREFILL_LEND_PROBE` and `KT_PREFILL_LEND_TABLE` are unset. The `hints/chunk-rank0.json` files are written by
  windows as records; nothing reads them to size a chunk (`kt_lend.table_chunk` reads only `KT_PREFILL_LEND_TABLE`).
  The served trial's `hints/` is empty.
- So there is nothing to port. It would matter only if a stage started calibrating with the cache on.

### Launch additions for the serving arms

Environment only, on the serving profile:

- **S1:** `KT_PREFILL_SWA_LEND=1`, `KT_PREFILL_SWA_LOAD_MAP=1`. Keep `KT_PREFILL_SWA_CLEAR_RELEASE_HOT=1` and
  `KT_PREFILL_PLAN_MARGIN_MB=4096`. `KT_KV_VMM=1` is already set. Leave `KT_PREFILL_SWA_LEND_KEEP` at its default.
  Add `SGLANG_HICACHE_STAGING_MB=32` (r1) if the arm should carry both VRAM gains.
- **S2:** S1 plus `KT_PREFILL_SWA_HOST_TRIM=1`.
- **Gate client:** a copy of the KV gate client that accepts `KT_PREFILL_SWA_LEND=1`.

### Bit-identity: yes, memory placement only

- The SWA lend changes which physical memory backs free pages. A page is never unmapped while in use, and a remapped
  page is zeroed and written before it is read. The C2 identity profile already runs it.
- (b1) maps the same pages one step earlier.
- (b2) moves cached SWA from the GPU to host or recompute. Load-backs are exact (host-reload gate, every tier
  comparison 0.0), and so is recompute.
- The extra hot slots change expert placement only, which is exact by the contract.

### Which arms need the serving profile

All of them:

- (b1) and (b2) only act with HiCache attached. On the performance and identity profiles (no HiCache) they are inert,
  and the SWA lend there is already the reference.
- The 11,496-score panel on the identity profile is unaffected and need not be rerun for this port.

### Identity run (serving profile, S1 then S2, against C2 serving)

- The host-reload gate and the disk-aware KV check, through the client copy: every tier comparison and replay differs
  by 0.0.
- **Exercise the race on purpose.** Four sessions in decode so the 17 kept pages run short, then at least three
  load-backs, from host and from disk, the third included.
- **Check the log shows the race happened.** For each load-back: `SWA pages back for alloc: ... mapped at the next
  forward`, then `SWA map at HiCache load`. Without those lines the run did not reach the race.
- No `illegal memory access`, no `cannot be mapped back`.
- `/flush_cache` while hot slots are held: no `SWA pages for clear cannot be mapped back`.
- Fixtures and greedy prompts: ids, logprobs and top 8 EXACT against C2.

### Timing run (serving profile, S0 = today, S1, S2, two launches each)

- **Hot slots.** The first `lend 1:` line: expect about 80 more slots than S0's 903, about 96 more with the staging
  cap.
- **Hot slots after a cached 35K prompt.** Read the window-end `SWA lend (end of window): ... lent` line (mapped =
  301 - lent) and the next decode's `lend` line. Expect slots within a few of the pre-prompt count in S1, and within
  one or two in S2. Repeat with four sessions holding prefix tails, where S2 should differ.
- Decode breadth with four sessions.
- 35K, 41K and 131K prefill time. The `SWA map at window start` cost was 26 ms on the performance run.
- Follow-up-turn time to first token from host and from disk. (b1) adds a sync only when maps are pending.
- **S2 only:** hit lengths of repeated prompts, where trimmed tails show up as shorter matches.
- Node memory guards unchanged.

### Tests

`python tests/test_swa_hicache_lend.py <sglang>`: 21 checks pass.

- It runs the real SWA-lend functions of both files on stand-ins: the allocator wrapper, `swa_map`, `swa_flush`,
  `bind_tree`, `swa_trim`.
- The served code, and the patched code with the switch unset, copy into the two unmapped lent pages.
  `KT_PREFILL_SWA_LOAD_MAP=1` maps them first; it wraps once across repeated binds and never without the lend or a
  controller.
- The trim matches the served code with its switch unset at every fill level. With the switch and a controller it
  works to 4,096 tokens; without a controller it keeps the budget.
- Four mutations (copy before map, trim default on, wrap without its switch, trim without the controller check) are
  each caught.
- r1's three suites still pass (262, 25, 50) after a two-line addition to `tests/faketorch.py`.
