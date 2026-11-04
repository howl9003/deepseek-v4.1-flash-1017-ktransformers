# Upstream picks r1: three of 05yuki's changes as overlays, 3 Nov 2026

Three apply scripts for the served C2 tree (`/scratch/aster/bench/ada-serve-c2-20261103-r3`). Each is off by default,
pins the served file hashes, and leaves behaviour exactly as today with its switch unset. All three are tested on
local copies with CPU-only tests. Nothing ran on fea01 beyond reads; no card runs yet.

| change | upstream | script | switch | value for our profile |
|---|---|---|---|---|
| 1. short prompts leave part of the stream to the CPU | `46da340`, patch 14 | `apply_stream_cpu_short.py <sglang>` | `KT_STREAM_CPU_SHORT` | `1024:48`, then sweep k |
| 2. prefill warm-up before ready, Engram staging retired behind an event | `77021a7`, patch 22 + adapter | `apply_prefill_warmup.py <sglang> <engram-adapter>` | `--warmups prefill_lengths` + `SGLANG_WARMUP_PREFILL_LENGTHS`; `DSV41_ENGRAM_RETIRE_EVENT` | `1000,8200`; event unset |
| 3. HiCache write-back staging cap | `cef35da`, patch 21, cap only | `apply_hicache_staging_cap.py <sglang>` | `SGLANG_HICACHE_STAGING_MB` | `32` |

The scripts touch disjoint files, so they apply in any order and alongside `swa-tail-r1`. Each one asserts every pin,
patches in memory, compiles, and only then writes. It then prints the new hashes. A second run refuses, because the pin
no longer matches.

**Staging the trial.** The served launch runs the module-origins observer, so a patched file whose hash is not in
`ORIGINS_CONFIG.json` (`source_sha256`) exits the server with code 86. Re-render `ORIGINS_CONFIG.json` with the hashes
below and update `FEATURE_ORIGINS_CONFIG_SHA256`. `engram_backend.py` is not checked by the observer, but
`PREPARED.json` and `PHASE_G_G1V_ADASC3_PREPARED_r1.json` pin it under `adapter_after`, so update those too.

**Tests.** Run from this folder with the served files: `python tests/test_stream_cpu_short.py <sglang>`,
`python tests/test_prefill_warmup.py <sglang> <engram-adapter>`, `python tests/test_hicache_staging_cap.py <sglang>`.
Each copies the served files, applies its script and tests the copy. They need numpy only: `tests/faketorch.py`
stands in for the parts of torch they use. Results: 262, 25 and 50 checks pass. Every test was also run against
deliberately broken scripts (5, 4 and 3 mutations), and each break was caught.

---

## 1. `KT_STREAM_CPU_SHORT`

### Files

| file (under `python/sglang/`) | sha256 before | sha256 after |
|---|---|---|
| `srt/layers/moe/kt_stream_prefill.py` | `7827b4961219076e8cce930e65ec673f7a215517065369f490626ab76f179f95` | `a8d65c6845a04490da67cb1726c5d51fb9d9e0427cb349f2f6b6cf06f47c7a63` |
| `srt/layers/moe/contract_v2/adapter.py` | `01beed62fdf94920144c42b939546b4b4612c6c44ba70ec1f13f9045e23687c8` | `aac793b2fa7d5c372dcba60dd31bcc03a9962848eb9a6b594a3a95639d3079a2` |
| `srt/layers/moe/contract_v2/stream_graph_candidate.py` | `6c757f1fea59dc1c1c241624fe33724f9dfc9c535b17a5e911f7f82bdd1d8627` | `87e0f7ff1da6e818e1b6fd069ba996dc6605497994df85518b36c8edc0cdf63f` |

### How our path splits a layer, and what the port changes

With `KT_CONTRACT_V2=1`, `kt_ep_wrapper.apply` hands every MoE call to `contract_v2/adapter.apply`. A forward of 512
rows or more (`KT_GPU_STREAM_PREFILL=512`) gets the streamer. The split is already there for the static
`KT_STREAM_CPU_GROUPS` share. `stream_cpu_cutoff` returns the first physical id the CPU takes:
`1 + (96 - k) * 4` for k groups of 4 over physical 1..383.

- Before the stream, the adapter sends routes with id >= cutoff to kt-kernel's per-route API (`submit_routes`).
- Groups `[1, cutoff)` go to the stream. That is `stream_graph_candidate.run` (path A, one graph per row count and
  group), or `adapter.stream_routes` when the graphs are off.
- After the stream, `sync_routes` waits for the CPU, and its rows replace the CPU-owned routes in the same
  `routes [T, 6, 5120]` buffer.
- An assert checks that every live route has exactly one owner. Then one FP32 combine runs in logical expert order.

So the CPU share already worked on our path, but only as a fixed count for every row count. His change makes the count
depend on the forward's rows. The port does this:

- `kt_stream_prefill.py`: parses `KT_STREAM_CPU_SHORT=<tokens>:<k>,...` (sorted, first entry that fits, malformed
  input refused at import). It adds `cpu_short_entry` and `cpu_groups_for`, and gives
  `stream_cpu_cutoff(method, streamer, tokens=None)` the row count. Unset, every function returns
  `KT_STREAM_CPU_GROUPS` behaviour exactly. It refuses the switch at import unless `KT_CONTRACT_V2=1`.
  `kt_ep_wrapper.py` is not touched: its own split serves only the non-contract path, which we do not run.
- `adapter.py` and `stream_graph_candidate.py`: all three cutoff calls now pass `x.shape[0]`, so the split, the
  stream's groups and the ownership check agree.
- Added by us: the CPU's rows are padded to a bucket (the next power of two, at most the entry's bound). The padding
  rows are copies of the last row, with id -1 and weight 0. kt-kernel's `KExpertsCPUBuffer` and the adapter's
  `_route_buffers` keep a buffer set for every row count, forever. Without the bucket, each new short prompt length
  would add about 147 MB of VRAM and about 170 MB of pinned memory at 1,024 rows. With it, there is one set per
  bucket: 126 MB of pinned per-route rows (two layer slots), plus kt-kernel's 1,024-row staging (21 MB of it on the
  GPU, made once). The device copy of the CPU's rows is a transient inside the prefill window.

### Value: `KT_STREAM_CPU_SHORT=1024:48` to start

- **Ours is a different machine from his.** He set `1024:64` (64 of 96 groups) on one 5070 Ti. Our stream moves
  17.75 MB an expert (the log's "2 packed DMA slots x 71 MB" for G=4). That is 272 GB over 40 layers, so a streamed
  forward has about 11 s of PCIe floor at about 25 GB/s. That floor is most of the 17.1 s cold time to first token
  measured on 2 Nov for a 1,024-token prompt.
- **What the CPU costs per group is not measured on this build.** At about 16 tokens an expert, our contract kernel
  (integer digit planes, AVX2, 42 workers on six nodes) may be bound by compute rather than memory bandwidth. Assume
  a CPU group costs between 0.5x and 1.5x what PCIe costs for it. Then k=48 gains in every case: about 5.5 s at 0.5x
  and about 2.7 s at 1.5x. His 64 gains nothing at 1.5x, and k=72 would lose 1.4 s.
- **Which experts the CPU gets.** It takes physical 193..383. In our placement, the first physical ids hold a chosen
  set of experts, and the rest follow in logical order. So the share falls outside that front set; it is not
  specifically the coldest experts.
- **Why the bound is 1,024.** Above it, the CPU's work per expert grows with rows, but PCIe costs the same per
  expert. He stopped at 1,024 because his NLL test could not resolve a difference above that. Our identity check is
  exact at any length, so the bound and a second entry (for example `2048:24`) can follow timing data.

### Launch additions

Environment only: `KT_STREAM_CPU_SHORT=1024:48`. No argv change. `KT_CONTRACT_V2=1` is already set.

### Bit-identity: expected, by construction, but not yet shown at prefill row counts

- Each live route is produced once, as a BF16 row with its routing weight applied, by its owner. There are no partial
  sums or group subtotals. The graph's `acc` is per route and copied by the owned-route select.
- The combine is the logical-order FP32 sum (`KT_CONTRACT_V2_COMBINE_ORDER=logical`), whoever owns each route. So the
  combine order and the buffers do not change with the share. The CPU tests confirm this: with one per-route function
  for every owner, every k and row count gives output bit-identical to no share.
- So the output stays bit-identical exactly when kt-kernel's contract v2 per-route row equals the GPU's for the same
  token and expert. The GPU side is path A for row blocks of 128 rows or more, and the static Triton route kernel for
  shorter tail blocks. Evidence for this:
  - Moving experts between hot slots and the CPU is exact against C1 and C2. That shows CPU = GPU static routes at
    decode batch sizes.
  - G1p showed path A prefill exact against the contract Triton stream (Phase G report 2).
- **Not yet shown:** CPU against path A on the same rows at prefill batch sizes of 512 to 1,024. The CPU also now
  sees padded batches, which is safe only if its per-row result does not depend on batch size
  (`KT_MXFP4_BATCH_INVARIANT=1` says it does not). The identity run must settle both.

### Risks

- **Digits.** As above. If the CPU and path A disagree at prefill batch sizes, the share changes outputs, but only for
  forwards of at most 1,024 rows.
- **Timing.** If the CPU is slower than its share of PCIe, the GPU waits at `sync_routes`, and short prompts get
  slower than today. Too large a k is the failure mode; hence the sweep.
- **Memory.** 126 MB of pinned memory per bucket, plus kt-kernel's bucket staging. Its 21 MB on the GPU is allocated
  in the first share window and kept, so the first relend after it may show about one hot slot fewer.
- **Scheduling.** The shared expert now forks after the CPU submit (`KT_SHARED_AFTER_SUBMIT`) in a share forward, as
  it does in decode. This changes scheduling only.
- **CPU capacity.** The kt-kernel row bound is 37,376, so any bucket up to the 36,864-row cap fits.

### Identity run (card)

- **Arms.** Share unset against `1024:48`, and a stress arm with `1024:95`, where the CPU computes all but one
  streamed group. That is the strongest check of CPU-against-path-A digits.
- **Prompts.** 512, 513, 640, 1000 and 1,024 tokens, where the share applies, and 1,025, 4,096 and 35K tokens, where
  it does not. Greedy decoding, 64 tokens.
- **Pass.** Output ids, selected logprobs and top 8 are EXACT against C2. The 11,496-score set is EXACT. The path A
  flag word is clear on 40 layers, and the ownership assert never fires.

### Timing run (card)

- A fresh 1,024-token prompt with `max_new_tokens=1`, cold and warm, for k = unset, 32, 48, 64, 72. Two launches per
  arm, alternated. Also 512 and 768 tokens.
- Confirm 1,025 tokens and 35K are unchanged, and decode breadth is unchanged.
- Watch the hot slot count in the `lend` lines.
- `KT_GPU_STREAM_TIMING=1` or a short Nsight capture shows whether the GPU waits at `sync_routes`.

---

## 2. Prefill warm-up, and the Engram retire event

### Files

| file | sha256 before | sha256 after |
|---|---|---|
| `python/sglang/srt/entrypoints/warmup.py` | `1c080ad3362b9131f344f31c03bed479a20e16507a9f53e31ec2bf80884e9610` | `be97f458fb6df8bfe7ded2dd4fd6826928d13787f99716fcb21dd943107b0334` |
| `engram-adapter/engram_backend.py` | `7e3eed1c65140b2cad9e716b5a86fc81ba7b66fec8542bdf3c1e7d427dbabb88` | `696d00da4bff9f21f5af852fee42a3186fab7e4fac1d52bb963b7d59ff17b04c` |

### What the port does

- **`warmup.py`.** His `prefill_lengths` warm-up is appended; the served text is otherwise unchanged. Before the
  server reports ready, it sends one random prompt per length, each producing one greedy token.
  - One change from his: the ids come from a per-launch seed. `SGLANG_WARMUP_PREFILL_SEED` fixes one.
  - Why: his fixed seed 0 sends the same prompt every launch, and our HiCache file tier keeps chains on `/scratch`
    across launches. A repeated warm-up prompt could then be matched from disk instead of computed. That is more
    likely with the swa-checkpoint-tail overlay. A matched prompt warms nothing.
- **`engram_backend.py`.** His change is made switchable. With `DSV41_ENGRAM_RETIRE_EVENT=1`, an eager staging set
  that must grow is held behind a CUDA event instead of a stream synchronize. Its Works stay with it, and both are
  dropped at the next eager lookup once the event has passed, never during a capture. Unset, it synchronizes, as
  today.

### Values and launch additions

- **Argv:** add `--warmups prefill_lengths`.
- **Env:** `SGLANG_WARMUP_PREFILL_LENGTHS=1000,8200`. Leave `SGLANG_WARMUP_PREFILL_SEED` unset.
- **Why these lengths.**
  - Both are above the 512-row stream threshold, so they go through the streamed path A prefill, and 1,000 rows hit
    the share's 1,024 bucket when change 1 is on.
  - Neither is divisible by 16, like most real prompt lengths. Triton specialises row-count arguments on that; his
    1,024 and 12,288 both are divisible.
  - 8,200 rows cover two full 4,096-row path A blocks plus a sub-128-row tail on the static kernel, and grow the
    Engram eager set to 2^18 rows.
- **Why not his chunk-size analogue, 36,864.** That would pre-size everything to the plan cap, but costs about 50 s
  of readiness every launch. On 1 Nov, in round 1, the first 35K prompt paid only about 1 s more than the second
  (51.86 against 50.81 s). Round 0's extra 14 s (69.10 against 55.24 s, the same in the gather trial) most likely
  came from the trial's fresh Triton and torch caches; that is inferred, not measured.
- **Expected cost:** about 40 s more to ready (the 2 Nov cold times to first token were 17.1 s at 1K and 23.3 s at
  8K). Allow that in the controller's readiness wait.
- **`DSV41_ENGRAM_RETIRE_EVENT`: leave unset.** On our build, path A's `check_flags` synchronizes the host with the
  stream at the end of every streamed layer. So the queue a growth synchronize waits for is at most one layer's tail,
  not his deep queue. Little to gain; test it in the timing run and keep it off unless it shows a gain.

### Bit-identity: yes

- Neither part changes arithmetic. The warm-up only fills caches: kernel modules, staging, the radix tree, the disk
  tier, Engram rows.
- A random prompt cannot match a real prompt's first 256-token page, so real requests are computed exactly as without
  it.
- The retire event changes only when memory is freed.

### Risks

- About 40 s more to ready.
- About 17 MB written to the disk tier per launch (1.9 KB a token). That is inside the 64 GiB cap with eviction.
- The warm-up chains sit in the tree until evicted. On this profile their KV is a few MB, and only the SWA tail of
  each is kept.
- Gate clients that count cache entries see two extra chains. None can match a fixture.

### Identity run

- With `--warmups prefill_lengths` on: the 11,496-score set and the fixtures are EXACT against C2.
- The host-reload and disk-aware KV gates pass.
- Repeat the check with `DSV41_ENGRAM_RETIRE_EVENT=1`.

### Timing run

- Time to ready, with and without the warm-up.
- First against second 1,024-token, 8K and 35K prompt of each launch, two launches each, warm Triton cache.
- One trial with a cold Triton cache, where his gain should show most.
- The hot slot count in the first `lend` line after the warm-up.
- Node 0 headroom during the 8,200-row warm-up.

---

## 3. HiCache write-back staging cap

### File

| file (under `python/sglang/`) | sha256 before | sha256 after |
|---|---|---|
| `srt/mem_cache/memory_pool_host.py` | `c428d0353505c9b1a7f27303fc3e0686461d0281f20b80092bed3dd43eaabd76` | `cb191f3f39dece533eb9ce53ca94cccb6b748798c8c795b3ce1af92e00434d6e` |

### What it does

- With `page_first`, ours and the default here, each V4 host pool keeps a device staging buffer of min(pages, 64)
  pages across all its layers.
- `SGLANG_HICACHE_STAGING_MB` also caps that buffer at the given number of MiB. Both pool classes use the same helper.
- Unset, the buffer is min(pages, 64) pages, as before.
- The served staged kernel `transfer_hicache_all_layer_mla_staged_lf_pf` (`kernels/ops/kvcache/hicache.py`, unchanged)
  already copies in chunks of the buffer's page capacity.

### Value: `SGLANG_HICACHE_STAGING_MB=32`, his value

- From the served `launcher.log`, the 'swa' pool is 40 layers x 135,168 bytes a page. Its 64-page buffer is 330 MiB;
  at 32 MiB it is 6 pages, 30.9 MiB.
- c1, c2 and their indexers are already under 8 MiB and keep 64 pages.
- That frees 299 MiB of VRAM, which the hot lend takes as about 16 more slots (903 now, at 18.8 MB an expert).
- Our SWA write-back moves only each request's tail window (`KT_HICACHE_SWA_TAIL_ONLY=1`), a few pages, so 6 pages a
  launch rarely splits one.

### Launch additions

Environment only: `SGLANG_HICACHE_STAGING_MB=32`.

### What else from patch 21 the cap needs: nothing

- The patch's other parts serve `KT_PREFILL_SWA_LEND=1`: mapping SWA pages before a load-back, and the host trim of
  the SWA budget. We serve `KT_PREFILL_SWA_LEND=0`.
- Its calibration change serves his first-launch chunk calibration.
- Nothing else reads `staging_buffer`; only the two write-back paths do.

### Bit-identity: yes

- The buffer carries bytes only. The test runs the served staged loop with a numpy stand-in for the kernel: 64-page
  and 6-page buffers write identical host pages, in ceil(n/6) launches instead of ceil(n/64).
- The extra hot slots move experts from the CPU to GPU slots. That is exact by the contract, as every placement change
  on C2 has been.

### Identity run

- The 11,496-score set is EXACT against C2.
- The host-reload gate and the disk-aware KV check pass, with logprob delta 0.0 on every tier comparison and replay.

### Timing and memory run

- The first `lend` line should show about 16 more slots.
- Decode breadth: his 36 slots on two cards were worth 2 to 4%; expect a fraction of that.
- A gate that pins the hot slot count or the VRAM after load will need its expected value updated.
- A 131K disk-tier round trip should take the same time as before.

---

## Found while porting, not changed: per-length buffers below 512 rows

- Prefills below 512 rows run the CPU path through `adapter.submit_routes`. Both `_route_buffers` and kt-kernel's
  `KExpertsCPUBuffer.get_buffer` keep one buffer set per row count, for the life of the server.
- That is about 143 KB of VRAM per row (`T x 6 x 5120 x 2 B` in two layer slots, plus kt-kernel's output), and a
  little more pinned memory. Each distinct short prefill length adds that once: about 43 MB of VRAM at 300 rows.
- The run behind this trial saw only 5 distinct sizes. Chat with varied turn lengths would see many, and the memory
  comes out of what the hot lend can use.
- Worth a separate, default-off fix with the same bucket idea as change 1.
