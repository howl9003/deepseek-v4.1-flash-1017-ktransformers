# Route 3 phase 2 results for Claude — revision 3

2026-10-28T08:55:23.566839+00:00. Package r2 from authenticated Zulip message 40 is tested. **The policy variant is bit-identical on the required matrix but fails the unchanged 12.0 ms gate.** In the matched session, default costs 12.764928 ms and policy costs 12.732160 ms per 208 calls. The separate read-sweep diagnostic costs 10.317824 ms. This is the “anything else: report and stop” branch of the requested decision rule. No model, production dispatch change, new ledger acceptance, push or publication followed.

## Row 0: compiler and binary evidence

The r2 default build (build 11) produces SASS byte-identical to original build 3, SHA256 `c336cdda06154131fc2159f8259525331179ba5b48c36b0b6420dcb21bb41901`. Its extension SHA is `51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649`.

The valid policy binary is **build 13**, extension `route3_phase2_l2p1_r2`, SHA `88ec38312bbd2317f01ae31d6e7535c19c9ea163d0356e410a755d00eeecc3be`. Its SASS SHA is `61eb5d782b20c6d3fa6e2cd50f8e537144420d5f27f81cc6fccce597128f2129`. It is SASS-identical to the initial build 12, which was later overwritten by the invalid attempt described below. The supplied hinted `cp.async` syntax compiled directly; no fallback was needed.

Policy SASS materializes `UR16=0`, `UR17=0x12f00000`. Of 66 static LDGSTS sites, 24 carry `desc[UR16]`: 21 BYPASS.128 weight-copy sites and 3 weight-scale sites. The 42 other sites have no descriptor. Both builds retain 72 registers/thread, 38,496 dynamic shared bytes/CTA, 16 zero-C QMMA sites, 64 separate FMUL/accumulating-FFMA pairs, no FTZ, no local spills and no stack. Resource footprints are unchanged; this was not a new occupancy measurement. Both have 2,784 static instructions. Register allocation and scheduling produce 1,946 position-wise instruction differences, so the SASS diff is not merely 24 changed lines. Natural K32 ordering, four fixed partitions and the reduction remain unchanged.

## Row 1: numerical and flag checks

Corrected validation 10 passes **597/597 identity checks and 361/361 flag checks** before pricing. Coverage includes raw partials before reduction, reduced outputs, row/batch invariance, changed graph inputs, permutations/partitions, default-off and enabled dispatch, prefill boundaries, invalid scales and changed scale graphs. Its original-method M1 price is 12.746496 ms; it is an independent run, not the matched comparison below.

The matched session separately passes **240/240 harness identity checks and 96/96 zero-flag checks** before timing: six shapes, M=1/4/8/16, two binaries, raw/eager/graph/changed/restored outputs. These are harness checks, not 240 independent model scenarios. Both runs import pinned binaries directly, disable compiler loading, and verify the actual loaded binary hashes before and after validation. The hashes match their build receipts.

## Row 2: unchanged original-method price

Milliseconds per 208 calls, calculated as the sum of each shape's median times its original call count. Six checkpoint weight shapes, 15 samples/cell, randomized condition order within each shape/batch round. Quantization, dispatch, native partials and unchanged reduction are included. Correctness checks are outside timing. The gate remains M1 <= 12.0 ms, with the original displacement and outer CUDA event boundaries.

| Binary | M1 | M4 | M8 | M16 |
|---|---:|---:|---:|---:|
| Default, original method | 12.764928 | 12.861696 | 13.106688 | 13.464064 |
| L2 policy, original method | 12.732160 | 12.815872 | 12.936448 | 13.233152 |

Policy M1 is only 0.257% lower in this session; no robust speedup is established from that small difference. Its gate deficit is 0.732160 ms. The valid comparison contains 48 gate cells and 6 separate diagnostic cells.

## Row 3: default-only read-sweep diagnostic

After the original 256 MiB displacement, an untimed `.sum()` reads a separate 128 MiB float32 buffer. Its result is synchronized before the start event. The input version is unchanged and the sum is checked; the reduction result/workspace may be written. No physically clean-L2 state is asserted.

The default drops from 12.764928 to 10.317824 ms, a 2.447104 ms (19.17%) reduction. **This is diagnostic only and cannot pass or replace the gate.** The read sweep's own time is excluded, so this is not an end-to-end optimization result.

M1 per-call medians, microseconds:

| Shape | Default/original | Policy/original | Default/read sweep |
|---|---:|---:|---:|
| wqkv | 30.400 | 30.528 | 27.488 |
| wqb | 95.008 | 95.104 | 74.752 |
| wob | 94.048 | 93.952 | 73.664 |
| shared_w13 | 58.176 | 57.344 | 47.104 |
| shared_w2 | 36.864 | 36.704 | 30.720 |
| indexer_wqb | 23.136 | 23.360 | 21.088 |

## Row 4: displacement and interpretation

The actual displacement **reads and writes**: `eviction.add_(1)` over 67,108,864 float32 elements, a 256 MiB read-modify-write, followed by CUDA synchronization outside the timing events. It is not `cache.zero_()`.

The matched read-sweep result establishes sensitivity to the pre-state left by this method. It is consistent with dirty-line eviction/writeback contributing to the cost, but does not uniquely identify that mechanism: the extra read kernel also changes other cache and GPU execution state. The policy hint did not recover the roughly 2.45 ms diagnostic difference. SASS confirms the hint was emitted; it does not establish whether/how hardware honored it. NVIDIA defines this operand as a performance hint that may not be respected ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)). Production KV/activation retention has not been measured. An exact causal decomposition cannot be inferred by subtracting the earlier unpaired ncu/trace/chain receipts.

## Invalid attempt and correction

Validation 9 is **invalid as policy evidence**. Its harness checked build 12's hash, then called the compiling loader without setting `ROUTE3_L2_POLICY`; that rebuilt the same extension as macro 0. The first matched attempt stopped before any cases or prices because its strict binary check detected the mismatch. Both attempts and the explicit invalidation audit are preserved. Builds 3–11 were audited and still matched their recorded hashes.

Fresh build 13 plus validation 10 and matched prices r2 correct this harness error. Direct pinned imports and before/after loaded-file hash checks prevent compiler substitution in both valid runs. The invalid validation 9 price must not be quoted as policy performance.

## State and handoff

This prescribed component test is complete; the performance objective remains incomplete. No blind variant rerun or model is queued. The next decision needs a targeted mechanism/implementation that can pass the existing method, or explicit human direction about the performance acceptance method; no gate has been changed here. The existing accepted full-model breadth medians remain 16.0 tok/s for stripped BF16 and 16.55 tok/s for contract, with 30+ still unproven. Revision 2 retains prior variant, counter, chain and graph-boundary evidence.

Both production selectors remain default-off; only the isolated validation selected the policy binary. PR6 stays held; nothing pushed/published. Host-reload SHA remains `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`. Six model CPU nodes/42 workers, 64 GiB FEA reserve, 416 GiB model limit/no swap and 320 W are unchanged. Component runs used CPU55, 8 GiB/no swap, the shared lock and GPU/resource guards. The matched run peaked at 671.8 MiB host cgroup memory and 523.1 MiB allocated CUDA memory, with no cgroup memory events. At 2026-10-28T08:52:08.097146+00:00, jobs had exited successfully, GPU was free and the shared lock was free. Qwen remains unloaded. No periodic automation was created.

## Receipt hashes

Paths below are relative to `six-node-performance-r1`; remote raw receipts also reside in the corresponding run directories without the local `route3-phase2-work-r1/` prefix.

- `ROUTE3_L2_DEFAULT_REVIEW_r1.json` — `9e2b4c7bc65808c464e45b31cd69412a42d62378c882e76a46e03fa81a075234`
- `ROUTE3_L2_SASS_DIFF_r1.json` — `a31831ca1877b69ac79e3d73f15c006a24219606788106950eeb19232404aa75`
- `ROUTE3_PHASE2_SASS_REVIEW_r10.json` — `ccec4208aadd54125f9ac4b437dee2636d874e0f1425a96ecbcfb23ecf66ff51`
- `ROUTE3_PHASE2_STRICT_REVIEW_r10.json` — `2426ce82d171b2a20b161b852c3a06cc9600aeaa6becbc2531026058eef0fe5c`
- `ROUTE3_L2_PRICES_REVIEW_r2.json` — `280ffec7250f611e0f7e020a2bfd3514503cef97fd250d8614e3a5f99c751276`
- `ROUTE3_L2_BINARY_INVALIDATION_r1.json` — `bf837d044669225a02ce80e0de32aff0afa0ed115eab058e01478b7d57d56da8`
- `ROUTE3_L2_RELEASE_r2.json` — `b3bd2ed58c22e9a2407954ba0e9ffbf3fcb55b5ad4a0860b166f98e820c46b29`
- `ROUTE3_L2_PRICES_PREPARATION_r2.json` — `9987eb57a7015e691f20793ea45604e7e78f0c53a8a519c1a8b9ca3cdb953207`
- `route3-phase2-work-r1/route3-phase2-validation-r10/RESULT.json` — `851866b6e6fa3c80193d1a4f4aa86c3ad0d665f35cf0d7a219b05b83c4ab4bd2`
- `route3-phase2-work-r1/route3-l2-prices-r2/RESULT.json` — `e79f43ebd65ed2e253cf2ab3af97ecb747073e585c7a704ac0dfd055c231200a`
