# Route 3 phase 1: exact, but too slow — 28 October 2026

The ordered brief's stop condition was reached. All **7,216 correctness checks passed**, but the best eligible geometry costs **27.446 ms per 208 dense calls** at M=1, above the **12 ms** stop threshold. No Route3 model or ledger probe was run. Please prepare the phase-2 CUDA kernel text specified in the design; the evidence below is ready for implementation review. Nothing was pushed or published.

## Measured results

Cold-weight CUDA graph medians, complete calls including activation quantization, scale validation for Route3, workspace and split reduction. Fifteen samples per cell; 256 MiB displacement occurs outside the measured interval. No diagnostic comparison runs inside timing. Totals use five shapes with40 calls each plus8 indexer calls; the2 Engram calls are excluded.

| Path | M=1 ms /208calls | M=4 | M=8 |
|---|---:|---:|---:|
| Requested starting geometry: BN16,1warp,6stages,no swap |44.958|31.702|31.743|
| Best eligible Route3: BN32,2warps,4stages,no swap |27.446|28.427|29.137|
| Existing fixed path, BN128 |30.536|30.626|31.357|
| Existing fixed path with matched transpose |22.998|30.242|30.897|
| Current K3 CUDA-core path, different arithmetic |12.322|13.488|20.683|
| Original dispatch, reference only/different arithmetic |12.418|12.829|31.436|

The selected geometry is fastest atM1 among configurations within10% of the best at each matched batch sizeM4/M8. It is also the fastest global configuration atM1. Even an unconstrained per-shape choice among all24 geometries totals **24.606 ms** atM1, so a dispatch table alone does not reach the target.

Payload is **5,200,199,680 bytes per208calls**. The selected path achieves **189,470 bytes/µs** (189.47 GB/s), or25.74% of the736 GB/s comparison ceiling. Its payload-only bandwidth floor is **7.0655 ms**. These are effective payload rates, not DRAM-counter measurements.

Per-call M=1 medians, µs:

| Shape | Starting geometry | Selected geometry | Existing transpose | Current CUDA core |
|---|---:|---:|---:|---:|
| wqkv1792x5120 |52.224|51.200|43.008|31.616|
| wqb32768x1280 |509.952|219.968|179.072|92.160|
| wob5120x8192 |280.384|213.728|183.296|92.160|
| shared_w13 4608x5120 |163.840|134.144|114.688|56.320|
| shared_w2 5120x2304 |106.496|61.440|50.176|31.744|
| indexer_wqb4096x1280 |55.296|28.352|23.552|20.288|

The assertion overhead is included, but not separately isolated; do not attribute the whole regression to it. The results do not establish a hardware bottleneck or validate the CTA-count explanation. The one-warp starting configuration particularly regresses wq_b. Inspect the saved PTX/TTIR and phase2 implementation before proposing another measurement.

## Exactness and arithmetic

All24 combinations of BN{16,32}, warps{1,2}, stages{4,6,8}, swap{false,true} passed on six real checkpoint shapes. Batches1,2,4,8,13,16,26,32,128,538,768; every raw `[4,M,N]` partial is compared before reduction. Public/default-off dispatch, per-row batch invariance, changed-input raw/public graphs at1/2/4, permutation and partition cases passed. Scale checks cover strided layouts, normal/subnormal powers of two and invalid values. All256 E8M0 code conversions are checked, including code0 andNaN255.

All150 saved specializations have a zero-C `tt.dot` in TTIR and the expected `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instruction. The accepted sequence remains `mul.f32(D,As)` then `fma.rn.f32(product,Bs,acc)` without `.ftz` in the epilogue. Ksplit4 and the original reduce kernel are unchanged. The matched-swap helper preserves As-before-Bs; the original swap branch does not preserve that ordering and was not substituted for it.

The corrected Claude package was authenticated against its02:49:40UTC manifest. One build-specific correction remains: this build's JIT quantizer floorsamax at **1e-10**, giving minimumAs2^-42, rather than the reference1e-4/2^-22 used in section1a. A payload scan of **all288 dense scale tensors** found E8M0 codes **114..121** (Bs2^-13..2^-6). With minimum nonzero E4M3 product2^-18, the lower bound is2^-73, **53 exponent bits above minimum FP32 normal2^-126**. Preserve the exact epilogue anyway; this is not a proof for arbitrary extreme inputs/overflow.

Prefill512/6144/36864 prices and bounded35K timing were deliberately not run after the price stop. Source and partial identity through768 passed; do not claim a new measured6.2s prefill result. The optional39-case AD103 ledger probe is prepared but unrun. The13-fractional-bit claim remains literature-derived here.

## Artifacts

Local root: `C:\Users\howl\Downloads\Claude\strata-deepseek1017\six-node-performance-r1`.
Remote rootP: `/scratch/aster/bench/six-node-performance-r1`.

- `ROUTE3_COMPONENT_PREPARATION_r1.json`: corrected package, source/runtime digests and guards. SHA2b51b204f92fdcfda7a99f5fb93f9dfa0d88442ea967237e935feb8d0c762d72.
- `ROUTE3_COMPONENT_REVIEW_r1.json`: complete evidence and rederived selection; statusSTOP_FOR_PHASE2_NO_MODEL.
- Local `evidence/ROUTE3_GEOMETRY_RESULT_r1.json`; remote `route3-geometry-r1/RESULT.json`:7216checks,504timingcells. SHA2f37f8b17b02bb95a5202fe46224a4d2a3d6259f06a3c97d0f3220e305e44604.
- Local `route3-work-r1/geometry-ptx-r1/`; remote `route3-geometry-r1/ptx/`:150PTX files and150TTIR files. Every local PTX digest authenticated.
- `route3-source-r2/fp8_kernel.py`, `fixed_fp8_decode.py`: private staged experimental source, **not installed**. The source's default experimental configuration remains the initial BN16/1warp/6stages setting; the sweep overrides its private namespace. No selected configuration is promoted after a failed price gate.
- `tools/test_route3_geometry_r1.py`, `tools/review_route3_geometry_r1.py`: executed harness and strict reviewer. `tools/probe_route3_ad103_r1.py`: optional probe, unrun.

Component owner62157bd14b784541a103a0ca0fb91ba5 completed successfully03:23:37UTC, with the requested price-stop result. Host-memory peak1.104GiB, GPU allocated peak1.635GiB, swap0, memory events unchanged. All model processes were gone before admission and the component owned `.bench.lock`. Power/boot unchanged.

## Phase2 boundary

Implement the explicitly specified CUDA decode kernel preserving zero-C k32 MMA, As-before-Bs exact epilogue, fixed four K partitions, and the unchanged reduce. Keep selectors default-off. Include scale validation costs; any removal or caching needs a sound immutable-weight and changed-activation contract. Do not use pointer-only caching that can bypass graph replay changes. Reuse the full raw-partial and public-dispatch matrix before any timing or model promotion.

The prospective Route3 full-model base remains acceptedK3, with only the dense route replaced. The separate owner/foldK4 run has now passed26 required checks plus7 added identity checks, including byte-identical11496-position NLL toK3; its historical performance is unmeasured and it must not silently enter this candidate.

PR6held; frozen host-reload SHAeb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4;416GiB/noSwap model;64GiB FEA; six compute NUMA nodes0,1,2,4,5,6 and42workers;320W; Qwenunloaded. No further GPU or model run is queued. The finite continuation has completed; the canceled recurring automation remains canceled.

## Static binary follow-up — no additional GPU execution

The saved cubins were inspected with Triton's bundled `cuobjdump`. Resource usage and disassembly are authenticated in `ROUTE3_COMPILER_RESOURCES_r2.json` and `route3-work-r1/binary-inspection-r1/`.

| wq_b specialization | Registers/thread | Dynamic shared bytes/CTA | Aggregate upper bound on resident warps |
|---|---:|---:|---:|
| Starting BN16, 1 warp, 6 stages |74|5120|16 /48|
| Selected BN32, 2 warps, 4 stages |80|4608|24 /48|
| wq_b's fastest BN32, 2 warps, 8 stages |80|10752|16 /48|
| Accepted prefill BM64/BN128, 4 warps, 3 stages |227|12800|8 /48|

These are capacity bounds, **not measured occupancy**. They assume the maximum 100 KiB shared-memory allocation, include CUDA's 1 KiB shared-memory reservation per CTA, and apply the published 24-block, 48-warp and 64K-register limits. Actual allocation granularity, shared-memory allocation and scheduler partitioning can lower them. The four-warps prefill entry is not a decode-timing comparison. Hardware limits: [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/).

All four binaries report **zero local memory and zero stack**, with no local loads/stores in SASS. All use native `QMMA.16832.F32.E4M3.E4M3`; there is no scalar-math fallback. The starting geometry already has 16-byte asynchronous copies, so a missing vectorized copy is not established. Increasing stage count increases shared storage; more CTAs do not imply more resident warps. The eight-stage wq_b result is faster despite its lower capacity bound, so occupancy alone is not a demonstrated cause either.

Phase2 should explicitly budget shared memory and registers, avoid unnecessary staging/epilogue work for padded rows, and amortize pipeline startup across useful work. Keep the exact arithmetic of each live row. These are implementation directions for testing, not measured speedups or permission to skip the raw-partial gate. No further timing, compilation or kernel execution was performed during this inspection.

The later r1d package adds a DSpark refit campaign. Its stated prerequisite is a final Route3 kernel and a passed Route3 full-model run; neither is established. It remains a downstream proposal and was not launched. The phase1 stop condition is unchanged.
