# Route 3 phase 2 results for Claude — 28 October 2026

**Identity passes; the price gate does not.** The unchanged phase-2 kernel is the fastest tested configuration at **12.790 ms per token for the 208 dense calls (M=1)**, 6.58% above the 12 ms gate. Both requested tunables are slower. All three variants passed the full 597-case matrix before any prices were collected. The prescribed one-shape hardware-counter capture is prepared but awaits interactive authentication. **No full-model run, model promotion, push or publication occurred.**

## Build and SASS

The received handover and both manifests were authenticated. The complete phase2-cuda folder was copied under `/scratch/aster/bench/six-node-performance-r1/kernel-steer-r1/route3-mma-r1/`, preserving `../route3_sequence_reference.py`. The five original CPU layout checks passed.

- Build r1 stopped before GPU execution: `cusparse.h` was absent from the compiler search path.
- Build r2 also stopped before GPU execution: adding the bundled headers with CPATH put CUDA 13.0 runtime headers ahead of the installed CUDA 13.3 toolkit headers.
- Build r3 succeeded using **CPLUS_INCLUDE_PATH** for the bundled library headers, after the explicit toolkit include paths. Claude's kernel, header and wrapper were unchanged. Both failed logs are preserved. CUDA 13.3.73, sm_89, `-fmad=false`, no fast-math; no driver or toolkit installation was changed.

Original build r3 SASS: **16 static QMMA.16832.F32.E4M3.E4M3 sites, all with RZ accumulators; 64 FMUL followed by 64 accumulating FFMA sites; no FTZ, no LDL/STL, zero local memory and zero stack.** It uses 72 registers/thread and **38,496 bytes dynamic shared memory/CTA**. Static SASS has 16 LDSM, one BAR.SYNC and three LDGDEPBAR sites; these are static counts, not measured dynamic rates. All scale multiplies remain separate before the accumulation FMA. The SASS review traces each FMUL product into an accumulating FFMA.

The four-stage build has 72 registers and 51,328 bytes/CTA. The 16-block build has 64 registers and 76,992 bytes/CTA, with 32 zero-C QMMA, 128 FMUL and 128 FFMA sites. Neither spills. Both tunables require an explicit host-side shared-memory opt-in above 48 KiB; this was added on initial module load, before graph capture. Their entire CUDA device namespace is byte-identical to the original source; only header geometry constants and that host setup differ. No arithmetic was changed.

The BPC16 trial uses the baseline three stages. BPC16 plus four stages would require 102,656 bytes, exceeding the 101,376-byte opt-in ceiling. The local layout checks were rerun with each variant's actual constants and the appropriate opt-in memory budget; the original two-CTA budget assertion is not claimed for these variants.

## Identity and flags

Each variant passed **597 cases**, with all raw-partial comparisons completed before reduced/public-output testing and all correctness completed before timing:

| Coverage per variant | Count |
|---|---:|
| Native raw `[4,M,N]` vs accepted Triton, six shapes x M={1,2,4,8,13,16} | 36 |
| Native 16-row partials vs Triton batches {26,32,128,538,768} | 30 |
| Reference raw batch invariance | 66 |
| Native permutation / partition | 6 + 6 |
| Changed-input raw graph replay, M={1,2,4}, two inputs per shape | 36 |
| Reduced native outputs | 36 |
| Reference reduced invariance / fixed public / default-off public | 66 each |
| Enabled native public / unchanged prefill public | 36 + 30 |
| Changed reduced / changed public graph replay | 36 each |
| E8M0 conversion and original strided scale-domain checks | 10 |
| Native normal/subnormal powers, invalid scales, changed-scale graph replay | 35 |

There were **361 flag observations per variant**. All matched expectations: invalid As=1, invalid Bs=2, both=3; all valid-domain cases and every timed cell ended clear. Invalid scales still produced output. Flags were reset between deliberate invalid tests, and the final protocol flag was **0**. Scale validation ran on changed-input graph replay; there is no pointer-only cache.

The candidate dispatcher is private and **SGLANG_FP8_ROUTE3_CUDA defaults to 0**. Removing its guarded addition leaves the phase-1 dispatcher AST unchanged. Accepted model/runtime source files were authenticated before each run and were not modified. Correctness establishes agreement with the accepted fixed-split Triton arithmetic for these inputs, not full-model acceptance relative to K3's different dense arithmetic.

## Matched prices

Cold-weight CUDA-graph medians, 15 samples/cell, 256 MiB displacement outside events. Full-call prices include production activation quantization, native scale validation, contiguous conversions where needed, partial output allocation in the captured call, and the unchanged split reduction. Synchronous flag reads occur outside timed events. The comparisons in each table use identical inputs. There are 78 price cells per run. No diagnostic comparisons run inside timing.

Totals below are ms per **208 batched calls**, not divided by M; at M=1 this is the requested per-token dense cost. The two Engram calls are excluded consistently with phase 1.

| Variant | M=1 | M=4 | M=8 | M=16 | Registers/thread | Shared bytes/CTA |
|---|---:|---:|---:|---:|---:|---:|
| Original: BPC8 / stages3 | 12.790 | 12.881 | 13.220 | 13.621 | 72 | 38,496 |
| Tunable 1: BPC8 / stages4 | 14.817 | 14.948 | 15.063 | 15.712 | 72 | 51,328 |
| Tunable 2: BPC16 / stages3 | 14.626 | 14.649 | 15.047 | 15.745 | 64 | 76,992 |

Original phase-2 run's matched comparison:

| Path | M=1 | M=4 | M=8 | M=16 |
|---|---:|---:|---:|---:|
| Phase 2 original + unchanged reduce | 12.790 | 12.881 | 13.220 | 13.621 |
| Phase 1 best Triton | 27.117 | 28.771 | 29.090 | not requested |
| Fixed path with matched transpose | 23.109 | 30.244 | 30.819 | not requested |
| K3 CUDA-core path, different arithmetic | 12.382 | 13.459 | 20.806 | not requested |

The original phase-2 path's **406,583 bytes/us (406.58 GB/s)** is 55.24% of the 736 GB/s comparison ceiling. Payload is 5,200,199,680 bytes; the payload-only floor is **7.0655 ms**. These are useful-payload rates, not hardware DRAM measurements. The path is about 3.3% slower than the matched K3 CUDA-core dense path while reproducing the fixed tensor-core partials. This is not a full-model tok/s estimate.

Per-call medians from the original matched run, microseconds:

| Shape | M | Phase 2 | Phase 1 best | Matched transpose | K3 CUDA core |
|---|---:|---:|---:|---:|---:|
| wqkv | 1 | 30.464 | 52.064 | 42.592 | 31.520 |
| wqkv | 4 | 30.464 | 52.224 | 53.248 | 32.768 |
| wqkv | 8 | 30.720 | 53.248 | 55.104 | 87.040 |
| wqb | 1 | 95.232 | 212.992 | 179.200 | 93.184 |
| wqb | 4 | 97.280 | 243.520 | 226.080 | 96.256 |
| wqb | 8 | 101.376 | 247.808 | 231.424 | 131.840 |
| wob | 1 | 94.112 | 213.792 | 185.120 | 92.160 |
| wob | 4 | 94.208 | 222.208 | 256.000 | 104.448 |
| wob | 8 | 95.232 | 222.880 | 261.120 | 147.456 |
| shared_w13 | 1 | 58.368 | 132.960 | 115.712 | 56.320 |
| shared_w13 | 4 | 58.368 | 134.144 | 154.624 | 64.512 |
| shared_w13 | 8 | 59.392 | 136.192 | 156.672 | 97.280 |
| shared_w2 | 1 | 36.864 | 60.416 | 50.176 | 32.480 |
| shared_w2 | 4 | 36.864 | 61.440 | 61.440 | 33.792 |
| shared_w2 | 8 | 38.752 | 61.440 | 61.440 | 50.176 |
| indexer_wqb | 1 | 23.552 | 28.448 | 24.576 | 19.456 |
| indexer_wqb | 4 | 24.160 | 28.672 | 23.552 | 23.552 |
| indexer_wqb | 8 | 25.184 | 28.448 | 23.552 | 31.744 |

The requested 16-row phase-2 cells and every sample for all variants are in the JSON receipts. No additional prefill price or full-model throughput was measured in this step.

## Counter capture: prepared, not executed

All three requested geometries miss 12 ms, so the next action is the prescribed **single K=1280 wq_b capture**, using the fastest original BPC8/stages3 variant. No further kernel modification was attempted after the tunables.

Nsight Compute 2026.3.1 was downloaded from the configured NVIDIA package repository, its package SHA256 verified, and extracted only inside `route3-phase2-ncu-tools-r1/`; it was not system-installed. AD103 support and the exact metric names were queried without running a kernel. The driver reports `RmProfilingAdminOnly: 1`, and `sudo -n true` reports that interactive authentication is required. [NVIDIA documents that this counter policy requires administrative privileges or CAP_SYS_ADMIN](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute).

Prepared command (requires the user's terminal authentication):

```sh
ssh -t fea01 'sudo python3 -B /scratch/aster/bench/six-node-performance-r1/tools/launch_route3_phase2_profile_r1.py'
```

The launcher starts a bounded service as `aster` with temporary CAP_SYS_ADMIN, 8 GiB/no-swap, CPU55, 96 GiB admission/64 GiB reserve, an idle-GPU check and the shared lock. It profiles one native wq_b kernel, with one warm repeat used for byte validation, and allows Nsight's hardware replay as necessary. It uses `--clock-control none`, changes no persistent capability, driver policy, boot or power setting, and launches no model. Metrics are DRAM throughput, DRAM byte rate, total executed warp instructions and tensor-pipe instructions. The latter is reported directly; expected source QMMA warp count is 163,840, not assumed to equal an unexamined hardware counter.

Until that capture is read, pipeline, occupancy and prologue explanations remain hypotheses. The 13-fractional-bit ledger claim is not newly measured by these tests; SASS receipts establish instruction sequence only.

## Evidence and standing state

Remote root: `/scratch/aster/bench/six-node-performance-r1`. Local root: `C:/Users/howl/Downloads/Claude/strata-deepseek1017/six-node-performance-r1`.

- `route3-phase2-build-r1/` and `r2/`: preserved environment failures and build logs.
- `route3-phase2-build-r3/`, `r4/`, `r5/`: original, four-stage and 16-block binaries, SASS, resources and build logs. Local copies are under `route3-phase2-work-r1/` (the original includes the binary; all remote binaries remain intact).
- `route3-phase2-validation-r1/`, `r2/`, `r3/`: raw/public/flag case lists, all samples and selected totals. Local RESULT.json copies are under `route3-phase2-work-r1/`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r1.json` through `r3.json`: independently verified exact coverage, flag expectations, 78 timing cells, medians and 208-call sums.
- `ROUTE3_PHASE2_SASS_REVIEW_r1.json` through `r3.json`: pinned binaries and SASS facts.
- `ROUTE3_PHASE2_PROFILE_PREPARATION_r1.json`: exact pending capture command, file hashes and resource/permission requirements.

Each validation peaked near 0.78 GiB host memory and 1.31 GiB allocated GPU memory, with no swap or changed memory events. Six-node/42-worker model topology, 64 GiB FEA reserve, 416 GiB/no-swap model limit, 320 W and boot remain unchanged. Frozen host-reload SHA: `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`. PR6 remains held. No full model is queued.
