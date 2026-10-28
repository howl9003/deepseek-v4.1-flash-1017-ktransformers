# Route 3 counter capture review

## Counter capture completed — 28 October 2026, 14:28 Singapore

The requested K=1280, N=32768, M=1 original BPC8/S3 capture completed. Exactly one native kernel was profiled in one hardware replay pass. The workload's warm and profiled raw partials were byte-identical and the scale flag was zero.

| Measured counter | Value |
|---|---:|
| DRAM throughput relative to sustained peak | 88.39% |
| DRAM byte rate | 635,480,945,489.63 bytes/s (635.48 GB/s) |
| Executed warp instructions | 13,492,224 |
| Tensor-pipe instructions | 163,840 |
| Total warp instructions / tensor-pipe instruction | 82.35 |
| Profiled native kernel duration | 66.336 microseconds |

The tensor-pipe count matches the expected source QMMA warp count of 163,840. Launch metadata confirms 4,096 CTAs, 64 threads/CTA, 72 registers/thread, 38,496 bytes dynamic shared memory, and a two-CTA/SM shared-memory limit on 80 SMs. These occupancy limits do not prove an occupancy bottleneck. The high measured DRAM rate applies to this sampled native kernel, not every shape or the full quantization/partial/reduce call.

The 66.336 us profiler duration must not be subtracted from the separate 95.232 us cold-graph full-call median as a measured overhead breakdown: their instrumentation, launch conditions and clocks were not paired. The existing uninstrumented 208-call price remains **12.790 ms**, above 12 ms. No new full-model speed claim follows from this capture.

**Recorded harness failure:** Nsight 2026.3 exported `profile.ncu-repz`; the original service checked only `profile.ncu-rep` and exited 1 after the successful capture. The original RESULT, failed unit state, manifest and all workload files are retained. No rerun was needed. An offline import recovered the counters and `review_route3_phase2_profile_r2.py` validated the actual wide CSV. The old long-format fixtures and six new rejection fixtures passed. No prepared capture file was changed.

Strict review: `ROUTE3_PHASE2_COUNTER_REVIEW_r1.json`. Binary report SHA256 `5a8985cfd9e90550573251a2b71648d68ef7283bb802b36de648208ad16dadcb`; offline CSV SHA256 `91321dbbbbadf006cb43de148f54c865297ebb7beb8c299c8769b48d233789a0`. Remote capture folder: `/scratch/aster/bench/six-node-performance-r1/route3-phase2-ncu-capture-r1/`. Local copy: `route3-phase2-work-r1/route3-phase2-ncu-capture-r1/`.

At the 14:39 Singapore release check, MainPID was zero, no GPU compute applications remained, the benchmark lock was free, and about 479.5 GiB was available. Peak capture memory was 1.107 GiB, swap zero and all memory events zero. Power stayed 320 W. Host-reload hash was independently rechecked unchanged.

**Next review request for Claude:** use these counters to prioritize the next arithmetic-neutral change. My proposed next measurement is a matched breakdown of quantization/layout, native partials, unchanged reduction and full public call across the six shapes, with identical inputs and cold-weight treatment and no correctness instrumentation inside timing. This would distinguish wrapper/reduction cost from kernel cost before committing to a persistent-grid implementation. The single high-bandwidth K=1280 profile does not establish the bottleneck of the K=5120 shapes. Any implementation must preserve the pinned MMA/epilogue/four-way split/reduce sequence and pass the complete component matrix before pricing. No full-model run while the price gate still fails.

