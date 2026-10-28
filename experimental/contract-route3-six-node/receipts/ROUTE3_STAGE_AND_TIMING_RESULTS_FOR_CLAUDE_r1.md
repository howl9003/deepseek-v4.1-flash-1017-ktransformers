# Route 3 stage and timing-boundary evidence — 28 October 2026

Two guarded component runs used the original BPC8/S3 binary unchanged. No kernel or serving-source change, full model or acceptance-gate substitution occurred. The first passed 126 output checks/42 flag observations and all108 timing cells. The boundary diagnostic passed156 output checks/66 flag observations and72 timing cells. Every cell has15 samples. Strict reviews are separate from the measurement scripts.

## What was found

- All q, As, B and Bs tensors were already contiguous for all six shapes at M=1/4/8. No conversion copy was needed. All six M=1 graph traces contain exactly three GPU kernels: quantization, native partials and unchanged reduction. No hidden copy kernel appeared.
- Weighted graph-internal gaps were **0.016176 ms per208 calls**. Quantization and reduce kernel durations in these instrumented traces summed to0.244992 and0.211160ms respectively; native kernels summed to10.975392ms. These single-replay diagnostics are not acceptance prices. They do not support a1ms inter-kernel-gap explanation.
- In uninstrumented matched prices, native-only cost12.227840ms, native+reduce12.449792ms and full public12.695808ms. Added reduce and quantization costs are approximately0.222 and0.246ms in this comparison. Standalone quantization/reduction graphs measured about2.5ms each because independent launch/event boundaries also enter those measurements; their medians must not be added as a decomposition.
- The original price method reproduced at12.695808ms and then12.659200ms, both above12ms. The original accepted-method receipt12.790016ms is retained.

## Per-shape M=1 prices and independent trace

Useful rates below include weight scales, as in the existing harness. They are payload rates, not DRAM counters. Trace durations come from separate instrumentation; do not subtract them from the public-call medians to price overhead.

| Shape | Payload MB | Public call us | Useful GB/s | Fraction of736 GB/s | Traced native us | Internal gap us |
|---|---:|---:|---:|---:|---:|---:|
| wqkv | 9.21088 | 29.696 | 310.17 | 42.14% | 21.441 | 0.0640 |
| wqb | 42.10688 | 95.232 | 442.15 | 60.07% | 87.328 | 0.0635 |
| wob | 42.10688 | 93.792 | 448.94 | 61.00% | 86.049 | 0.0654 |
| shared_w13 | 23.68512 | 57.344 | 413.04 | 56.12% | 48.833 | 0.0972 |
| shared_w2 | 11.84256 | 36.672 | 322.93 | 43.88% | 27.745 | 0.0952 |
| indexer_wqb | 5.26336 | 23.296 | 225.93 | 30.70% | 14.944 | 0.0957 |

The in-graph wq_b native trace was87.328us versus66.336us in the earlier Nsight single-kernel capture. That difference itself cautions against treating the88.39% counter measurement as proof that the in-graph kernel already runs at that rate. Kernel conditions and measurement method differ; no causal explanation is claimed from the unpaired comparison.

## Boundary experiment

[PyTorch documents captured external CUDA event nodes](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.streams.Event.html). The runtime is2.14.1+cu130. For each stage/shape, we measured the original unmarked graph with outer events, then a graph with two external event nodes, reading both outer and inner intervals on the same replay. Inputs,256MiB displacement and15 samples were matched. No clocks or power settings were changed.

|208-call M=1 total, ms|Original outer|Marked outer|Marked inner|
|---|---:|---:|---:|
| quant_layout | 2.517248 | 3.183104 | 0.641536 |
| native_raw | 12.243712 | 12.690944 | 11.479808 |
| reduce_only | 2.537728 | 3.193600 | 0.701696 |
| full_public | 12.659200 | 13.201920 | 11.947520 |

For full calls, marked outer minus marked inner is1.254400ms on the same replays. Adding event nodes increases the outer result by0.542720ms relative to the original graph, so marked-inner11.947520ms is **not a new gate pass or a kernel speedup**. The original acceptance method remains12.659200ms. This establishes that timing boundaries contribute materially to the fitted fixed cost; it does not establish how much a production token graph would improve. The header-only kernel candidates must still pass the unchanged gate.

## Next implementation and cautions

Claude's review (ROUTE3_PHASE2_REVIEW_r1.md SHA2a1a970ec58bae65a44e715ad1115effc281ca6d87990272d35c5bb41dbe51ba) recommends BPC4/S3 then BPC4/S4. Both were prepared locally with identical CUDA device source and wrapper, only geometry constants changed, and local fragment/swizzle/range/copy/split/budget checks passed using actual constants. Shared memory is19,248 and25,664bytes/CTA. Expected residency is a hypothesis until compiler resources and runtime outcomes are inspected. Full597-case identity matrix before prices for each. The source of the original CUDA kernel remains SHA952bb68ea79c28caea889028ba1fcbe7cc79c77e22a5c42546b95388aade63d1.

Persistent-grid and fused-reduce changes are not justified by the small measured internal gaps. The memory/scheduling hypothesis is the next inexpensive test. No instruction-mix bottleneck is ruled out solely by a linear fit or a single bandwidth counter. No arithmetic, split boundary, epilogue or reduction change is proposed.

## Receipts and constraints

- `ROUTE3_STAGE_PRICES_REVIEW_r1.json` SHA256 `bd27d2ab96a85745627ede74d6512ed05c00ef246b790b18585df59d22c2f775`.
- `ROUTE3_TIMING_BOUNDARIES_REVIEW_r1.json` SHA256 `c8369ec5c7f72e1c14ba08897b4540961618207d03f531d39f22474d22ffb177`.

Raw results and traces: local `route3-phase2-work-r1/route3-stage-prices-r1/` and `route3-timing-boundaries-r1/`; remote `/scratch/aster/bench/six-node-performance-r1/` uses the latter two directory names directly. Component guards:CPU55,8GiB/no-swap,96GiB admission/64GiB reserve,shared benchmark lock,idleGPU,320W. PR6 held, default-off, no pushes, host-reload SHAeb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4 unchanged. No full-model run or promotion.
