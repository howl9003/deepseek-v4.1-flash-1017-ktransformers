# Comparison items for the 5070 Tis

Things to measure once the two cards are in and the single route is qualified on them. Nothing here runs before that. `SM120_PATH_r1.md` stays the plan; this file only holds the queue.

## 1. NVFP4 W4A4 (filed 1 Nov 2026, owner)

**Candidate:** [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4), released 30 Oct 2026, MIT, same source revision as ours (1bae0e10). 429 GB, about 400 GiB.

**Question:** does W4A4 on Blackwell's native FP4 tensor core beat our integer-exact path by enough to leave it?

**Why it is not simply better:** experts convert losslessly from the MXFP4 weights we hold, but activations go to 4-bit on calibrated scales and the Engram tables go FP8 to MXFP4, which is lossy. So it is not bit-identical to the 1017 reference or to contract v2. Using it would be the owner's numerics decision and needs a precision ledger entry.

**Compare against** contract v2 on the same two cards, same power limits:

| Measure | How |
|---|---|
| Decode | Breadth median of the six workloads, two rounds |
| Prefill | Both 35K prompts, two rounds |
| Distance from the reference | The method in `storage-path` r2 (other-grid paths came out about 2.3% from the 1017 reference; only contract v2 lands on its grid), plus AtomicChat's teacher-forced protocol below so the numbers compare with theirs |
| Accuracy | NVIDIA's table against the source, percent: GPQA 92.93 / 92.77, AA-LCR 80.19 / 80.41, SciCode 57.32 / 57.58, IFBench 76.47 / 76.33, MMMU-Pro 78.73 / 78.38, Terminal-Bench 81.11 / 81.81. Theirs, on four GB300s, not independently checked |
| VRAM | Slots a layer: scale bytes double, about 6% more per expert |
| CPU routes | Price a W4A4 CPU kernel against the contract v2 kernel on real weights, no model load. It drops two digit planes, and memory is at most about 30% of a task on the 4080 build, so it could be materially faster. Untested estimate |

**Published fidelity to beat.** [AtomicChat's V4.1-Flash build](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia) (0909 weights, not 1017) was measured against the original on one vLLM build on 4 x B200: 24 windows of 4,096 tokens on each of neutral, code and agentic corpora, 49,152 scored positions, teacher-forced. Against repeats of the original at 96.13 / 97.69 / 98.67% top-1, the NVFP4 build scored 94.10 / 96.65 / 98.34%, with perplexity +0.85 / +0.38 / +0.09%. The gap is the whole execution path, not FP4 activations alone. Raw dumps are in their metrics dataset.

**Measured here on the public panel (1 Nov), emulated, no Blackwell:** `six-node-performance-r1/reference-panel-r1/`. W4A4 activations in DeepSeek's reference forward on the 4080, vLLM's activation rule, flat `input_scale`, FP8 Engram: KL 0.0163, top-1 96.19%, NLL 0.47293 against 0.47077, 254,846 positions. With NVIDIA's Engram rule, byte-verified, the release itself: KL 0.0167, top-1 96.09%. Reference against itself: KL 0.00996, top-1 97.01%. Best public 2.8 bpw pack: 0.0198, 95.94%, against its own GB10 reference. So NVFP4's fidelity cost is known before the cards arrive; what stays open is speed. `NVFP4_PANEL_RESULTS_r1.md`.

**Calibration may not matter.** The same run found no convincing benefit from NVIDIA's calibration: a flat cast with `input_scale = 1.0` scored within its interval on every corpus but one. That is one model on three runs each.

**Single-stream speed may be small.** Unsloth's table for Qwen 3.6 on one B200 shows W4A4 on the tensor core well ahead on batch throughput (6,863 against 2,403 tok/s on 27B) but only 9-23% on single-stream decode. We serve single-stream. Different model, their figures. Measure, do not assume.

**Needs first:**

- Disk: `/scratch` had 186 GB free on 1 Nov, so the full repo does not fit, and a first test does not need it. Cast the experts from our copy with `input_scale = 1.0` and keep the FP8 Engram tables, as AtomicChat did. Fetch NVIDIA's `input_scale` and MXFP4 Engram only if that test justifies it.
- Runtime: NVIDIA runs it through vLLM with PR 61530 on Blackwell. SGLang is not validated, so an SM120 FP4 expert path and a W4A4 CPU contract would have to be built or the comparison run under vLLM.
- Calibration: 24 of 46,080 activation scales are fallbacks; check whether those projections matter.

Background: memory notes `nvfp4-1017-review`, `qwen-nvfp4-cpu`.
