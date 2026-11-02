# NVFP4 1017 on the public panel, 2 Nov 2026

Nobody had published NVFP4 1017 on a token-level panel. This is a measurement of an emulation on the 4080, not NVIDIA's kernels on Blackwell: DeepSeek's reference forward with W4A4 activations on the routed experts and, in the second arm, NVIDIA's Engram conversion. **It scores KL 0.0167, top-1 96.09%, about 1.7 times the reference's own noise floor on KL, and ahead of the best public pack on these numbers.**

All arms use sudoingX's panel: 199 conversations, 254,846 scored positions, sha256 39e0db2f, scored with p4_kld.py's KL (native top 128 plus one tail bucket) on the full distribution.

| Arm | KL | KL p99 | Top-1 | NLL | Perplexity |
|---|---|---|---|---|---|
| Reference | | | | 0.47077 | 1.6012 |
| Reference, MoE batch 2,048 (floor) | 0.00996 | 0.123 | 97.01% | 0.47105 | |
| W4A4, FP8 Engram | 0.01633 | 0.207 | 96.19% | 0.47293 | 1.6047 |
| **W4A4, MXFP4 Engram (the release)** | **0.01668** | **0.217** | **96.09%** | **0.47270** | **1.6043** |

For scale, measured by others on the same panel against a GB10 reference: sudoingx 2.8 bpw 0.0198 / 95.94%, HeNryous 2.75 bpw 0.0201 / 95.96%, LibertAIDAI 2.9 bpw 0.0304 / 94.76%. sudoingX's GB10 reference has NLL 0.47031, ours 0.47077.

By domain, W4A4 with FP8 Engram against the floor, KL / top-1:

| Domain | Positions | Floor | W4A4 |
|---|---|---|---|
| code | 165,450 | 0.0070 / 97.41% | 0.0112 / 96.70% |
| agentic | 16,686 | 0.0119 / 96.71% | 0.0192 / 95.86% |
| chat | 17,976 | 0.0217 / 95.34% | 0.0375 / 93.88% |
| knowledge | 14,093 | 0.0171 / 95.52% | 0.0290 / 94.27% |
| long | 16,457 | 0.0144 / 96.96% | 0.0214 / 96.18% |
| math | 10,625 | 0.0079 / 96.76% | 0.0122 / 96.22% |
| multilingual | 8,620 | 0.0148 / 97.09% | 0.0281 / 96.01% |
| writing | 4,939 | 0.0218 / 95.44% | 0.0374 / 94.37% |

KL excess over the floor, averaged by conversation: 0.00775, SE 0.00061, so the loss is real. It runs about 1.5-1.9x the floor in every domain. On the 24 pieces: NLL +0.0122 (SE 0.0063), mean absolute change 0.178 against the floor's 0.121.

## Reading it

- **FP4 activations cost something, but less than cutting weights to under 3 bits.** NVFP4 keeps every weight bit, and its KL is about 1.7 times the floor against about 2 times for the best 2.8 bpw pack. The gap between those two is small and across references, so read it as "similar", not "better".
- **The lossy Engram tables barely matter.** About 12% relative error on the table moves KL 0.0163 to 0.0167 and top-1 by 0.10 points; NLL moved the other way, inside the noise.
- **NLL barely moves:** +0.0019 nats. This is why task scores do not show it.
- **Compare with AtomicChat:** 0909 weights, their corpora, their calibrated and flat casts: general text top-1 94.1%, code 96.65%, agentic 98.34%. Not the same panel or weights; the direction agrees.

## What the emulation is and is not

- Activation rule: vLLM's `cvt_warp_fp16_to_fp4` with SFScale 1: per 16, E4M3(amax / 6), then E2M1 ties to even, saturating at 6. `w4a4.py`.
- `input_scale` is flat at 1.0, not NVIDIA's calibrated values. AtomicChat found no convincing benefit from calibration on 0909; I have not measured it on 1017.
- Weights: the same values NVFP4 stores. Expert products run in FP32 on CUDA, not on Blackwell's FP4 tensor core with 25 fractional bits. I expect that gap to be small, but it is not measured.
- Engram: `local_ckpt.mxfp4_row`, checked against NVIDIA's real tables: 96 sampled rows, every scale and weight byte equal. Scale is the smallest power of two at or above amax / 6 per 32, ties toward zero, -0 to +0.
- Everything else is DeepSeek's code on Ada, so the reference and every arm carry Ada's dense arithmetic.
- Not measured: speed.

Runs: `w4a4-panel-r1-part*`, `nvfp4-release-r1-part*`, about 41 minutes each.
