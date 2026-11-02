# Our build against DeepSeek's reference, 1 Nov 2026

First real measurement of the served build against DeepSeek's own reference code. On the 24 identity pieces, 11,496 positions, **our build is about as far from the reference as the reference is from itself when only its batching changes.** Mean NLL is +0.0034 nats a token (SE 0.0044). The same reference with different MoE batching moves it +0.0036 (SE 0.0041).

The reference here is DeepSeek's unmodified `inference/` code on the 4080, so it carries Ada's tensor-core arithmetic. It is not their Hopper or Blackwell bits. Selected-token log-probabilities only: this is NLL, not KL or top-1. Those need the panel.

| Compared with reference run 1 | NLL diff (SE) | Exact | Mean abs | Median abs | p90 | p99 | Largest |
|---|---|---|---|---|---|---|---|
| Reference, same settings, repeated | 0 | 100% | 0 | 0 | 0 | 0 | 0 |
| Reference, MoE batch per piece | +0.0036 (0.0041) | 6.1% | 0.1209 | 0.0002 | 0.358 | 1.78 | 8.62 |
| **Served build (Engram, C1 bits)** | **+0.0034 (0.0044)** | **5.5%** | **0.1301** | **0.0002** | **0.394** | **1.96** | **8.96** |
| Path A dense plus single route (arm 2) | +0.0031 (0.0036) | 5.6% | 0.1244 | 0.0002 | 0.370 | 1.86 | 4.89 |
| Nearest saved build (`pr6-main-r4`) | +0.0001 (0.0055) | 5.9% | 0.1173 | 0.0002 | 0.351 | 1.73 | 7.03 |

Reference NLL 0.7143, served 0.7178. All 21 distinct saved builds sit between 0.117 and 0.130.

Between our own builds: Engram against arm 2 0.1281, Engram against `pr6-main-r4` 0.1129, `pr6-main-r4` against `math-main-r8` 0.0928.

## Reading it

- **The reference is deterministic on this card.** Same settings, every position bit-identical. So this noise is not a race and not run-to-run.
- **Rounding alone produces this spread.** Changing nothing but how tokens batch through the MoE moves 94% of positions. Most moves are tiny (median 0.0002), but a tenth are over 0.36 nats, and they are symmetric. I read this as small rounding changes flipping routing or index choices at near-ties, which then shift a few positions a lot. That mechanism is not measured.
- **We are near that floor, slightly above it on this sample.** 0.130 against 0.121 is one draw on one set; builds range 0.117 to 0.130 and I have no error bar on the mean absolute figure. Path A dense at 0.124 is a little closer than Route 3, within the same range.
- **This changes what the panel can show.** A build sitting at the floor cannot be told from the reference by these scores. Expect the panel to give the same shape with tighter numbers.

The probe found rows independent of row count in `fp8_gemm` and `fp4_gemm`, yet batching changes the output. So the sensitivity is elsewhere: the torch-side linears and sums are the obvious candidates, not yet separated.

## Panel reference

`ref-panel-r2-part*.safetensors`: all 199 conversations, 254,846 scored positions, about 32 minutes in three parts, peak 18.7 GiB. Mean NLL 0.47077, perplexity 1.6012. sudoingX's published native figures, on GB10: 0.47031 and 1.6005. So DeepSeek's code on Ada lands within 0.0005 nats of DeepSeek's code on Blackwell over the panel. Their logits are not public, so that is the whole of the cross-check.

Noise floor on the panel: the reference again with MoE groups of 2,048 tokens instead of 65,536, scored with p4_kld's KL on the full distribution. KL 0.00996, p99 0.123, top-1 97.01%, NLL 0.47105 against 0.47077 (`ref-floor-r1-part*.safetensors`). This is the reference against itself, on Ada. By part: 0.0068 / 97.45%, 0.0087 / 97.04%, 0.0178 / 96.15%. For scale, sudoingX's best pack reads 0.0198 / 95.94% against their GB10 reference, so those packs sit about twice the floor on KL. They are measured on another GPU, so that gap is indicative.

The first panel run died in layer 0: `local_ckpt.py` unpacked experts on the GPU, and the panel's streams left no room. `ref-panel-part0.log` keeps the failure.

## Loader evidence

- `verify_ckpt.py` against the Hub at revision 1bae0e10: 72 expert tensors from 12 experts, 41 dense tensors (first MiB of each), and 64 rows in each of four Engram tables, all byte-identical. This also closes the storage review's open item: the GGUF experts match the Hub on these samples, and `original-engram-*` are the 1017 tables.
- Panel rebuilt from the public repo with the checkpoint's own encoder: `panel.safetensors` sha256 39e0db2f, matching the published hash.

Run command, `run_ref.py --pieces`: 224 s, peak 23 GiB. DeepSeek's kernels are unmodified apart from sudoingX's 16-head split, which the card needs.
