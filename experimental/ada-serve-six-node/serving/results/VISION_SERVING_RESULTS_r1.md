# C2 served with vision, 4 November 2026

**Served from 4 Nov 12:38 UTC.** The owner asked to serve with vision ("can we serve with vision please") from the original release files ("download original"). Text stays exact vs C2, except prompts that hold the bare image token 129264.

## What changed

- **Weights.** `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e10`: shard 1 of 48 with the 263 BF16 vision and aligner tensors, plus the image start, end and newline embeddings. The served checkpoint has no vision tensors but has all 40 `bias_vl`.
- **View.** `/scratch/aster/models/DeepSeek-V4.1-Flash-1017-c2-vision-view-r1` links the served checkpoint and adds the 266 tensors through a merged index (`profiles/VISION_VIEW.json`). The served checkpoint is untouched.
- **Launch.** The model path points at the view and `--language-model-only` is dropped. `SGLANG_DSV4_VISION_ON_DEMAND=1` was already set: the tower, 925.7 MiB, stays in pinned host memory and goes to the GPU only while images encode. `profiles/serving-as-run.json`.
- **Source.** None: the SGLang tree and natives equal served C2's.

## Evidence

| Run | Result |
|---|---|
| adac2v, `ada-identity-c2-vision-20261104-r1` | Failed on the first 35K case: its random prompt holds token 129264 at position 8519, which vision reads as an image |
| adac2w, `ada-identity-c2-vision-20261104-r2` | `ADA_NUMERICAL_EXACT_VS_C2`: 11,496 scores, both replays, the 4K and second 35K cases with top logprobs, rotation, and replay 0 after an image, all exact. The first 35K case is skipped for the reason above. Image test: "red and blue" for a red and blue image. `receipts/identity-vision/` |
| adasv, `ada-serve-c2-vision-20261104-r1` | Serving breadth 33.7 and 34.15 (C2 serving: 33.2, 34.1); one, two and four sessions 32.1, 42.3 and 52.0 tok/s in total (C2: 31.8, 41.7, 51.5); repeats exact after timing; tool call fine. `receipts/serving/` |

## Limits

- A prompt that holds token 129264 as text is read as an image. Test prompts must avoid that id.
- The vision trial shares served C2's KV disk store.
