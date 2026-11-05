# DeepSeek-V4.1-Flash-1017 on one RTX 4080 SUPER, bit-exact across CPU and GPU

The build we serve: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on SGLang + KTransformers, on [05yuki's recipe](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers), with the routed experts computed in the same integers on the CPU and the GPU. Replays, KV caching, sessions and hot-slot swaps are bit-exact.

- **Hardware:** one modded 32 GB RTX 4080 SUPER (SM89) at 320 W, 2 x EPYC 7543, 512 GB DDR4-3200.
- **Decode:** 33.3 tok/s breadth median on six NUMA nodes, 35.05 on all eight. Served with radix, HiCache and vision: 33.7 and 34.15 in two rounds on one session, 52 tok/s in total on four.
- **Accuracy:** KL 0.0102 against DeepSeek's own inference code on sudoingX's public panel, where the reference against itself (only MoE batching changed) reads 0.0100.

## What it changes

- **Integer-exact routed experts:** CPU and GPU compute each expert in the same integers on the reference's activation grid, with one rounding at the end, combined in logical expert order as the reference sums them.
- **Fixed-sequence dense:** one fixed sequence of FP8 tensor-core instructions, so decode and prefill share bits.
- **KV and logits on the reference's grid:** V41 window KV, FP4 compressed KV and FP32 logits.
- **Speed on top, bit-exact:** slot rules, fused route glue, a `wo_a` CUDA kernel, live indexer top-k, staged attention gather and more.

Everything is in [`experimental/ada-serve-six-node`](experimental/ada-serve-six-node): the [README](experimental/ada-serve-six-node/README.md) with results and receipts, [`BUILD.md`](experimental/ada-serve-six-node/BUILD.md) to rebuild it on the recipe, [`SERVED_SELECTORS.md`](experimental/ada-serve-six-node/SERVED_SELECTORS.md) and the [`PRECISION_LEDGER.md`](experimental/ada-serve-six-node/PRECISION_LEDGER.md).

Tested on SM89 only, not on SM120 or with TP2.

## Upstream

- The write-up for 05yuki: [05yuki/deepseek-v4.1-flash-1017-ktransformers#12](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers/issues/12). Our live top-k kernels are his tier-1 patch 27.
- SGLang fix for the V4.1 KV scale floor: [sgl-project/sglang#48500](https://github.com/sgl-project/sglang/pull/48500).
- The recipe itself, setup and serving: [05yuki's README](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers#readme). The files at the top of this branch are his recipe at upstream `7e199f2`; our build is based on his `5d53f727`.

## Credits and license

Built on 05yuki's recipe, KTransformers and SGLang, against DeepSeek's reference inference code; [`CREDITS.md`](CREDITS.md) has the full list. Apache-2.0 (see [`LICENSE`](LICENSE)), except `engram-adapter/`, which is derived from 0xSero's engram adapter under the MIT License (see the headers of its files). The model weights are under DeepSeek's license (see the model card).
