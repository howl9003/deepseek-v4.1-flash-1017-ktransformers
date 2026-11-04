# Served selectors

C2 with vision, trial `ada-serve-c2-vision-20261104-r1`. Every selector is off by default in the source; the profiles turn them on. Full launches: `profiles/performance-as-run.json` (bench timing) and `profiles/serving-as-run.json` (what is served). Values below are read from them.

## Numerics

| Selector | Value | What it does |
|---|---|---|
| `KT_CONTRACT_V2` | `1` | Integer-exact routed experts: CPU and GPU compute each expert in the same integers on the reference's activation grid, one rounding at the end |
| `KT_CONTRACT_V2_COMBINE_ORDER` | `logical` | Routed experts combined in ascending logical expert id, as the reference sums them; bits do not depend on placement |
| `KT_GPU_STREAM_PATHA` | `1` | Integer-exact prefill experts on the GPU stream |
| `KT_GPU_STREAM_PATHA_DOWN_PLANES` | `3` | Integer prefill path setting, as served |
| `KT_GPU_STREAM_PATHA_MAX_ROWS` | `0` | Integer prefill path setting, as served |
| `SGLANG_FP8_FIXED_SPLIT_K` | `4` | Fixed-sequence dense: four fixed K ranges, the same at decode and prefill |
| `SGLANG_FP8_ROUTE3_CUDA` | `1` | The CUDA kernel for that sequence (Route 3) |
| `SGLANG_FP8_FIXED_DECODE_TRANSPOSE` | `0` | Fixed-sequence decode setting, as served |
| `SGLANG_DSV41_FIXED_PROJECTIONS` | `1` | Dense projections through the fixed sequence |
| `SGLANG_DSV41_FIXED_WO_A` | `1` | `wo_a` kept in FP8 blocks, through the fixed sequence |
| `SGLANG_DSV41_FIXED_LM_HEAD` | `1` | Head through the fixed projection |
| `SGLANG_SM89_FP4_COMPRESSED_KV` | `1` | Compressed KV keeps the reference's FP4 values |
| `SGLANG_SM89_V41_KV` | `1` | Window KV in the V41 layout, on the reference's grid; implies the FP4 caches |
| `--enable-fp32-lm-head` | on | FP32 logits, no BF16 cast |

## Speed, bit-exact

Each gave the same 11,496 scores as the build it was added to.

| Selector | Value | What it does |
|---|---|---|
| `KT_EXPERT_ARENA_SPILL_STAGGER` | `1` | Spilled expert routes stay on the socket and are staggered, so the workers do not hit one node at once |
| `KT_KV_HOT_DYN_WORKER` | `1` | Dynamic slot rounds on a worker thread |
| `KT_KV_HOT_DYN_HALF_LIFE` | `64` | Slot score half-life |
| `KT_KV_HOT_DYN_HYST` | `1.5` | Slot swap hysteresis |
| `KT_KV_HOT_DYN_EVERY` | `4` | Slot rounds every 4 tokens |
| `KT_KV_HOT_LAYER_QUOTA` | `profiles/LAYER_QUOTA.json` | Per-layer slot quota |
| `KT_CONTRACT_V2_ROUTE_REF` | `1` | Routes pass by reference, no scatter copy |
| `KT_CONTRACT_V2_GLUE_FUSED` | `1` | Fused route glue in captured decode, 51 fewer launches a layer |
| `SGLANG_DSV41_FIXED_WO_A_CUDA` | `1` | `wo_a` on a CUDA kernel, sm_89 |
| `KT_IDX_LIVE_TOPK` | `1` | Indexer top-k on the live length |
| `KT_IDX_LIVE_TOPK_MAX_L` | `1048580` | Live top-k at the served capacity, with the `-inf` fix |
| `KT_EXPERT_ARENA_REBALANCE_NODE0` | `3` | Node 0 gives up a sixteenth of its expert arena |
| `DSV41_ENGRAM_AIO` | `1` | A call's Engram misses in one polled `io_submit` |
| `KT_PREFILL_KV_LEND_ALL_TENSORS` | `1` | KV lend reaches every pool tensor laid out by the allocator's pages, after 05yuki `7fef854e` |
| `SGLANG_SM89_ATTN_SPLITK` | `0` | Unsplit SM89 attention |
| `SGLANG_SM89_ATTN_STAGED` | `1` | The unsplit gather as three staged kernels, same bits |
| `SGLANG_ROUTE3_ENGRAM` | `1` | Engram `wkv` through the Route 3 kernel |

## Serving profile

On top of the performance profile, the served launch sets:

| Selector | Value | What it does |
|---|---|---|
| `KT_STREAMING_SESSION_ADMISSION` | `1` | Retained-session admission and ownership fixes (#6) |
| `SGLANG_CACHE_TTL` | `1` | Prefix cache lifetimes by request class |
| `SGLANG_CACHE_TTL_AGENT_S` | `1800` | Agent prefix lifetime, seconds |
| `SGLANG_CACHE_TTL_USER_S` | `43200` | User prefix lifetime, seconds |
| `KT_PREFILL_SWA_LEND` | `0` | Off: SWA stays mapped while serving |
| `KT_HICACHE_FULL_HOST_TOKENS` | `1048576` | HiCache host tokens, full layers |
| `KT_HICACHE_SWA_HOST_TOKENS` | `65536` | HiCache host tokens, SWA layers |
| `KT_HICACHE_SWA_TAIL_ONLY` | `1` | Only the last SWA window goes to host and disk |

and sets `--max-running-requests` `4`, `--enable-streaming-session`, `--prefill-max-requests` `4`, `--max-queued-requests` `32`, `--enable-hierarchical-cache`, `--enable-cache-report`, `--hicache-ratio` `4.0`, `--hicache-write-policy` `write_through`, `--hicache-storage-backend` `file`; it drops `--disable-radix-cache`, `--language-model-only`.

## Vision

| Setting | Value | What it does |
|---|---|---|
| `--model-path` | checkpoint view `DeepSeek-V4.1-Flash-1017-c2-vision-view-r1` | The served checkpoint plus 266 tensors from the release at `1bae0e10`: the 263 BF16 vision and aligner tensors and the image start, end and newline embeddings (`profiles/VISION_VIEW.json`) |
| `--language-model-only` | dropped | Loads the vision tower |
| `SGLANG_DSV4_VISION_ON_DEMAND` | `1` | 05yuki's vision on demand: the tower, 925.7 MiB, stays in pinned host memory and is on the GPU only while images encode |

Text is exact vs C2 with vision on, except prompts holding the bare image token 129264, which vision reads as an image.
