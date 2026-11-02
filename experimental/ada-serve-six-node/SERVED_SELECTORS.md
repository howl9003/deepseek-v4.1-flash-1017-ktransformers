# Served selectors

The owner chose these settings for serving on 2 November 2026 ("we can use the 32.98 settings"). They stay default-off in the source files; the profiles here turn them on. `profiles/performance-as-run.json` and `profiles/serving-gate-as-run.json` hold the full environment, 181 and 189 variables.

Breadth is the six-workload median on six nodes at 320 W after that step. Each step was exact vs C1 with the gate passed.

| Selector | Value | What it does | Breadth tok/s | Write-up |
|---|---|---|---:|---|
| `KT_EXPERT_ARENA_SPILL_STAGGER` | 1 | Spilled expert routes stay on the socket and are staggered, so 42 workers do not hit one node at once | 23.78 | `results/ada-spill-r1/ADA_OPTIMISATION_RESULTS_r1.md` |
| `KT_KV_HOT_DYN_WORKER` | 1 | Dynamic slot rounds run on a worker thread | 23.78 | same |
| `KT_KV_HOT_DYN_HALF_LIFE`, `KT_KV_HOT_DYN_HYST` | 64, 1.5 | Faster slot scoring; half-life 256 was the miss | 24.98 | `ADA_TIMELINE_RESULTS_r1.md` |
| `KT_CONTRACT_V2_ROUTE_REF` | 1 | Routes pass by reference, removing the scatter copy | 24.98 | same |
| `KT_CONTRACT_V2_GLUE_FUSED` | 1 | Fused route glue in captured decode, 51 fewer launches a layer | 25.95 | `ADA_GLUE_RESULTS_r1.md` |
| `SGLANG_DSV41_FIXED_WO_A_CUDA` | 1 | `_parts` on a CUDA kernel, 97 to 50 us | 27.65 | `ADA_WO_A_RESULTS_r1.md` |
| `KT_IDX_LIVE_TOPK`, `KT_IDX_LIVE_TOPK_MAX_L` | 1, 1048580 | Live indexer top-k at the served capacity, with the `-inf` overlay | 29.68 | `ADA_TOPK_RESULTS_r1.md` |
| `KT_EXPERT_ARENA_REBALANCE_NODE0` | 3 | Node 0 gives up a sixteenth of its arena | 31.25 | `ADA_CPU_WINDOW_RESULTS_r1.md` |
| `DSV41_ENGRAM_AIO` | 1 | A call's Engram misses in one polled `io_submit` | 31.25 | same |
| `KT_PREFILL_KV_LEND_ALL_TENSORS` | 1 | KV lend reaches every pool tensor laid out by the allocator's pages, after 05yuki `7fef854e` | 31.25 | same |
| `KT_KV_HOT_LAYER_QUOTA` | `profiles/LAYER_QUOTA.json` | Per-layer slot quota | 31.25 | same |
| `KT_KV_HOT_DYN_EVERY` | 4 | Slot rounds every 4 tokens | 31.25 | same |
| `SGLANG_SM89_ATTN_STAGED` | 1 | The unsplit gather as three staged kernels, same bits | 32.0 | `ADA_GATHER_RESULTS_r1.md` |
| `SGLANG_ROUTE3_ENGRAM` | 1 | Engram `wkv` through the Route 3 kernel | **32.98** | `ADA_ENGRAM_RESULTS_r1.md` |
| `KT_GPU_STREAM_PATHA` | 1 | Prefill path A, with `KT_GPU_STREAM_PATHA_DOWN_PLANES=3` and `KT_GPU_STREAM_PATHA_MAX_ROWS=0` | prefill | `../contract-route3-six-node/` history, `tools/stage_phase_g_g1v_adaygate_r1.py` |

The serving-gate profile adds `--enable-streaming-session`, `--prefill-max-requests 4`, `--max-queued-requests 4`, `--max-running-requests 4`, `--enable-hierarchical-cache`, `--enable-cache-report`, `--hicache-ratio 4.0`, `--hicache-write-policy write_through`, drops `--disable-radix-cache`, and sets the cache policy `SGLANG_CACHE_TTL=1`, `KT_STREAMING_SESSION_ADMISSION=1`, `KT_PREFILL_SWA_LEND=0`, `KT_HICACHE_FULL_HOST_TOKENS=1048576`, `KT_HICACHE_SWA_HOST_TOKENS=65536`, `KT_HICACHE_SWA_TAIL_ONLY=1`. Its TTLs, `SGLANG_CACHE_TTL_AGENT_S=6` and `SGLANG_CACHE_TTL_USER_S=120`, are test lifetimes, not serving values.

Not served: `KT_KV_HOT_DYN_COPY_CHUNK` (20% slower), `KT_KV_HOT_DYN_FREEZE_AFTER` (diagnostic), `SGLANG_FP8_PATHA_DENSE` and `SGLANG_SINGLE_ROUTE` (parked).
