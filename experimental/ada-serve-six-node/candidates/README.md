# Candidates: tested, not served

Each is exact vs C2 (11,496 scores, both replays, the 4K and two 35K cases), off by default, and applies over the served tree.

| Folder | What | Selector | Result |
|---|---|---|---|
| `cpu-batch/` | One input quantisation per part at one token, the decode phases as one pool job, the merge inside each part's own job. Apply `apply_shared_x.py` first; `build_ada_cpu_batch_r1.py` builds the native | `KT_CONTRACT_V2_SHARED_X`, `KT_CONTRACT_V2_FUSED_PHASES`, `KT_CONTRACT_V2_MERGE_IN_PARTS` | Serving breadth 34.15 against 33.65 (+1.5%); one session +2.4%, two and four flat |
| `upstream-picks/` | 05yuki's short-prompt CPU share (`46da340`) ported to the contract v2 stream, his prefill warm-up (`77021a7`) and HiCache staging cap (`cef35da`) | `KT_STREAM_CPU_SHORT=1024:48,2048:34,4096:20,8192:12`, `SGLANG_WARMUP_PREFILL_LENGTHS`, `SGLANG_HICACHE_STAGING_MB` | Fresh 2,048-token prompt 17.4 s against 21.1 s; 1,024 tokens 13.45 s; the 8K tier gains nothing |
| `swa-tail/` | 05yuki's patch 23, SWA checkpoint tail (`3e1fea9`), and his cache test | `apply_swa_tail.py` | Cache test scaled to 128K, 64K, 32K and 16K histories: 23 of 23 fact checks over 28 requests |
| `dspark-swa-match/` | DSpark's draft SWA pool sized to the pool the SWA lend widens; without it the draft writes past its buffers once a prompt's SWA slots pass 7,680 | `KT_DSPARK_SWA_MATCH=1` | Fixes the illegal address on prompts over about 7K tokens; sizing only, about 110 MB |

The CPU share pads its rows to a power of two, because kt-kernel keeps one buffer set per row count (about 150 MB of VRAM each at 1,024 rows).
