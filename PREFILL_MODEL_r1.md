# Cold prefill, packing and prefix restore

A 35,000-token cold prompt can spend several seconds uploading experts at each chunk boundary. First read the actual planner decision from the serving log. Available memory has produced both three chunks near 12K rows and two near 18K on this hardware. Charging 460 KiB per row in the measured planner fit moved the two-chunk boundary to roughly 12,000 MiB free. The configured cost tuple takes precedence over the separate row-size hint.

## One layer

The model has 40 MoE layers, hidden size 5,120, expert width 2,304, 384 routed experts, six routes per token and one shared expert. Early layers process every row. On the final chunk the late half uses a tail of up to 2,816 rows. When no intermediate result requires the tail, the release skips the eligible non-final boundaries. It skipped all eight in the two timed cold traces.

In the reference full sweep, each of the two GPUs pulls its socket-local half of 383 non-resident experts. The 96 groups of up to four experts use two buffers so upload can overlap compute. With byte scales, a full sweep transfers 3.58–3.60 GB per card. At 27.6 GB/s the upload floor is about 130 ms.

| Rows | Backlog before the sweep | Device sweep | Source of the distinction |
|---:|---:|---:|---|
| 2,816 | 23 ms | 130 ms | Start/end GPU events around the group graphs |
| 9,000 | 70 ms | 145 ms | Same trace |
| 12,288 | 100 ms | 152 ms | Same trace |
| 18,432 | 155 ms | 187 ms | Same trace |

A useful baseline fit to the host-reported stream line was `max(137, 120 + 11.3 × rows/1000)` ms. Its slope includes roughly 8.2 ms of backlog per 1,000 rows and 3 ms of device sweep. At 12,288 rows a full device layer took about 277 ms, with about 125 ms between sweeps. At 18,432 rows the group graph exceeded the upload period. Sampling the host total alone would overprice a change to that graph.

## Calibrating CPU share and packing

Use the actual routed groups and both rank timings. Skip empty groups, and assign CPU work until its predicted completion approaches the remaining GPU sweep. Track CPU lag as well as end-to-end time. The integrated cold traces had a median CPU lag of 6.14 ms. Cold wall times on table B averaged 9.45 s at 12,288 tokens and 21.03 s at 35,000. The latter pools four cold observations from two client protocols.

Co-admit compatible prompts subject to the live memory estimate and a bounded prefetch wait. In the measured packs, all four timed pairs or groups of four completed in one forward per group. All 43 requests matched their solo output. Both ranks recorded 23, 4 and 3 complete streamed forwards at widths 2,048, 4,096 and 8,192 respectively. Mean packed/serial TTFT ratios were 0.745 for pairs and 0.691 for fours. For fours the means were 6.682 and 9.673 s. Estimate each denominator on the serving build being compared.

## Reusing a long prefix

Separate lookup, disk read, host-to-device copy and the remaining forward. For one measured follow-up, L2 and L3 each restored 36,620,800 bytes per rank, reused 35,072 tokens and processed 244 new rows. Output matched the cold twin in both cases. L2 took 2.037 s and L3 took 1.970 s at the client. L3 still spent 0.252–0.254 s per rank reading from storage. Total request time alone does not identify the storage penalty.
