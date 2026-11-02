# Decode timeline on the 32.98 build, 2 Nov 2026

**A breadth token is 13.3 ms serial GPU, 7.4 ms GPU in the expert window, 6.9 ms waiting for the CPU and 2.2 ms head. The inferred split I had been quoting was right. The stalls seen on the 1 Nov capture are gone, so that item closes.** Trial `ada-nsys-engram-20261102-r1`, 32.98 settings, 64 steady tokens from token 512 on each of three rows, one load. FEA job 1716 was running on its own nodes during this capture.

| ms per token | English technical | Python | Japanese |
|---|---:|---:|---:|
| Serial GPU | 13.46 | 13.30 | 13.32 |
| GPU in the expert window | 7.35 | 7.73 | 9.65 |
| GPU idle, waiting for the CPU | 6.94 | 6.20 | 1.74 |
| LM head and tail | 2.17 | 2.17 | 2.17 |
| **Total** | **29.93** | **29.41** | **26.89** |
| Median token period | 30.36 | 29.64 | 26.64 |
| Windows where the GPU finished last | 21% | 26% | 61% |

Against the 31 Oct capture, serial fell from 19.9 to 13.3 ms and CPU wait from about 8 to about 6.6 on breadth rows. Note this measures steady tokens, so it reads about 3-5% faster than breadth, which is a whole request including slot adaptation.

## What serial is

Serial sum 13.2 ms, 1,924 events:

| Kernel | ms | Calls a token |
|---|---:|---:|
| `route3_decode_kernel` | 6.67 | 130 |
| `wo_a_parts_kernel` | 2.08 | 40 |
| `_output_kernel` | 0.51 | 40 |
| `_hc_mix_stats_partial_kernel` | 0.49 | 79 |
| `_partials` | 0.38 | 59 |
| `_pack_kernel` | 0.35 | 40 |
| `_scores_kernel` | 0.26 | 40 |
| `hc_split_sinkhorn` | 0.25 | 79 |
| torch small kernels | 0.23 | 142 |

Route 3 is half of serial, and with its 3.3 ms in the window it is about 10 ms of every token, the largest single item. Layers 1, 14 and 20-ish stand out in serial (548-692 us against about 300), which are the Engram and indexer layers.

## Stalls

Token periods over the 63 steady intervals:

| Row | Median | Mean | p90 | Max | Over 1.3x median |
|---|---:|---:|---:|---:|---:|
| English technical | 30.33 | 30.55 | 32.59 | 50.0 | 1 |
| Python | 29.57 | 29.70 | 31.94 | 33.0 | 0 |
| Japanese | 26.68 | 27.20 | 29.10 | 30.2 | 0 |

The 1 Nov capture showed 57-77 ms tokens against 33 and a mean 3 ms above the median. Here the gap is 0.1-0.5 ms. So I am closing the stall item: there is nothing left to win on this stretch. Scope is 64 tokens from token 512 of three rows, so an early-request stall would not show.

## What the replay says

Levers replayed on these windows, English technical, ms a token, from 29.91:

| Lever | Token | Saved |
|---|---:|---:|
| 1 ms off serial | 28.91 | 1.00 |
| 50 us off each window's GPU part | 29.50 | 0.41 |
| CPU 20% faster | 28.63 | 1.28 |
| CPU twice as fast | 26.69 | 3.22 |
| CPU never last | 23.47 | 6.44 |

On Japanese the CPU barely matters (26.22 at twice as fast) and the window's GPU part does: 50 us saves 1.23 ms. Breadth is the rate we quote, so the CPU window is still the first lever, and that fits the eight-node estimate: three quarters of the CPU work gives about -2 to -2.5 ms.

The three small CPU fixes at 0.35-0.85 ms in total stay worthwhile but small, and now the timeline supports the order: eight nodes, then those.
