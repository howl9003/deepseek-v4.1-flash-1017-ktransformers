# Serving profile timing, 2 Nov 2026

**The serving profile measures breadth 33.15 tok/s on one session, level with the 32.98 performance profile. Four sessions at once give 51.1 tok/s in total, 13.5 each.** Trial `ada-serve-20261102-r2`, six nodes, 42 workers, 320 W, two rounds on one load, same frozen benchmark. Radix prefix cache on, HiCache host and disk tiers on, four running requests, real cache lifetimes (agent 30 min, user 12 h).

| Measure | Performance profile (32.98) | Serving profile |
|---|---:|---:|
| Breadth, rounds | 33.05, 32.90 | 32.65, 33.65 |
| **Breadth** | **32.98** | **33.15** |
| Japanese | 35.1, 35.8 | 35.0, 35.9 |
| 35K prefill, first round | 69.50, 55.46 s | 78.03, 61.06 s |
| 35K prefill, second round | 51.93, 50.80 s | 4.18, 2.59 s, read from the prefix cache |

Prefill is about 11% slower cold: the serving profile keeps SWA mapped and plans smaller chunks. The second-round rows are the cache working, since the benchmark resends the same prompts. They are not a cold rate.

## Sessions at once

512 tokens each, four different prompts, streamed through `/v1/chat/completions`:

| Sessions | Total tok/s | Per session | First token |
|---|---:|---:|---:|
| 1 | 31.0 | 31.6 | 0.4 s |
| 2 | 41.7 | 22.3 | 1.6 s |
| 4 | 51.1 | 13.5 | 2.2 s |

Four sessions scale total throughput 1.65x. The GPU serial part batches well; the CPU window and slots are what each extra session shares.

## Four long sessions, follow-up turn

Four 16K sessions at once, 128 tokens each, then each sends its next turn:

| Turn | Cached | Wall |
|---|---:|---:|
| First | 0 of 16,384 | 129 s |
| Second | 16,384 of 16,576 | 35 s |

The second turn's 18-22 s to first token is high for 192 new tokens across four sessions; one session from disk took 3.3 s. Not yet explained. Likely slot lend and prefill window turnarounds under load, worth a look on the timeline.

## dsh

DeepSeek Harness against this server (direct to the port, not yet through the selector):

- One session: a real tool-use task, read a file and quote it, 38 s from cold, right answer. The second turn read 7,936 of its tokens from the prefix cache.
- Four sessions at once: all four complete in 105 s, cold, right answers. Trial `ada-serve-20261102-r4`.
- dsh's system prompt alone is about 7.9K tokens, so a cold first turn costs about 20 s.

The first four-session try failed on the gate's queue of 4: dsh sends more than one request per session at the start, and two sessions got "The request queue is full". Serving now waits on 32. The first launcher test failed too: my resident loop treated a slow `/health` as fatal, and the server cannot answer it during a prefill window. Misses are now counted, not fatal; the memory guards are unchanged.

Stopping takes about 110 s until the GPU is free, clean, nothing left in the scope. A load takes about 18 minutes.

## Also checked

- **Tool calls work:** a chat request with one tool returned `finish_reason: tool_calls` with parsed arguments.
- **Memory:** lowest node 7.4 GiB (node 3), the lowest yet, guards passed. Lowest was 8.4 on the performance gate. Watch this one.
- Post-timing repeat in the frozen client passed in both rounds.

Serving differs from the 32.98 profile in the radix cache, HiCache, four running requests, `KT_PREFILL_SWA_LEND=0`, the cache lifetimes, the plan margin, the disk tier and two inert overlays. The 32.98 selectors themselves are unchanged.
