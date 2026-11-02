# Slot copies and the CPU window, 1 Nov 2026

The CPU window split found a slow state worth about 7.5 ms a token while it lasts, lining up with slot copy rounds, but only as correlation. Admission is held by an FEA job, so this tests the mechanism without the server: the real contract kernel on the served shape, seven pinned workers a node, cold passes, against GPU copies from pinned memory. No model.

**The GPU reading a node's memory slows that node's workers 47%, and no other node's.** Taking copies a small chunk from each part in turn nearly removes the cost to the window. This is not a served measurement yet.

## One node

Kernel on node 1, copies from each node in turn. `test_copy_contend.cpp`, `contend-r1.log`.

| Copy source | DMA GB/s | Pass median us | p90 us |
|---|---|---|---|
| None, four baselines | | 29.1-30.7 | 31-33 |
| **Same node** | 20.5 | **42.8** | 46.1 |
| Node 2, same socket | 24.3 | 28.8 | 30.1 |
| Node 5, other socket | 22.6 | 30.7 | 32.5 |
| Node 6, the GPU's | 23.1 | 29.1 | 31.1 |
| Same node, paced to 4.6 GB/s | 4.6 | 28.5 | 41.9 |
| Same node, paced to 1.7 GB/s | 1.7 | 29.3 | 42.2 |

The kernel reads 33 GB/s alone and 23 GB/s beside a 20.5 GB/s copy, so the node tops out near 44 GB/s. That is two channels of DDR4-3200 at about 85%. It is memory bandwidth on that node and nothing in the fabric, which the earlier guess named. Pacing only lowers how often a pass is hit: the p90 stays near 42 us because each 983 KB piece still arrives at full speed.

## Six parts

All six compute nodes, seven workers each, started together, window = the last part's finish. `test_copy_window.cpp`, `rotate-r3.log`. Runs r1 and r2 pinned the copy helper to core 22, a worker core, and show false 2 ms stalls; r3 uses core 15.

| Copy order | DMA GB/s | Window median us | p90 us |
|---|---|---|---|
| None | | 34.3-34.9 | 37.2-37.7 |
| A part at a time, the served order | 21.8 | 41.3 (+19%) | 47.8 |
| One 983 KB piece from each node in turn | 10.7 | 42.6 (+23%) | 134 |
| **One 256 KB chunk from each node in turn** | **15.4** | **35.2 (+2%)** | **39.1** |
| One 64 KB chunk from each node in turn | 5.3 | 32.7 | 51.3 |

The served order holds one node for about 180 us (3.7 MB). In a window that long, that node's part runs 47% slower for most of the window, and the window is the slowest part, which fits the +26-35% measured on served windows. With 256 KB chunks each node is touched for about 17 us at a time, which is short against a task.

Limits: these passes are 35 us against about 100-300 us of served window, so a 983 KB rotation may do better served than it does here. Rates differ by row, so the rows are not at equal load. The 64 KB row reads below baseline, which I cannot explain, so I am not leaning on it. Mean part time also sits about 2 us lower whenever the helper runs.

## The change

`KT_KV_HOT_DYN_COPY_CHUNK=262144`, `copy-rotate-r1/`, `arena_stream_candidate.py` 30fa5011 to de364f88. `load_hot` takes one chunk from each part in turn; scale and down bytes wait in their own scratch, finished once all are queued. Unset, the function is unchanged. Same bytes, same destinations.

`test_rotate.py` on the card, served shapes, six parts, both row orders, five experts: 160 tensors byte-equal at 64 KB, 256 KB, 983 KB and 4 MB. A copy takes 1.76 ms an expert at 256 KB against 1.56 plain.

Timing chain generated: tag `adacperf`, trial `ada-perf-rotate-20261101-r1`. Not run: the FEA job holds admission. The frozen-swap chain `adavperf` stays ready as the diagnostic if this does not move breadth.

If the 0.9-1.8 ms of copy cost in the window split holds and most of it goes, this is worth about 0.7-1.5 ms, 2-5% of breadth. That is an estimate.

## Other engines, checked 2 Nov

A read of 0xSero's offload, SAEM, the partitioned KV paper, KTransformers, TensorFold and 05yuki's docs gave no bit-neutral lever above this one. Already on here: resident swaps, PSHUFB conversion, shared expert after submit, handoff pack. 0xSero's 19-27 tok/s is a total across 1-4 streams on a 3.0 bpw quant. Verified drafting is the only idea that could beat 2 ms, and it needs the owner's decision on speculation.

VRAM for slots, priced with `tools/replay_dyn_slots_r5.py` on the 31 Oct trace: slots hold about 17.7 GiB, and moving `embed.weight` (1.23 GiB) to the host would add about 70.

| More slots | Window ms a token | CPU routes a token |
|---|---|---|
| 0 | 15.66 | 105.7 |
| +70 | 15.41 | 103.4 |
| +140 | 15.16 | 101.0 |
| +280 | 14.64 | 96.2 |

So +70 slots is about 0.25 ms, near what two timed rounds can see, for a change to the embedding path in prefill and decode. Not worth building yet. No other tensor on the card is idle in decode: `wo_b`, `wq_b`, `wo_a`, shared experts and the head are all read every token.
