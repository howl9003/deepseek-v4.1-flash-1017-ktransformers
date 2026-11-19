# Expert placement, VRAM and thermal balance

Placement affects the share of decode routes served on the GPU. That share changes CPU service per layer and, at high concurrency, the utilization of the shared CPU pool.

## From free bytes to routes

One expert slot holds half an expert on each GPU. Weight and scale tensors total 9,400,320 bytes per card or 17.93 MiB across both. Allocation granularity raised the measured physical cost to about 9.23 MiB per slot per card in one 687-slot allocation. The entire non-resident set would take about 268.2 GiB across the GPUs. Use the live allocation after accounting for checkpoints, graph pools, KV, scratch and the configured spare floor.

Across representative decode traffic the mean lend was 569 slots across 40 layers, about 14.2 per layer. The table gives the measured-route replay's percentage of routes served on GPU with the serving policy. Count the resident route in this metric. Expert reuse and the admission delay make the benefit per added slot diminish.

| Sessions and lane split | 4 slots/layer | 14 | 28 | 57 | 100 |
|---|---:|---:|---:|---:|---:|
| One | 19.6% | 38.3% | 52.0% | 67.9% | 80.9% |
| Two, 1+1 | 15.3% | 32.2% | 45.4% | 62.1% | 77.0% |
| Four, 2+2 | 12.2% | 28.1% | 41.2% | 58.3% | 73.9% |

For GPU share `s`, predict `6 × (1 − s)` CPU routes per row and layer. The observed ratios of distinct experts to CPU routes were 0.992 at two rows and 0.972 at four. At one row a marginal fit for table B gave 50.7 µs per CPU expert plus about 98 µs fixed. A sensitivity fit gave 72.6 µs per expert plus 25 µs fixed. Refit both intercept and slope against measured service times on a different CPU.

## Choosing a slot budget

On the measured topology an expert upload occupied about 0.35 ms of link time. An asynchronous swap also contended with serving. The fitted incremental cost was about 0.44 ms per swap at one session, 0.18 ms at two and 0.40 ms at four. At one session roughly nine avoided CPU routes were needed to repay a swap. Estimate the number of routes served *after* admission. Count evictions and cancelled pending admissions.

For the four-session split the crossover out of pool-limited execution was near 59% GPU share, with a sensitivity range of 53–72%. With the serving policy that was about 59 slots per layer, or 41 GiB across the cards. The test server's two 16 GB cards had about 6.2–6.4 GiB flexible space each in decode-only periods. Reserve room for KV growth and prefill before choosing a slot count.

## Partitioning CPU work

Each of eight NUMA parts gets whole blocks of 32 expert rows. Table A gives each part nine blocks, or 288 rows. Table B uses 10,10,10,10,6,6,10,10 blocks. It reduces memory load on two parts and raises the longest part's work by approximately 10/9. Scale the exposed CPU service by this ratio, then recompute the step with its overlap. The decode model gives the measured B rates and conditional A estimates.

Thermal state changes the result. The A soak at one session recorded three assertions at 85 °C. The B soak reached 81 °C on the monitored P2 bank and sustained four sessions without an assertion. A throughput comparison requires matched session count, output length, duration, prompt mix and starting DIMM temperature.
