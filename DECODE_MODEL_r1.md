# Decode time for the plain 1017 build

A token crosses 40 mixture-of-experts layers. At one row, the layer computes the attention and dense path, routes six experts, forms GPU and CPU partials, adds the shared expert and reduces the int64 sum across the two ranks. The model rounds once after the reduction. The final projection, collective and sampler run once per token. The head uses FP32 logits.

## Calibrating one row

On table B, the median of 31 interior decode reports was 47.98 tokens/s, or 20.84 ms/token. The matched CPU intervals gave 274 µs per layer for queue plus run, including a median queue wait of 5 µs. The head, all-gather and sampler budget from the reference trace was 1.30 ms. The FP32 head kernel contributed 0.81 ms of that budget.

| Component | Time per token | Basis |
|---|---:|---|
| 40 CPU calls | 10.96 ms | Measured service on the test server |
| Remaining layer path | 8.58 ms | Residual, including GPU work, synchronization and transfers |
| Head, collective and sampler | 1.30 ms | Reference trace |
| Total | 20.84 ms | Interior decode median |

Calculate the residual per layer as `(20.84 − 1.30 − 40 × 0.274) / 40 = 0.2146 ms`. Across repetitions its range was 192–232 µs. A queue wait can grow when the other lane is using the workers. Do not count that wait twice in a two-lane estimate.

## Extending to concurrent requests

Two decode lanes fork at the start of a step and join before sampling. Both use one CPU worker pool. The pool serializes their CPU calls while the GPU streams can overlap other work. At two sessions the split is 1+1, at four it is 2+2. Component measurements without lanes gave 218.5, 424.4 and 902.9 µs for one, two and four CPU rows. The integrated build's CPU service and cache share vary with traffic.

To turn aggregate rate into effective step time use `1000 × sessions / aggregate tokens per second`. For the three 512-token trials at each width, table B gave 21.08, 29.94 and 52.51 ms at 1, 2 and 4 sessions. This includes startup of each request. The interior one-row estimate above excludes it.

| Sessions | Table B measured tok/s | CPU service used in the A projection | A projected tok/s before throttling |
|---:|---:|---|---:|
| 1 | 47.43 | 40 × 274 µs queue + run | 50.0 (47.7–52.2) |
| 2 | 66.79 | 40 × 297 µs run exposed by the fitted overlap | 69.6 (68.5–70.4) |
| 4 | 76.17 | 80 × 593 µs run at the pool limit | 83.7 (81.7–85.7) |

Table A takes approximately 90% of table B's CPU service time. Apply the factor to the exposed calls and leave the other stages fixed. The sensitivity intervals vary the measured rates, call medians and the call-scale factor. A's measured single-session soak was 48.42 tok/s, with three memory thermal assertions. A matched throughput ratio at each width remains an open measurement.

At three sessions B measured 73.82 tok/s. At three and four sessions, lower CPU service coincided with higher throughput than the preceding plain build. No individual switch was isolated in those comparisons. Refit against the same traffic and cooling state before projecting to another machine.
