# Shared input digits, 2 Nov 2026

**Breadth 33.1 (33.15, 33.05) against 32.98 (33.05, 32.90): about 0.1 ms a token, inside run-to-run noise. Kept default off.** Selector `KT_CONTRACT_V2_SHARED_X`, trial `ada-perf-shared-x-20261102-r2`, six nodes, 320 W, two rounds, FEA job 1716 running on its own nodes.

At one token every route in a part reads the same input row, so one quantisation serves them all by pointer. Same digits, so the bits should not move, but no identity run was made since the fix is not being kept alone.

| Measure | Engram build | Shared digits |
|---|---:|---:|
| Breadth | 32.98 | 33.1 |
| Japanese | 35.1, 35.8 | 35.1, 35.8 |
| 35K prefill | 69.50, 55.46, 51.93, 50.80 s | 69.85, 55.62, 51.91, 50.84 s |

The estimate was 0.07-0.25 ms. Quantisation of x was already parallel, at about 10 us a call, so only the n term goes. The hoped-for cache gain did not show.

Patch `apply_shared_x.py`, native `ada-native-shared-x-20261102-r1`. Worth including only as part of a batch with the other two CPU fixes, where the total could be measured.
