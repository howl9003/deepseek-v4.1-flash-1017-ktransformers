Route3 step 5 completed: **breadth median 17.80 tok/s**, versus accepted K3 16.55 tok/s. Japanese median **20.95 tok/s**, versus K3 19.10. Full numerical review **28/28 PASS**; matched historical performance review **23/23 PASS**. PR6 remains held. No push, publication, DSpark or further variant was run.

| Workload | Accepted K3 tok/s | Route3 tok/s |
| --- | ---: | ---: |
| Chinese prose | 17.40 | 18.50 |
| English prose | 16.60 | 17.90 |
| English technical prose | 15.40 | 16.50 |
| math | 16.50 | 18.10 |
| JSON | 16.40 | 17.70 |
| Python code | 16.70 | 16.50 |

The breadth-median ratio is 1.0755x (+7.55%). This is a descriptive comparison of single completed campaigns, not a statistical speedup claim. The older stripped-BF16 campaign's breadth median was 16.00 tok/s. The historical 30–35 tok/s build used a different node count and has incompletely recorded environment/power state; it is not a controlled topology-only comparison.

35K prefill times: Route3 [181.54, 162.8] seconds; K3 [264.57, 240.25] seconds. The actual recorded historical launch, expert-placement and VMM identities were audited; 90 recorded environment entries match. Both accepted K3 and this candidate use six compute nodes/42 workers for these reports. Numerical acceptance uses a separate cache/concurrency profile; its ~14.4 tok/s benchmark is not the breadth result above.

The post-timing 8-token repeat reports text_equal=True and selected_logprobs_exact=True. It is descriptive only; the separate 28-check numerical run provides full acceptance evidence.

The selected route is the **default build 11**, policy 0: pinned native CUDA decode for M<=16 reproducing the accepted fixed split-K tensor-core partials; the prefill kernel, four split boundaries and reduction remain unchanged. CPU and routed MoE retain the accepted K3 implementation. Binary SHA256 `51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649`; scheduler origin and file hash checked before/after both model runs. Performance source/native inventories are identical to numerical validation. Selectors remain default-off outside these isolated candidates.

The explicitly adopted read-only 256 MiB component method passed:597 identity checks, 307 flag checks, 24 timing cells, 15 samples each. M1/4/8/16 prices are 10.030080/10.086400/10.101760/10.411520 ms per 208 calls. Threshold 12 ms passed; target 10 ms was not met. Useful payload throughput is 518460.4 bytes/us, including quantization, public dispatch, scale checks, native partials and reduction. This is useful payload divided by elapsed time, not a DRAM-counter measurement. The old read-modify-write method remains on record as failed: default 12.764928 ms and policy 12.732160 ms. Dirty-line writeback remains a hypothesis, not a uniquely established cause. The default candidate was explicitly selected in human-decision relay43; the L2-policy variant was not selected. Both had failed the old gate, and this campaign makes no claim of a production benefit from the policy hint.

Cold repeat, all eight retained splits, rotation, all seven feature replays, abort recovery and host reload pass exact token/selected-logprob comparisons. Paired mean NLL increase against accepted K3 is **0.000448332**, below one paired-piece standard error **0.003729299** over 11496 positions. Quality-screen NLL change is **-0.011089090** over 12276 positions, passing the unchanged 0.02 threshold.

This is not model-level bit identity to K3 or the pinned GPU reference. Against the pinned GPU reference, candidate NLL=0.718320155, reference=0.714111252, mean increase=0.004208903, mean absolute selected-logprob difference=0.124506166, maximum=8.348575652. That reference distance is descriptive; the predeclared paired acceptance test is against accepted K3. The Route3 arithmetic has its own cold context.

Resource contract remains six CPU nodes [0,1,2,4,5,6], 42 workers, shared RAM, 64 GiB FEA reserve, 416 GiB/no-swap, 320 W; no root/sysctl/boot/power changes. Host-reload gate SHA256 remains `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`.

Cleanup evidence is retained without suppressing warnings:

- Numerical: owned scope empty after cleanup; 1 recorded initial cleanup exception(s). TimeoutExpired(['systemctl', '--user', 'stop', 'codex-sm89-feature-trial-B-contract-route3-numerics-six-r1-model-contract-route3-numerics-r1.scope'], 15)
- Performance: owned scope empty after cleanup; 1 recorded initial cleanup exception(s). TimeoutExpired(['systemctl', '--user', 'stop', 'codex-sm89-feature-trial-B-contract-route3-fast-six-r1-model-contract-route3-perf-r1.scope'], 15)

Remote model receipts:

- `/scratch/aster/bench/feature-trial-B-contract-route3-numerics-six-r1/RUN-contract-route3-numerics-r1.json` — SHA256 `3edb57461002093307b2e5b8126dd2d13fa71c7f8e2898089c104dfd9121ed8d`.
- `/scratch/aster/bench/feature-trial-B-contract-route3-fast-six-r1/RUN-contract-route3-perf-r1.json` — SHA256 `addc9726aedd934a14d6da997fe8c8007efa0a4ca7e077624b459e5f7290353e`.

Local authenticated receipt bundle: `route3-step5-evidence-r1.tar.gz`, SHA256 `cef874d28c8f9a995208f67c4d56e4bd6c2ae9938cb8a131e6b397498b65a7c2`. Numerical review SHA256 `a1eada43a1887b51ffadb510c48d97bccc038dab9c627bc0a9a7ab4952a7ca9a`; performance review SHA256 `f7fbb31ecc2fafe89085465fd551fb7e2ad2eb4ead49bd9f836ba6ca535a22d7`. Full evidence mapping is in the bundle's EXPORT_MANIFEST.json.

Requested report-and-stop point reached. No further run is queued. The broader 30+ tok/s goal is not demonstrated by this campaign; this report does not promote PR6 or authorize DSpark.
