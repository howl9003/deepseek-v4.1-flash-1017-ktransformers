# Route 3 phase 2 results for Claude — revision 2

All eight variants pass the full597-case identity matrix and361 scale-flag checks before78 price cells each. None passes the unchanged12ms isolated-call gate. The useful new finding is that the original and two-stage kernels cost8.817ms and8.594ms when208 distinct-weight projections share one captured replay, against12.780ms and12.680ms in separately launched graphs from the same diagnostic. This is a measurement/execution-context finding, not a promoted gate pass or a full-model speedup.

## Compiler and identity

Each variant preserves zero-C QMMA.16832.F32.E4M3.E4M3, separate FMUL(D,As) followed by accumulating FFMA(product,Bs,acc), no FTZ, no local spills or stack, fixed natural K32 order and four partitions, and the unchanged Triton reduce. BPC4 has8 static MMA sites and32 multiply/FMA pairs; BPC8 has16/64; BPC16 has32/128. SASS reviews1–8 pin actual binaries and instruction sequences. Strict reviews1–8 pin raw receipts. Each597 matrix includes raw partials before reduce, row/batch invariance, partition/permutation, public/default-off dispatch, larger-prefill boundary, changed graphs and invalid/changed scales. This is the same matrix on each variant, not4776 independent input scenarios.

## Original-method cold per-call price

Milliseconds per208 calls;15 samples per shape/batch/method. Quantization, complete dispatch, native partial workspace and unchanged reduction included.256MiB displacement outside events. Rows are independent runs; tiny differences need a matched comparison. All M1 values exceed12ms.

| Candidate | Registers/thread | Shared bytes/CTA | M1 | M4 | M8 | M16 |
|---|---:|---:|---:|---:|---:|---:|
| Original BPC8/S3 | 72 | 38496 | 12.790016 | 12.880640 | 13.220352 | 13.621248 |
| BPC8/S4 | 72 | 51328 | 14.816512 | 14.947584 | 15.063296 | 15.712256 |
| BPC16/S3 | 64 | 76992 | 14.626304 | 14.649344 | 15.046656 | 15.745280 |
| BPC4/S3 | 94 | 19248 | 13.103872 | 13.328128 | 13.611776 | 14.155520 |
| BPC4/S4 | 94 | 25664 | 12.914944 | 13.117696 | 13.424640 | 13.873920 |
| Live-row epilogue, BPC8/S3 | 64 | 38496 | 12.967680 | 13.057280 | 13.339136 | 13.764352 |
| BPC8/S2 | 72 | 25664 | 12.704256 | 12.882944 | 13.156608 | 13.546240 |
| Persistent grid, BPC8/S3 | 128 | 38496 | 12.964608 | 13.068288 | 13.328640 | 13.833472 |

The live-row candidate skips only unstored padded-row scalar epilogues; both MMAs remain warp-uniform. The persistent candidate preserves every original (split,column tile), uses a fixed160-block grid for large shapes, and drains pending copies plus a CTA barrier before shared-stage reuse. Both are exact but slower here. BPC8/S2 is header-only; it does not establish a material speedup in isolated-call measurements. No candidate is installed in production.

## Matched stage and boundary evidence

Stage diagnostic:126 exact checks,108 timing cells,72 layout observations. No contiguous copy for q/As/B/Bs in any of six shapes at M1/4/8. Each M1 graph has precisely quantize/native/reduce. Internal inter-kernel gaps total0.016176ms/208 calls, so they do not explain a millisecond-scale fitted intercept. Instrumented quant/reduce durations total0.245/0.211ms; these are not standalone stage prices.

Boundary diagnostic:156 exact checks,72 timing cells. Legacy outer12.65920ms; adding captured event nodes gives outer13.20192ms and inner11.94752ms on those marked replays. Added nodes perturb timing. The inner number never replaced the gate. Standalone stage medians are not additive. The linear fit in review1 is descriptive and does not establish that all fixed/marginal cost belongs to wrappers; one high-bandwidth capture does not prove every in-graph kernel is already at the same rate.

## One captured208-projection component diagnostic

This uses208 separate weight/scale allocations, with byte-exact clones of six checkpoint archetypes (5,200,199,680 useful bytes total), independent M1 inputs, and a fixed interleaved synthetic order. Inputs are changed and restored through each graph. Three implementations share exactly the same inputs, payload addresses, cold displacement and original outer-event method;15 samples per cell in randomized round-robin order. The isolated total weights one representative call per archetype; the chain visits all208 distinct allocations and input rows. It is a matched implementation comparison within each workload, not a208-pair causal isolation of graph grouping. No profiler/correctness checks occur inside timing.

| Implementation | Sum of isolated-call medians, ms | One208-projection graph, ms |
|---|---:|---:|
| original_b8s3 | 12.780032 | 8.816640 |
| b8s2 | 12.679936 | 8.594432 |
| current_core | 12.395264 | 8.144512 |

Strict review passes2930 checks (416 raw,624 eager,624 graph base,624 changed,624 restored,18 isolated-vs-chain),20 zero flags,21 timing cells and208 unique weight/scale allocations. The CUDA-core control is checked against its own arithmetic, never claimed bit-identical to Route3. Host peak0.639GiB; CUDA peak5.285GiB; no swap or cgroup memory events. No model weights beyond these bounded dense archetypes are loaded.

The gap combines execution conditions and amortized launch/timing boundaries; it is not proof of4ms removable work in production, and no single cause is isolated by this comparison. Synthetic weights repeat archetypes at distinct addresses; actual layer weights, attention/MoE, dependencies, CPU host callbacks and KV are absent. No end-to-end throughput prediction or full-model gate result follows.

## Existing runtime already uses full decode graphs

Read-only pinned K3 PROFILE.json selects decode backend `full`, batches1/2/4, prefill disabled. `runner/decode_cuda_graph_runner.py` lines1399–1468 wraps the model forward in `run_once`; `runner_backend/full_cuda_graph_backend.py` lines198–215 captures `forward_fn()` and lines225–233 replay one graph. Source hashes and excerpts are saved in ROUTE3_GRAPH_BOUNDARY_AUDIT_r1.json. This establishes configured/source behavior, not a new live model capture audit. Simply recommending enabling full graphs would be redundant. CPU/MoE work and host nodes can still affect within-graph conditions; that is not measured by the synthetic chain.

## Proposed next decision

Stop blind geometry sweeps. Review whether the original per-call12ms price is appropriately predictive under the already-configured full decode graph, without changing that gate. Propose one concrete next component implementation/measurement that can improve the unchanged original-method price or localize the remaining kernel/context cost. If another kernel is justified, compacting the padded activation/shared-stage layout is an untested direction that preserves every MMA and rounding boundary; it requires an updated lane/layout proof and is not yet implemented. Fused reduction is not justified by the measured ~1us reduce plus tiny gap, and must not be introduced silently.

Codex retains runtime ownership/validation. Claude can return a local-only decision/design/package. No further full-model run is allowed in this phase2 step; no threshold or host-reload edit is authorized by these results.

## State and evidence

All owned build/validation services are terminal; latest chain and persistent validation have MainPID0/success. GPU and shared bench lock verified free.320W/boot settings unchanged, Qwen not restored. Six compute nodes0,1,2,4,5,6;42workers;64GiB FEA reserve; model416GiB/no-swap and component guards unchanged. PR6 held, no push/publication, selectors default-off. Host-reload SHA `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4` reverified. Full-model breadth medians remain strippedBF16=16.0tok/s and contract=16.55tok/s;30+ is not demonstrated.

Evidence root local `six-node-performance-r1`, remote `/scratch/aster/bench/six-node-performance-r1`:

- `ROUTE3_CHAIN_PRICES_REVIEW_r1.json` SHA256 `38491dff99f374953390bccabad3c36997039e76599a1f7e8349fcb34cb6c7d7`.
- `ROUTE3_GRAPH_BOUNDARY_AUDIT_r1.json` SHA256 `01d500a5396c368c1e2cdb7cacbdd7cf140972bce9ae0589a9537dd8a6cf9c45`.
- `ROUTE3_STAGE_AND_TIMING_RESULTS_FOR_CLAUDE_r1.md` SHA256 `5bd11b285c995acbc19c39b15f50d98618f5617da8eccc54e5a277bd314f1785`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r1.json` SHA256 `c637923cb0ce7202c4ee4f75c433830a679a71a9606ecaf26030ede16ded5f97`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r2.json` SHA256 `685cf9562d513ca16f954d6c6c3663a0f2c5ad45238d2fec06134e13b9a0b23d`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r3.json` SHA256 `983daffb6efb0cbe3fa944de19bcf6f272935b5ac58cfcbe5e2888663967a673`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r4.json` SHA256 `77215817b7d39d4e0d8169727a364eb8ab876f973cf2ad2d4ca4bf15bccb343f`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r5.json` SHA256 `2486c1e507c1cc6c7cb134d310a4c14de2fc0a9e9ad2b54c4bae96417d14bf1c`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r6.json` SHA256 `49b8a3dcac5ac1b704d52ee12f867a8fb0eb9f4828bc672baf611ede20a8cf54`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r7.json` SHA256 `848adbd6d92588d0f7d701befe37009ba6eb2c50cf5fd7e4d5ffe1cff8b89a9e`.
- `ROUTE3_PHASE2_STRICT_REVIEW_r8.json` SHA256 `0cf0fd0c23d80fffa10e5372d8e8bf57911d09a705a64a44d356908a49ca3183`.
- Chain raw RESULT SHA256 `08a15d39d0c06da283f132642da6b18c5c2cf70412daff288b77a0c878415005`; local under `route3-phase2-work-r1/route3-chain-prices-r1`, remote `route3-chain-prices-r1`.
- Previous counter report preserved separately; one-shape ncu receipt and failed extension-only postcheck retained. No duplicate profiler run.
