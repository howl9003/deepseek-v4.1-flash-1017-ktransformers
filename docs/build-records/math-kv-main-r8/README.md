# Main-based math/KV build record (R8)

This branch records an experimental build, not an accepted release or an upstream
pull request. The tested recipe source is commit `e269c463f76ceaa1308f5c8937155183461dd87f`,
tree `f32435e39afaf6d179683b82d9e658c30e786a08`, directly on upstream main
`e8608a224c108c06c8dbe636bf0bfef295e82161`. Later commits on this branch add records only.

## Differences outside PRs 5, 6, 7 and 8

1. Canonical routed-MoE arithmetic: fixed router reduction, matching CPU/GPU
   unweighted expert outputs, fixed route accumulation, live packed-byte views
   and explicit route ownership. Implemented by tier-1 patch 19 and the native
   kt-kernel patch; the runtime gate defaults off.
2. Complete expert-payload restoration in tier-1 patch 13, including both scale
   tensors when GPU resident experts are rebound after streamed prefill.
3. Explicit prefix TTL/cache-class and SWA retention policy in tier-1 patch 18,
   with its feature prerequisites, contracts and documentation.

The source change is 11 recipe files, including supporting tests, CI selection and
feature dependencies. None of PR5 overlap/packing, PR6 admission/ownership,
PR7 SM89 live-top-k/merge, or PR8 dynamic-hot refinements is included in this
isolated build. Main plus those four PRs is a different configuration. Their exact
heads and changed files are listed in [SOURCE_DELTA_AUDIT.json](SOURCE_DELTA_AUDIT.json).

## Runtime and validation boundary

Pinned SGLang: `8b9f72417fa85a04de6193b0a52f4cec63370d77`.
Pinned ktransformers: `95009ea6856c0799e517e93cb12be5e8494bc7ce`.
All 211 main runtime requirements and actual attention imports passed preflight.
Three missing dependencies were supplied in a private runtime overlay:
`sgl-flash-mla==1.0.0`, `sgl-deep-gemm==0.2.3.post2`, and `tml-renderers==0.1.0`.
These already belong to main's requirements; they are not extra feature patches.

The guard harness, origin observer, input fixtures and machine-specific launch
profile are validation infrastructure, not changes to the serving recipe.
The model uses a 440 GiB/no-swap scope, one SM89 GPU, eight NUMA pools and 56 workers.
All 542 resolved server arguments matched the frozen configuration. Source,
client and workflow hashes were verified before starting the single R8 attempt.

The host-reload gate is unchanged: `kv_checks.py` SHA256
`eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`.
It still requires a positive host reload of at least one page and retained user
KV, alongside its other protocol checks. Completion is not inferred from NLL.

## Current result

R8 passed startup with the corrected runtime, 24-piece NLL recording/reference
comparison and the longer-prompt quality screen. The initial retained/cold exact
history replays still change selected token IDs. Numerical equivalence has not
passed; the upstream math/KV draft remains held. The full attempt's terminal
receipt will be recorded separately after its remaining checks and cleanup.

The older R7 startup failure described in `docs/canonical-moe-math.md` remains
historical evidence; this record supersedes its current-status paragraph.
