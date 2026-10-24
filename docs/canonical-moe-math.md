# Canonical routed-MoE arithmetic

Changing a prefix from cold prefill to cached decode can change where its routed
experts execute. CPU, resident GPU, hot GPU and streamed GPU previously used
different arithmetic and accumulation orders. Correct payload bytes alone did not
make those paths numerically identical.

`KT_MOE_CANONICAL_MATH=1` opts into a common route contract. Every backend emits
unweighted BF16 expert outputs. The adapter combines them in logical top-k order
using FP32 fused multiply-add, then rounds once to BF16. The GPU expert kernel
matches the AVX2 group-32 reduction, activation and eight-part down-projection
rounding. Router logits use a fixed reduction tree independent of batch size.
The native route API is separate from the existing default-off API.

The corrected adapter reinterprets packed int8 weights and E8M0 scales as bytes
without conversion or stale cached aliases. Main's static hot routing leaves
resident IDs for the old native skip mask; the adapter explicitly excludes those
routes from CPU ownership. Each active route must have exactly one owner.

The accompanying prefill-lend repair restores all four expert tensors, including
small scale tensors, when resident experts are rebound. The request-prefix policy
is an explicit prerequisite of this validation campaign; see
[prefix cache policy](prefix-cache-policy.md). Host-reload acceptance is unchanged.

Apply the `canonical-moe-math` tier-1 feature and rebuild kt-kernel with the recipe's
native patch. The environment switch defaults off. The validated experimental
layout is one GPU rank, hidden size 5120, expert width 2304, top-6 routing, eight
CPU NUMA parts, AVX2 MXFP4, and zero deferred experts. Streamed prefill requires
eager zero-copy W16 staging. This is a correctness experiment, with no speed claim.

The candidate is based on main `e8608a224c108c06c8dbe636bf0bfef295e82161`, with
SGLang `8b9f72417fa85a04de6193b0a52f4cec63370d77` and ktransformers
`95009ea6856c0799e517e93cb12be5e8494bc7ce`. It contains no PR5 overlap/packing,
poll-hint, or retained-session admission changes.

Main-based component checks pass exact CPU/GPU route and combination comparisons,
live storage and rebinding, static hot routing, changed-input CUDA graph replays,
router batch/permutation invariance through 4,224 rows, and unchanged default-off
native outputs. The guarded full-model run stopped during attention initialization because the reused
runtime lacked main's `sgl-flash-mla==1.0.0` dependency. Numerical and host-reload
checks were not reached. Publication remains held pending a completed validation
with all of main's runtime requirements.
