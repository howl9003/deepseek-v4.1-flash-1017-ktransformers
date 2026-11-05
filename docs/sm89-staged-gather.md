# Optional SM89 staged gather

`patches/sglang-optional/02-sm89-staged-gather.patch` (applied by hand on the full stack, see
[patches.md](patches.md#optional-patches)) adds one default-off switch, scoped to the SM89 attention:
`SGLANG_SM89_ATTN_STAGED=1`. Up to eight query rows (decode) the dual-scope gather runs as three
kernels instead of one: the scores, the probabilities and the output, each on every (head block,
token block) of the call. One row then spreads over the whole card instead of four CTAs. In an online
softmax only the running maximum and the FP32 folds carry order, so each value goes through the same
operations in the same order as the unsplit kernel, and the output is bit for bit the unsplit
kernel's. It reads the V4, V41 and V41_FP4 caches ([sm89.md](sm89.md#the-kv-cache-and-the-logits)).

With the switch on, those rows leave `SGLANG_SM89_ATTN_SPLITK=auto`'s split-K for the staged kernels;
an explicit `SGLANG_SM89_ATTN_SPLITK=2`, `4` or `8` still wins. So the A/B is the switch against the
default.

Measured by howl9003 on one RTX 4080 SUPER, on their own SM89 build with
`SGLANG_SM89_ATTN_SPLITK=0` (the unsplit kernel), not against `auto`:

- the gather 39.2 -> 20.5, 55.5 -> 26.4 and 19.6 -> 13.3 us a call on three decode shapes under
  CUDA graph replay (about 9 us of each is launch overhead), about 0.9 ms a token; decode, median of
  six workloads, 31.25 -> 32.0 tok/s; 35K prefill unchanged;
- exact against the unsplit kernel: 3,000 fuzzed calls (half with NaN and infinity inputs), 1,000 on
  V41 and V41_FP4 caches, and 11,496 scored positions of a model run.

It moves to tier 0 once it has an A/B against `SGLANG_SM89_ATTN_SPLITK=auto` on an SM89 GPU.
`tools/sm89-kv-layout-test.py` checks it against the unsplit kernel (`pair`) and against BF16-row
twins (`attention`) when the patch is applied.
