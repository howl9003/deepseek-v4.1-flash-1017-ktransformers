# Brief for Codex: route 3, dense FP8 decode on the tensor core with the prefill kernel's sequence

From Claude (Fable 5.1), 28 Oct 2026. Approved by the user as the dense route. Attachments in this folder:
`ROUTE3_MMA_DESIGN.md` (the specification), `route3_sequence_reference.py` (NumPy oracle for everything around the
MMA; self-test passes locally), `LEDGER_ENTRY_route3.md` (already inserted in `PRECISION_LEDGER.md`), `DSPARK_CAMPAIGN_r1.md` (the MTP campaign
that follows route 3, step 6 below), and `PREPARATION.json` (sha256 of each file). Nothing in this folder was run
on fea01.

## The ask

Replace the dense CUDA-core family (`SGLANG_FP8_FIXED_CUDA_CORE`) with a decode kernel that reproduces, bit for
bit, the fixed split-K tensor-core path the prefill already uses, so that decode runs at the weight-bandwidth
floor and prefill returns to the fixed split-K speed (6.2 s bounded 35K instead of 63 s). The arithmetic reverts to
the bits accepted in r18 and contract r5; no new rounding is introduced.

## Do this, in order; stop at the first failure

1. **Pin the sequence (half a day, no GPU).** Read the accepted prefill kernel's PTX for one shape and record in
   `ROUTE3_PREPARATION_r1.json`: one `mma.sync.m16n8k32.e4m3` per K block from a zero accumulator; the FP32
   epilogue instructions including any `.ftz` modifier (power-of-two scales make the multiplies exact only while
   nothing underflows, so the sequence is retained exactly rather than argued away); `SPLIT_K = 4` with
   `tiles_per_split = cdiv(cdiv(K, 32), 4)`; the unchanged `_reduce_block_fp8_split_k`. Add the power-of-two
   assertion on `As` and `Bs` to the new selector's dispatch, and record the minimum E8M0 code over the dense
   weight scale tensors from the checkpoint headers (the underflow margin in the design, section 1a).
2. **Phase 1 geometry (one guarded component session).** In the `fixed_split` branch add a default-off decode
   configuration for M <= 16: `BLOCK_SIZE_M 16, BLOCK_SIZE_N 16, BLOCK_SIZE_K 32, GROUP_SIZE_M 1, num_warps 1,
   num_stages 6, SPLIT_K 4`, then sweep `BLOCK_SIZE_N` {16, 32}, `num_warps` {1, 2}, `num_stages` {4, 6, 8}, with
   and without `SWAP_AB`. Identity first, speed second (section 5 of the design): every row equal to the prefill
   kernel's row at every M in {1, 2, 4, 8, 13, 16, 26, 32, 128, 538, 768}, changed-input graph replay at batch 1,
   2, 4, permutation and partition cases, raw `[4, M, N]` partials compared before the reduce, the five checkpoint
   shapes plus the indexer shape, K = 1280 and K = 8192 as the split-boundary edges.
3. **Prices (same session).** Cold-weight CUDA-graph medians for the current fixed path, the best phase 1
   configuration and (reference only) the original dispatch at M = 1, 4, 8; the unchanged prefill kernel at 512,
   6144 and 36864 rows. Report per-token ms over 208 calls and bytes per microsecond against 736 GB/s. Target:
   208 calls at or below 10 ms per token with the prefill component unchanged at 6.2 s. If the best Triton
   configuration stays above 12 ms, say so and stop; Claude then writes the phase 2 CUDA kernel text into
   `six-node-performance-r1/kernel-steer-r1/route3-mma-r1/phase2-cuda/` (kernel source, launcher, the pinned
   sequence as a header, CPU-side tests, and its own PREPARATION.json). That folder does not exist until that
   result arrives; until then the phase 2 design is section 4 of `ROUTE3_MMA_DESIGN.md` and the fragment map in
   `route3_sequence_reference.py`.
4. **Ledger probe (half a day, optional but wanted).** Khattak and Mikaitis's fp8 vectors and SageAttention2's
   accumulator-input test on the 4080 SUPER, to confirm 13 fractional bits and truncation on AD103. A different
   result changes `LEDGER_ENTRY_route3.md`, not the route.
5. **One guarded full-model candidate.** K3 with dense r3 replaced by this route, under the r18 protocol: cold
   repeat, eight retained splits, rotation, quality screen, paired NLL within one standard error of accepted K3,
   feature and host-reload checks exact, resource and settle receipts unchanged; then the matched historical
   performance campaign, reported as the breadth median first, Japanese second. No acceptance inferred from the
   component results.
6. **DSpark (MTP), only after step 5 passes and the kernel is final.** The user considers MTP the last big lever
   once the GPU chain is reduced, and chose "DSpark re-fitted": shipped draft weights, the verification-length
   policy refitted to the hybrid's measured cost curve. Do the refit first (`DSPARK_CAMPAIGN_r1.md` section 3a:
   forced-length cost curve T = 1..6 on the final runtime, offline acceptance traces on the twelve workloads,
   confidence-head calibration by one scalar per domain, offline policy sweep), and only then the paired arms.
   Design, prerequisites and arms are in `DSPARK_CAMPAIGN_r1.md`: decode graphs at batch 8
   and the DP4A route (or its priced fallback) at rows up to 8; the two blockers the 24 October study named closed
   (draft scope isolation, capture-safe MXFP4 LUT fix); six-row verify bit-identical to single-row decode through
   the whole layer; the draft running entirely on the CPU pool (its experts cannot be streamed over PCIe and do not
   fit in VRAM); four arms (plain at natural capacity, static block 3, gated K=5 at p_min 0.5, capacity-matched
   plain control), histories required identical to plain, breadth median first, and the observed fixed per-step
   cost at six rows against one row reported as the number that decides whether the lever has more room. Stop on
   any history mismatch or resource gate.

## What to report back

The preparation receipt with the pinned sequence, the identity and price receipts, the ledger probe result, and
the two model receipts. Say explicitly which configuration was selected and why, and give the bytes-per-microsecond
figure, not only tokens per second.

## Constraints

Default-off selectors only; frozen host-reload gate `eb31f84a…97b4`, 416 GiB no-swap, 64 GiB FEA reserve, six
nodes, 42 workers, 320 W and boot unchanged; PR6 held; no push or publish; `.bench.lock` owned for every
component; no model resident during components; failed receipts preserved, never overwritten.
