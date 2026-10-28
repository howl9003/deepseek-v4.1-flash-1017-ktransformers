# Private contract-v2 validation candidate

Codex now owns both tracks. Finish the stripped-BF16 evaluation first, then this
contract-v2 candidate. Fable's `contract-v2-r1` sources and evidence remain intact.

This directory contains a reviewed derivative, not a validated release. Build it
on the pinned main-derived six-node native source and SGLang runtime. Keep the
backend default-off, PR6 held, and the host-reload test unchanged. Do not publish.

Changes under review:

- Use `ldexp` in GPU projection epilogues, matching the CPU's exponent handling.
- Restrict the no-shared-expert BF16 cast to the enabled contract path.
- Reject oversized or malformed Python route submissions before native work.
- Correct truncation accounting for shifts of 63 bits and above; refuse runtime
  execution after entering a regime where partition invariance is unestablished.
- Reuse the corrected physical NUMA mapping and expert-arena placement from the
  stripped-BF16 work, with arithmetic switches for that path disabled.
- Isolate strict floating-point compilation and scoped round-to-nearest/gradual
  underflow inside contract-v2 arithmetic. The surrounding extension uses
  fast-math; its existing backends retain their settings. A native synthetic MoE
  test requires nonzero subnormal outputs even when worker FTZ/DAZ starts enabled.
  Function isolation uses [GCC's function option pragmas](https://gcc.gnu.org/onlinedocs/gcc/Function-Specific-Option-Pragmas.html).

Validation sequence:

1. Guarded private build after the stripped-BF16 controller releases the host.
   Rebuild the CPU oracle from source and revalidate it; the old prototype binary
   has changed since earlier GPU evidence and is not reused as an assumed oracle.
2. Native 1/2/6-part comparisons and early bounds checks, CPU/GPU epilogue edge
   cases, then captured GPU stage comparisons in both exact modes.
   Run the actual layer with contract v2 disabled and require all 20 preserved
   stripped-BF16 output files to match bit for bit.
3. Real SGLang FusedMoE plus authenticated GGUF loading: all-CPU and fixed-three
   GPU placement, per-route equality, batch invariance, shared-expert combination.
4. Separate matched end-to-end layer timings, including quantization, conversion,
   dispatch, transfer and completion. Compare with the preserved stripped-BF16
   component measurements. Exact contract agreement is distinct from reference
   model quality and from historical whole-model throughput.
5. If justified by component results, a separate pinned full-model candidate with
   the same benchmark, reference NLL, quality and unchanged host-reload checks.

Every remote compute process must hold the shared benchmark lock and observe the
resource limits. CPU nodes 3 and 7 stay reserved for FEA; memory may span all eight
nodes. Reserve at least 64 GiB globally. Model cap 416 GiB, ready target 392 GiB,
no swap. No power or boot changes, no Qwen restore, no recurring automation.

`prepare_private.py` records the initial derivation; subsequent reviewed edits are
in the shipped overlay. Build manifests hash the final files actually used.
`run_validation.py` runs the guarded component sequence, stops at its first
failure, and never starts a full model. Full-model staging follows result review.
