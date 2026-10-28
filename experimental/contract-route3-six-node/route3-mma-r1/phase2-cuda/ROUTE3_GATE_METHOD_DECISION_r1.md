# Route 3 gate method decision (r1)

28 Oct 2026. Human decision ("ok adopt and forward step 4"), relayed by Claude (Fable 5.1). Context:
`ROUTE3_PHASE2_RESULTS_FOR_CLAUDE_r3.md` (sha256 d392a3a0…, verified) and `ROUTE3_PHASE2_REVIEW_r2.md` (2f0b784f…).

## Decision

1. **Gate method.** The component gate for the route 3 decode kernel is measured with a read-only displacement: the
   same 256 MiB pass over the displacement buffer, reading only (a reduction such as `.sum()` whose result is
   consumed), followed by the same synchronization outside the events. Everything else of the original method is
   unchanged: cold weights, outer CUDA events, 15 samples per cell, six shapes weighted by their 208-call counts,
   M = 1/4/8/16, quantization + dispatch + native partials + unchanged reduce inside the window, correctness checks
   outside it. Threshold unchanged: M = 1 at or below 12.0 ms per 208 calls, target 10.
   Reason: the read-modify-write displacement leaves the 64 MB L2 full of dirty lines that every kernel, including
   the CUDA-core control, pays to evict (2.447 ms recovered by an untimed read sweep, report r3 row 3). A production
   dense call follows read-dominated kernels, so that dirty footprint is not part of what the gate was meant to
   measure. Weights stay cold under either displacement.
2. **Candidate.** The default build: r1 source `952bb68e…`, build 11, SASS identical to build 3 (`c336cdda…`).
   The `ROUTE3_L2_POLICY` switch stays in the package, default off, not selected (exact, no robust speedup).
3. **Record.** Original method: default 12.765 ms, policy 12.732 ms, not met and kept on record as such. Default with
   the untimed read sweep: 10.318 ms. Chain of 208 projections: 8.817 ms.
4. **Next.** The brief's step 5, one guarded full-model candidate (`CODEX_BRIEF_route3.md`), with the default build.
   The user and Claude called this "step 4" in Zulip; the brief numbers it 5 (its step 4 is the optional ledger probe).

## Order for Codex

A. Price the default build under the adopted method (read-only 256 MiB displacement) in one guarded component
   session, M = 1/4/8/16, identity checks before timing as before, and record it as the gate row. Expected about
   10.3 ms at M = 1. If it is above 12.0 ms, stop and report before step 5 (fail closed).
B. Step 5 as written in the brief: K3 with dense r3 replaced by the route 3 default build, under the r18 protocol
   (cold repeat, eight retained splits, rotation, quality screen, paired NLL within one standard error of accepted
   K3, feature and host-reload checks exact, resource and settle receipts unchanged), then the matched historical
   performance campaign reported as the breadth median first, Japanese second. No acceptance inferred from component
   results. Standing constraints unchanged: six compute nodes / 42 workers, 64 GiB FEA reserve, 416 GiB no-swap cap,
   320 W, no root/sysctl/boot/power changes, never kill the server first, no second server, a new receipt namespace
   per retry, never overwrite failed-run evidence, PR6 held, no push or publication, host-reload gate SHA
   `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4` unchanged, selectors default-off outside the
   candidate's own run. Report the two model receipts and the breadth median. No step 6 (DSpark) without a further
   human decision.

## Not changed

The arithmetic contract, the unchanged reduce, and `LEDGER_ENTRY_route3.md` (route 3 bit-identical to the fixed
split-K path; its row moves from proposed to candidate only after step 5's identity checks).
