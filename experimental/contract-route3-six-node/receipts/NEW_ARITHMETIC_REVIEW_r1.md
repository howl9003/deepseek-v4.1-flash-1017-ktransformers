# Numerical review for new arithmetic candidates

Fable's kernel steer permits a new fixed reduction order to establish its own cold context. This applies to a new dense family and, if the actual-input observation justifies it, a fixed output head. It does not permit drift between cold, retained, concurrent or host-reloaded execution of that candidate.

The existing 14 model checks remain required: completed benchmark/client lifecycle, clean owned cleanup, unchanged source/gate pins, authenticated reference inputs, the unchanged quality threshold, exact early cold repeat and eight retained splits, exact session/abort replays, and the complete unchanged host-reload protocol with its four exact comparisons. The resident-rotation regression must also preserve the candidate's own cold output before/after rotation and across the retained split.

The extra byte-identity assertions in the r4 lender-only review establish that a lifecycle repair did not change arithmetic. They are specific to that repair, not a replacement for reviewing a new arithmetic candidate. Preserve the r4 receipts and its failure; do not edit its review or the frozen tests to make a later candidate pass.

For Fable's additional NLL requirement, compare the candidate with the corresponding accepted build on the authenticated 24 pieces of 479 scored positions. Report each paired piece mean, the mean NLL increase and the paired standard error (sample standard deviation across pieces divided by sqrt(24)). Use one standard error as written in the steer: hold a candidate whose positive mean degradation exceeds that value. Do not substitute a token-level standard error, round away a tiny positive excess, or increase the multiplier after seeing results. Report distance and standard error against the pinned GPU reference separately. The existing quality threshold remains 0.02 nats; this additional comparison does not replace it.

The performance profile remains separate from the four-session numerical profile. Use the historical fast parameters with six nodes and 42 workers, record all additional differences, include conversion/quantization costs, and measure actual decode and long-prefill throughput. A component price estimate is not proof of 30 tok/s.

All candidates remain isolated and default-off. PR6 stays held. Host-reload gate SHA: `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`. No power/boot changes, no swap, 416 GiB model cap and 64 GiB FEA reserve.
