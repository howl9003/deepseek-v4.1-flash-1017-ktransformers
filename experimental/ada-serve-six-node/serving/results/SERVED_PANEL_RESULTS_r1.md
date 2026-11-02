# The served build on the public panel, 2 Nov 2026

**The 32.98 build scores KL 0.0111 and top-1 96.74% against DeepSeek's reference on sudoingX's panel. That is 1.11 times the reference's own noise floor.** NLL is +0.0007 nats. So the Ada kernels, Route 3 dense, contract v2 experts and the Engram path together sit close to what changing only the reference's MoE batch size does.

Panel: 199 conversations, 254,846 scored positions, sha256 39e0db2f, p4_kld.py's KL (native top 128 plus one tail bucket). Trial `ada-panel-engram-20261101-r1`, 32.98 settings, teacher-forced through `/generate`, top 256 kept per position. Scorer `score_panel.py`, result `adapanel-score-r1.json` on fea01.

| Arm | KL | KL p99 | Top-1 | NLL | Perplexity |
|---|---|---|---|---|---|
| Reference on Ada | | | | 0.47077 | 1.6012 |
| Reference, MoE batch 2,048 (noise floor) | 0.00996 | 0.123 | 97.01% | 0.47105 | |
| **Served 32.98 build** | **0.01106** | **0.137** | **96.74%** | **0.47150** | **1.6024** |
| NVFP4 release, emulated | 0.01668 | 0.217 | 96.09% | 0.47270 | 1.6043 |
| Best public pack, sudoingx 2.8 bpw, against its GB10 reference | 0.0198 | | 95.94% | | |

By domain, KL / top-1:

| Domain | Positions | Floor | Served |
|---|---|---|---|
| code | 165,450 | 0.0070 / 97.41% | 0.0077 / 97.15% |
| agentic | 16,686 | 0.0119 / 96.71% | 0.0137 / 96.43% |
| chat | 17,976 | 0.0217 / 95.34% | 0.0246 / 94.95% |
| knowledge | 14,093 | 0.0171 / 95.52% | 0.0181 / 95.30% |
| long | 16,457 | 0.0144 / 96.96% | 0.0158 / 96.78% |
| math | 10,625 | 0.0079 / 96.76% | 0.0087 / 96.74% |
| multilingual | 8,620 | 0.0148 / 97.09% | 0.0167 / 96.68% |
| writing | 4,939 | 0.0218 / 95.44% | 0.0247 / 94.74% |

## Reading it

- **Near the floor, not at it.** Every domain runs 1.06-1.15x the floor. A measured excess of about 0.0011 is real on this many positions, though I have not computed its standard error by conversation for this arm.
- **The numerics are not where the loss is.** NVFP4 and the published packs are 1.7-2x the floor; ours is 1.11x. This is the case for keeping the integer contract.
- **This is the first number that says so.** Identity runs say exact vs C1, our accepted build. This one is against DeepSeek's code running on Ada.

## What it does not cover

- **Top 256, not the full distribution.** A native id outside our kept top 256 takes our smallest kept probability, which overstates Q there. That applied to 138,489 ids out of about 32.6 million, 0.4%, and it nudges KL down a little. The floor and NVFP4 arms were scored on the full distribution.
- **Prefill path.** Teacher forcing runs the panel through prefill, not decode. The check against decode crashed in this capture on an off-by-one of mine: the forced prompt left out the last generated token. An earlier version of this note blamed a short conversation, which was wrong. Run separately on 2 Nov on the serving profile: eight conversations at 256 tokens and eight at 1,024, 32 tokens each. All 512 positions agree within 4.5e-7 nats, about one float32 step of the logprob, and 246 are bit-equal. So this score applies to decode.
- **Reference on Ada.** DeepSeek's kernels running on an Ada card, not Hopper or Blackwell. On sudoingX's GB10 the same panel reads NLL 0.47031, ours 0.47077.
- **One capture,** performance-profile settings, one session. Seven conversations repeated on the serving profile a day later match this capture bit for bit.

## Context-free rows

A few rows predict `<`, newline, `#` and `##` in nearly the same proportions wherever it happens, although the next token is obvious, for example "Yellow leaves can be" followed by " caused". Reference 3 rows, this build 3, FP4 KV 5, all between tokens 47 and 104, which hold 4.7% of the panel. Two rows are common to all three. They cost under 1% of the KL.

- **The reference does it too.** Changing only its MoE batch size turns row 76,101 bad and row 152,61 good.
- **They are deterministic and depend only on earlier tokens.** On the serving profile, six rows asked with the prompt cut at up to twelve lengths gave identical bits at every length, including with nothing after the row.
- **So it reaches real output.** Row 41,104 is context-free with the prompt ending right there, so a reply could hit it. It is not an effect of teacher forcing.

The cause is open: see `BLANK_ROWS_RESULTS_r1.md` when that report lands. Everything run so far shares this box's weights, tables and card, so this does not show the model itself does it. Evidence: `blank_rows.py`, `kl_outliers.py`, `blank_by_length.py`, `decode_prefill_check.py`, `blank-by-length-served-r1.json` and `decode-vs-prefill-served-r1.json` on fea01.
