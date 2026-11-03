# Context-free rows on the panel: not on our end, 2 Nov 2026

**The context-free rows are DeepSeek V4.1-Flash-1017's own behaviour.** DeepSeek's `inference/model.py`, run on the CPU with exact FP64 sums in place of every one of its kernels (no Tilelang, no Ada arithmetic, no 16-head split, no GPU reader), reproduces row 157,47 as the generic distribution: NLL 14.34 against the GPU reference's 14.32, and the same six top tokens within 0.02 nats of each other. The weights, Engram rows, embedding rows and head rows that row touches are byte-identical to the Hub. So nothing of ours is in the loop when the row goes wrong.

**What the row is.** The model turns that token into an attention-sink token, the kind the literature calls a massive activation. From layer 13 its MoE routing joins the path the lone `<｜begin▁of▁sentence｜>` token takes; at layer 20 expert 180, with 0.85 of the routing weight, blows the residual stream up (RMS 2.9 → 131 → 508 → about 5,600 by layer 37, while the neighbouring tokens stay at 0.7-3.4), on the same hidden dimensions BOS carries (2455, 4987, 568). After layer 20 the row's stream has cosine 0.95 to the BOS stream, 0.996 by layer 25. The final norm and head then read out what they read out for BOS alone: `#`, `\n`, `##`, `<`, `\n\n`. That is why the distribution is "context-free": it is the model's empty-context prior, through a token the model has decided to use as a sink.

**Cost.** Under 1% of the panel KL (the session that started this one measured it). The rows are rare: 5 in 254,846 reference positions have the generic top three with NLL above 3.

## Evidence that it is the model

Method: `cpu_ref.py` imports the checkpoint's own `model.py` and `engram.py` unmodified and replaces the six `kernel.py` functions with FP64 implementations of the sums their docstrings state (the model's own quantisation steps stay: FP8 activations per 32 with E8M0 scales, E2M1 ties to even, BF16 outputs). `blank_cpu_r1.py` runs one conversation's prefix through it layer by layer with a readout of the final norm and head after every layer. One sequence, one process, one core, experts read per expert from the GGUF. About 60-90 s a layer. Scripts and outputs under `reference-panel-r1/` locally and `/scratch/aster/bench/reference-panel-r1/` on fea01 (`cpu-trace-*.json`, `.safetensors`, `.report.txt`, `.sink.txt`).

| Row | GPU reference (Ada, Tilelang, 16-head split) | CPU exact sums | Agreement |
|---|---|---|---|
| 157,47 target " caused" | NLL 14.317; top `\n` -2.27, `#` -2.41, `<` -2.76, `##` -3.09, `\n\n` -3.72, `\n\n\n` -4.06 | NLL 14.342; `\n` -2.27, `#` -2.39, `<` -2.76, `##` -3.08, `\n\n` -3.72, `\n\n\n` -4.06 | same row, same distribution |
| 157,46 control | fine | NLL 0.55 | |
| 157,48 control | fine | NLL 0.00 | |
| 64,69 target "32" after "Vec<(i32, i" | NLL 9.185; `\n` -2.40, `#` -2.76, `<` -2.79, `##` -3.22, `\n\n` -3.85, `\n\n\n` -4.05 | NLL 9.225; `\n` -2.39, `#` -2.73, `<` -2.75, `##` -3.20, `\n\n` -3.80, `\n\n\n` -4.03 | same row, same distribution; same path (BOS's experts at layer 13 with 0.84 of the weight, collapse at 20 to RMS 23 → 397, cosine to BOS 0.91 at 20 and 0.99 at 25, 80 of 240 picks shared with BOS, neighbours 3) |
| 64,68 and 64,70 controls | fine | NLL 0.00, 0.00 | |
| BOS alone (position 0, no scored row) | not scored | top `#` -1.60, `\n` -2.34, `##` -2.93, `<` -3.39, 1897 -3.80, `\n\n` -3.81 | the generic set |

The 48-token prefix is the short case the task asked for: the served build gives the identical row at every prompt cut from 48 to 2,048 tokens (`blank-by-length-served-r1.json`, from the session that started this one), and the CPU run uses the first 56 tokens.

### The layer where the row parts from its neighbours

From `cpu-trace-157-47.sink.txt` (routing against the BOS run `cpu-trace-bos-r1`) and `.report.txt`:

| Layer | Row 47 | Controls 46 and 48 |
|---|---|---|
| 0-12 | ordinary: out RMS 0.03-0.14, 0 or 1 expert shared with BOS a layer, largest dimension 1-2 | same |
| **13** | **first parting**: experts 182 and 221, BOS's top two at this layer, take 0.82 of the routing weight; MoE output RMS 0.67 (controls 0.15); largest dimension 3136 at 11.9 (controls 5.3, 4.2) | 0 shared |
| 14-18 | 1-3 experts shared with BOS a layer, 0.9-1.3 of routing weight on them; stream RMS 0.4-0.8 | 0 shared |
| 19 | dimension 2455, the BOS massive dimension, appears at 39.8; cosine to the BOS stream 0.39 | 0.001 |
| **20** | **collapse**: expert 180 at 0.85, MoE output RMS 95, stream RMS 131, cosine to BOS 0.95 | RMS 0.8, cosine 0.003 |
| 21-25 | experts 320, 121, ... (BOS's), RMS 508 → 1,625, cosine 0.996 at 25 | 0.8-1.2 |
| 26-39 | all six experts shared with BOS from layer 36; RMS 5,500; early readout top token `#` from layer 20, `\n` from 38 | 1.4-3.4, readout converges on the real next token |

Over 40 layers the row shares 84 of its 240 expert picks with BOS; its two neighbours share 1 each.

Attention is not what collapses: the row's sink share (probability on `attn_sink`) is 0.002-0.011 at every layer, the same as its neighbours, and its attention output RMS is ordinary. The Engram deltas at layers 1 and 14 are 0.01-0.02 RMS for the row and its neighbours alike. The event is a routing decision at layer 13 followed by the sink-making experts at 20-22.

## Suspects on our end

| Suspect | Verdict | Evidence |
|---|---|---|
| `local_ckpt.py`, GGUF experts, `original-dense-*`, Engram tables | out | `verify_rows_r1.py` against the Hub at revision 1bae0e10, byte for byte: all 24 Engram hash rows at the row's position and its two neighbours in both tables (72 rows a table, weight and scale), for all six rows; every embedding row of each prefix; the head rows of the generic tokens and each target. 4,375,296 bytes, 0 mismatches (`verify-rows-r1.json`). All 240 routed experts row 157,47 hits over its 40 layers, all three projections, weight and scale, read from the GGUF through the same inversion `local_ckpt.py` uses: 1,440 tensors, 4,512,683,136 bytes, 0 mismatches (`verify-experts-157-r1.json`). The CPU run reads the same files by a different path (`pread` per tensor, per expert), so a reader fault would have to be in both readers. |
| sudoingX's 16-head `sparse_attn` split | out | The CPU path runs all 64 heads in one softmax with no split and reproduces the row. Independently, the kernel treats heads independently by construction: every reduction in `sparse_attn_kernel` is indexed by head (`reduce_max` and `reduce_sum` over the slot axis per head, `sum_exp[i] += exp(attn_sink[i] - scores_max[i])` per head, `GemmWarpPolicy.FullRow`), so a 16-head call computes each head's sum the same way a 64-head call would. |
| Ada FP8 and BF16 arithmetic, batch sensitivity | out for the row; it decides knife-edge rows | FP64 sums reproduce the row, so no tensor-core truncation is needed to produce it. For the rows that flip with MoE batch size see the next section. |
| `DiskEngramEmbedding`, Engram hash path | out | The CPU path uses DeepSeek's `NgramHashState` unmodified and its own row reads; rows byte-identical to the Hub; Engram deltas at the bad row are the size of its neighbours'. |
| Panel tokens, tokenizer | out | `panel.safetensors` sha256 39e0db2f matches the published hash (`REFERENCE_PIECES_RESULTS_r1.md`); the rows depend only on the tokens before them (identical at every prompt cut). |
| Anything else shared by every arm | out for 157,47 | The only things the CPU run shares with the GPU arms are the weight bytes (verified) and DeepSeek's `model.py` and `engram.py`. |

## Model-side suspects

| Mechanism | Finding |
|---|---|
| Engram gate `clamp_min(1e-6).sqrt()` | not involved: Engram deltas at the row are 0.01-0.02 RMS, as at its neighbours, at both Engram layers |
| `hc_split_sinkhorn` | not the trigger: the row's hc mixes are ordinary through layer 13; they move after the collapse, downstream of it |
| sparse attention with -1 slots while the window fills | not involved: sink share and attention output are ordinary at every layer; at positions below 128 every query sees every earlier token, so there is no slot selection to go wrong |
| `attn_sink` | not involved, same measurement |
| **MoE routing at layer 13, sink-making experts at layers 20-22** | **the mechanism**, measured above |

Why these rows are early in their conversations (6 of 7 generic rows at tokens 47-109, though only 4.7% of scored positions lie below 128) is not measured. A plausible reading is that the model keeps a small number of sink tokens per window and picks one early when BOS is the only one; it is a guess and I have not tested it.

## Rows that flip with arithmetic

These rows sit on the same decision. Row 152,61 (target "s" after "It’"), bad in the GPU reference at MoE batch 65,536 and in the NVFP4 emulation, fine at batch 2,048, in W4A4, on the served build and on FP4 KV:

| Layer | Row 152,61 in exact FP64 sums | Its neighbours |
|---|---|---|
| 17-18 | ordinary routing, out RMS 0.4-0.7 | same |
| **19** | routes to experts 370 (weight 1.02 of 1.5) and 259, both in BOS's six at this layer; dimension 2455 rises to 25.4; out RMS 1.6 | 0 shared with BOS, RMS 0.5-0.8 |
| **20** | expert 180, the sink maker, is **not** among its six (34, 51, 239, 358, 136, 165); RMS falls back to 1.4 and the row recovers | |
| 39 | NLL 0.00, top "s" with the runner-up at -14.6 | |

Row 76,101 (fine in the reference at batch 65,536, bad at 2,048 and in W4A4) is fine in exact sums too, NLL 0.015: at layer 13 it takes BOS's top two experts, 221 and 182, with 0.65 of its weight, shares one or two BOS experts a layer through 19 (0.35-0.76), but never gives one expert the dominant weight the collapsing rows do at layer 19, never meets expert 180 at layer 20, and its stream RMS stays at or below 1.3.

Row 136,64 (bad only on the FP4 KV build) is fine in exact sums, NLL 0.00: BOS's experts at layer 13 with 0.59 of its weight, one or two BOS experts a layer through 18, stream RMS at or below 0.9, no collapse.

Row 41,104 (fine in the reference at both batch sizes and in W4A4; generic on the served, NVFP4 and FP4 KV builds) is fine in exact sums, NLL 0.00, and is the one row that shows nothing of the path in exact arithmetic: 2 of its 240 picks coincide with BOS's, cosine to the BOS stream at most 0.01 at every layer, RMS ordinary throughout. The three builds that break it reach the same end state (their top tokens are the generic set, `\n` -2.51, `<` -2.64, `#` -2.95, `##` -3.26 on the served build), so the collapse is the same; the decision that gets them there is not one the exact trace comes near, and I have not measured where in those builds it happens.

So exact arithmetic puts 152,61 one routing decision away from the collapse: it steps onto the BOS track at layer 19 and steps off at layer 20. Six rows, different depths on the same path: 157,47 and 64,69 go all the way, 152,61 to layer 19, 76,101 and 136,64 to layer 13, 41,104 not at all. No row is on any other path, and every neighbouring token shares at most one expert a layer with BOS. Whichever side of that gate a given arithmetic lands decides whether the row is near-certain or context-free; nothing in between. That is why the reference flips it with its MoE batch size, and why one build each flips 41,104 and 136,64. Which side the row lands on is arbitrary; that it has two sides is the model's.

**How close the edge is.** The same prefix of 152 again with FP32 sums in place of FP64 (`cpu-trace-152-61-fp32`, with the gate's top eight recorded): the row ends fine again, NLL 0.00, but the two roundings do not take the same path to it.

| Layer | Row 152,61: FP32 against FP64 | Neighbours 60 and 62 |
|---|---|---|
| 2-12 | routing already differs: 4-5 of its six experts the same at layers 9-12; stream differs by 7-31% of its norm | 5-6 of 6 the same, 3-18% |
| 13 | FP32 takes BOS's 221 and 182 (0.61 of weight), FP64 one of them (0.24); stream differs by 44%; gate margin (sixth minus seventh biased score) 0.080 on scores of 11.4 | 15%, 10% |
| 17-18 | 1-2 of 6 experts the same; stream differs by 92-93%; margin at 18: 0.006 on 12.6 | 28-30%, 10% |
| 19 | both roundings route to 370 with 0.95-1.02 of the weight and light dimension 2455 (FP64 to 25.4, FP32 to about 9); margin 0.022 | |
| 20 | 0 of 6 experts the same between the roundings; expert 180 in neither top eight; margin 0.004 on 14.5 | 6 of 6 |
| 35-37 | FP32's early readout wobbles (target at -29 then -17) and recovers at 38 | |

So this is a token whose gate decisions are near ties from layer 2 onwards (margins of 0.004-0.08 on scores of 11-15), whose hidden state is as different between two roundings as its own size by layer 17, and which is one pick away from the collapse at layers 19-20. Both CPU roundings land fine; the reference's batch-65,536 rounding lands on the other side. The layer where the reference's own two settings part is not measured here: that needs the two GPU runs with per-layer dumps.

Summary of the six rows in exact sums against the arms (bad = the generic distribution):

| Row | Reference 65,536 | Reference 2,048 | W4A4 | NVFP4 | Served | FP4 KV | **CPU exact FP64** | CPU FP32 | Depth on the BOS path in exact sums |
|---|---|---|---|---|---|---|---|---|---|
| 157,47 | bad | bad | bad | bad | bad | bad | **bad, 14.34** | | collapse at 20 |
| 64,69 | bad | bad | bad | bad | bad | bad | **bad, 9.23** | | collapse at 20 |
| 152,61 | bad | fine | fine | bad | fine | fine | **fine, 0.00** | fine, 0.00 | to layer 19 |
| 76,101 | fine | bad | bad | fine | fine | bad | **fine, 0.02** | | to layer 13 |
| 136,64 | fine | fine | fine | fine | fine | bad | **fine, 0.00** | | to layer 13 |
| 41,104 | fine | fine | fine | bad | bad | bad | **fine, 0.00** | | none seen |

## The milder later class

Rows whose top token is `\n\n` (271), `<｜end▁of▁sentence｜>` (1) or `">` (3320) with NLL above 4: 149 in the reference. 146 of them are the `<think>` (91) or `</think>` (55) template token right after `<｜Assistant｜>`: 85 at tokens below 64 (the first turn), 35 at tokens 650-1,800 (later turns of multi-turn conversations). The chat template inserts those tokens; the model never predicts them, so teacher-forcing them scores NLL 8-15 there. The remaining three are ordinary high-NLL rows. Their distributions are far from the generic class (Jensen-Shannon distance 0.72-0.79 against row 157,47, where the generic rows sit 0.06-0.16 from each other), so this is a different thing: a scoring artefact of the template, not a model behaviour and not ours.

## Other checks from the session that started this one

- Served build: all six rows bit-identical at every prompt length up to 2,048 (`blank-by-length-served-r1.json`). Row 41,104 is context-free with the prompt ending just before the token, so it can reach real output; rows 76, 136 and 152 are healthy that way on the served build.
- Decode and prefill agree within 4.5e-7 nats on 512 positions on the served build (`decode-vs-prefill-served-r1.json`).

## What remains open

- The layer where the reference's own two settings (MoE batch 65,536 and 2,048) part on rows 152,61 and 76,101. The CPU pair (FP64 against FP32) did not flip either row, so it shows the sensitivity but not that flip. The GPU pair is written and smoke-tested except for the card: `blank_gpu_r1.py` re-runs exactly the group of conversations each row shared an MoE batch with (152: sequences 135-181, 64,186 tokens; 76: 62-99, 64,412 tokens; at batch 2,048 each conversation alone), since attention, Engram and the hc mixes are per sequence in the Streamer and only the MoE sees the batch, with `tap_hook`, `block_hook` and a gate wrapper dumping the row's stream, routing, gate margins and early readout every layer, and the final top 128 checked against the stored part. `run_gpu_pair_r1.sh` runs the four cases under the bench lock, about 40 minutes of card time in all. On 3 Nov the card went from the adavr2 timing chain to adavr3 and then back to the served model (down since about 01:00), so there is no free window: the pair now needs the served model unloaded, which is the owner's call. If OK'd: after `SERVING.json` exists, `curl -s http://127.0.0.1:8082/unload`, wait for the `ada-serve-*` unit to exit and the lock to be free, run the chain, then bring the model back with a chat request for `deepseek-v4.1-flash-sglang` through the gateway.
- Update from the session that started this one, 3 Nov: on the V41-window + FP4 KV + FP32-head capture (adavr1), row 41,104 is no longer context-free, while the head-only arm still has it; that build's context-free rows are 157,47, 64,69 and 136,64. That is another build landing on another side of the same near-ties, consistent with the reading above.
- Where the served, NVFP4 and FP4 KV builds make 41,104 collapse. The exact trace never comes near the path on that row. The served build has no per-layer taps; a capture overlay would be needed.
- Why the collapsing rows are early in their conversations (tokens 47-109) is not measured.
- The window-KV V41 overlay (`ada-spill-r1/v41-window-r1/`) changes K values on every layer; the early rows should be re-scored on it when it is served.
- The later-class rows (`<think>` and `</think>` after `<｜Assistant｜>`) could be excluded from the panel score, or scored as the template positions they are; that is sudoingX's panel's choice, not ours.

## What this means for the build

Nothing to fix. The rows are the model choosing an extra sink token; a build either lands on the same side of a near-tie as the reference or not, and the reference does not agree with itself across batch sizes on those rows. The six rows together are under 1% of the panel KL. A build that made many more rows collapse than the reference would be a finding; the served build has the same count as the reference (one more, one fewer, in the rows above).

## Notes

- The served model was stopped at 22:52:49 local by a `RESIDENT_STOP` file and SIGTERM, not by a memory guard: every node held 8-11 GiB free plus clean at the time (floor 4 GiB). A reload (`runs/20261102T153447Z`) was admitted at 23:34:48 while four of my CPU runs were still resident (about 26 GiB in all) and was stopped at 23:37:20 by another `RESIDENT_STOP`, with no guard message in its log. My CPU runs were pinned one per node at about 6.5 GiB each under `MemoryMax=8G`, with every file read dropped from the page cache behind it; the last one ended at 23:58 and none left a `.pyc`. At 23:58 the card held 12.6 GB from a process that is not mine and llama-swap reported nothing loaded; I did not touch the card at any point.
- Run times on one core: 35-50 min a trace at 56-112 tokens; the eight traces read 0.8 TB from the GGUF in all, dropped behind them.
- Precision wording: the reference numbers here are DeepSeek's code on Ada; "exact" above means FP64 sums on the CPU, and nothing here is a claim against C1.
