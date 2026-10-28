# Handover to Codex: Route 3 phase 2 CUDA decode kernel

From Claude (Fable 5.1) via the user, 28 Oct 2026. Reply to `ROUTE3_PHASE1_RESULTS_FOR_CLAUDE.md` (phase 1 best
27.446 ms per 208 calls at M = 1 against the 12 ms gate; 7,216 exactness checks passed; GPU and lock released).

## Package

Folder `six-node-performance-r1/kernel-steer-r1/route3-mma-r1/phase2-cuda/` (local copy; copy it next to the
route3-mma-r1 package on fea01, the test imports `../route3_sequence_reference.py`).

| file | bytes | sha256 |
|---|---|---|
| `route3_sequence.h` | 2829 | `e2f7fccc9b167d4a390e4ecf24a5c5d8f921a8f654ffed047f4ca303982f3995` |
| `route3_decode.cu` | 13994 | `952bb68ea79c28caea889028ba1fcbe7cc79c77e22a5c42546b95388aade63d1` |
| `route3_decode.py` | 2393 | `18bbaba811ce206fb95ed02238f02f54c7394de2a61514fc6ad3f8b8e7f3428c` |
| `test_route3_layout_local.py` | 5809 | `7f6ccffead4d2dfdf940aad2e616ee80d8fae7e006012778b364acb99255a460` |
| `INTEGRATION.md` | 5943 | `58dadbf6f5d5c611b9e5c5a489c99ce6567da6f831dbe526ba8574017b8fd6f6` |

Read `INTEGRATION.md` first; `PREPARATION.json` is the manifest. Nothing in the folder has been compiled or run on a
GPU: the only local evidence is the CPU-side layout self-test (`SELFTEST_PASS`, five checks).

## What the kernel preserves (your pinned sequence, unchanged)

- One `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` per 32-wide K block, zero accumulator, natural k order.
- Epilogue `t = mul.rn.f32(D, As)`, `acc = fma.rn.f32(t, Bs, acc)` as explicit `__fmul_rn` / `__fmaf_rn`, built with
  `-fmad=false`, no fast-math, no ftz.
- Four fixed K partitions by `tiles_per_split = cdiv(cdiv(K, 32), 4)`; partials `[4, M, N]` float32 from 0.0f.
- The reduce is the unchanged Triton `_reduce_block_fp8_split_k`, called from the host side.
- Scales are validated in-kernel (every As and Bs used; exact power of two, subnormal accepted; zero, negative, inf,
  NaN, other values set a device flag). Nothing is cached by pointer; graph replay with changed inputs re-validates.

What changed is only geometry: grid (N/32, 4 splits), 2 warps x 16 columns, one cp.async group and one block barrier
per 8 K blocks, 3 swizzled stages (38,496 B per CTA, two CTAs per SM), `ldmatrix.x4` fragments.

## Steps, in order

1. Build with `route3_decode.py` in the trial venv (`-gencode=arch=compute_89,code=sm_89`). Keep the SASS: confirm
   QMMA.16832 with RZ accumulator inputs, FMUL then FFMA in the epilogue, no FFMA contraction of the scale multiply,
   no spills. Record `smem_bytes()`.
2. Raw-partial identity before any reduce: this kernel's `[4, M, N]` against the Triton kernel's partials, byte for
   byte, six shapes, M in {1, 2, 4, 8, 13, 16}; then the reduced outputs; then the batch-invariance rows (rows of a
   16-row batch against the same rows in Triton batches of 26, 32, 128, 538, 768). The full raw-partial and
   public-dispatch matrix from phase 1, unchanged, plus changed-input graph replay at batch 1, 2, 4, and the scale
   flag cases (invalid scale flags and still produces output; flag clear across the protocol).
3. Only after step 2 is clean: prices in the same guarded session as phase 1 (cold-weight CUDA-graph medians, 15
   samples, 256 MiB displacement), the four-row table in `INTEGRATION.md` section 4. Gate at or below 12 ms per 208
   calls at M = 1, target at or below 10 (floor 7.07 ms for the 5.20 GB payload).
4. If it misses: `STAGES` 3 to 4, then `BPC` 8 to 16 (both in `route3_sequence.h`, neither touches the arithmetic),
   re-running step 2 after each change; then one `ncu` capture of the K = 1280 shape before any other change.
5. No full-model run in this step. Model promotion follows the brief's step 4 (r18 protocol, new cold context
   relative to K3, breadth median) only after the gate passes and the ledger entry is updated with the SASS facts.

## Standing constraints

Default-off selector only; PR6 held; nothing pushed or published; host-reload gate and provenance checks
unchanged (kv_checks.py SHA256 eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4); six compute
nodes, the 64 GiB FEA reserve, resource guards and power settings untouched; new receipt per attempt, never
overwrite failed-run evidence.

## Report back

`ROUTE3_PHASE2_RESULTS_FOR_CLAUDE.md` with: build log and SASS facts, the identity matrix counts, the price table,
and the flag state; if a check fails, the first failing case with both partial vectors.
