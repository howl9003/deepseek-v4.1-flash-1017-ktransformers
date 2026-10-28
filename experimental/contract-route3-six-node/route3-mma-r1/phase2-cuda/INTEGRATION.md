# Route 3 phase 2: integrating and validating the CUDA decode kernel

Claude (Fable 5.1), 28 Oct 2026, in reply to `ROUTE3_PHASE1_RESULTS_FOR_CLAUDE.md`. Local draft; nothing here has
been compiled or run (no local GPU). The kernel reproduces the sequence Codex pinned in `ROUTE3_PREPARATION_r1.json`.

## 1. What the kernel does differently from the phase 1 geometries

Phase 1's binaries had no spills and native QMMA, and even the one-warp geometry kept 16-byte asynchronous copies and
tens of KB in flight per SM; what they paid for was per-block overhead: per 32-wide K step the selected geometry
issued 7 block barriers, 9 LDS, 4 LDSM, 6 LDG and 8 LDGDEPBAR around 2 QMMA (`ROUTE3_COMPILER_ANALYSIS_r1.json`).
`route3_decode.cu` moves the per-step overhead to per 8 blocks: one `cp.async` group and one `__syncthreads` per
chunk of 8 K blocks, then per block one `ldmatrix.x4` for A, one for B, two MMAs and the eight FMUL/FFMA of the
pinned epilogue. Geometry: 2 warps per CTA, 16 output columns per warp (one 32-column weight-scale block per CTA),
grid `(N/32, 4 splits)`, 3 stages of 12.8 KB, so two CTAs per SM and roughly 50 KB in flight per SM. The 16-byte
chunks of each 256-byte smem row are swizzled by `chunk ^ (row & 7)`, which keeps every ldmatrix phase conflict free
(`test_route3_layout_local.py`, check 2).

## 2. Build and dispatch

- Build: `route3_decode.py` compiles the kernel with torch's extension loader using the venv's nvcc
  (`-gencode=arch=compute_89,code=sm_89 -fmad=false`, no fast-math). First build takes a minute; the cache lives in
  torch's extension directory. `smem_bytes()` reports 38,496 bytes per CTA (no opt-in attribute needed).
- Dispatch (private, default-off, suggested selector `SGLANG_FP8_ROUTE3_CUDA=1`), inside the `fixed_split` branch of
  `w8a8_block_fp8_matmul_triton`, before the Triton launch:

```python
if _ROUTE3_CUDA and fixed_split == 4 and route3_decode.eligible(M, N, K, As, Bs):
    partials = torch.empty((4, M, N), dtype=torch.float32, device=A.device)
    route3_decode.route3_partials(A.view(M, K), As.view(M, K // 32), B, Bs, out=partials)
    _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](partials, C, M * N, 4, 256)   # unchanged
    return C
```

  `B` is the weight tensor `[N, K]` and `Bs` its scale tensor `[N/32, K/32]`, exactly the objects the Triton call site
  already holds (its `stride_bn = K`, `stride_bk = 1`, `stride_Bs_n = K/32`, `stride_Bs_k = 1`). Batches above 16 rows
  keep the unchanged Triton kernel. Shapes outside the six verified ones keep the existing path.
- The launcher rejects misaligned views (A and B 16-byte, scales 4-byte) and non-contiguous tensors with a
  TORCH_CHECK, so a slice with an odd storage offset fails loudly instead of silently taking the Triton path.
- Scales: the kernel validates every activation and weight scale it uses (exact power of two; subnormal powers of two
  accepted; zero, sign, inf, NaN and non-power values rejected: the quantiser floor keeps zero out) and sets a device flag; `error_flag(reset=True)` reads it
  between runs. Inside a CUDA graph the check still runs; the flag is read after replay. This replaces the Python-side
  validation cost that phase 1 counted inside its timings; nothing is cached by pointer.

## 3. Exactness matrix to rerun (Codex's harness, unchanged cases)

- Raw `[4, M, N]` partials of this kernel equal the Triton kernel's partials byte for byte, before the reduce, for the
  six shapes at M in {1, 2, 4, 8, 13, 16}; the reduced outputs equal for the same cases; rows of a 16-row batch equal
  the same rows computed in Triton batches of 26, 32, 128, 538, 768 (batch invariance across the kernel boundary).
- Changed-input CUDA graph replay at batch 1, 2, 4; permutation and partition cases; the K = 1280 (10 blocks per split,
  two chunks of 8 and 2) and K = 8192 (64 blocks, 8 full chunks) edges; K = 2304 (18 per split: chunks 8, 8, 2).
- Scale validation: invalid scale sets the flag and the output is still produced; normal and subnormal powers of two
  pass; the flag is clear across the whole protocol.
- Error flag clear after every timing run.

## 4. Prices (same guarded session as phase 1, cold-weight CUDA-graph medians, 15 samples, 256 MiB displacement)

| row | kernel | M |
|---|---|---|
| 1 | this kernel + unchanged reduce | 1, 4, 8, 16 |
| 2 | phase 1 best (BN32, 2 warps, 4 stages) | 1, 4, 8 |
| 3 | existing fixed path with matched transpose | 1, 4, 8 |
| 4 | K3 CUDA-core path (different arithmetic, reference only) | 1, 4, 8 |

Report per-call medians for the six shapes, the 208-call total, bytes per microsecond against 736 GB/s. The gate is
unchanged: at or below 12 ms per 208 calls at M = 1, target at or below 10 (floor 7.07 ms for the 5.20 GB payload).

## 5. If it misses

Two tunables are compile-time constants in `route3_sequence.h` and do not touch the arithmetic: `STAGES` (3 to 4) and
`BPC` (8 to 16; 16 doubles the stage to 25.6 KB and halves the barrier count). If the kernel is still above 12 ms,
profile one shape with `ncu` for `dram__throughput` and `smsp__inst_executed` per QMMA before changing anything
else; a result near the floor on `wq_b` (K = 1280, 10 blocks per split, short pipelines) but not on the K = 5120
shapes would point at the prologue cost, cured by a persistent grid, not by geometry.

## 6. Acceptance

Unchanged from the brief: this kernel adds no rounding, so a bit-identical result against the Triton kernel means the
model candidate carries the already-accepted fixed split-K arithmetic; the full-model run still goes through the r18
protocol as a new cold context relative to K3 (dense r3 replaced), then the matched historical campaign reported as
the breadth median. `LEDGER_ENTRY_route3.md` stands; Codex's scale scan (E8M0 codes 114..121, quantiser floor 1e-10,
margin 53 bits) belongs in its evidence column.

## 7. Build switch `ROUTE3_L2_POLICY` (revision r2, 28 Oct 2026)

`ROUTE3_L2_POLICY=1` in the environment of the process that builds the extension adds `-DROUTE3_L2_POLICY=1` and names
the extension `route3_decode_sm89_l2p1`. The kernel then creates one L2 `evict_first` policy per thread and issues the
B and Bs copies with `cp.async … .L2::cache_hint`; A, As, the MMA sequence, the epilogue, the split rule and the reduce
are untouched, and the default build reduces token for token to the r1 source (`test_route3_l2policy_local.py`,
archive copy `archive/route3_decode_r1.cu`). Rationale, expected effect and the measurement plan:
`ROUTE3_PHASE2_REVIEW_r2.md`.
