// route3_sequence.h - the pinned per-row sequence of the dense FP8 projections (Route 3, phase 2).
//
// Pinned by Codex from the accepted prefill kernel's PTX/SASS (ROUTE3_PREPARATION_r1.json, 28 Oct 2026):
//   * K is cut into 32-wide blocks t = 0..T-1 (T = K/32); SPLIT_K = 4 fixed partitions, split s owns blocks
//     [s*ceil(T/4), min((s+1)*ceil(T/4), T)).
//   * Per block: ONE  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32  with a ZERO accumulator input.
//     The instruction's k index 0..31 is the block's k in natural order (no permutation inside the block).
//   * Per block, in FP32, per output element:  t = mul.rn.f32(D, As[m,t]);  acc = fma.rn.f32(t, Bs[n/32,t], acc)
//     (no .ftz, no contraction of the mul: the product is a multiplicand of the fma, never an addend).
//   * Each split's accumulator starts at 0.0f and is stored to a [4, M, N] float32 workspace;
//     the four partials are reduced by the UNCHANGED Triton kernel _reduce_block_fp8_split_k, which also casts.
//   * Rows beyond M inside the 16-row MMA tile do not influence real rows (the fused adder is per element).
// Anything outside this list (tile sizes, warps, stages, swizzles, chunking, CTA order) is free.
#pragma once
#include <cstdint>

namespace route3 {

constexpr int BLOCK_K        = 32;   // one MMA per block, k32
constexpr int SPLIT_K        = 4;    // fixed by the sequence
constexpr int TILE_M         = 16;   // the MMA's M; decode rows M <= 16 share one tile
constexpr int SCALE_GROUP_N  = 32;   // one weight scale per 32 output rows (32x32 blocks, K side == BLOCK_K)

// Geometry (free parameters, chosen for bandwidth at M <= 16; see INTEGRATION.md for the sweep)
constexpr int WARPS          = 2;                 // warps per CTA
constexpr int THREADS        = 32 * WARPS;
constexpr int COLS_PER_WARP  = 16;                // two n8 MMA tiles per warp
constexpr int BN             = WARPS * COLS_PER_WARP;   // 32 output columns per CTA = one weight-scale block
constexpr int BPC            = 8;                 // K blocks per chunk (one cp.async group, one barrier)
constexpr int CHUNK_BYTES    = BPC * BLOCK_K;     // 256 bytes of K per row per chunk
constexpr int CHUNKS16       = CHUNK_BYTES / 16;  // 16-byte copies per row per chunk
constexpr int STAGES         = 3;                 // chunks in flight per CTA

inline __host__ __device__ int tiles_total(int K) { return (K + BLOCK_K - 1) / BLOCK_K; }
inline __host__ __device__ int tiles_per_split(int K) { return (tiles_total(K) + SPLIT_K - 1) / SPLIT_K; }

// Swizzle of the 16-byte chunks inside one 256-byte smem row: physical = logical ^ (row & 7).
// Makes every ldmatrix phase (8 rows of one 8x16-byte tile) hit 8 distinct bank groups.
inline __device__ int swz(int row, int chunk16) { return chunk16 ^ (row & 7); }

}  // namespace route3
