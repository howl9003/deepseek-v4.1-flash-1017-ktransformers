// Contract v3 CPU kernels, arena / row-major input mode (spec r2). C ABI, no OpenMP: every entry point computes a
// column range (or a row / block range) on the calling thread, so kt-kernel's own worker pool can drive it.
//
// Weights stay in the checkpoint layout that kt-kernel's arena already holds (the same bytes are exported to the GPU):
//   nibbles: row n at b + n * row_bytes, 32-block g at byte offset 16 * g, low nibble = even k, high nibble = odd k;
//   scales:  FP32 exact powers of two [N][G] in the arena, converted ONCE at load (v3_rm_scales_to_codes) into E8M0
//            codes in panel order [N/8][G][8] (the same n*G bytes the contract v2 BufferB already reserves).
// Each (8-column panel, 32-block) is transposed from the 8 rows into the tier's lane order in registers, once per
// block, and reused for all M rows (M <= 4 in registers; M > 4 from an L1/L2 buffer of the panel's decoded blocks).
// The arithmetic is the frozen oracle's (oracle_v3.py spec r2): gate/up = FP32 chain over all blocks in ascending
// order then BF16 RNE; down = exact int64 partial over any block subset on the layer grid bL; finish = f32_rne(S)*2^bL
// then BF16 RNE. Every entry point sets MXCSR to RNE without FTZ/DAZ for its own duration and restores it.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "v3_cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { V3_ERR_SCALE_FORMAT = 6 /* an arena scale is not an exact power of two in [2^-127, 2^127] */ };

typedef struct {
  int N, K, G;           // output columns (multiple of 8), reduction length (multiple of 32), K / 32
  const uint8_t* b;      // checkpoint-order nibbles, row stride row_bytes (>= K / 2)
  size_t row_bytes;
  const uint8_t* s;      // panel-order E8M0 codes [N/8][G][8]
} v3_rm_mat;

// FP32 scales [N][G] (exact powers of two; 2^-127 is the FP32 subnormal 0x00400000 = E8M0 code 0) -> panel-order codes.
// Returns V3_ERR_SCALE_FORMAT (and writes nothing further) at the first scale that is not one.
int v3_rm_scales_to_codes(const float* d, int N, int G, uint8_t* s_panel);
// Same from row-major E8M0 bytes [N][G] (tests, GGUF).
int v3_rm_e8m0_to_codes(const uint8_t* e8m0, int N, int G, uint8_t* s_panel);

// Quantiser on BF16 rows: x [M][ldx] (first K values), q int16 (C) / int8 (D) [M][ldq], ea [M][ldea] (K / 32 used).
int v3_quantise_bf16(const uint16_t* x, int M, int K, int ldx, int tier, void* q, int ldq, int32_t* ea, int ldea);

// Gate / up: columns [n0, n1) (multiples of 8), all M rows, FP32 chain over all G blocks ascending.
int v3_rm_gemm_bf16(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                    int n1, uint16_t* out, int ldo);
int v3_rm_gemm_acc(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M, int n0,
                   int n1, float* acc, int ldacc);   // pre-BF16 accumulators (tests)

// SwiGLU (table SiLU, two FP32 multiplies, BF16 RNE) on intermediate blocks [g0, g1) of gate/up [M][ldgu]; writes
// xd (BF16, may be null) and, if q is non-null, the down-input quantiser output for those blocks.
int v3_swiglu_quant(const uint16_t* gate, const uint16_t* up, int ldgu, int M, const float* w_route, const float* silu,
                    int g0, int g1, int tier, uint16_t* xd, int ldxd, void* q, int ldq, int32_t* ea, int ldea);

// Down: S [M][ldS] (columns n0..n1) += exact int64 partial over the listed blocks (blocks == NULL: all G).
int v3_rm_down_partial(const v3_rm_mat* w, int tier, const void* q, int ldq, const int32_t* ea, int ldea, int M,
                       int bL, const int* blocks, int nb, int n0, int n1, int64_t* S, int ldS);

// Finish: y = bf16_rne(f32_rne(S) * 2^bL) for rows [0, M), columns [n0, n1); comb (may be null) += floor(y * 2^-bL).
int v3_finish_rows(const int64_t* S, int ldS, int M, int n0, int n1, int bL, uint16_t* y, int ldy, int64_t* comb,
                   int ldc);

#ifdef __cplusplus
}
#endif
