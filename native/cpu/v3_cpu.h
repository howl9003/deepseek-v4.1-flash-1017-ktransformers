// Contract v3 CPU expert kernel (tiers C and D), AVX2 + FMA3 (Zen3). Spec: ../oracle_v3.py (frozen r1).
// Tier C: int16 activations per 32 (QMAX 32767), vpmaddwd.  Tier D: int8 per 32 (QMAX 127), vpmaddubsw.
// Both: exact int32 block dot -> cvtdq2ps (exact, |dot| < 2^24) -> vfmadd231ps(dot, 2^(ea+ew-1), acc), ascending K.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { V3_TIER_C = 0, V3_TIER_D = 1 };
enum { V3_OK = 0, V3_ERR_SCALE_OVERFLOW = 1, V3_ERR_SHAPE = 2, V3_ERR_FIXED_OVERFLOW = 3, V3_ERR_NONFINITE = 4,
       V3_ERR_DOWN_OVERFLOW = 5 /* r2: s - bL + 24 > 62 or |S| >= 2^62 */ };

// Packed weight matrix [N, K]: panels of 8 output columns.
//   w: [N/8][G][128] bytes; s: [N/8][G][8] UE8M0 bytes.  128 B per (panel, 32-block) in a tier-specific nibble order.
typedef struct {
  int N, K, G, tier;
  uint8_t* w;
  uint8_t* s;
} v3_mat;

// codes: [N][K] E2M1 nibble codes (0..15), e8m0: [N][G].
int v3_pack_codes(v3_mat* m, const uint8_t* codes, const uint8_t* e8m0, int N, int K, int tier);
// ggml MXFP4 blocks [N][G][17]: byte 0 = UE8M0, bytes 1..16: element j low nibble, element j+16 high nibble.
int v3_pack_gguf(v3_mat* m, const uint8_t* blocks17, int N, int K, int tier);
void v3_free(v3_mat* m);

// Quantise M rows of FP32 x [M][K]. q: int16 [M][K] (tier C) or int8 [M][K] (tier D); ea: int32 [M][K/32].
void v3_quantise(const float* x, int M, int K, int tier, void* q, int32_t* ea);

// FP32 accumulators acc [M][N] (row stride ldacc) for output columns [n0, n1), multiples of 8. Returns V3_OK or error.
int v3_gemm_acc(const v3_mat* m, const void* q, const int32_t* ea, int M, float* acc, int ldacc, int n0, int n1);

// Spec r2 down projection: S[M][ldS] (columns n0..n1) += sum over the listed 32-blocks of floor(dot * 2^(s - bL)),
// s = ea + e8m0 - 128, exact int64. Any partition of the blocks, summed in any order, gives the same S.
int v3_down_partial(const v3_mat* w2, const void* q, const int32_t* ea, int M, int bL, const int* blocks, int nb,
                    int n0, int n1, int64_t* S, int ldS);
// y = bf16_rne(f32_rne(S) * 2^bL); if combine is non-null, combine[i] += floor(y * 2^-bL).
int v3_finish(const int64_t* S, int64_t n, int bL, uint16_t* y, int64_t* combine);
// One NUMA part (spec r2): gate/up/SwiGLU/quantiser for the intermediate columns of its down blocks, then the int64
// down partial over those blocks for all H outputs, ADDED into S [M][H]. Optional gate/up/xdown: only its columns.
int v3_expert_part(const v3_mat* w1, const v3_mat* w3, const v3_mat* w2, const uint16_t* x_bf16, int M,
                   const float* w_route, const float* silu, int bL, const int* blocks, int nb, int nthreads,
                   int64_t* S, uint16_t* gate, uint16_t* up, uint16_t* xdown);

// One routed expert, whole (spec r2: one part with all down blocks, then the finish), on nthreads OpenMP threads.
//   x_bf16 [M][H]; w_route [M]; silu [65536] FP32 table indexed by BF16 bits.
//   partial [M][H] int64: F(y) = floor(y * 2^-bL) is ADDED (the device's exact combine partial).
//   Optional outputs (may be null): y [M][H] BF16 bits, gate/up/xdown [M][I] BF16 bits.
int v3_expert(const v3_mat* w1, const v3_mat* w3, const v3_mat* w2, const uint16_t* x_bf16, int M,
              const float* w_route, const float* silu, int bL, int nthreads, int64_t* partial, uint16_t* y,
              uint16_t* gate, uint16_t* up, uint16_t* xdown);

// BF16 values -> int64 floor(y * 2^-bL), added into acc.  Returns V3_ERR_FIXED_OVERFLOW if |value| >= 2^62.
int v3_fixed_add(const uint16_t* y, int64_t n, int bL, int64_t* acc);

#ifdef __cplusplus
}
#endif
