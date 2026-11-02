// patha_decode.cu - Path A kernel 1: exact integer decode GEMV (M <= 8) for the dense FP8 projections and the
// hot-slot experts of DeepSeek V4.1-Flash-1017 on sm_89 (DESIGN_r1.md section 1; oracle contract_a.py).
//
// Arithmetic (must equal the oracle bit for bit, in this order):
//   activation  n[m,k]  int32, |n| < 2^18 (E4M3 code -> N_OF_CODE * sign, 2^-9 units; prep kernel)
//   dense       w[n,k]  int32, |w| < 2^18 (E4M3 code -> same table, SW = 9)       products < 2^36, int64 lane sums
//   expert      w[n,k]  int32, |w| <= 12  (E2M1 nibble -> W_INT, halves, SW = 1)    products < 2^22, int32 lane sums
//   lane partial: 16 consecutive k of one weight row against each of the M activation rows (mad.wide.s32 chain for
//   dense, IMAD chain for experts); the two lanes of a group combine by one shfl.xor into the exact group dot D_g.
//   per group:  sh = ea[m,g] + ew[n,g] - B[m,n],  B = EA[m] + EW[n] - MARGIN;  A[m,n] += sh >= 0 ? D_g << sh : D_g >> -sh
//               (arithmetic shift, floor; right shifts clamped at 63 which is the exact floor for |D| < 2^63).
//               MARGIN per kind (CONTRACT_FIXES_r1.md H1): 13 dense, 20 expert (contract v2, locked). Dense:
//               |A| <= G * 32 * 229376^2 * 2^13 < 2^61.62 for K <= 8192, expert |A| < 2^54.4: no row sum, split
//               partial or merge can overflow a signed int64 on a legal input; the overflow flag is a defence only.
//   metadata:   sh <= MARGIN for every staged group (ea_g <= EA and ew_g <= EW); a larger shift means a stale EA / EW
//               table and sets the meta flag (L5) instead of silently changing bits.
//   split-K:    grid (N/32, S); split s owns groups [s*tps, min((s+1)*tps, G)), tps = cdiv(G, S) (route 3's rule);
//               int64 partials P[S, M, N]; merge_finish sums the S partials in index order and finishes.
//   finish:     f = __ll2float_rn(A) (one direct rounding; never through double); e2 = B - SA - SW (SA + SW = 18
//               dense, 10 expert); the finish rule (M1): A == 0 -> +0; biased exponent of f plus e2 in 1..254 -> the
//               normal f * 2^e2 formed on the exponent field (bit-identical to ldexpf there); above 254 -> signed
//               infinity and the finish flag; below 1 -> signed zero and the finish flag. Unreachable on legal inputs
//               (quantiser amax floor 1e-4 so EA >= -22; weight scale codes >= 54 dense / 53 expert, enforced at load).
//               bf16 = RNE(y) as contract_v2.bf: (u + 0x7fff + ((u >> 16) & 1)) >> 16 (identical to __float2bfloat16_rn).
// Flags (plain stores of 1u, no atomics; read with error_flag(reset)): bit 0 int64 accumulation overflow (GEMV or
//   merge), bit 1 finish outside the normal range (overflow or underflow under the rule), bit 2 NaN activation code,
//   bit 3 exponent metadata inconsistent. The kernels never silently round: every flagged case still produces an
//   output, and the flag is checked between runs.
//
// Data movement (route 3's phase 2 geometry): one CTA of 2 warps per 32 output rows, 256-byte (dense) or 128-byte
// (expert) cp.async chunks of 8 groups per weight row, 16-byte pieces swizzled by chunk16 ^ (row & 7), the M x 256
// activation integers of the chunk staged alongside (M KB), the group exponents of the chunk staged as 4-byte copies
// from the 4-aligned-down group index, three stages in flight. Lane -> (row = (lane & 7) | ((lane >> 4) << 3),
// half = (lane >> 3) & 1): every 8-lane LDS.128 phase reads 8 distinct rows at one logical 16-byte piece (conflict
// free under the swizzle), every 16-lane LDS.64 phase 8 rows x 2 halves (all 32 banks once), and the activation
// reads of a phase are one address (broadcast). Partner lane = lane ^ 8.
//
// Build: nvcc -O3 -gencode=arch=compute_89,code=sm_89 -fmad=false (patha_decode.py); no fast-math anywhere.
// Macros (microbenchmark knobs, every variant is exact): PATHA_DENSE_LUT (1 = 256-entry int32 smem table replicated
// PATHA_LUT_REPL times, interleaved so lane l reads replica l % R; 0 = six-op arithmetic form), PATHA_EXPERT_LUT
// (1 = 16-entry int32 smem table replicated 32 times, conflict free; 0 = prmt byte-table form), PATHA_STAGES.
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef PATHA_DENSE_LUT
#define PATHA_DENSE_LUT 1
#endif
#ifndef PATHA_LUT_REPL
#define PATHA_LUT_REPL 8
#endif
#ifndef PATHA_EXPERT_LUT
#define PATHA_EXPERT_LUT 1
#endif
#ifndef PATHA_STAGES
#define PATHA_STAGES 3
#endif

namespace patha {

constexpr int MARGIN_DENSE  = 13;                 // path A dense (H1): int64-safe for K <= 8192
constexpr int MARGIN_EXPERT = 20;                 // contract v2 (locked)
constexpr int SA            = 9;                  // E4M3 activation integers are in 2^-9 units
constexpr int BLOCK_K       = 32;                 // scale group
constexpr int WARPS         = 2;
constexpr int THREADS       = 32 * WARPS;
constexpr int ROWS_PER_WARP = 16;
constexpr int BN            = WARPS * ROWS_PER_WARP;   // 32 output rows per CTA = one dense scale block
constexpr int BPC           = 8;                  // groups per chunk (one cp.async group, one barrier)
constexpr int STAGES        = PATHA_STAGES;
constexpr int MAX_M         = 8;
constexpr int X_CHUNK_INTS  = BPC * BLOCK_K;      // 256 activation integers per row per chunk
constexpr int X_CHUNKS16    = X_CHUNK_INTS * 4 / 16;   // 64 sixteen-byte copies per activation row per chunk
constexpr int EXP_WORDS     = 3;                  // 4-byte exponent words per row per chunk (covers off + 8 <= 12)
constexpr int EXP_ROW_BYTES = 16;
constexpr int LUT_REPL      = PATHA_LUT_REPL;
constexpr int EXPERT_LUT_REPL = 32;
static_assert(LUT_REPL >= 1 && LUT_REPL <= 32 && (32 % LUT_REPL) == 0, "replication must divide the warp");
static_assert(STAGES >= 2, "two stages minimum");

inline __host__ __device__ int tiles_total(int K) { return (K + BLOCK_K - 1) / BLOCK_K; }
inline __host__ __device__ int tiles_per_split(int K, int S) { return (tiles_total(K) + S - 1) / S; }
inline __device__ int swz(int row, int chunk16) { return chunk16 ^ (row & 7); }

// Weight formats. BYTES_PER_GROUP: weight bytes per row per 32-wide group. ROW_SCALES: one E8M0 scale per (row, group)
// (experts) against one per (32-row block, group) (dense).
struct Dense {
  static constexpr int  BYTES_PER_GROUP = 32;
  static constexpr int  CHUNK_BYTES     = BPC * BYTES_PER_GROUP;   // 256
  static constexpr int  CHUNKS16        = CHUNK_BYTES / 16;        // 16
  static constexpr int  SW              = 9;
  static constexpr int  MARGIN          = MARGIN_DENSE;
  static constexpr bool ROW_SCALES      = false;
  static constexpr int  EW_ROWS         = 1;
  static constexpr int  LUT_INTS        = PATHA_DENSE_LUT ? 256 * LUT_REPL : 0;
  static __host__ __device__ size_t row_bytes(int K) { return static_cast<size_t>(K); }
};
struct Expert {
  static constexpr int  BYTES_PER_GROUP = 16;
  static constexpr int  CHUNK_BYTES     = BPC * BYTES_PER_GROUP;   // 128
  static constexpr int  CHUNKS16        = CHUNK_BYTES / 16;        // 8
  static constexpr int  SW              = 1;
  static constexpr int  MARGIN          = MARGIN_EXPERT;
  static constexpr bool ROW_SCALES      = true;
  static constexpr int  EW_ROWS         = BN;
  static constexpr int  LUT_INTS        = PATHA_EXPERT_LUT ? 16 * EXPERT_LUT_REPL : 0;
  static __host__ __device__ size_t row_bytes(int K) { return static_cast<size_t>(K) / 2; }
};

template <int MT, class F>
struct __align__(16) Stage {
  uint8_t W[BN][F::CHUNK_BYTES];          // weight bytes, 16-byte pieces swizzled by row
  int32_t X[MT][X_CHUNK_INTS];            // activation integers of the chunk, natural order
  uint8_t EA[MT][EXP_ROW_BYTES];          // ea bytes [a0, a0 + 12) with a0 = t_first & ~3 (int8 exponents)
  uint8_t EW[F::EW_ROWS][EXP_ROW_BYTES];  // weight scale codes (E8M0 uint8) of the same group window
};
template <int MT, class F>
__host__ __device__ constexpr size_t smem_bytes_for() {
  return sizeof(Stage<MT, F>) * STAGES + static_cast<size_t>(F::LUT_INTS) * 4;
}

// --------------------------------------------------------------------------------------------------- tables
// m of an E4M3 code: N_OF_CODE[c] * SIGN_OF_CODE[c] (contract_v2); NaN codes 127 and 255 map to 0 as in the table.
__device__ const int32_t DENSE_M_OF_CODE[256] = {
        0,       1,       2,       3,       4,       5,       6,       7,       8,       9,      10,      11,      12,      13,      14,      15,
       16,      18,      20,      22,      24,      26,      28,      30,      32,      36,      40,      44,      48,      52,      56,      60,
       64,      72,      80,      88,      96,     104,     112,     120,     128,     144,     160,     176,     192,     208,     224,     240,
      256,     288,     320,     352,     384,     416,     448,     480,     512,     576,     640,     704,     768,     832,     896,     960,
     1024,    1152,    1280,    1408,    1536,    1664,    1792,    1920,    2048,    2304,    2560,    2816,    3072,    3328,    3584,    3840,
     4096,    4608,    5120,    5632,    6144,    6656,    7168,    7680,    8192,    9216,   10240,   11264,   12288,   13312,   14336,   15360,
    16384,   18432,   20480,   22528,   24576,   26624,   28672,   30720,   32768,   36864,   40960,   45056,   49152,   53248,   57344,   61440,
    65536,   73728,   81920,   90112,   98304,  106496,  114688,  122880,  131072,  147456,  163840,  180224,  196608,  212992,  229376,       0,
        0,      -1,      -2,      -3,      -4,      -5,      -6,      -7,      -8,      -9,     -10,     -11,     -12,     -13,     -14,     -15,
      -16,     -18,     -20,     -22,     -24,     -26,     -28,     -30,     -32,     -36,     -40,     -44,     -48,     -52,     -56,     -60,
      -64,     -72,     -80,     -88,     -96,    -104,    -112,    -120,    -128,    -144,    -160,    -176,    -192,    -208,    -224,    -240,
     -256,    -288,    -320,    -352,    -384,    -416,    -448,    -480,    -512,    -576,    -640,    -704,    -768,    -832,    -896,    -960,
    -1024,   -1152,   -1280,   -1408,   -1536,   -1664,   -1792,   -1920,   -2048,   -2304,   -2560,   -2816,   -3072,   -3328,   -3584,   -3840,
    -4096,   -4608,   -5120,   -5632,   -6144,   -6656,   -7168,   -7680,   -8192,   -9216,  -10240,  -11264,  -12288,  -13312,  -14336,  -15360,
   -16384,  -18432,  -20480,  -22528,  -24576,  -26624,  -28672,  -30720,  -32768,  -36864,  -40960,  -45056,  -49152,  -53248,  -57344,  -61440,
   -65536,  -73728,  -81920,  -90112,  -98304, -106496, -114688, -122880, -131072, -147456, -163840, -180224, -196608, -212992, -229376,       0,
};
// w' of an E2M1 nibble: W_INT (contract_v2), halves.
__device__ const int32_t EXPERT_W_OF_NIBBLE[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
// The same 16 signed bytes packed for prmt (little-endian): entries 0-3, 4-7, 8-11, 12-15.
constexpr uint32_t EXPERT_PRMT_T0 = 0x03020100u;
constexpr uint32_t EXPERT_PRMT_T1 = 0x0C080604u;
constexpr uint32_t EXPERT_PRMT_T2 = 0xFDFEFF00u;
constexpr uint32_t EXPERT_PRMT_T3 = 0xF4F8FAFCu;
// Sign byte masks of elements 0..3 of a 16-bit nibble window v: prmt(v << 4, v, 0xD9C8) replicates bits 3, 7, 11, 15.
constexpr uint32_t EXPERT_PRMT_SIGNSEL = 0xD9C8u;
// Sign extension of byte i of a word: prmt(x, 0, SEXT[i]).
constexpr uint32_t EXPERT_PRMT_SEXT0 = 0x8880u, EXPERT_PRMT_SEXT1 = 0x9991u, EXPERT_PRMT_SEXT2 = 0xAAA2u, EXPERT_PRMT_SEXT3 = 0xBBB3u;

// --------------------------------------------------------------------------------------------------- flags
__device__ unsigned int g_flag_overflow = 0u;   // bit 0
__device__ unsigned int g_flag_finish   = 0u;   // bit 1
__device__ unsigned int g_flag_nan_code = 0u;   // bit 2
__device__ unsigned int g_flag_meta     = 0u;   // bit 3: a staged ea_g > EA or ew_g > EW (shift above the margin)

// --------------------------------------------------------------------------------------------------- device helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int PENDING> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING));
}
// d = a * b + c with a, b int32 and c, d int64: one IMAD.WIDE (products < 2^36 here, chains < 2^40).
__device__ __forceinline__ int64_t mad_wide_s32(int32_t a, int32_t b, int64_t c) {
  int64_t d;
  asm("mad.wide.s32 %0, %1, %2, %3;\n" : "=l"(d) : "r"(a), "r"(b), "l"(c));
  return d;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// Six-op arithmetic form of DENSE_M_OF_CODE (same values for all 256 codes, NaN codes included).
__device__ __forceinline__ int32_t dense_m_arith(uint32_t c) {
  const uint32_t e = (c >> 3) & 15u, mant = c & 7u;
  int32_t mag = e ? static_cast<int32_t>((8u | mant) << (e - 1u)) : static_cast<int32_t>(mant);
  mag = ((c & 127u) == 127u) ? 0 : mag;
  const int32_t s = -static_cast<int32_t>(c >> 7);           // 0 or -1
  return (mag ^ s) - s;
}
// Exact two's-complement add with overflow detection folded into a running sign mask (negative iff any overflow).
__device__ __forceinline__ int64_t add_ovf(int64_t a, int64_t t, int64_t& ovf) {
  const int64_t r = static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(t));
  ovf |= (a ^ r) & (t ^ r);
  return r;
}
// A * 2^(sh): left shift (sh <= MARGIN by the metadata check, |D| < 2^41: no overflow possible), or the floor right
// shift clamped at 63 (M5; exact: |D| < 2^41 floors to 0 or -1 at any shift of 41 or more; a shift of 64 or more
// would be undefined behaviour in C++ and poison in LLVM even though PTX shr clamps).
__device__ __forceinline__ int64_t shift_term(int64_t D, int sh) {
  return sh >= 0 ? static_cast<int64_t>(static_cast<uint64_t>(D) << sh) : (D >> min(-sh, 63));
}
// The finish rule (M1; identical in contract_a.finish_with_flag, both NumPy models and patha_prefill.cu):
//   f = __ll2float_rn(A): one direct int64 -> FP32 rounding (pin a: never through double).
//   A == 0: y = +0, never flagged.  Otherwise ex = biased exponent of f plus e2:
//   1 <= ex <= 254: y = f * 2^e2 exactly, formed by writing ex into the exponent field (bit-identical to ldexpf,
//                   which the normal range makes exact; no floating-point multiply is emitted);
//   ex > 254 ("overflow"):  y = signed infinity, bad = true;
//   ex < 1   ("underflow"): y = signed zero,     bad = true.
//   Then one RNE to BF16 (contract_v2.bf); an FP32 y above the BF16 maximum rounds to infinity unflagged (IEEE).
// Neither flagged case is reachable on legal inputs (CONTRACT_PINS_r1.md): the quantiser's 1e-4 amax floor gives
// EA >= -22 and the weight-load admission (scale codes >= 54 dense, 53 expert) gives ex >= 1 for the smallest
// nonzero A; the largest (|A| < 2^62) stays below ex = 255 while EA + EW <= 95.
__device__ __forceinline__ uint16_t finish_bf16(int64_t A, int e2, bool& bad) {
  const float f = __ll2float_rn(A);
  uint32_t u = __float_as_uint(f);
  if (f != 0.0f) {
    const int ex = static_cast<int>((u >> 23) & 0xFFu) + e2;      // biased exponent of f * 2^e2
    if (ex > 254) {
      bad = true;
      u = (u & 0x80000000u) | 0x7F800000u;                         // overflow: signed infinity
    } else if (ex < 1) {
      bad = true;
      u = (u & 0x80000000u);                                       // underflow: signed zero
    } else {
      u = (u & 0x807FFFFFu) | (static_cast<uint32_t>(ex) << 23);
    }
  }
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// --------------------------------------------------------------------------------------------------- prep kernel
// n[m,k] = N_OF_CODE[c] * SIGN_OF_CODE[c] (int32), EA[m] = max_g ea[m,g]. One block per row.
__global__ void __launch_bounds__(256) prep_kernel(const uint8_t* __restrict__ codes, const int8_t* __restrict__ ea,
                                                   int32_t* __restrict__ n, int8_t* __restrict__ EA, int K, int G) {
  __shared__ int red[8];
  const int m = blockIdx.x;
  bool nan_seen = false;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const uint32_t c = codes[static_cast<size_t>(m) * K + k];
    nan_seen |= ((c & 127u) == 127u);
    n[static_cast<size_t>(m) * K + k] = dense_m_arith(c);
  }
  int mx = -128;
  for (int g = threadIdx.x; g < G; g += blockDim.x) mx = max(mx, static_cast<int>(ea[static_cast<size_t>(m) * G + g]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int r = red[0];
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w) r = max(r, red[w]);
    EA[m] = static_cast<int8_t>(r);
  }
  if (nan_seen) g_flag_nan_code = 1u;
}

// --------------------------------------------------------------------------------------------------- GEMV kernel
template <int MT, class F>
__device__ __forceinline__ void load_chunk(Stage<MT, F>* st, const uint8_t* __restrict__ W, const uint8_t* __restrict__ ws,
                                           const int32_t* __restrict__ X, const int8_t* __restrict__ ea,
                                           int K, int G, int row0, int t_first, int nb, int tid) {
  const size_t rb = F::row_bytes(K);
  const int kb0 = t_first * F::BYTES_PER_GROUP;
  // weight rows: BN x CHUNKS16 sixteen-byte pieces, swizzled; pieces beyond the live groups are zero-filled
  for (int i = tid; i < BN * F::CHUNKS16; i += THREADS) {
    const int r = i / F::CHUNKS16, ch = i % F::CHUNKS16;
    const bool live = (ch * 16) < nb * F::BYTES_PER_GROUP;
    const uint8_t* src = W + static_cast<size_t>(row0 + r) * rb + kb0 + (live ? ch * 16 : 0);
    cp_async16(smem_u32(&st->W[r][swz(r, ch) * 16]), src, live ? 16 : 0);
  }
  // activation integers: MT rows x 64 pieces (4 ints each), natural order
  for (int i = tid; i < MT * X_CHUNKS16; i += THREADS) {
    const int m = i / X_CHUNKS16, ch = i % X_CHUNKS16;
    const bool live = ch < nb * (BLOCK_K * 4 / 16);
    const int32_t* src = X + static_cast<size_t>(m) * K + static_cast<size_t>(t_first) * BLOCK_K + (live ? ch * 4 : 0);
    cp_async16(smem_u32(&st->X[m][ch * 4]), src, live ? 16 : 0);
  }
  // exponent windows: 3 four-byte words from the 4-aligned-down group index; bytes past the row end are zero-filled
  const int a0 = t_first & ~3;
  for (int i = tid; i < MT * EXP_WORDS; i += THREADS) {
    const int m = i / EXP_WORDS, w = i % EXP_WORDS;
    const int g = a0 + 4 * w;
    const int valid = max(0, min(4, G - g));
    const int8_t* src = ea + static_cast<size_t>(m) * G + (valid ? g : 0);
    cp_async4(smem_u32(&st->EA[m][4 * w]), src, valid);
  }
  for (int i = tid; i < F::EW_ROWS * EXP_WORDS; i += THREADS) {
    const int r = i / EXP_WORDS, w = i % EXP_WORDS;
    const int g = a0 + 4 * w;
    const int valid = max(0, min(4, G - g));
    const int srow = F::ROW_SCALES ? (row0 + r) : (row0 / BN);
    const uint8_t* src = ws + static_cast<size_t>(srow) * G + (valid ? g : 0);
    cp_async4(smem_u32(&st->EW[r][4 * w]), src, valid);
  }
}

// Convert this lane's 16 weight elements of group b to int32 w[16] (k = 32 b + 16 half + j).
template <class F>
__device__ __forceinline__ void convert_weights(const uint8_t* wrow, int r, int half, int b, const int32_t* lut,
                                                int lane, int32_t (&w)[16]);

template <>
__device__ __forceinline__ void convert_weights<Dense>(const uint8_t* wrow, int r, int half, int b, const int32_t* lut,
                                                       int lane, int32_t (&w)[16]) {
  const uint4 v = *reinterpret_cast<const uint4*>(wrow + swz(r, 2 * b + half) * 16);
  const uint32_t wd[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t code = (wd[q] >> (8 * j)) & 0xFFu;
#if PATHA_DENSE_LUT
      w[4 * q + j] = lut[code * LUT_REPL + (lane % LUT_REPL)];
#else
      (void)lut; (void)lane;
      w[4 * q + j] = dense_m_arith(code);
#endif
    }
  }
}

template <>
__device__ __forceinline__ void convert_weights<Expert>(const uint8_t* wrow, int r, int half, int b, const int32_t* lut,
                                                        int lane, int32_t (&w)[16]) {
  const uint2 v = *reinterpret_cast<const uint2*>(wrow + swz(r, b) * 16 + half * 8);
  const uint32_t wd[2] = {v.x, v.y};
#pragma unroll
  for (int q = 0; q < 2; ++q) {
#if PATHA_EXPERT_LUT
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t nib = (wd[q] >> (4 * j)) & 15u;
      w[8 * q + j] = lut[nib * EXPERT_LUT_REPL + lane];
    }
#else
    (void)lut; (void)lane;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {                 // nibble windows: elements 8q+0..3 (low 16 bits), 8q+4..7 (high)
      const uint32_t vv = hh ? (wd[q] >> 16) : wd[q];
      const uint32_t sel = vv & 0x7777u;
      const uint32_t pos = prmt(EXPERT_PRMT_T0, EXPERT_PRMT_T1, sel);
      const uint32_t neg = prmt(EXPERT_PRMT_T2, EXPERT_PRMT_T3, sel);
      const uint32_t bm = prmt(vv << 4, vv, EXPERT_PRMT_SIGNSEL);
      const uint32_t packed = (pos & ~bm) | (neg & bm);  // four signed bytes = W_INT of elements 8q+4hh+0..3
      w[8 * q + 4 * hh + 0] = static_cast<int32_t>(prmt(packed, 0u, EXPERT_PRMT_SEXT0));
      w[8 * q + 4 * hh + 1] = static_cast<int32_t>(prmt(packed, 0u, EXPERT_PRMT_SEXT1));
      w[8 * q + 4 * hh + 2] = static_cast<int32_t>(prmt(packed, 0u, EXPERT_PRMT_SEXT2));
      w[8 * q + 4 * hh + 3] = static_cast<int32_t>(prmt(packed, 0u, EXPERT_PRMT_SEXT3));
    }
#endif
  }
}

// Lane partial over 16 elements: dense int64 (mad.wide), expert int32 (IMAD); returned widened to int64.
template <class F>
__device__ __forceinline__ int64_t lane_partial(const int32_t (&x)[16], const int32_t (&w)[16]);
template <>
__device__ __forceinline__ int64_t lane_partial<Dense>(const int32_t (&x)[16], const int32_t (&w)[16]) {
  int64_t p = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) p = mad_wide_s32(x[j], w[j], p);
  return p;
}
template <>
__device__ __forceinline__ int64_t lane_partial<Expert>(const int32_t (&x)[16], const int32_t (&w)[16]) {
  int32_t p = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) p += x[j] * w[j];
  return static_cast<int64_t>(p);
}

template <int MT, class F, bool FUSED>
__global__ void __launch_bounds__(THREADS, 2)
gemv_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ ws, const int8_t* __restrict__ EW,
            const int32_t* __restrict__ X, const int8_t* __restrict__ ea, const int8_t* __restrict__ EA,
            int64_t* __restrict__ P, uint16_t* __restrict__ Y, int N, int K, int S) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Stage<MT, F>* stages = reinterpret_cast<Stage<MT, F>*>(smem_raw);
  int32_t* lut = reinterpret_cast<int32_t*>(smem_raw + sizeof(Stage<MT, F>) * STAGES);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int lrow = (lane & 7) | ((lane >> 4) << 3);
  const int half = (lane >> 3) & 1;
  const int r = warp * ROWS_PER_WARP + lrow;            // row within the CTA
  const int row0 = blockIdx.x * BN;
  const int n = row0 + r;
  const int split = blockIdx.y;
  const int G = K / BLOCK_K;
  const int tps = tiles_per_split(K, S);
  const int t0 = split * tps;
  const int t1 = min(t0 + tps, G);
  const int nblk = max(t1 - t0, 0);
  const int nchunk = (nblk + BPC - 1) / BPC;
  const int off = t0 & 3;                                // exponent byte offset inside the staged window

  // conversion table (interleaved: entry c of replica rep at c * R + rep)
  if (F::LUT_INTS > 0) {
    constexpr int R = F::ROW_SCALES ? EXPERT_LUT_REPL : LUT_REPL;
    const int32_t* src = F::ROW_SCALES ? EXPERT_W_OF_NIBBLE : DENSE_M_OF_CODE;
    for (int i = tid; i < F::LUT_INTS; i += THREADS) lut[i] = src[i / R];
  }

  const int ew_row = static_cast<int>(F::ROW_SCALES ? EW[n] : EW[row0 / BN]);
  int cm[MT], eam[MT];
  int64_t acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) { eam[m] = static_cast<int>(EA[m]); cm[m] = F::MARGIN - eam[m] - ew_row; acc[m] = 0; }
  int64_t ovf = 0;
  bool meta_bad = false;                                 // L5: a staged ea_g > EA[m] or ew_g > EW[n] (stale table)

#pragma unroll
  for (int c = 0; c < STAGES - 1; ++c) {
    if (c < nchunk)
      load_chunk<MT, F>(&stages[c], W, ws, X, ea, K, G, row0, t0 + c * BPC, min(BPC, nblk - c * BPC), tid);
    cp_commit();
  }

  for (int c = 0; c < nchunk; ++c) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int cn = c + STAGES - 1;
      if (cn < nchunk)
        load_chunk<MT, F>(&stages[cn % STAGES], W, ws, X, ea, K, G, row0, t0 + cn * BPC, min(BPC, nblk - cn * BPC), tid);
      cp_commit();
    }
    const Stage<MT, F>& st = stages[c % STAGES];
    const int nb = min(BPC, nblk - c * BPC);

#pragma unroll
    for (int b = 0; b < BPC; ++b) {
      if (b < nb) {
        int32_t w[16];
        convert_weights<F>(&st.W[r][0], r, half, b, lut, lane, w);
        const int ew_g = static_cast<int>(st.EW[F::ROW_SCALES ? r : 0][off + b]) - 127;
        meta_bad |= (ew_g > ew_row);                               // L5: a staged ew_g above the row's EW
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const int4* xp = reinterpret_cast<const int4*>(&st.X[m][b * BLOCK_K + half * 16]);
          int32_t x[16];
#pragma unroll
          for (int q = 0; q < 4; ++q) { const int4 v = xp[q]; x[4 * q] = v.x; x[4 * q + 1] = v.y; x[4 * q + 2] = v.z; x[4 * q + 3] = v.w; }
          const int64_t p = lane_partial<F>(x, w);
          const int64_t D = p + __shfl_xor_sync(0xFFFFFFFFu, p, 8);      // exact group dot (both lanes hold it)
          const int ea_g = static_cast<int>(static_cast<int8_t>(st.EA[m][off + b]));
          meta_bad |= (ea_g > eam[m]);                             // L5: a staged ea_g above EA[m]
          const int sh = ea_g + ew_g + cm[m];                      // <= F::MARGIN on consistent tables
          acc[m] = add_ovf(acc[m], shift_term(D, sh), ovf);
        }
      }
    }
  }

  if (ovf < 0) g_flag_overflow = 1u;
  if (meta_bad) g_flag_meta = 1u;
  if (half == 0) {
    if (FUSED) {
      bool bad = false;
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int B = static_cast<int>(EA[m]) + ew_row - F::MARGIN;
        Y[static_cast<size_t>(m) * N + n] = finish_bf16(acc[m], B - SA - F::SW, bad);
      }
      if (bad) g_flag_finish = 1u;
    } else {
#pragma unroll
      for (int m = 0; m < MT; ++m) P[(static_cast<size_t>(split) * MT + m) * N + n] = acc[m];
    }
  }
}

// --------------------------------------------------------------------------------------------------- merge kernel
template <bool ROW_SCALES>
__global__ void __launch_bounds__(256) merge_kernel(const int64_t* __restrict__ P, const int8_t* __restrict__ EA,
                                                    const int8_t* __restrict__ EW, uint16_t* __restrict__ Y,
                                                    int M, int N, int S, int sasw) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  const int m = idx / N, n = idx - m * N;
  constexpr int MARGIN = ROW_SCALES ? MARGIN_EXPERT : MARGIN_DENSE;
  int64_t a = 0, ovf = 0;
  for (int s = 0; s < S; ++s) a = add_ovf(a, P[(static_cast<size_t>(s) * M + m) * N + n], ovf);
  const int B = static_cast<int>(EA[m]) + static_cast<int>(ROW_SCALES ? EW[n] : EW[n / BN]) - MARGIN;
  bool bad = false;
  Y[idx] = finish_bf16(a, B - sasw, bad);
  if (ovf < 0) g_flag_overflow = 1u;
  if (bad) g_flag_finish = 1u;
}

// --------------------------------------------------------------------------------------------------- host launch
template <int MT, class F, bool FUSED>
static void launch_mt(const uint8_t* W, const uint8_t* ws, const int8_t* EW, const int32_t* X, const int8_t* ea,
                      const int8_t* EA, int64_t* P, uint16_t* Y, int N, int K, int S, cudaStream_t stream) {
  const size_t smem = smem_bytes_for<MT, F>();
  static bool attr_set = false;                        // per instantiation; first launch happens outside any capture
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel<MT, F, FUSED>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(smem)));
    attr_set = true;
  }
  const dim3 grid(N / BN, S), block(THREADS);
  gemv_kernel<MT, F, FUSED><<<grid, block, smem, stream>>>(W, ws, EW, X, ea, EA, P, Y, N, K, S);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class F, bool FUSED>
static void launch(int M, const uint8_t* W, const uint8_t* ws, const int8_t* EW, const int32_t* X, const int8_t* ea,
                   const int8_t* EA, int64_t* P, uint16_t* Y, int N, int K, int S, cudaStream_t stream) {
  switch (M) {
    case 1: launch_mt<1, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 2: launch_mt<2, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 3: launch_mt<3, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 4: launch_mt<4, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 5: launch_mt<5, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 6: launch_mt<6, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 7: launch_mt<7, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    case 8: launch_mt<8, F, FUSED>(W, ws, EW, X, ea, EA, P, Y, N, K, S, stream); break;
    default: TORCH_CHECK(false, "patha: 1 <= M <= 8");
  }
}

template <class F>
static size_t smem_bytes_of(int M) {
  switch (M) {
    case 1: return smem_bytes_for<1, F>(); case 2: return smem_bytes_for<2, F>();
    case 3: return smem_bytes_for<3, F>(); case 4: return smem_bytes_for<4, F>();
    case 5: return smem_bytes_for<5, F>(); case 6: return smem_bytes_for<6, F>();
    case 7: return smem_bytes_for<7, F>(); case 8: return smem_bytes_for<8, F>();
    default: return 0;
  }
}

}  // namespace patha

// --------------------------------------------------------------------------------------------------- torch binding
namespace {

using namespace patha;

void check_common(const torch::Tensor& X, const torch::Tensor& ea, const torch::Tensor& EA, int& M, int& K) {
  TORCH_CHECK(X.is_cuda() && ea.is_cuda() && EA.is_cuda(), "patha: CUDA tensors required");
  TORCH_CHECK(X.scalar_type() == at::kInt, "patha: activation integers must be int32");
  TORCH_CHECK(ea.scalar_type() == at::kChar && EA.scalar_type() == at::kChar, "patha: ea and EA must be int8");
  TORCH_CHECK(X.is_contiguous() && ea.is_contiguous() && EA.is_contiguous(), "patha: contiguous tensors required");
  TORCH_CHECK(X.dim() == 2 && ea.dim() == 2 && EA.dim() == 1, "patha: ranks (X [M,K], ea [M,K/32], EA [M])");
  M = static_cast<int>(X.size(0));
  K = static_cast<int>(X.size(1));
  TORCH_CHECK(M >= 1 && M <= MAX_M, "patha: 1 <= M <= 8");
  TORCH_CHECK(K % (BLOCK_K * 4) == 0, "patha: K % 128 == 0 (four-byte exponent copies need K/32 % 4 == 0)");
  TORCH_CHECK(ea.size(0) == M && ea.size(1) == K / BLOCK_K, "patha: ea must be [M, K/32]");
  TORCH_CHECK(EA.size(0) == M, "patha: EA must be [M]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0, "patha: X must be 16-byte aligned (cp.async)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(ea.data_ptr()) % 4 == 0, "patha: ea must be 4-byte aligned");
}

template <class F>
void check_weights(const torch::Tensor& W, const torch::Tensor& ws, const torch::Tensor& EW, int K, int& N) {
  TORCH_CHECK(W.is_cuda() && ws.is_cuda() && EW.is_cuda(), "patha: CUDA weight tensors required");
  TORCH_CHECK(W.scalar_type() == at::kByte && ws.scalar_type() == at::kByte, "patha: weight codes and scale codes must be uint8");
  TORCH_CHECK(EW.scalar_type() == at::kChar, "patha: EW must be int8");
  TORCH_CHECK(W.is_contiguous() && ws.is_contiguous() && EW.is_contiguous(), "patha: contiguous weight tensors required");
  TORCH_CHECK(W.dim() == 2 && ws.dim() == 2 && EW.dim() == 1, "patha: weight ranks");
  N = static_cast<int>(W.size(0));
  TORCH_CHECK(N % BN == 0, "patha: N % 32 == 0");
  TORCH_CHECK(W.size(1) == static_cast<int64_t>(F::row_bytes(K)), "patha: weight row length mismatch (K or K/2 bytes)");
  const int G = K / BLOCK_K;
  if (F::ROW_SCALES) {
    TORCH_CHECK(ws.size(0) == N && ws.size(1) == G, "patha: expert scales must be [N, K/32]");
    TORCH_CHECK(EW.size(0) == N, "patha: expert EW must be [N]");
  } else {
    TORCH_CHECK(ws.size(0) == N / BN && ws.size(1) == G, "patha: dense scales must be [N/32, K/32]");
    TORCH_CHECK(EW.size(0) == N / BN, "patha: dense EW must be [N/32]");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, "patha: weights must be 16-byte aligned (cp.async)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(ws.data_ptr()) % 4 == 0, "patha: weight scales must be 4-byte aligned");
}

void prep_activations(torch::Tensor codes, torch::Tensor ea, torch::Tensor n, torch::Tensor EA) {
  TORCH_CHECK(codes.is_cuda() && ea.is_cuda() && n.is_cuda() && EA.is_cuda(), "patha.prep: CUDA tensors required");
  TORCH_CHECK(codes.scalar_type() == at::kByte, "patha.prep: codes must be uint8 (view float8_e4m3fn as uint8)");
  TORCH_CHECK(ea.scalar_type() == at::kChar && EA.scalar_type() == at::kChar && n.scalar_type() == at::kInt, "patha.prep: dtypes");
  TORCH_CHECK(codes.is_contiguous() && ea.is_contiguous() && n.is_contiguous() && EA.is_contiguous(), "patha.prep: contiguous");
  TORCH_CHECK(codes.dim() == 2 && ea.dim() == 2 && n.dim() == 2 && EA.dim() == 1, "patha.prep: ranks");
  const int M = static_cast<int>(codes.size(0)), K = static_cast<int>(codes.size(1));
  TORCH_CHECK(M >= 1 && K % BLOCK_K == 0, "patha.prep: K % 32 == 0");
  TORCH_CHECK(ea.size(0) == M && ea.size(1) == K / BLOCK_K, "patha.prep: ea must be [M, K/32]");
  TORCH_CHECK(n.size(0) == M && n.size(1) == K && EA.size(0) == M, "patha.prep: outputs n [M,K] int32, EA [M] int8");
  const c10::cuda::CUDAGuard guard(codes.device());
  prep_kernel<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<uint8_t>(), ea.data_ptr<int8_t>(), n.data_ptr<int32_t>(), EA.data_ptr<int8_t>(), K, K / BLOCK_K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class F>
void partials_impl(torch::Tensor X, torch::Tensor ea, torch::Tensor EA, torch::Tensor W, torch::Tensor ws,
                   torch::Tensor EW, torch::Tensor P, int64_t S) {
  int M, K, N;
  check_common(X, ea, EA, M, K);
  check_weights<F>(W, ws, EW, K, N);
  TORCH_CHECK(S >= 1 && S <= 8, "patha: 1 <= splits <= 8 (1, 2 or 4 in the design)");
  TORCH_CHECK(P.is_cuda() && P.scalar_type() == at::kLong && P.is_contiguous() && P.dim() == 3, "patha: P must be contiguous int64 [S, M, N]");
  TORCH_CHECK(P.size(0) == S && P.size(1) == M && P.size(2) == N, "patha: P must be [S, M, N]");
  const c10::cuda::CUDAGuard guard(X.device());
  launch<F, false>(M, W.data_ptr<uint8_t>(), ws.data_ptr<uint8_t>(), EW.data_ptr<int8_t>(), X.data_ptr<int32_t>(),
                   ea.data_ptr<int8_t>(), EA.data_ptr<int8_t>(), P.data_ptr<int64_t>(), nullptr, N, K,
                   static_cast<int>(S), at::cuda::getCurrentCUDAStream());
}

template <class F>
void fused_impl(torch::Tensor X, torch::Tensor ea, torch::Tensor EA, torch::Tensor W, torch::Tensor ws,
                torch::Tensor EW, torch::Tensor Y) {
  int M, K, N;
  check_common(X, ea, EA, M, K);
  check_weights<F>(W, ws, EW, K, N);
  TORCH_CHECK(Y.is_cuda() && Y.scalar_type() == at::kBFloat16 && Y.is_contiguous() && Y.dim() == 2, "patha: Y must be contiguous bfloat16 [M, N]");
  TORCH_CHECK(Y.size(0) == M && Y.size(1) == N, "patha: Y must be [M, N]");
  const c10::cuda::CUDAGuard guard(X.device());
  launch<F, true>(M, W.data_ptr<uint8_t>(), ws.data_ptr<uint8_t>(), EW.data_ptr<int8_t>(), X.data_ptr<int32_t>(),
                  ea.data_ptr<int8_t>(), EA.data_ptr<int8_t>(), nullptr, reinterpret_cast<uint16_t*>(Y.data_ptr()),
                  N, K, 1, at::cuda::getCurrentCUDAStream());
}

void merge_finish(torch::Tensor P, torch::Tensor EA, torch::Tensor EW, int64_t kind, torch::Tensor Y) {
  TORCH_CHECK(P.is_cuda() && EA.is_cuda() && EW.is_cuda() && Y.is_cuda(), "patha.merge: CUDA tensors required");
  TORCH_CHECK(P.scalar_type() == at::kLong && EA.scalar_type() == at::kChar && EW.scalar_type() == at::kChar
              && Y.scalar_type() == at::kBFloat16, "patha.merge: dtypes (P int64, EA/EW int8, Y bfloat16)");
  TORCH_CHECK(P.is_contiguous() && EA.is_contiguous() && EW.is_contiguous() && Y.is_contiguous(), "patha.merge: contiguous");
  TORCH_CHECK(P.dim() == 3 && EA.dim() == 1 && EW.dim() == 1 && Y.dim() == 2, "patha.merge: ranks");
  TORCH_CHECK(kind == 0 || kind == 1, "patha.merge: kind 0 (dense) or 1 (expert)");
  const int S = static_cast<int>(P.size(0)), M = static_cast<int>(P.size(1)), N = static_cast<int>(P.size(2));
  TORCH_CHECK(M >= 1 && N % BN == 0 && S >= 1, "patha.merge: shapes");
  TORCH_CHECK(EA.size(0) == M && Y.size(0) == M && Y.size(1) == N, "patha.merge: EA [M], Y [M, N]");
  TORCH_CHECK(EW.size(0) == (kind == 1 ? N : N / BN), "patha.merge: EW [N] (expert) or [N/32] (dense)");
  const c10::cuda::CUDAGuard guard(P.device());
  const int total = M * N, threads = 256, blocks = (total + threads - 1) / threads;
  const int sasw = SA + (kind == 1 ? Expert::SW : Dense::SW);
  if (kind == 1)
    merge_kernel<true><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        P.data_ptr<int64_t>(), EA.data_ptr<int8_t>(), EW.data_ptr<int8_t>(), reinterpret_cast<uint16_t*>(Y.data_ptr()), M, N, S, sasw);
  else
    merge_kernel<false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        P.data_ptr<int64_t>(), EA.data_ptr<int8_t>(), EW.data_ptr<int8_t>(), reinterpret_cast<uint16_t*>(Y.data_ptr()), M, N, S, sasw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Read (and optionally clear) the flags. Synchronous; call between runs, never inside a capture.
int64_t error_flag(bool reset) {
  unsigned int o = 0u, f = 0u, q = 0u, t = 0u;
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(&o, g_flag_overflow, sizeof(o)));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(&f, g_flag_finish, sizeof(f)));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(&q, g_flag_nan_code, sizeof(q)));
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(&t, g_flag_meta, sizeof(t)));
  if (reset) {
    const unsigned int z = 0u;
    C10_CUDA_CHECK(cudaMemcpyToSymbol(g_flag_overflow, &z, sizeof(z)));
    C10_CUDA_CHECK(cudaMemcpyToSymbol(g_flag_finish, &z, sizeof(z)));
    C10_CUDA_CHECK(cudaMemcpyToSymbol(g_flag_nan_code, &z, sizeof(z)));
    C10_CUDA_CHECK(cudaMemcpyToSymbol(g_flag_meta, &z, sizeof(z)));
  }
  return static_cast<int64_t>((o ? 1u : 0u) | (f ? 2u : 0u) | (q ? 4u : 0u) | (t ? 8u : 0u));
}

int64_t smem_bytes(int64_t M, int64_t kind) {
  return static_cast<int64_t>(kind == 1 ? smem_bytes_of<Expert>(static_cast<int>(M)) : smem_bytes_of<Dense>(static_cast<int>(M)));
}

int64_t build_config(int64_t which) {
  switch (which) {
    case 0: return PATHA_DENSE_LUT; case 1: return PATHA_LUT_REPL; case 2: return PATHA_EXPERT_LUT; case 3: return PATHA_STAGES;
    default: return -1;
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep_activations", &prep_activations, "codes u8 [M,K], ea i8 [M,K/32] -> n i32 [M,K], EA i8 [M] (outputs preallocated)");
  m.def("dense_partials", &partials_impl<Dense>, "n, ea, EA, wc u8 [N,K], ws u8 [N/32,K/32], EW i8 [N/32], P i64 [S,M,N], S");
  m.def("dense_finish", &fused_impl<Dense>, "S = 1 with the finish fused: n, ea, EA, wc, ws, EW, Y bf16 [M,N]");
  m.def("expert_partials", &partials_impl<Expert>, "n, ea, EA, w u8 [N,K/2], ws u8 [N,K/32], EW i8 [N], P i64 [S,M,N], S");
  m.def("expert_finish", &fused_impl<Expert>, "S = 1 with the finish fused: n, ea, EA, w, ws, EW, Y bf16 [M,N]");
  m.def("merge_finish", &merge_finish, "P i64 [S,M,N], EA i8 [M], EW i8 [N/32 or N], kind (0 dense, 1 expert), Y bf16 [M,N]");
  m.def("error_flag", &error_flag, "bit0 int64 overflow, bit1 finish out of range (rule M1), bit2 NaN activation code, bit3 exponent metadata inconsistent; reset=True clears");
  m.def("smem_bytes", &smem_bytes, "dynamic shared memory per CTA for (M, kind)");
  m.def("build_config", &build_config, "0: PATHA_DENSE_LUT, 1: PATHA_LUT_REPL, 2: PATHA_EXPERT_LUT, 3: PATHA_STAGES");
}
