// patha_expert_static.cu - Path A r3-E step 1: one static launch per layer per projection for the hot-slot GPU
// experts of DeepSeek V4.1-Flash-1017 on sm_89 (DESIGN_r3_dense.md section 2; oracle contract_a.py / contract_v2.expert).
//
// What it replaces: `dp4a_routes_candidate.expert_routes_splitk`'s `project` + `_merge_projection` (both projections),
// the `_active_weight_metadata` launches and the `quant_planes` prep. Kept outside: the `_quant_codes` quantiser
// (patha_expert_static.py) and the unchanged `_activate_routes`.
//
// Arithmetic: textually r2's expert kind (patha_decode.cu, sha fba8e996..., contract fixes r1), copied, not rewritten:
//   activation  n[m,k]  int32, |n| < 2^18 (E4M3 code -> N_OF_CODE * sign, 2^-9 units; dense_m_arith, converted in-kernel)
//   expert      w[n,k]  int32, |w| <= 12  (E2M1 nibble -> W_INT, halves, SW = 1)    products < 2^22, int32 lane sums
//   lane partial: 16 consecutive k of one weight row against each gathered token slot (IMAD chain); the two lanes of a
//   group combine by one shfl.xor into the exact group dot D_g.
//   per group:  sh = ea[m,g] + ew[n,g] - B[m,n],  B = EA[m] + EW[n] - MARGIN (20, contract v2, locked);
//               A[m,n] += sh >= 0 ? D_g << sh : D_g >> min(-sh, 63) (floor); int64 with per-add overflow detection
//               (a defence: |A| < 2^54.4 on a legal input).
//   metadata:   EA[m] = max_g ea[m,g] and EW[n] = max_g ws[n,g] - 127 are computed in the prologue from the same
//               tensors the chunks stage (no host table, no _active_weight_metadata launch); the staged-exponent checks
//               ea_g <= EA and ew_g <= EW of the contract fixes (L5) stay as a defence and set the meta flag.
//   finish:     S = 1 fused: f = __ll2float_rn(A) (one direct rounding); e2 = B - SA - SW = B - 10; the finish rule
//               (M1): A == 0 -> +0; biased exponent of f plus e2 in 1..254 -> the normal f * 2^e2 formed on the exponent
//               field; above 254 -> signed infinity and the finish flag; below 1 -> signed zero and the finish flag.
//               bf16 = RNE(y) as contract_v2.bf. Gate/up writes the BF16 bits widened to FP32 (bits << 16, exact) so
//               `_activate_routes` reads them unchanged; down writes BF16 [count, H].
//
// Launch shape (design section 2.2): grid (N/32, 1, count), count = M x topk fixed per captured graph; block 64 threads
// (2 warps x 16 rows, r2's lane mapping and swizzle). CTA z = route: eid = active[z]; exit if eid < 0; leader rule:
// exit if any route r' < z has active[r'] == eid; otherwise gather every route with this eid (at most M <= 8 slots,
// one per token, since topk is a set) into a shared slot list and serve them all: each expert's weights stream from
// DRAM once per layer. Rows: gate/up (FIRST) row = route / topk into the x codes; down row = route into the h codes.
// Outputs are written per route (out[route]), so duplicate-expert routes get the bits of a per-expert launch by
// construction. Split-K S = 1 everywhere (fused finish; no partials, no merge, no counter).
//
// Stage (cp.async, STAGES in flight): 32 weight rows x BPC groups of 16 nibble bytes (16-byte pieces swizzled by
// piece ^ (row & 7)), the chunk's E4M3 codes of each slot (BPC x 32 bytes), the slot exponent windows (BPC bytes as
// BPC/4 four-byte copies) and the 32 rows' scale codes of the window. After the wait the chunk's codes are converted
// cooperatively (64 threads x 4 codes per slot, the six-op form) into one int32 X buffer shared by the stages, then
// r2's inner loop runs on it. Shared memory per CTA: 3 x (4096 + 512 + 256 MT + 16 MT) + 1024 MT + 2048 (LUT) bytes:
// 17,712 at M = 1 (5 CTAs per SM), 30,592 at M = 8 (3 CTAs per SM); BPC = 16 doubles the weight and code parts.
//
// Flags: one 32-bit word per stream, passed in (review M6), or-ed by atomicOr only when a bit is set (no atomic runs
// on a clean call; the outputs are deterministic and the word is not an output). Bits: 1 int64 overflow, 2 finish
// outside the normal range, 4 NaN activation code (127 / 255) in a staged code, 8 exponent metadata inconsistent,
// 16 weight scale code outside the contract admission (255, or below MIN_SCALE_CODE = 53: the finish rule's bound,
// enforced here because arena-DMA'd slots never pass prepare_expert_weights), 32 more routes share one expert than M
// slots allow (a token routed twice to the same expert: the surplus routes are not served). Every flagged case still
// produces an output; the word is read between runs (eager: after every call; graph: once after the replay).
//
// Build: nvcc -O3 -gencode=arch=compute_89,code=sm_89 -fmad=false (patha_expert_static.py); no fast-math anywhere.
// Macros (every variant is exact): PATHA_EXPERT_LUT (1 = 16-entry int32 smem table replicated 32 times; 0 = prmt
// byte-table form), PATHA_STAGES (3), PATHA_EXPERT_BPC (8; 16 = one barrier pair per 16 groups, design knob for the
// 9-chunk down projection), PATHA_EXPERT_TC (0; 1 is the step 2 tensor-digit knob, designed, not implemented here).
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#ifndef PATHA_EXPERT_LUT
#define PATHA_EXPERT_LUT 1
#endif
#ifndef PATHA_STAGES
#define PATHA_STAGES 3
#endif
#ifndef PATHA_EXPERT_BPC
#define PATHA_EXPERT_BPC 8
#endif
#ifndef PATHA_EXPERT_TC
#define PATHA_EXPERT_TC 0
#endif

#if PATHA_EXPERT_TC
// ------------------------------------------------------------------------------------------ step 2 knob (not built)
// PATHA_EXPERT_TC=1 (DESIGN_r3_dense.md section 2.3, direction B applied to experts) replaces lane_partial<Expert> and
// the LUT / prmt nibble conversion with tensor-core digit planes; everything else in this file (gather, staging, the
// per-group shift and checked add, the finish, the flags) is shared. Fragment map, for the record:
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 per warp per 16-row half-group, two k16 steps per 32-wide group.
//   A (16 x 16, row-major) = this warp's 16 weight rows x 16 k of w' as FP16 integers (|w'| <= 12, exact; the seven
//     nonzero magnitudes have FP16 patterns 0x3C00..0x4A00 with a zero low byte: a prmt high-byte table plus the sign
//     moved to bit 15). Lane holds a0 = (row g, k 2c..2c+1), a1 = (row g+8, same k), a2 = (row g, k 8+2c..), a3 =
//     (row g+8, k 8+2c..) with g = lane >> 2, c = lane & 3; k16 step s covers group elements 16 s .. 16 s + 15.
//   B (16 x 8, col-major) = 16 k x 8 token slots: two FP16 digit planes of the activation integer n of slot j =
//     lane >> 2: hi = RN(n / 512) (<= 448), lo = n - 512 hi (|lo| <= 256), both exact FP16 integers, from
//     cvt.rn.f16x2.e4m3x2 + 2 HADD2 + HSUB2 + HMUL2 per code pair; b0 = (k 2c..2c+1, slot j), b1 = (k 8+2c.., slot j).
//     Slots j >= cnt hold zeros (no load).
//   C/D (16 x 8, f32) zero per plane per group: d0, d1 = (row g, slots 2c, 2c+1), d2, d3 = (row g+8, same). Plane sums
//     per group are below 2^23 (hi: 32 x 448 x 12 = 172,032; lo: 98,304), so the FP32 accumulate is exact; readback
//     by cvt.rzi.s32.f32 and D = hi_sum * 512 + lo_sum (two mad.wide.s32) feeds the unchanged shift_term / add_ovf.
//   Per warp per group: 1 ldmatrix.x2 + 28 nibble ops + 1 LDS.64 + 2 cvt + 8 split + 4 MMA + 8 F2I + 4 mad.wide +
//     the epilogue (section 2.3). The per-group path is the oracle's accumulate verbatim; the chunked fold (budget
//     2 bits over 8 groups, 5.5 M < 2^23) is an option, not part of the knob's first build.
// Gates to add before any price (design section 6, step 2): RN digit split over all 256 codes (hi x 512 + lo == n,
// |lo| <= 256), plane bounds, fragment permutation bijection, ldmatrix.x2 phase conflicts, the FP16-plane expert
// realisation in digit_exactness_oracle.py (fp16_digits_rn) on 200,000 groups, the nibble-to-f16 prmt constants on all
// 65,536 windows, and the GPU plane readback against the oracle on 1,000,005 groups.
#error "PATHA_EXPERT_TC=1 is not implemented in r3-E step 1 (design knob only; see the fragment map above)"
#endif

namespace patha_es {

constexpr int MARGIN_EXPERT = 20;                 // contract v2 (locked); experts keep MARGIN 20 (CONTRACT_PINS_r1 section 0)
constexpr int SA            = 9;                  // E4M3 activation integers are in 2^-9 units
constexpr int BLOCK_K       = 32;                 // scale group
constexpr int WARPS         = 2;
constexpr int THREADS       = 32 * WARPS;
constexpr int ROWS_PER_WARP = 16;
constexpr int BN            = WARPS * ROWS_PER_WARP;   // 32 output rows per CTA
constexpr int BPC           = PATHA_EXPERT_BPC;   // groups per chunk (one cp.async group, one barrier pair)
constexpr int STAGES        = PATHA_STAGES;
constexpr int MAX_M         = 8;                  // token rows per step (template bound); one slot per token per expert
constexpr int MAX_TOPK      = 6;
constexpr int MAX_COUNT     = THREADS;            // the prologue reads one route per thread; admission gives M x topk <= 48
constexpr int X_CHUNK_INTS  = BPC * BLOCK_K;      // activation integers per slot per chunk (256 at BPC = 8)
constexpr int C_CHUNK_BYTES = X_CHUNK_INTS;       // one E4M3 code per element
constexpr int C_CHUNKS16    = C_CHUNK_BYTES / 16; // sixteen-byte code pieces per slot per chunk
constexpr int EXP_WORDS     = BPC / 4;            // 4-byte exponent words per row per chunk (S = 1: the window starts at t_first)
constexpr int EXP_ROW_BYTES = 16;
constexpr int EXPERT_LUT_REPL = 32;
constexpr int MIN_SCALE_CODE  = 53;               // contract_a.MIN_SCALE_CODE_EXPERT: the finish rule's admission bound
constexpr unsigned int FLAG_OVERFLOW  = 1u;       // bit 0 int64 accumulation overflow (defence)
constexpr unsigned int FLAG_FINISH    = 2u;       // bit 1 finish outside the normal range (rule M1)
constexpr unsigned int FLAG_NAN_CODE  = 4u;       // bit 2 NaN activation code in a staged chunk
constexpr unsigned int FLAG_META      = 8u;       // bit 3 a staged ea_g > EA or ew_g > EW (L5 defence)
constexpr unsigned int FLAG_BAD_SCALE = 16u;      // bit 4 weight scale code 255 or below MIN_SCALE_CODE
constexpr unsigned int FLAG_SLOTS     = 32u;      // bit 5 more routes share an expert than the M slots of this launch
static_assert(BPC == 8 || BPC == 16, "PATHA_EXPERT_BPC must be 8 or 16");
static_assert(EXP_WORDS * 4 <= EXP_ROW_BYTES, "exponent window must fit the staged row");
static_assert(STAGES >= 2, "two stages minimum");

inline __device__ int swz(int row, int chunk16) { return chunk16 ^ (row & 7); }

// Weight format tag (r2's Expert format: 16 nibble bytes per row per 32-wide group, one E8M0 scale per (row, group)).
struct Expert {
  static constexpr int  BYTES_PER_GROUP = 16;
  static constexpr int  CHUNK_BYTES     = BPC * BYTES_PER_GROUP;   // 128 at BPC = 8
  static constexpr int  CHUNKS16        = CHUNK_BYTES / 16;        // 8
  static constexpr int  SW              = 1;
  static constexpr int  MARGIN          = MARGIN_EXPERT;
  static constexpr int  LUT_INTS        = PATHA_EXPERT_LUT ? 16 * EXPERT_LUT_REPL : 0;
  static __host__ __device__ size_t row_bytes(int K) { return static_cast<size_t>(K) / 2; }
};
using F = Expert;

template <int MT>
struct __align__(16) Stage {
  uint8_t W[BN][F::CHUNK_BYTES];          // weight bytes, 16-byte pieces swizzled by row
  uint8_t C[MT][C_CHUNK_BYTES];           // E4M3 codes of the chunk per slot, natural order
  uint8_t EA[MT][EXP_ROW_BYTES];          // ea bytes [t_first, t_first + BPC) per slot (int8 exponents)
  uint8_t EW[BN][EXP_ROW_BYTES];          // weight scale codes (E8M0 uint8) of the same group window
};
template <int MT>
__host__ __device__ constexpr size_t x_bytes_for() { return sizeof(int32_t) * MT * X_CHUNK_INTS; }
template <int MT>
__host__ __device__ constexpr size_t smem_bytes_for() {
  return sizeof(Stage<MT>) * STAGES + x_bytes_for<MT>() + static_cast<size_t>(F::LUT_INTS) * 4;
}

// --------------------------------------------------------------------------------------------------- tables (r2)
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

// --------------------------------------------------------------------------------------------------- device helpers (r2)
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
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// Six-op arithmetic form of DENSE_M_OF_CODE (same values for all 256 codes, NaN codes included): the prep kernel's
// activation integer n = N_OF_CODE[c] * SIGN_OF_CODE[c], computed in-kernel on the staged codes.
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
// The route table is int32 or int64 as the router hands it (dp4a_routes_candidate accepts both); read either width.
__device__ __forceinline__ int read_active(const void* active, int active_i64, int i) {
  return active_i64 ? static_cast<int>(reinterpret_cast<const int64_t*>(active)[i])
                    : reinterpret_cast<const int32_t*>(active)[i];
}

// --------------------------------------------------------------------------------------------------- staging
template <int MT>
__device__ __forceinline__ void load_chunk(Stage<MT>* st, const uint8_t* __restrict__ Wx, const uint8_t* __restrict__ wsx,
                                           const uint8_t* __restrict__ C, const int8_t* __restrict__ ea, const int* s_tok,
                                           int cnt, int K, int G, int row0, int t_first, int nb, int tid) {
  const size_t rb = F::row_bytes(K);
  const int kb0 = t_first * F::BYTES_PER_GROUP;
  // weight rows of this expert: BN x CHUNKS16 sixteen-byte pieces, swizzled; pieces beyond the live groups are zero-filled
  for (int i = tid; i < BN * F::CHUNKS16; i += THREADS) {
    const int r = i / F::CHUNKS16, ch = i % F::CHUNKS16;
    const bool live = (ch * 16) < nb * F::BYTES_PER_GROUP;
    const uint8_t* src = Wx + static_cast<size_t>(row0 + r) * rb + kb0 + (live ? ch * 16 : 0);
    cp_async16(smem_u32(&st->W[r][swz(r, ch) * 16]), src, live ? 16 : 0);
  }
  // activation codes of the gathered slots: cnt rows x C_CHUNKS16 pieces of 16 codes, natural order
  for (int i = tid; i < MT * C_CHUNKS16; i += THREADS) {
    const int m = i / C_CHUNKS16, ch = i % C_CHUNKS16;
    if (m < cnt) {
      const bool live = ch < nb * (BLOCK_K / 16);
      const uint8_t* src = C + static_cast<size_t>(s_tok[m]) * K + static_cast<size_t>(t_first) * BLOCK_K + (live ? ch * 16 : 0);
      cp_async16(smem_u32(&st->C[m][ch * 16]), src, live ? 16 : 0);
    }
  }
  // exponent windows: EXP_WORDS four-byte words from t_first (4-aligned: S = 1, BPC % 4 == 0); bytes past the row end
  // are zero-filled (K % 128 == 0 keeps every row start 4-aligned)
  for (int i = tid; i < MT * EXP_WORDS; i += THREADS) {
    const int m = i / EXP_WORDS, w = i % EXP_WORDS;
    if (m < cnt) {
      const int g = t_first + 4 * w;
      const int valid = max(0, min(4, G - g));
      const int8_t* src = ea + static_cast<size_t>(s_tok[m]) * G + (valid ? g : 0);
      cp_async4(smem_u32(&st->EA[m][4 * w]), src, valid);
    }
  }
  for (int i = tid; i < BN * EXP_WORDS; i += THREADS) {
    const int r = i / EXP_WORDS, w = i % EXP_WORDS;
    const int g = t_first + 4 * w;
    const int valid = max(0, min(4, G - g));
    const uint8_t* src = wsx + static_cast<size_t>(row0 + r) * G + (valid ? g : 0);
    cp_async4(smem_u32(&st->EW[r][4 * w]), src, valid);
  }
}

// Convert this lane's 16 weight elements of group b to int32 w[16] (k = 32 b + 16 half + j).  (r2)
template <class FF>
__device__ __forceinline__ void convert_weights(const uint8_t* wrow, int r, int half, int b, const int32_t* lut,
                                                int lane, int32_t (&w)[16]);

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

// Lane partial over 16 elements: expert int32 (IMAD); returned widened to int64.  (r2)
template <class FF>
__device__ __forceinline__ int64_t lane_partial(const int32_t (&x)[16], const int32_t (&w)[16]);
template <>
__device__ __forceinline__ int64_t lane_partial<Expert>(const int32_t (&x)[16], const int32_t (&w)[16]) {
  int32_t p = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) p += x[j] * w[j];
  return static_cast<int64_t>(p);
}

// --------------------------------------------------------------------------------------------------- the kernel
// W  [E, N, K/2] nibbles, ws [E, N, K/32] E8M0 codes (whole expert tables; the CTA indexes expert eid);
// C  [R, K] E4M3 codes and ea [R, K/32] int8 group exponents of the activation rows (R = M for gate/up, count for down);
// active [count] int32 or int64 (-1 = inactive route); Y [count, N] float32 (FIRST: BF16 bits << 16) or uint16 BF16;
// A_dbg [count, N] int64 or nullptr (the raw row sums A, for the identity matrix); flag: the stream's flag word.
template <int MT, bool FIRST>
__global__ void __launch_bounds__(THREADS, 2)
expert_static_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ ws,
                     const uint8_t* __restrict__ C, const int8_t* __restrict__ ea,
                     const void* __restrict__ active, int active_i64, int count, int topk,
                     void* __restrict__ Y, int64_t* __restrict__ A_dbg, unsigned int* __restrict__ flag, int N, int K) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  Stage<MT>* stages = reinterpret_cast<Stage<MT>*>(smem_raw);
  int32_t (*X)[X_CHUNK_INTS] = reinterpret_cast<int32_t (*)[X_CHUNK_INTS]>(smem_raw + sizeof(Stage<MT>) * STAGES);
  int32_t* lut = reinterpret_cast<int32_t*>(smem_raw + sizeof(Stage<MT>) * STAGES + x_bytes_for<MT>());
  __shared__ int s_slot[MT];                             // route index of each gathered slot, ascending
  __shared__ int s_tok[MT];                              // activation row of each slot (FIRST: route / topk; else route)
  __shared__ int s_EA[MT];                               // EA per slot, computed in the prologue
  __shared__ int s_cnt_w0;                               // gathered routes among 0..31

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int z = static_cast<int>(blockIdx.z);            // this CTA's route
  const int eid = read_active(active, active_i64, z);
  if (eid < 0) return;                                   // inactive route (uniform exit, before any barrier)

  // leader rule and gather (design section 2.2): one route per thread; the leader is the smallest route with this eid
  const int a_t = (tid < count) ? read_active(active, active_i64, tid) : -1;
  const bool same = (a_t == eid);
  if (__syncthreads_or(same && tid < z)) return;        // an earlier route serves this expert (uniform exit)
  const unsigned int m_warp = __ballot_sync(0xFFFFFFFFu, same);
  const int cnt_all = __syncthreads_count(same);         // routes sharing this expert (<= M on a legal routing)
  if (warp == 0 && lane == 0) s_cnt_w0 = __popc(m_warp);
  __syncthreads();
  if (same) {
    const int idx = (warp ? s_cnt_w0 : 0) + __popc(m_warp & ((1u << lane) - 1u));
    if (idx < MT) { s_slot[idx] = tid; s_tok[idx] = FIRST ? tid / topk : tid; }
  }
  const int cnt = min(cnt_all, MT);
  __syncthreads();                                       // s_slot / s_tok visible

  const int G = K / BLOCK_K;
  const int lrow = (lane & 7) | ((lane >> 4) << 3);
  const int half = (lane >> 3) & 1;
  const int r = warp * ROWS_PER_WARP + lrow;            // row within the CTA
  const int row0 = static_cast<int>(blockIdx.x) * BN;
  const int n = row0 + r;
  const uint8_t* Wx = W + static_cast<size_t>(eid) * N * F::row_bytes(K);
  const uint8_t* wsx = ws + static_cast<size_t>(eid) * N * G;

  // EA[m] = max_g ea[tok[m], g] (the prep kernel's reduction, in-kernel): warp w takes slots w, w + WARPS, ...
  for (int m = warp; m < cnt; m += WARPS) {
    const int8_t* row = ea + static_cast<size_t>(s_tok[m]) * G;
    int mx = -128;
    for (int g = lane; g < G; g += 32) mx = max(mx, static_cast<int>(row[g]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
    if (lane == 0) s_EA[m] = mx;
  }
  // EW[n] = max_g ws[eid, n, g] - 127 (contract_v2.prepare_weights, in-kernel) from this row's G scale codes, the two
  // lanes of the row sharing the words; the admission check of the scale codes raises FLAG_BAD_SCALE
  int ew_max = 0;
  bool bad_scale = false;
  {
    const uint8_t* wsrow = wsx + static_cast<size_t>(n) * G;
    for (int w = half; w < G / 4; w += 2) {
      const uint32_t v = *reinterpret_cast<const uint32_t*>(wsrow + 4 * w);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = static_cast<int>((v >> (8 * j)) & 0xFFu);
        ew_max = max(ew_max, c);
        bad_scale |= (c == 255) || (c < MIN_SCALE_CODE);
      }
    }
    ew_max = max(ew_max, __shfl_xor_sync(0xFFFFFFFFu, ew_max, 8));
    bad_scale |= (__shfl_xor_sync(0xFFFFFFFFu, bad_scale ? 1 : 0, 8) != 0);
  }
  const int ew_row = ew_max - 127;

  // conversion table (interleaved: entry c of replica rep at c * R + rep)
  if (F::LUT_INTS > 0) {
    for (int i = tid; i < F::LUT_INTS; i += THREADS) lut[i] = EXPERT_W_OF_NIBBLE[i / EXPERT_LUT_REPL];
  }
  __syncthreads();                                       // s_EA and the table visible

  int cm[MT], eam[MT];
  int64_t acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) { eam[m] = (m < cnt) ? s_EA[m] : 0; cm[m] = F::MARGIN - eam[m] - ew_row; acc[m] = 0; }
  int64_t ovf = 0;
  bool meta_bad = false;                                 // L5: a staged ea_g > EA[m] or ew_g > EW[n] (defence)
  bool nan_seen = false;                                 // a staged activation code 127 / 255
  const int nchunk = (G + BPC - 1) / BPC;
  constexpr int off = 0;                                 // S = 1: the staged exponent window starts at the chunk's first group

#pragma unroll
  for (int c = 0; c < STAGES - 1; ++c) {
    if (c < nchunk)
      load_chunk<MT>(&stages[c], Wx, wsx, C, ea, s_tok, cnt, K, G, row0, c * BPC, min(BPC, G - c * BPC), tid);
    cp_commit();
  }

  for (int c = 0; c < nchunk; ++c) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int cn = c + STAGES - 1;
      if (cn < nchunk)
        load_chunk<MT>(&stages[cn % STAGES], Wx, wsx, C, ea, s_tok, cnt, K, G, row0, cn * BPC, min(BPC, G - cn * BPC), tid);
      cp_commit();
    }
    const Stage<MT>& st = stages[c % STAGES];
    const int nb = min(BPC, G - c * BPC);

    // fused prep: the chunk's staged codes -> int32 integers in X (dense_m_arith, the prep kernel's form), cooperatively:
    // 64 threads x 4 consecutive codes per slot per pass, MT passes; NaN codes raise the flag as the prep kernel did
    for (int i = tid; i < MT * (X_CHUNK_INTS / 4); i += THREADS) {
      const int m = i / (X_CHUNK_INTS / 4), q = i % (X_CHUNK_INTS / 4);
      if (m < cnt) {
        const uint32_t cw = *reinterpret_cast<const uint32_t*>(&st.C[m][4 * q]);
        nan_seen |= ((cw & 0x7Fu) == 0x7Fu) || (((cw >> 8) & 0x7Fu) == 0x7Fu)
                 || (((cw >> 16) & 0x7Fu) == 0x7Fu) || (((cw >> 24) & 0x7Fu) == 0x7Fu);
        int4 v;
        v.x = dense_m_arith(cw & 0xFFu);
        v.y = dense_m_arith((cw >> 8) & 0xFFu);
        v.z = dense_m_arith((cw >> 16) & 0xFFu);
        v.w = dense_m_arith(cw >> 24);
        *reinterpret_cast<int4*>(&X[m][4 * q]) = v;
      }
    }
    __syncthreads();                                     // X of this chunk visible

#pragma unroll
    for (int b = 0; b < BPC; ++b) {
      if (b < nb) {
        int32_t w[16];
        convert_weights<F>(&st.W[r][0], r, half, b, lut, lane, w);
        const int ew_g = static_cast<int>(st.EW[r][off + b]) - 127;
        meta_bad |= (ew_g > ew_row);                               // L5: a staged ew_g above the row's EW
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m < cnt) {                                           // CTA-uniform: no divergence inside the warp
            const int4* xp = reinterpret_cast<const int4*>(&X[m][b * BLOCK_K + half * 16]);
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
  }

  unsigned int my_flags = 0u;
  if (ovf < 0) my_flags |= FLAG_OVERFLOW;
  if (meta_bad) my_flags |= FLAG_META;
  if (nan_seen) my_flags |= FLAG_NAN_CODE;
  if (bad_scale) my_flags |= FLAG_BAD_SCALE;
  if (cnt_all > MT) my_flags |= FLAG_SLOTS;
  if (half == 0) {
    bool bad = false;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < cnt) {
        const int B = eam[m] + ew_row - F::MARGIN;
        const uint16_t bits = finish_bf16(acc[m], B - SA - F::SW, bad);
        const size_t o = static_cast<size_t>(s_slot[m]) * N + n;          // outputs per route
        if (FIRST) reinterpret_cast<uint32_t*>(Y)[o] = static_cast<uint32_t>(bits) << 16;   // BF16 value as FP32 (exact)
        else       reinterpret_cast<uint16_t*>(Y)[o] = bits;
        if (A_dbg != nullptr) A_dbg[o] = acc[m];
      }
    }
    if (bad) my_flags |= FLAG_FINISH;
  }
  if (my_flags) atomicOr(flag, my_flags);                // only on a flagged case; no atomic runs on a clean call
}

// --------------------------------------------------------------------------------------------------- host launch
static bool g_configured = false;                        // configure() ran (dynamic smem attribute of every instantiation)

template <int MT>
static void configure_mt() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(expert_static_kernel<MT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem_bytes_for<MT>())));
  C10_CUDA_CHECK(cudaFuncSetAttribute(expert_static_kernel<MT, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem_bytes_for<MT>())));
}

template <int MT, bool FIRST>
static void launch_mt(const uint8_t* W, const uint8_t* ws, const uint8_t* C, const int8_t* ea, const void* active,
                      int active_i64, int count, int topk, void* Y, int64_t* A_dbg, unsigned int* flag, int N, int K,
                      cudaStream_t stream) {
  TORCH_CHECK(g_configured, "patha_expert_static: configure() must run before the first launch (patha_expert_static._load does, outside any capture)");
  const dim3 grid(N / BN, 1, count), block(THREADS);
  expert_static_kernel<MT, FIRST><<<grid, block, smem_bytes_for<MT>(), stream>>>(W, ws, C, ea, active, active_i64, count, topk,
                                                                               Y, A_dbg, flag, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool FIRST>
static void launch(int M, const uint8_t* W, const uint8_t* ws, const uint8_t* C, const int8_t* ea, const void* active,
                   int active_i64, int count, int topk, void* Y, int64_t* A_dbg, unsigned int* flag, int N, int K,
                   cudaStream_t stream) {
  switch (M) {
    case 1: launch_mt<1, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 2: launch_mt<2, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 3: launch_mt<3, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 4: launch_mt<4, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 5: launch_mt<5, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 6: launch_mt<6, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 7: launch_mt<7, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    case 8: launch_mt<8, FIRST>(W, ws, C, ea, active, active_i64, count, topk, Y, A_dbg, flag, N, K, stream); break;
    default: TORCH_CHECK(false, "patha_expert_static: 1 <= M <= 8");
  }
}

static size_t smem_bytes_of(int M) {
  switch (M) {
    case 1: return smem_bytes_for<1>(); case 2: return smem_bytes_for<2>();
    case 3: return smem_bytes_for<3>(); case 4: return smem_bytes_for<4>();
    case 5: return smem_bytes_for<5>(); case 6: return smem_bytes_for<6>();
    case 7: return smem_bytes_for<7>(); case 8: return smem_bytes_for<8>();
    default: return 0;
  }
}

}  // namespace patha_es

// --------------------------------------------------------------------------------------------------- torch binding
namespace {

using namespace patha_es;

// Set the dynamic shared memory attribute of all 16 instantiations. Call once after loading, outside any capture.
void configure() {
  configure_mt<1>(); configure_mt<2>(); configure_mt<3>(); configure_mt<4>();
  configure_mt<5>(); configure_mt<6>(); configure_mt<7>(); configure_mt<8>();
  g_configured = true;
}

void expert_static_impl(const torch::Tensor& codes, const torch::Tensor& ea, const torch::Tensor& W, const torch::Tensor& ws,
                        const torch::Tensor& active, const torch::Tensor& Y, int64_t* A_dbg, const torch::Tensor& flag,
                        bool first, int64_t count, int64_t topk, int64_t M) {
  TORCH_CHECK(codes.is_cuda() && ea.is_cuda() && W.is_cuda() && ws.is_cuda() && active.is_cuda() && Y.is_cuda() && flag.is_cuda(),
              "patha_expert_static: CUDA tensors required");
  TORCH_CHECK(codes.scalar_type() == at::kByte, "patha_expert_static: activation codes must be uint8 (view float8_e4m3fn as uint8)");
  TORCH_CHECK(ea.scalar_type() == at::kChar, "patha_expert_static: ea must be int8");
  TORCH_CHECK(W.scalar_type() == at::kByte && ws.scalar_type() == at::kByte, "patha_expert_static: nibbles and scale codes must be uint8");
  TORCH_CHECK(active.scalar_type() == at::kInt || active.scalar_type() == at::kLong, "patha_expert_static: active must be int32 or int64");
  TORCH_CHECK(flag.scalar_type() == at::kInt && flag.numel() == 1, "patha_expert_static: flag must be one int32 word");
  TORCH_CHECK(codes.is_contiguous() && ea.is_contiguous() && W.is_contiguous() && ws.is_contiguous() && active.is_contiguous()
              && Y.is_contiguous() && flag.is_contiguous(), "patha_expert_static: contiguous tensors required");
  TORCH_CHECK(codes.dim() == 2 && ea.dim() == 2 && W.dim() == 3 && ws.dim() == 3 && Y.dim() == 2, "patha_expert_static: ranks (codes [R,K], ea [R,K/32], W [E,N,K/2], ws [E,N,K/32], Y [count,N])");
  TORCH_CHECK(M >= 1 && M <= MAX_M, "patha_expert_static: 1 <= M <= 8");
  TORCH_CHECK(topk >= 1 && topk <= MAX_TOPK, "patha_expert_static: 1 <= topk <= 6");
  TORCH_CHECK(count == M * topk && count >= 1 && count <= MAX_COUNT, "patha_expert_static: count = M x topk <= 64");
  TORCH_CHECK(active.numel() == count, "patha_expert_static: active must hold count routes");
  const int R = static_cast<int>(codes.size(0)), K = static_cast<int>(codes.size(1));
  TORCH_CHECK(R == (first ? M : count), "patha_expert_static: activation rows must be M (gate/up) or count (down)");
  TORCH_CHECK(K % (BLOCK_K * 4) == 0, "patha_expert_static: K % 128 == 0 (four-byte exponent copies need K/32 % 4 == 0)");
  const int G = K / BLOCK_K;
  TORCH_CHECK(ea.size(0) == R && ea.size(1) == G, "patha_expert_static: ea must be [R, K/32]");
  const int N = static_cast<int>(W.size(1));
  TORCH_CHECK(N % BN == 0, "patha_expert_static: N % 32 == 0");
  TORCH_CHECK(W.size(2) == static_cast<int64_t>(F::row_bytes(K)), "patha_expert_static: nibble row length must be K/2 bytes");
  TORCH_CHECK(ws.size(0) == W.size(0) && ws.size(1) == N && ws.size(2) == G, "patha_expert_static: scales must be [E, N, K/32]");
  TORCH_CHECK(Y.size(0) == count && Y.size(1) == N, "patha_expert_static: Y must be [count, N]");
  if (first) {
    TORCH_CHECK(Y.scalar_type() == at::kFloat, "patha_expert_static: gate/up output must be float32 (BF16 values)");
  } else {
    TORCH_CHECK(Y.scalar_type() == at::kBFloat16, "patha_expert_static: down output must be bfloat16");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0, "patha_expert_static: codes must be 16-byte aligned (cp.async)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, "patha_expert_static: nibbles must be 16-byte aligned (cp.async)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(ea.data_ptr()) % 4 == 0 && reinterpret_cast<uintptr_t>(ws.data_ptr()) % 4 == 0,
              "patha_expert_static: ea and scale codes must be 4-byte aligned");
  const c10::cuda::CUDAGuard guard(codes.device());
  const int active_i64 = active.scalar_type() == at::kLong ? 1 : 0;
  unsigned int* fl = reinterpret_cast<unsigned int*>(flag.data_ptr<int32_t>());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (first)
    launch<true>(static_cast<int>(M), W.data_ptr<uint8_t>(), ws.data_ptr<uint8_t>(), codes.data_ptr<uint8_t>(), ea.data_ptr<int8_t>(),
                 active.data_ptr(), active_i64, static_cast<int>(count), static_cast<int>(topk), Y.data_ptr(), A_dbg, fl, N, K, stream);
  else
    launch<false>(static_cast<int>(M), W.data_ptr<uint8_t>(), ws.data_ptr<uint8_t>(), codes.data_ptr<uint8_t>(), ea.data_ptr<int8_t>(),
                  active.data_ptr(), active_i64, static_cast<int>(count), static_cast<int>(topk), Y.data_ptr(), A_dbg, fl, N, K, stream);
}

void expert_static(torch::Tensor codes, torch::Tensor ea, torch::Tensor W, torch::Tensor ws, torch::Tensor active,
                   torch::Tensor Y, torch::Tensor flag, bool first, int64_t count, int64_t topk, int64_t M) {
  expert_static_impl(codes, ea, W, ws, active, Y, nullptr, flag, first, count, topk, M);
}

// The same launch also storing the raw int64 row sums A [count, N] (identity matrix: the oracle's accumulate_exact).
void expert_static_debug(torch::Tensor codes, torch::Tensor ea, torch::Tensor W, torch::Tensor ws, torch::Tensor active,
                         torch::Tensor Y, torch::Tensor A_dbg, torch::Tensor flag, bool first, int64_t count, int64_t topk, int64_t M) {
  TORCH_CHECK(A_dbg.is_cuda() && A_dbg.scalar_type() == at::kLong && A_dbg.is_contiguous() && A_dbg.dim() == 2
              && A_dbg.size(0) == count && A_dbg.size(1) == W.size(1), "patha_expert_static: A_dbg must be contiguous int64 [count, N]");
  expert_static_impl(codes, ea, W, ws, active, Y, A_dbg.data_ptr<int64_t>(), flag, first, count, topk, M);
}

int64_t smem_bytes(int64_t M) { return static_cast<int64_t>(smem_bytes_of(static_cast<int>(M))); }

int64_t build_config(int64_t which) {
  switch (which) {
    case 0: return PATHA_EXPERT_LUT; case 1: return PATHA_STAGES; case 2: return PATHA_EXPERT_BPC; case 3: return PATHA_EXPERT_TC;
    default: return -1;
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("configure", &configure, "set the dynamic shared memory attribute of every (M, projection) instantiation; outside any capture");
  m.def("expert_static", &expert_static, "codes u8 [R,K], ea i8 [R,K/32], W u8 [E,N,K/2], ws u8 [E,N,K/32], active i32/i64 [count], Y f32 (gate/up) or bf16 (down) [count,N], flag i32 [1], first, count, topk, M");
  m.def("expert_static_debug", &expert_static_debug, "as expert_static, also storing the raw int64 row sums A [count, N]");
  m.def("smem_bytes", &smem_bytes, "dynamic shared memory per CTA for M");
  m.def("build_config", &build_config, "0: PATHA_EXPERT_LUT, 1: PATHA_STAGES, 2: PATHA_EXPERT_BPC, 3: PATHA_EXPERT_TC");
}
