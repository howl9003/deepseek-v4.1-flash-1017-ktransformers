// patha_prefill.cu - path A kernel 2: chunked-fold INT8 tensor-core prefill GEMM (DESIGN_r1.md section 2) for the
// DeepSeek V4.1-Flash-1017 dense FP8 projections and the streamed routed experts on sm_89 (RTX 4080 SUPER).
// Revision r3-P3 (29 Oct 2026): r3-P2 plus a device-side live row count for the prep kernel and the ungrouped GEMM
// (the streamed-prefill integration runs both over a static 16,636-row workspace with about 300 live rows a group:
// rows at or beyond the count are skipped, their planes, metadata and outputs left untouched; live rows produce the
// same bytes as the unbounded kernels). Codex's r4 report (sha256 23c9e81e...): r3-P2 exact on 2,466 comparisons,
// gate/up 0.295 ms at M = 256 and 31.7 percent of peak at M = 6144 (both gates pass), down3 0.199 ms (miss), down4
// 0.162 ms (pass); group32 favours down3.
// Revision r3-P2 (29 Oct 2026): r3-P with two corrections from Codex's r3 build report
// (PATHA_PREFILL_RESULTS_FOR_CLAUDE_r3.md, sha256 b7020022...): (1) the expert expander's sign masks use inline PTX
// prmt.b32 (CUDA's __byte_perm intrinsic masks bit 3 of every selector nibble, so the sign-replicate selectors 0x9D8C /
// 0xBFAE reached the hardware as 0x1504 / 0x3726 and the first expert identity case failed; __byte_perm stays for the
// magnitude lookups, whose selectors are masked to 0x7777 by construction); (2) the gate/up projection uses the fold
// chunk of one stage (F = 1) like down, the configuration Codex built clean (151 registers, zero stack, no LDL/STL;
// F = 2 compiled to 199 registers with 8 bytes of stack), so the two expert projections share one instantiation and
// the 8-group fold chunk stays a modelled, gated, uninstantiated knob.
// Revision r3-P (29 Oct 2026): the price revision of the expert path after Codex's r2 receipt
// (PATHA_PREFILL_RESULTS_FOR_CLAUDE_r2.md: exact everywhere, zero spills, dense at 28 to 29.5 percent of peak, every
// expert row at 12 to 18 percent). The expert kernels now run a 64 x 64 CTA tile with eight warps (two per SM
// sub-partition: the warps of the upper half-tile expand the next stage's weights after their MMAs, the lower half
// before, so one warp's integer work overlaps the other's tensor work), expand the E2M1 nibbles by byte permutation
// (prmt, about 1.5 integer ops per nibble instead of about 20), decide the fold per 16 token rows x 16 weight rows
// (one m16 MMA tile by one warp's columns) instead of per CTA tile, run the fold chunk of the gate/up projection over
// two pipeline stages (one combine + shift + add per output per 8 groups), fold three bits of activation spread with
// three digit planes (the top digit uses the full int8 range), and hoist every exponent-metadata check to one row per
// lane per stage. The dense instantiation keeps r2's 64 x 32 tile and four warps. The arithmetic is r1's, unchanged.
//
// Exact integer arithmetic only. Per 32-wide group the oracle (contract_a.py) is D_g = sum n_k m_k with the E4M3
// activation integer n (< 2^18) and the weight integer m (dense E4M3 < 2^18, expert E2M1 halves |m| <= 12); rows
// accumulate as A = sum_g D_g << (ea_g + ew_g - B) in int64 (floor for negative shifts, right shifts clamped at 63),
// B = EA + EW - MARGIN with MARGIN 13 for the dense kernel and 20 for the expert kernel (patha_sequence.h; H1 of
// CONTRACT_FIXES_r1.md: the dense row sum stays below 2^61.62 at K <= 8192, so the int64 overflow flag is a defence
// no legal input can raise). Finish: the finish rule of CONTRACT_PINS_r1.md (M1), the same bits and flag as
// patha_decode.cu, contract_a.finish_with_flag and the NumPy models: f = __ll2float_rn(A) (one direct rounding),
// y = f * 2^(B - SA - SW) formed on the exponent field; A == 0 -> +0; overflow -> signed infinity and
// ERR_FINISH_RANGE; underflow -> signed zero and ERR_FINISH_RANGE; then the integer RNE to BF16 (contract_v2.bf,
// bit-identical to __float2bfloat16_rn). No floating-point multiply is emitted in the finish. Exponent metadata
// (EA from the prep kernel, EW from the host) is checked against every staged group exponent (ERR_META, L5): the
// expander checks the fold chunk's weight exponents of its row once per stage, each lane checks the stage's four
// activation exponents of one token row once per stage; the per-output "shift above the margin" check of r2 is
// implied by these two (ea_min <= max_g ea_g <= EA and ew_min <= EW give every fold shift <= MARGIN) and is no
// longer evaluated per output; the per-group path keeps its per-term check.
// Here the integers are split into base-128 digits (three int8 planes for n and dense m, with the top digit in the
// full int8 range; four planes for the down knob; one plane for the expert m shifted by at most 3 bits), the digit
// planes multiply on mma.sync.m16n8k32.s32.s8.s8.s32 (IMMA.16832, exact int32), and within a fold chunk of 4F groups
// the per-group shifts are folded into the integers before the split whenever the row's exponent spread is within
// budget, so the plane accumulators chain across the chunk.
//
// Data movement is route 3's (route3_decode.cu): cp.async 16-byte pieces into 128-byte plane rows swizzled by
// piece ^ (row & 7), ldmatrix fragments with the same lane addressing (the .s8 fragment map is the .e4m3 one),
// one cp.async group and one __syncthreads per stage. The weight code tile of stage x+1 is staged by cp.async one
// stage ahead and expanded into the next buffer's plane region during stage x.
//
// Layouts (binding): activation planes int8 [AP, Mtot, K] (plane-major, K innermost); ameta int8 [NCF, Mtot, 4] =
// {ea_min, eligible, EA, 0} per (fold chunk of 128 F, row); ea int8 [Mtot, G]; dense weights uint8 [N, K] E4M3 codes
// (row stride ldw); expert weights uint8 [E, N, K/2] adjacent nibbles (expert stride wstride, row stride ldw);
// ew int8 [E?, N, G] (per-row group exponents, scale code - 127); EW int8 [E?, N] (row maxima); row_off int32 [E+1]
// (token rows sorted by expert; absent = one expert, rows [0, Mtot)); out_bf16 [Mtot, N] and/or out_i64 [Mtot, N]
// (A before the finish).
#include <cuda_runtime.h>
#include <cstdint>
#include <optional>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "patha_sequence.h"

namespace patha {

__device__ unsigned int g_error = 0u;

// ------------------------------------------------------------------------------------------ PTX helpers (route 3)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(src_size));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int PENDING> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(PENDING));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t (&r)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
// prmt.b32 in its default mode: result byte i = byte sel_i[2:0] of {a, b}, and with sel_i bit 3 set the msb of that
// byte replicated over the result byte. CUDA's __byte_perm intrinsic discards bit 3 of every selector nibble (Codex's
// r3 SASS: 0x9D8C and 0xBFAE arrived as 0x1504 and 0x3726), so every selector that relies on sign replication goes
// through this inline instruction; __byte_perm is used only with selectors masked to 0x7777.
__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// The pinned instruction: one k32 INT8 MMA accumulating into int32 (IMMA.16832.S8.S8). The accumulators start from
// zero per plane group per fold chunk (fold) or per group (per-group path); their magnitude stays below 2^21 F.
__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------------------------------ integer helpers

// The checked int64 add of the per-group path: the flag is set per add (rare path).
__device__ __forceinline__ long long add_checked(long long a, long long b, unsigned& err) {
  const long long r = static_cast<long long>(static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b));
  if (((a ^ r) & (b ^ r)) < 0) err |= ERR_ACC_OVERFLOW;
  return r;
}
// The checked int64 add of the fold path: the overflow indicator (a ^ r) & (b ^ r) has its sign bit set exactly when
// the add overflowed; its high word is OR-ed into one register and tested once at the end (ERR_ACC_OVERFLOW is set if
// any add overflowed, the same condition as add_checked's, one integer op per add instead of a compare and a branch).
__device__ __forceinline__ long long add_acc(long long a, long long b, int& ovf) {
  const long long r = static_cast<long long>(static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b));
  ovf |= static_cast<int>(static_cast<unsigned long long>((a ^ r) & (b ^ r)) >> 32);
  return r;
}
__device__ __forceinline__ long long shl64(long long x, int s) {
  return static_cast<long long>(static_cast<unsigned long long>(x) << s);
}
__device__ __forceinline__ int shl32(int x, int s) { return static_cast<int>(static_cast<unsigned>(x) << s); }
// The oracle's per-term shift: left for non-negative shifts (s <= MARGIN by the metadata check; clamped at 63 so a
// flagged stale table cannot reach an undefined shift), arithmetic right (floor) otherwise, clamped at 63 as the
// decode kernel clamps it (M5): exact, since |x| < 2^47 floors to 0 or -1 at any shift of 47 or more, and a shift of
// 64 or more would be undefined in C++ and poison in LLVM although PTX clamps.
__device__ __forceinline__ long long shift_term(long long x, int s) {
  return s >= 0 ? shl64(x, min(s, 63)) : (x >> min(-s, 63));
}

// Base-128 digits of x, each placed as byte `byte` of the packed words: the low digits balanced in [-64, 63], the top
// digit the remaining carry in the full int8 range [-128, 127] (ERR_DIGIT_RANGE otherwise). Three digits hold
// |x| <= 2,089,024 (an E4M3 integer shifted by up to 3 bits), four hold |x| <= 267,386,944 (up to 10 bits).
__device__ __forceinline__ void digits3(int x, int byte, uint32_t& p0, uint32_t& p1, uint32_t& p2, unsigned& err) {
  const int d0 = digit0(x);
  const int r1 = carry(x, d0);
  const int d1 = digit0(r1);
  const int d2 = carry(r1, d1);
  if (d2 < TOP_DIGIT_MIN || d2 > TOP_DIGIT_MAX) err |= ERR_DIGIT_RANGE;
  p0 |= static_cast<uint32_t>(d0 & 255) << (8 * byte);
  p1 |= static_cast<uint32_t>(d1 & 255) << (8 * byte);
  p2 |= static_cast<uint32_t>(d2 & 255) << (8 * byte);
}
__device__ __forceinline__ void digits4(int x, int byte, uint32_t& p0, uint32_t& p1, uint32_t& p2, uint32_t& p3, unsigned& err) {
  const int d0 = digit0(x);
  const int r1 = carry(x, d0);
  const int d1 = digit0(r1);
  const int r2 = carry(r1, d1);
  const int d2 = digit0(r2);
  const int d3 = carry(r2, d2);
  if (d3 < TOP_DIGIT_MIN || d3 > TOP_DIGIT_MAX) err |= ERR_DIGIT_RANGE;
  p0 |= static_cast<uint32_t>(d0 & 255) << (8 * byte);
  p1 |= static_cast<uint32_t>(d1 & 255) << (8 * byte);
  p2 |= static_cast<uint32_t>(d2 & 255) << (8 * byte);
  p3 |= static_cast<uint32_t>(d3 & 255) << (8 * byte);
}
__device__ __forceinline__ int sbyte(uint32_t w, int b) {
  return static_cast<int>(static_cast<int8_t>((w >> (8 * b)) & 255u));
}

// The finish rule (M1; identical text in patha_decode.cu, and the same bits and flag as contract_a.finish_with_flag
// and the NumPy models):
//   f = __ll2float_rn(A): one direct int64 -> FP32 rounding (pin a: never through double).
//   A == 0: y = +0, never flagged.  Otherwise ex = biased exponent of f plus e2:
//   1 <= ex <= 254: y = f * 2^e2 exactly, formed by writing ex into the exponent field (bit-identical to ldexpf,
//                   which the normal range makes exact; no floating-point multiply is emitted);
//   ex > 254 ("overflow"):  y = signed infinity, bad = true;
//   ex < 1   ("underflow"): y = signed zero,     bad = true.
//   Then one RNE to BF16 (contract_v2.bf, the bits of __float2bfloat16_rn); an FP32 y above the BF16 maximum rounds
//   to infinity unflagged (IEEE).
// Neither flagged case is reachable on legal inputs (CONTRACT_PINS_r1.md): the quantiser's 1e-4 amax floor gives
// EA >= -22 and the weight-load admission (scale codes >= 54 dense, 53 expert) gives ex >= 1 for the smallest
// nonzero A; the largest (|A| < 2^62) stays below ex = 255 while EA + EW <= 95.
__device__ __forceinline__ uint16_t finish_bf16(long long A, int e2, bool& bad) {
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

// ------------------------------------------------------------------------------------------ prep kernel (activations)
//
// One thread per (token row, fold chunk of 4F groups): reads 128 F E4M3 codes and the row's G exponents, writes the
// AP digit planes of the chunk-shifted integer (n << (ea_g - ea_min) when the row's spread inside the chunk is within
// the budget of AP digits, the digits of n otherwise) and ameta = {ea_min, eligible, EA, 0}. r3-P3: rows_dev (a device
// int32 scalar, or nullptr) is the live row count: rows at or beyond it are skipped and their planes and ameta left
// untouched (the GEMM never reads them: its grid is bounded by the same count or by the row-offset table).
template <int AP, int F>
__global__ void __launch_bounds__(128)
prep_kernel(const uint8_t* __restrict__ codes, const int8_t* __restrict__ ea, int M, int K,
            int8_t* __restrict__ planes, int8_t* __restrict__ ameta, const int32_t* __restrict__ rows_dev) {
  constexpr int CG = STAGE_GROUPS * F, CK = STAGE_K * F, PIECES = CK / 16, BUDGET = ActBudgetOf<AP>::value;
  const int NC = K / CK, G = K / GROUP;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * NC) return;
  const int m = idx / NC, c = idx % NC;
  const int live = rows_dev ? min(M, max(0, *rows_dev)) : M;
  if (m >= live) return;
  unsigned err = 0u;
  const int8_t* er = ea + static_cast<size_t>(m) * G;
  int EA = -128;
  for (int g = 0; g < G; ++g) EA = max(EA, static_cast<int>(er[g]));
  int e[CG];
  int emin = 127, emax = -128;
#pragma unroll
  for (int g = 0; g < CG; ++g) {
    e[g] = er[c * CG + g];
    emin = min(emin, e[g]);
    emax = max(emax, e[g]);
  }
  const bool elig = (emax - emin) <= BUDGET;
  const uint4* src = reinterpret_cast<const uint4*>(codes + static_cast<size_t>(m) * K + c * CK);
  const size_t plane_stride = static_cast<size_t>(M) * K;
  int8_t* dst0 = planes + static_cast<size_t>(m) * K + c * CK;
#pragma unroll
  for (int piece = 0; piece < PIECES; ++piece) {
    const int s = elig ? (e[piece >> 1] - emin) : 0;          // 16 elements per piece, 2 pieces per group
    const uint4 w = __ldg(src + piece);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    uint32_t p0[4] = {0u, 0u, 0u, 0u}, p1[4] = {0u, 0u, 0u, 0u}, p2[4] = {0u, 0u, 0u, 0u}, p3[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const unsigned cb = (ww[k] >> (8 * b)) & 255u;
        if (e4m3_is_nan(cb)) err |= ERR_NAN_ACT;
        const int mag = e4m3_mag(cb);
        const int x = shl32((cb & 128u) ? -mag : mag, s);
        if constexpr (AP == 3) digits3(x, b, p0[k], p1[k], p2[k], err);
        else digits4(x, b, p0[k], p1[k], p2[k], p3[k], err);
      }
    }
    *reinterpret_cast<uint4*>(dst0 + piece * 16) = make_uint4(p0[0], p0[1], p0[2], p0[3]);
    *reinterpret_cast<uint4*>(dst0 + plane_stride + piece * 16) = make_uint4(p1[0], p1[1], p1[2], p1[3]);
    *reinterpret_cast<uint4*>(dst0 + 2 * plane_stride + piece * 16) = make_uint4(p2[0], p2[1], p2[2], p2[3]);
    if constexpr (AP == 4) {
      *reinterpret_cast<uint4*>(dst0 + 3 * plane_stride + piece * 16) = make_uint4(p3[0], p3[1], p3[2], p3[3]);
    }
  }
  const uint32_t meta = (static_cast<uint32_t>(emin & 255)) | (static_cast<uint32_t>(elig ? 1 : 0) << 8) |
                        (static_cast<uint32_t>(EA & 255) << 16);
  *reinterpret_cast<uint32_t*>(ameta + (static_cast<size_t>(c) * M + m) * META_BYTES) = meta;
  if (err) atomicOr(&g_error, err);
}

// ------------------------------------------------------------------------------------------ GEMM kernel

struct GemmArgs {
  const int8_t* planes;      // [AP][Mtot][K]
  const int8_t* ameta;       // [NCF][Mtot][4]
  const int8_t* ea;          // [Mtot][G]
  const uint8_t* w;          // dense [N][K] codes; expert [E][N][K/2] nibbles
  const int8_t* ew;          // [E?][N][G]
  const int8_t* EW;          // [E?][N]
  const int32_t* row_off;    // [E+1] or nullptr
  const int32_t* rows;       // device live row count for the ungrouped launch (nullptr = Mtot); r3-P3
  uint16_t* out_bf16;        // [Mtot][N] BF16 bits or nullptr
  long long* out_i64;        // [Mtot][N] or nullptr
  int Mtot, N, K;
  long long ldw, wstride;    // bytes
  long long ldew, ewstride;  // bytes
  int finish_base;           // -(SA + SW): e2 = B + finish_base in finish_bf16
};

// Shared-memory stage (patha_sequence.h): A region | B region | W region
template <class C> struct StageView {
  uint8_t* base;
  __device__ __forceinline__ uint8_t* aplane(int i) const { return base + i * C::PLANE_BYTES; }
  __device__ __forceinline__ uint32_t* ameta() const { return reinterpret_cast<uint32_t*>(base + C::AP * C::PLANE_BYTES); }
  __device__ __forceinline__ uint32_t* aexp() const { return ameta() + BM; }
  __device__ __forceinline__ uint8_t* bplane(int j) const { return base + C::A_REGION + j * C::BPLANE_BYTES; }
  __device__ __forceinline__ uint32_t* bmeta() const { return reinterpret_cast<uint32_t*>(base + C::A_REGION + C::WP * C::BPLANE_BYTES); }
  // the stage's four weight-row group exponents, copied by the expander from the W region (which the next cp.async
  // overwrites during this stage's MMAs) so the per-group path can read them
  __device__ __forceinline__ uint32_t* bexpb() const { return bmeta() + C::BN; }
  __device__ __forceinline__ uint8_t* wcodes() const { return base + C::A_REGION + C::B_REGION; }
  __device__ __forceinline__ uint32_t* bexp() const { return reinterpret_cast<uint32_t*>(wcodes() + C::WCODE_BYTES); }   // [BN][F]
};

// Asynchronous copies of stage x's activation planes, the fold chunk's ameta and the stage's four group exponents
// per token row.
template <class C>
__device__ __forceinline__ void load_a(const GemmArgs& a, StageView<C> st, int x, int row0, int nrows, int G, int tid) {
  constexpr int NPIECE = C::AP * BM * CHUNKS16;
  const size_t plane_stride = static_cast<size_t>(a.Mtot) * a.K;
#pragma unroll
  for (int i = tid; i < NPIECE; i += C::THREADS) {
    const int plane = i / (BM * CHUNKS16), rem = i % (BM * CHUNKS16);
    const int row = rem / CHUNKS16, piece = rem % CHUNKS16;
    const bool live = row < nrows;
    const int8_t* src = a.planes + plane * plane_stride + static_cast<size_t>(row0 + (live ? row : 0)) * a.K +
                        static_cast<size_t>(x) * STAGE_K + piece * 16;
    cp_async16(smem_u32(st.aplane(plane) + row * STAGE_K + swz(row, piece) * 16), src, live ? 16 : 0);
  }
  if (tid < BM) {
    const int row = tid;
    const bool live = row < nrows;
    const int8_t* src = a.ameta + (static_cast<size_t>(x / C::F) * a.Mtot + row0 + (live ? row : 0)) * META_BYTES;
    cp_async4(smem_u32(st.ameta() + row), src, live ? 4 : 0);
  } else if (tid < 2 * BM) {
    const int row = tid - BM;
    const bool live = row < nrows;
    const int8_t* src = a.ea + static_cast<size_t>(row0 + (live ? row : 0)) * G + x * STAGE_GROUPS;
    cp_async4(smem_u32(st.aexp() + row), src, live ? 4 : 0);
  }
}

// Asynchronous copies of stage x's weight codes (BN rows, 128 dense / 64 expert bytes each, swizzled pieces) and
// the fold chunk's 4F group exponents per weight row, into the buffer's W region.
template <class C>
__device__ __forceinline__ void load_w(const GemmArgs& a, StageView<C> st, int x, int col0, int e, int tid) {
  constexpr int NP = C::WPIECES;              // 16-byte pieces per code row: 8 dense, 4 expert
  constexpr int ROW_BYTES = NP * 16;
  const uint8_t* W = a.w + static_cast<long long>(e) * a.wstride;
#pragma unroll
  for (int i = tid; i < C::BN * NP; i += C::THREADS) {
    const int row = i / NP, piece = i % NP;
    const uint8_t* src = W + static_cast<long long>(col0 + row) * a.ldw + static_cast<long long>(x) * ROW_BYTES + piece * 16;
    cp_async16(smem_u32(st.wcodes() + row * ROW_BYTES + swz_w(row, piece, NP) * 16), src, 16);
  }
  if (tid < C::BN) {
    const int8_t* src = a.ew + static_cast<long long>(e) * a.ewstride + static_cast<long long>(col0 + tid) * a.ldew +
                        (x / C::F) * C::CHUNK_GROUPS;
    if constexpr (C::F == 1) cp_async4(smem_u32(st.bexp() + tid), src, 4);
    else cp_async8(smem_u32(st.bexp() + tid * C::F), src, 8);
  }
}

// Expand the staged codes of the buffer's W region into its B region: thread (row = tid / 4, group q = tid % 4)
// handles the row's 32 elements of group q of the stage (sub = the stage's index inside its fold chunk). Dense:
// m = +-N_OF_CODE shifted by ew_g - ew_min when the row's chunk spread is within budget, three digit planes;
// expert: w' (halves) shifted by up to 3 bits, one plane, by byte permutation (patha_sequence.h). bmeta (the fold
// chunk's {ew_min, eligible, EW, 0}) and bexpb (the stage's four exponents) by q == 0. L5: the chunk's exponents of
// the row are checked against EW here, once per row per stage, never per element.
template <class C>
__device__ __forceinline__ void expand_w(StageView<C> st, int tid, int EWr, int sub, unsigned& err) {
  constexpr int NP = C::WPIECES;
  constexpr int ROW_BYTES = NP * 16;
  const int wr = tid >> 2, q = tid & 3;
  uint32_t ews[C::F];
  int emin = 127, emax = -128;
#pragma unroll
  for (int f = 0; f < C::F; ++f) {
    ews[f] = st.bexp()[wr * C::F + f];
#pragma unroll
    for (int g = 0; g < STAGE_GROUPS; ++g) {
      const int eg = sbyte(ews[f], g);
      emin = min(emin, eg);
      emax = max(emax, eg);
    }
  }
  if (emax > EWr) err |= ERR_META;                                 // L5: a group exponent above the row's EW (stale EW)
  const bool elig = (emax - emin) <= C::W_BUDGET;
  const uint32_t ews_sub = (C::F == 1) ? ews[0] : ews[sub & (C::F - 1)];
  const int s = elig ? (sbyte(ews_sub, q) - emin) : 0;
  const uint8_t* crow = st.wcodes() + wr * ROW_BYTES;
  if constexpr (C::WP == 3) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {                                   // two code pieces = 32 codes = group q
      const int piece = 2 * q + k;
      const uint4 w = *reinterpret_cast<const uint4*>(crow + swz_w(wr, piece, NP) * 16);
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      uint32_t p0[4] = {0u, 0u, 0u, 0u}, p1[4] = {0u, 0u, 0u, 0u}, p2[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const unsigned cb = (ww[wd] >> (8 * b)) & 255u;
          if (e4m3_is_nan(cb)) err |= ERR_NAN_WEIGHT;
          const int mag = e4m3_mag(cb);
          const int x = shl32((cb & 128u) ? -mag : mag, s);
          digits3(x, b, p0[wd], p1[wd], p2[wd], err);
        }
      }
      const int off = wr * STAGE_K + swz(wr, piece) * 16;
      *reinterpret_cast<uint4*>(st.bplane(0) + off) = make_uint4(p0[0], p0[1], p0[2], p0[3]);
      *reinterpret_cast<uint4*>(st.bplane(1) + off) = make_uint4(p1[0], p1[1], p1[2], p1[3]);
      *reinterpret_cast<uint4*>(st.bplane(2) + off) = make_uint4(p2[0], p2[1], p2[2], p2[3]);
    }
  } else {
    // byte lookup: magnitudes 0,1,2,3 | 4,6,8,12 shifted by s (no byte carries: 12 << 3 = 96), negatives as the
    // per-byte two's complement (0x80 - v) ^ 0x80, the sign mask by prmt's sign-replicate selectors (inline PTX; the
    // magnitude selectors below are masked to 0x7777 and may use the __byte_perm intrinsic)
    const uint32_t T0 = LUT_POS0 << s, T1 = LUT_POS1 << s;
    const uint32_t T2 = (0x80808080u - T0) ^ 0x80808080u, T3 = (0x80808080u - T1) ^ 0x80808080u;
    const uint4 w = *reinterpret_cast<const uint4*>(crow + swz_w(wr, q, NP) * 16);   // 32 nibbles = group q
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    uint32_t p[8];                                                   // 32 elements -> 8 words
#pragma unroll
    for (int wd = 0; wd < 4; ++wd) {
      const uint32_t v = ww[wd], vs = v << 4;
      const uint32_t sel_lo = v & LUT_MAG_MASK, sel_hi = (v >> 16) & LUT_MAG_MASK;
      const uint32_t m_lo = prmt_b32(v, vs, LUT_SIGN_SEL_LO), m_hi = prmt_b32(v, vs, LUT_SIGN_SEL_HI);   // sign-replicate: inline prmt, never __byte_perm
      const uint32_t pos_lo = __byte_perm(T0, T1, sel_lo), neg_lo = __byte_perm(T2, T3, sel_lo);
      const uint32_t pos_hi = __byte_perm(T0, T1, sel_hi), neg_hi = __byte_perm(T2, T3, sel_hi);
      p[2 * wd] = (pos_lo & ~m_lo) | (neg_lo & m_lo);              // elements 8 wd .. 8 wd + 3
      p[2 * wd + 1] = (pos_hi & ~m_hi) | (neg_hi & m_hi);          // elements 8 wd + 4 .. 8 wd + 7
    }
    *reinterpret_cast<uint4*>(st.bplane(0) + wr * STAGE_K + swz(wr, 2 * q) * 16) = make_uint4(p[0], p[1], p[2], p[3]);
    *reinterpret_cast<uint4*>(st.bplane(0) + wr * STAGE_K + swz(wr, 2 * q + 1) * 16) = make_uint4(p[4], p[5], p[6], p[7]);
  }
  if (q == 0) {
    st.bmeta()[wr] = (static_cast<uint32_t>(emin & 255)) | (static_cast<uint32_t>(elig ? 1 : 0) << 8) |
                     (static_cast<uint32_t>(EWr & 255) << 16);
    st.bexpb()[wr] = ews_sub;
  }
}

// Combine the plane-group accumulators Q_s (s = i + j, |Q_s| < 2^21 F) into the int64 chunk (or group) dot:
// lo = Q0 + Q1 << 7 and mid = Q2 + Q3 << 7 are exact in int32; Dc = lo + mid << 14 (+ Q4 << 28) in int64.
template <class C>
__device__ __forceinline__ long long combine(const int (&q)[C::NS]) {
  const int lo = q[0] + shl32(q[1], 7);
  if constexpr (C::NS == 5) {
    const int mid = q[2] + shl32(q[3], 7);
    return static_cast<long long>(lo) + shl64(static_cast<long long>(mid), 14) + shl64(static_cast<long long>(q[4]), 28);
  } else if constexpr (C::NS == 4) {
    const int mid = q[2] + shl32(q[3], 7);
    return static_cast<long long>(lo) + shl64(static_cast<long long>(mid), 14);
  } else {
    return static_cast<long long>(lo) + shl64(static_cast<long long>(q[2]), 14);
  }
}

// One k32 block for one m16 tile and NTP n8 tiles: AP A ldmatrix.x4, WP B ldmatrix (x4 for a tile pair, x2 for one
// tile), AP x WP x NTP MMAs into the plane groups q[i + j][nt]. Piece offsets are computed, never indexed.
template <class C>
__device__ __forceinline__ void mma_block(const StageView<C>& st, int a_off, int b_off, int b, int lane7, int a_ch, int b_ch,
                                          int (&q)[C::NS][C::NTP][4]) {
  const int ap = ((2 * b + a_ch) ^ lane7) * 16;
  const int bp = ((2 * b + b_ch) ^ lane7) * 16;
  uint32_t af[C::AP][4];
#pragma unroll
  for (int i = 0; i < C::AP; ++i) ldmatrix_x4(smem_u32(st.aplane(i) + a_off + ap), af[i]);
  if constexpr (C::NTP == 2) {
    uint32_t bf[C::WP][4];
#pragma unroll
    for (int j = 0; j < C::WP; ++j) ldmatrix_x4(smem_u32(st.bplane(j) + b_off + bp), bf[j]);
#pragma unroll
    for (int i = 0; i < C::AP; ++i)
#pragma unroll
      for (int j = 0; j < C::WP; ++j) {
        mma_s8(q[i + j][0], af[i], bf[j][0], bf[j][1]);
        mma_s8(q[i + j][1], af[i], bf[j][2], bf[j][3]);
      }
  } else {
    uint32_t bf[C::WP][2];
#pragma unroll
    for (int j = 0; j < C::WP; ++j) ldmatrix_x2(smem_u32(st.bplane(j) + b_off + bp), bf[j]);
#pragma unroll
    for (int i = 0; i < C::AP; ++i)
#pragma unroll
      for (int j = 0; j < C::WP; ++j) mma_s8(q[i + j][0], af[i], bf[j][0], bf[j][1]);
  }
}

template <class C>
__global__ void __launch_bounds__(C::THREADS, 1)
gemm_kernel(GemmArgs a) {
  extern __shared__ __align__(16) uint8_t smem_raw[];
  constexpr int WP = C::WP;
  constexpr int S = C::STAGES;
  constexpr int F = C::F;
  constexpr int SB = C::STAGE_BYTES;
  constexpr int NSV = C::NS;
  constexpr int NTP = C::NTP;                         // n8 tiles per pass: 1 dense, 2 expert
  constexpr int NPASS = C::NPASS;
  constexpr int MARGIN = MarginOf<WP>::value;         // 13 dense, 20 expert (H1)

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tig = lane & 3, lane7 = lane & 7;
  const int wm = warp / C::WARPS_N, wn = warp % C::WARPS_N;
  const bool late = (wm == 1);                        // phase class: the upper half-tile's warps expand after their MMAs
  const int warp_m = wm * WARP_M, warp_n = wn * WARP_N;
  const int e = blockIdx.z;
  const int row_lo = a.row_off ? a.row_off[e] : 0;
  const int row_hi = a.row_off ? a.row_off[e + 1] : (a.rows ? min(a.Mtot, max(0, *a.rows)) : a.Mtot);   // r3-P3 bound
  const int row0 = row_lo + blockIdx.y * BM;
  if (row0 >= row_hi) return;                               // uniform across the CTA, before any barrier
  const int nrows = min(BM, row_hi - row0);
  const int col0 = blockIdx.x * C::BN;
  const int NST = a.K / STAGE_K, G = a.K / GROUP;
  const int EWr = a.EW[static_cast<long long>(e) * a.N + col0 + (tid >> 2)];   // the expander's row
  unsigned err = 0u;
  int ovf = 0;                                              // fold-path overflow indicator (sign bit)

  // ldmatrix lane addressing (route 3): A tiles (rows 0-7 | 8-15) x (k 0-15 | 16-31) -> a0..a3. B: with a tile pair
  // (x4) lanes 0-7 | 8-15 | 16-23 | 24-31 address (tile 0, k lo) | (tile 0, k hi) | (tile 1, k lo) | (tile 1, k hi);
  // with one tile (x2) lanes 0-7 | 8-15 address (k lo) | (k hi) and the upper lanes repeat them (ignored).
  const int a_row = lane7 + 8 * ((lane >> 3) & 1), a_ch = (lane >> 4) & 1;
  const int b_row = (NTP == 2) ? (lane7 + 8 * ((lane >> 4) & 1)) : lane7;
  const int b_ch = (lane >> 3) & 1;

  long long acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[mt][nt][k] = 0ll;
  // fold plane groups, live across the F stages of a fold chunk (F = 1: one (mt, np) at a time)
  int q[MT][NPASS][NSV][NTP][4];
  bool fold_mt[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) fold_mt[mt] = false;

  // buffer s lives at smem_raw + s * SB (computed, never an indexed array of views)
  auto stage = [&](int s) { StageView<C> v; v.base = smem_raw + s * SB; return v; };

  // prologue: codes of stage 0 staged, expanded into buffer 0's planes; then activation planes of stages 0..S-2 and
  // the codes of stages 1..S-1 in flight (one commit group per stage)
  load_w<C>(a, stage(0), 0, col0, e, tid);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  expand_w<C>(stage(0), tid, EWr, 0, err);
#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    if (s < NST) load_a<C>(a, stage(s), s, row0, nrows, G, tid);
    if (s + 1 < NST) load_w<C>(a, stage((s + 1) % S), s + 1, col0, e, tid);
    cp_commit();
  }

  for (int x = 0; x < NST; ++x) {
    cp_wait<S - 2>();                                       // stage x's planes and stage x+1's codes have landed
    __syncthreads();                                        // and every warp has finished stage x-1
    {
      const int xn = x + S - 1;                             // planes of stage xn -> buffer xn % S (last read at x-1)
      if (xn < NST) load_a<C>(a, stage(xn % S), xn, row0, nrows, G, tid);
      if (xn + 1 < NST) load_w<C>(a, stage((xn + 1) % S), xn + 1, col0, e, tid);   // codes of stage xn+1 -> W region of buffer x % S (expanded at x-1)
      cp_commit();
    }
    const bool expand_next = (x + 1 < NST);
    if (!late && expand_next) expand_w<C>(stage((x + 1) % S), tid, EWr, (x + 1) % F, err);   // planes of stage x+1 (B region last read at x+1-S)
    const StageView<C> st = stage(x % S);
    const uint32_t* am = st.ameta();
    const uint32_t* bm = st.bmeta();
    const uint32_t* ax = st.aexp();                         // token rows' four group exponents of this stage
    const uint32_t* bx = st.bexpb();                        // weight rows' four group exponents of this stage
    const int sub = x % F;                                  // the stage's index inside its fold chunk

    // L5: the stage's four activation group exponents of one live row per lane are checked against the row's EA
    // (the weight side is checked in the expander), so a stale ea / ameta pair is flagged on every stage.
    {
      const int r = warp_m + lane;
      if (r < nrows) {
        const uint32_t xe = ax[r];
        const int emax = max(max(sbyte(xe, 0), sbyte(xe, 1)), max(sbyte(xe, 2), sbyte(xe, 3)));
        if (emax > sbyte(am[r], 2)) err |= ERR_META;
      }
    }

    // fold decision per m16 tile of this warp x the warp's 16 columns, once per fold chunk (warp-uniform per tile):
    // all live token rows of the tile and all 16 weight rows eligible, and the chunk shift
    // (ea_min - EA) + (ew_min - EW) + MARGIN non-negative at its minimum over the tile. Dead rows (beyond nrows)
    // are neutral. Both paths produce the same bits; the decision only selects the cheaper one.
    if (sub == 0) {
      const int r = warp_m + lane;
      const bool live = r < nrows;
      const uint32_t xa = live ? am[r] : 0u;
      const bool okr = !live || ((xa >> 8) & 1u);
      const int da = live ? (sbyte(xa, 0) - sbyte(xa, 2)) : 0;
      const uint32_t yb = bm[warp_n + (lane & 15)];
      const bool okc_all = __all_sync(0xffffffffu, (yb >> 8) & 1u);
      const int dw_min = __reduce_min_sync(0xffffffffu, sbyte(yb, 0) - sbyte(yb, 2));
      const unsigned balr = __ballot_sync(0xffffffffu, okr);
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const bool okr_mt = ((balr >> (16 * mt)) & 0xFFFFu) == 0xFFFFu;
        const int da_min = __reduce_min_sync(0xffffffffu, ((lane >> 4) == mt) ? da : 0);
        const int smin = da_min + dw_min + MARGIN;
        fold_mt[mt] = okr_mt && okc_all && (smin >= 0);
        // ERR_NEG_FOLD_SHIFT: the fold path is entered only with smin >= 0, so a negative chunk shift on it is
        // impossible by construction; the check is the decision itself (retained bit).
        if (fold_mt[mt] && smin < 0) err |= ERR_NEG_FOLD_SHIFT;
      }
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int a_off = (warp_m + mt * 16 + a_row) * STAGE_K;
#pragma unroll
      for (int np = 0; np < NPASS; ++np) {
        const int b_off = (warp_n + np * (8 * NTP) + b_row) * STAGE_K;
        if (fold_mt[mt]) {
          if (sub == 0) {
#pragma unroll
            for (int s = 0; s < NSV; ++s)
#pragma unroll
              for (int ntl = 0; ntl < NTP; ++ntl)
#pragma unroll
                for (int k = 0; k < 4; ++k) q[mt][np][s][ntl][k] = 0;
          }
#pragma unroll 1
          for (int b = 0; b < STAGE_GROUPS; ++b) mma_block<C>(st, a_off, b_off, b, lane7, a_ch, b_ch, q[mt][np]);
          if (sub == F - 1) {
            // fold epilogue, once per output per fold chunk: shift = (ea_min - EA) + (ew_min - EW) + MARGIN, in
            // [0, MARGIN] by the decision and the metadata checks (clamped at 63 so a flagged stale table cannot
            // reach an undefined shift); Dc << shift added to the int64 row sum with the overflow indicator.
            int sh_r[2], sh_c[NTP][2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t xr = am[warp_m + mt * 16 + g + 8 * h];
              sh_r[h] = sbyte(xr, 0) - sbyte(xr, 2) + MARGIN;
            }
#pragma unroll
            for (int ntl = 0; ntl < NTP; ++ntl)
#pragma unroll
              for (int j = 0; j < 2; ++j) {
                const uint32_t yc = bm[warp_n + (np * NTP + ntl) * 8 + tig * 2 + j];
                sh_c[ntl][j] = sbyte(yc, 0) - sbyte(yc, 2);
              }
#pragma unroll
            for (int ntl = 0; ntl < NTP; ++ntl) {
              const int nt = np * NTP + ntl;
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                int qs[NSV];
#pragma unroll
                for (int s = 0; s < NSV; ++s) qs[s] = q[mt][np][s][ntl][k];
                const long long dc = combine<C>(qs);
                const int shift = min(sh_r[k >> 1] + sh_c[ntl][k & 1], 63);
                acc[mt][nt][k] = add_acc(acc[mt][nt][k], shl64(dc, shift), ovf);
              }
            }
          }
        } else {
          // per-group path: the oracle's accumulate per group with effective exponents (the shift actually applied
          // to each row is undone: ea_eff = ea_min when the row was folded, ea_g otherwise; the same for the weight
          // row), floor for negative shifts, the per-term metadata check kept (rare path).
#pragma unroll 1
          for (int b = 0; b < STAGE_GROUPS; ++b) {
            int qg[NSV][NTP][4];
#pragma unroll
            for (int s = 0; s < NSV; ++s)
#pragma unroll
              for (int ntl = 0; ntl < NTP; ++ntl)
#pragma unroll
                for (int k = 0; k < 4; ++k) qg[s][ntl][k] = 0;
            mma_block<C>(st, a_off, b_off, b, lane7, a_ch, b_ch, qg);
#pragma unroll
            for (int ntl = 0; ntl < NTP; ++ntl) {
              const int nt = np * NTP + ntl;
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                const int rl = warp_m + mt * 16 + g + 8 * (k >> 1);
                const int cl = warp_n + nt * 8 + tig * 2 + (k & 1);
                const uint32_t xw = am[rl], yw = bm[cl];
                int qs[NSV];
#pragma unroll
                for (int s = 0; s < NSV; ++s) qs[s] = qg[s][ntl][k];
                const long long dg = combine<C>(qs);
                const int ea_eff = ((xw >> 8) & 1u) ? sbyte(xw, 0) : sbyte(ax[rl], b);
                const int ew_eff = ((yw >> 8) & 1u) ? sbyte(yw, 0) : sbyte(bx[cl], b);
                const int Bb = sbyte(xw, 2) + sbyte(yw, 2) - MARGIN;
                const int u = ea_eff + ew_eff - Bb;
                if (u > MARGIN && rl < nrows) err |= ERR_META;      // L5: ea_eff <= EA and ew_eff <= EW on consistent tables
                acc[mt][nt][k] = add_checked(acc[mt][nt][k], shift_term(dg, u), err);
              }
            }
          }
        }
      }
    }
    if (late && expand_next) expand_w<C>(stage((x + 1) % S), tid, EWr, (x + 1) % F, err);
  }

  // finish and store: raw int64 row sums and/or the finish rule (finish_bf16: one __ll2float_rn, the scale on the
  // exponent field, flagged signed infinity / zero outside the normal range, integer RNE to BF16)
  const StageView<C> last = stage((NST - 1) % S);    // its metadata is intact: nothing wrote it afterwards
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = warp_m + mt * 16 + g + 8 * h;
      if (r >= nrows) continue;
      const long long row = static_cast<long long>(row0 + r);
      const int EAr = sbyte(last.ameta()[r], 2);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int cl = warp_n + nt * 8 + tig * 2;
        const long long v0 = acc[mt][nt][2 * h], v1 = acc[mt][nt][2 * h + 1];
        if (a.out_i64) {
          *reinterpret_cast<longlong2*>(a.out_i64 + row * a.N + col0 + cl) = make_longlong2(v0, v1);
        }
        if (a.out_bf16) {
          const uint32_t* bmf = last.bmeta();
          const int B0 = EAr + sbyte(bmf[cl], 2) - MARGIN, B1 = EAr + sbyte(bmf[cl + 1], 2) - MARGIN;
          bool bad = false;
          const uint32_t o0 = finish_bf16(v0, B0 + a.finish_base, bad);
          const uint32_t o1 = finish_bf16(v1, B1 + a.finish_base, bad);
          if (bad) err |= ERR_FINISH_RANGE;
          *reinterpret_cast<uint32_t*>(a.out_bf16 + row * a.N + col0 + cl) = o0 | (o1 << 16);   // cl even, N % 32 == 0
        }
      }
    }
  if (ovf < 0) err |= ERR_ACC_OVERFLOW;
  if (err) atomicOr(&g_error, err);
}

}  // namespace patha

// ------------------------------------------------------------------------------------------ torch binding

static int patha_fold_stages(int K, const torch::Tensor& ameta) {
  // F from the ameta leading dimension: ameta is [K / (128 F), M, 4]
  const int NST = K / patha::STAGE_K;
  const int NCF = static_cast<int>(ameta.size(0));
  TORCH_CHECK(NCF == NST, "patha: ameta leading dimension must be K/128 (fold chunk of 4 groups; the 8-group chunk is a "
                          "modelled knob that r3-P2 does not instantiate: Codex's build 1 gave it 8 bytes of stack)");
  return 1;
}

static const int32_t* patha_rows_ptr(const std::optional<torch::Tensor>& rows, const torch::Device& dev) {
  if (!rows.has_value()) return nullptr;
  const torch::Tensor r = rows.value();
  TORCH_CHECK(r.is_cuda() && r.device() == dev && r.scalar_type() == at::kInt && r.numel() == 1,
              "patha: rows must be an int32 device scalar (the live row count) on the operands' device");
  return r.data_ptr<int32_t>();
}

static void patha_prep_activations(torch::Tensor codes, torch::Tensor ea, torch::Tensor planes, torch::Tensor ameta,
                                   std::optional<torch::Tensor> rows) {
  using namespace patha;
  TORCH_CHECK(codes.is_cuda() && ea.is_cuda() && planes.is_cuda() && ameta.is_cuda(), "patha: CUDA tensors required");
  TORCH_CHECK(codes.scalar_type() == at::kByte, "patha: codes must be uint8 (E4M3 bit patterns)");
  TORCH_CHECK(ea.scalar_type() == at::kChar && planes.scalar_type() == at::kChar && ameta.scalar_type() == at::kChar,
              "patha: ea, planes and ameta must be int8");
  TORCH_CHECK(codes.is_contiguous() && ea.is_contiguous() && planes.is_contiguous() && ameta.is_contiguous(),
              "patha: contiguous tensors required");
  TORCH_CHECK(codes.dim() == 2 && ea.dim() == 2 && planes.dim() == 3 && ameta.dim() == 3, "patha: ranks");
  const int M = static_cast<int>(codes.size(0)), K = static_cast<int>(codes.size(1));
  const int AP = static_cast<int>(planes.size(0));
  TORCH_CHECK(AP == ACT_PLANES || AP == ACT_PLANES_4, "patha: planes must be [3 or 4, M, K]");
  TORCH_CHECK(K % STAGE_K == 0 && K > 0, "patha: K % 128 == 0");
  const int F = patha_fold_stages(K, ameta);
  TORCH_CHECK(K % (STAGE_K * F) == 0, "patha: K % (128 F) == 0");
  const int NC = K / (STAGE_K * F), G = K / GROUP;
  TORCH_CHECK(ea.size(0) == M && ea.size(1) == G, "patha: ea must be [M, K/32]");
  TORCH_CHECK(planes.size(1) == M && planes.size(2) == K, "patha: planes must be [AP, M, K]");
  TORCH_CHECK(ameta.size(0) == NC && ameta.size(1) == M && ameta.size(2) == META_BYTES, "patha: ameta must be [K/(128 F), M, 4]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(planes.data_ptr()) % 16 == 0,
              "patha: codes and planes must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(codes.device());
  const long long total = static_cast<long long>(M) * NC;
  const dim3 grid(static_cast<unsigned>((total + 127) / 128)), block(128);
  const uint8_t* cp = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  const int32_t* rows_dev = patha_rows_ptr(rows, codes.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(F == 1, "patha: fold chunk of 4 groups only (r3-P2)");
  if (AP == 3) prep_kernel<3, 1><<<grid, block, 0, stream>>>(cp, ea.data_ptr<int8_t>(), M, K, planes.data_ptr<int8_t>(), ameta.data_ptr<int8_t>(), rows_dev);
  else prep_kernel<4, 1><<<grid, block, 0, stream>>>(cp, ea.data_ptr<int8_t>(), M, K, planes.data_ptr<int8_t>(), ameta.data_ptr<int8_t>(), rows_dev);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class C>
static void launch_gemm(const patha::GemmArgs& args, int E, int max_rows, cudaStream_t stream) {
  using namespace patha;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    attr_set = true;
  }
  const dim3 grid(args.N / C::BN, (max_rows + BM - 1) / BM, E), block(C::THREADS);
  gemm_kernel<C><<<grid, block, C::SMEM, stream>>>(args);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Dense (WP = 3) or expert (WP = 1) GEMM. w: dense [N, K] uint8, expert [E, N, K/2] uint8 (or [N, K/2] with E = 1).
// The configuration follows the operands: planes [AP, Mtot, K] and ameta [K / 128, Mtot, 4] select dense (3 planes),
// the expert kernel of gate/up and down (3 planes, one instantiation) or the four-plane down knob (4 planes).
static void patha_gemm(torch::Tensor planes, torch::Tensor ameta, torch::Tensor ea, torch::Tensor w, torch::Tensor ew,
                       torch::Tensor EW, std::optional<torch::Tensor> row_off, int64_t max_rows,
                       std::optional<torch::Tensor> out_bf16, std::optional<torch::Tensor> out_i64, bool expert,
                       std::optional<torch::Tensor> rows) {
  using namespace patha;
  TORCH_CHECK(planes.is_cuda() && ameta.is_cuda() && ea.is_cuda() && w.is_cuda() && ew.is_cuda() && EW.is_cuda(),
              "patha: CUDA tensors required");
  TORCH_CHECK(planes.scalar_type() == at::kChar && ameta.scalar_type() == at::kChar && ea.scalar_type() == at::kChar &&
              ew.scalar_type() == at::kChar && EW.scalar_type() == at::kChar, "patha: planes, ameta, ea, ew, EW must be int8");
  TORCH_CHECK(w.scalar_type() == at::kByte, "patha: weights must be uint8 (E4M3 codes or E2M1 nibble pairs)");
  TORCH_CHECK(planes.is_contiguous() && ameta.is_contiguous() && ea.is_contiguous() && EW.is_contiguous(),
              "patha: planes, ameta, ea, EW must be contiguous");
  TORCH_CHECK(planes.dim() == 3 && (planes.size(0) == ACT_PLANES || planes.size(0) == ACT_PLANES_4), "patha: planes must be [3 or 4, Mtot, K]");
  const int AP = static_cast<int>(planes.size(0));
  const int Mtot = static_cast<int>(planes.size(1)), K = static_cast<int>(planes.size(2));
  TORCH_CHECK(K % STAGE_K == 0 && K > 0, "patha: K % 128 == 0");
  const int F = patha_fold_stages(K, ameta);
  const int NCF = K / (STAGE_K * F), G = K / GROUP;
  TORCH_CHECK(ameta.dim() == 3 && ameta.size(0) == NCF && ameta.size(1) == Mtot && ameta.size(2) == META_BYTES, "patha: ameta [K/(128 F), Mtot, 4]");
  TORCH_CHECK(ea.dim() == 2 && ea.size(0) == Mtot && ea.size(1) == G, "patha: ea [Mtot, K/32]");
  int E = 1, N = 0;
  long long ldw = 0, wstride = 0, ldew = 0, ewstride = 0;
  if (expert) {
    TORCH_CHECK(w.dim() == 3 || w.dim() == 2, "patha: expert weights [E, N, K/2] or [N, K/2]");
    torch::Tensor w3 = w.dim() == 3 ? w : w.unsqueeze(0);
    torch::Tensor ew3 = ew.dim() == 3 ? ew : ew.unsqueeze(0);
    torch::Tensor EW2 = EW.dim() == 2 ? EW : EW.unsqueeze(0);
    E = static_cast<int>(w3.size(0));
    N = static_cast<int>(w3.size(1));
    TORCH_CHECK(w3.size(2) == K / 2 && w3.stride(2) == 1, "patha: expert weights must be [E, N, K/2] with contiguous nibble rows");
    TORCH_CHECK(ew3.size(0) == E && ew3.size(1) == N && ew3.size(2) == G && ew3.stride(2) == 1, "patha: ew [E, N, K/32]");
    TORCH_CHECK(EW2.size(0) == E && EW2.size(1) == N && EW2.is_contiguous(), "patha: EW [E, N]");
    ldw = w3.stride(1); wstride = w3.stride(0);
    ldew = ew3.stride(1); ewstride = ew3.stride(0);
    w = w3; ew = ew3; EW = EW2;
    TORCH_CHECK(F == 1 && (AP == 3 || AP == 4), "patha: expert configurations are planes 3 or 4 with a fold chunk of 4 groups");
  } else {
    TORCH_CHECK(w.dim() == 2 && w.size(1) == K && w.stride(1) == 1, "patha: dense weights [N, K] uint8, K contiguous");
    N = static_cast<int>(w.size(0));
    TORCH_CHECK(ew.dim() == 2 && ew.size(0) == N && ew.size(1) == G && ew.stride(1) == 1, "patha: ew [N, K/32]");
    TORCH_CHECK(EW.dim() == 1 && EW.size(0) == N, "patha: EW [N]");
    ldw = w.stride(0); wstride = 0;
    ldew = ew.stride(0); ewstride = 0;
    TORCH_CHECK(AP == 3 && F == 1, "patha: the dense kernel takes three planes and a fold chunk of 4 groups");
  }
  const int BN_req = expert ? CfgGateUp::BN : CfgDense::BN;
  TORCH_CHECK(N % BN_req == 0 && N > 0, "patha: N must be a multiple of the CTA's output rows (32 dense, 64 expert)");
  TORCH_CHECK(ldw % 16 == 0 && wstride % 16 == 0, "patha: weight row and expert strides must be multiples of 16 bytes");
  TORCH_CHECK(ldew % (4 * F) == 0 && ewstride % (4 * F) == 0, "patha: exponent row strides must be multiples of 4 F bytes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(planes.data_ptr()) % 16 == 0,
              "patha: weights and planes must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(ew.data_ptr()) % (4 * F) == 0 && reinterpret_cast<uintptr_t>(ea.data_ptr()) % 4 == 0 &&
              reinterpret_cast<uintptr_t>(ameta.data_ptr()) % 4 == 0, "patha: exponent tensors must be 4-byte (ew: 4 F) aligned");
  const int32_t* row_off_ptr = nullptr;
  if (row_off.has_value()) {
    const torch::Tensor ro = row_off.value();
    TORCH_CHECK(ro.is_cuda() && ro.scalar_type() == at::kInt && ro.is_contiguous() && ro.dim() == 1 && ro.size(0) == E + 1,
                "patha: row_off must be int32 [E+1] on the device");
    row_off_ptr = ro.data_ptr<int32_t>();
  } else {
    TORCH_CHECK(E == 1, "patha: several experts need row_off");
    max_rows = Mtot;
  }
  TORCH_CHECK(max_rows >= 1 && max_rows <= Mtot, "patha: max_rows must be in 1..Mtot");
  TORCH_CHECK(!(row_off.has_value() && rows.has_value()), "patha: rows (the device live row count) applies to the ungrouped launch; a grouped launch is bounded by row_off");
  const int32_t* rows_dev = patha_rows_ptr(rows, planes.device());
  TORCH_CHECK(out_bf16.has_value() || out_i64.has_value(), "patha: at least one output");
  uint16_t* ob = nullptr;
  long long* oi = nullptr;
  if (out_bf16.has_value()) {
    const torch::Tensor o = out_bf16.value();
    TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kBFloat16 && o.is_contiguous() && o.dim() == 2 && o.size(0) == Mtot && o.size(1) == N,
                "patha: out_bf16 must be bf16 [Mtot, N] contiguous");
    ob = reinterpret_cast<uint16_t*>(o.data_ptr());
  }
  if (out_i64.has_value()) {
    const torch::Tensor o = out_i64.value();
    TORCH_CHECK(o.is_cuda() && o.scalar_type() == at::kLong && o.is_contiguous() && o.dim() == 2 && o.size(0) == Mtot && o.size(1) == N,
                "patha: out_i64 must be int64 [Mtot, N] contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(o.data_ptr()) % 16 == 0, "patha: out_i64 must be 16-byte aligned");
    oi = reinterpret_cast<long long*>(o.data_ptr());
  }
  const c10::cuda::CUDAGuard guard(planes.device());
  GemmArgs args;
  args.planes = planes.data_ptr<int8_t>();
  args.ameta = ameta.data_ptr<int8_t>();
  args.ea = ea.data_ptr<int8_t>();
  args.w = reinterpret_cast<const uint8_t*>(w.data_ptr());
  args.ew = ew.data_ptr<int8_t>();
  args.EW = EW.data_ptr<int8_t>();
  args.row_off = row_off_ptr;
  args.rows = rows_dev;
  args.out_bf16 = ob;
  args.out_i64 = oi;
  args.Mtot = Mtot; args.N = N; args.K = K;
  args.ldw = ldw; args.wstride = wstride; args.ldew = ldew; args.ewstride = ewstride;
  args.finish_base = -(SA + (expert ? SW_EXPERT : SW_DENSE));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int mr = static_cast<int>(max_rows);
  if (!expert) launch_gemm<CfgDense>(args, E, mr, stream);
  else if (AP == 3) launch_gemm<CfgExpert3>(args, E, mr, stream);     // gate/up and down (r3-P2: one instantiation)
  else launch_gemm<CfgDown4>(args, E, mr, stream);
}

// Read (and optionally clear) the device flag. Synchronous; call between runs, never inside a capture.
static int64_t patha_error_flag(bool reset) {
  unsigned int v = 0u;
  C10_CUDA_CHECK(cudaMemcpyFromSymbol(&v, patha::g_error, sizeof(v)));
  if (reset) {
    const unsigned int z = 0u;
    C10_CUDA_CHECK(cudaMemcpyToSymbol(patha::g_error, &z, sizeof(z)));
  }
  return static_cast<int64_t>(v);
}

// Dynamic shared memory of a configuration: dense (expert = false), or the expert kernel selected by the activation
// plane count (3 / 4) and the fold chunk in groups (4 / 8).
static int64_t patha_smem_bytes(bool expert, int64_t planes, int64_t fold_groups) {
  using namespace patha;
  if (!expert) return static_cast<int64_t>(CfgDense::SMEM);
  if (planes == 3 && fold_groups == 8) return static_cast<int64_t>(CfgGateUpF2::SMEM);   // the uninstantiated knob's budget
  if (planes == 3 && fold_groups == 4) return static_cast<int64_t>(CfgExpert3::SMEM);
  if (planes == 4 && fold_groups == 4) return static_cast<int64_t>(CfgDown4::SMEM);
  TORCH_CHECK(false, "patha: no expert configuration with these planes / fold groups");
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prep_activations", &patha_prep_activations,
        "E4M3 codes [M,K] u8 + ea [M,K/32] i8 -> planes [3|4,M,K] i8, ameta [K/128,M,4] i8; rows (optional int32 device scalar): "
        "the live row count, rows at or beyond it are skipped and left untouched",
        py::arg("codes"), py::arg("ea"), py::arg("planes"), py::arg("ameta"), py::arg("rows") = py::none());
  m.def("gemm", &patha_gemm,
        "planes, ameta, ea, w, ew, EW, row_off (optional int32 [E+1]), max_rows, out_bf16 (optional), out_i64 (optional), expert (bool), "
        "rows (optional int32 device scalar: the live row count of an ungrouped launch; CTAs beyond it exit before any barrier)",
        py::arg("planes"), py::arg("ameta"), py::arg("ea"), py::arg("w"), py::arg("ew"), py::arg("EW"), py::arg("row_off"),
        py::arg("max_rows"), py::arg("out_bf16"), py::arg("out_i64"), py::arg("expert"), py::arg("rows") = py::none());
  m.def("error_flag", &patha_error_flag, "device flag (bits in patha_sequence.h); reset=True clears");
  m.def("smem_bytes", &patha_smem_bytes, "dynamic shared memory per CTA: (expert, planes, fold_groups)");
}
