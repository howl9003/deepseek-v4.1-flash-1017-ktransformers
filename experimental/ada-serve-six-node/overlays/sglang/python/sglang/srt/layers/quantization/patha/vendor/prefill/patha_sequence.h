// patha_sequence.h - constants, configurations, split rule, swizzle and digit helpers of the path A chunked-fold INT8
// prefill GEMM (kernel 2 of DESIGN_r1.md section 2). Everything the NumPy model (patha_prefill_model.py) mirrors is here.
// Revision r3-P (29 Oct 2026, price revision of the expert path after Codex's r2 receipt): the expert kernels run a
// 64 x 64 CTA tile with eight warps (two per SM sub-partition, phase-shifted), the E2M1 expansion is a byte lookup,
// the fold decision is per 16 token rows x 16 weight rows, the fold chunk may span two pipeline stages (gate/up) and
// the down projection uses four activation digit planes (fold budget 9 bits). The arithmetic (digits, fold, fallback,
// combine, shift, finish, flags) is unchanged from r1/r2 and equals contract_a.py bit for bit.
//
// Arithmetic (contract_a.py, the oracle):
//   activation a_k = n_k * 2^(ea_g - SA), SA = 9 (E4M3 integer n < 2^18 in 2^-9 units)
//   weight     w_k = m_k * 2^(ew_g - SW), SW = 9 (dense E4M3, m < 2^18) or 1 (expert E2M1 halves, |m| <= 12)
//   row base   B[m, n] = EA[m] + EW[n] - MARGIN, MARGIN per kind: 13 dense (WP = 3), 20 expert (WP = 1)
//              (CONTRACT_FIXES_r1.md H1: |A| <= G * 32 * 229376^2 * 2^13 < 2^61.62 at K <= 8192, so the int64 row
//              accumulator cannot overflow on a legal input; the expert margin is contract v2's, locked)
//   A[m, n]    = sum_g D_g << (ea_g + ew_g - B)   (floor for negative shifts, right shifts clamped at 63), D_g the
//              exact 32-wide group dot
//   y          = bf16_rne( f32_rne(A) * 2^(B - SA - SW) ) under the finish rule (M1): one direct int64 -> FP32
//              rounding, the scale applied on the exponent field; A == 0 -> +0; biased exponent outside 1..254:
//              overflow -> signed infinity and ERR_FINISH_RANGE, underflow -> signed zero and ERR_FINISH_RANGE
//              (unreachable on legal inputs: quantiser amax floor 1e-4 and weight scale codes >= 54 / 53)
//   metadata   ea_g <= EA and ew_g <= EW for every staged group, else ERR_META (L5: a stale exponent table)
//
// Fold (per decision tile and fold chunk): the chunk's 4F groups share one shift when every live token row's and
// every weight row's exponent spread inside the chunk is within budget (ACT_BUDGET bits for the AP-digit activation
// operand: 3 bits with three digits, 10 with four; 2 bits for the 3-digit dense weight; 3 bits for the one-plane
// expert weight) and the chunk shift ea_min[m] + ew_min[n] - B[m, n] is non-negative for every element of the tile;
// the integers are shifted by (e_g - e_min) before the digit split, so the plane products chain across the chunk in
// int32 and one combine + shift + add per element per chunk suffices. Otherwise the same MMAs run per group with
// the oracle's per-group shift, corrected by the shift actually applied to each row (effective exponents).
// r3-P: the decision tile is 16 token rows (one m16 MMA tile, aligned to the expert's first row) x WARP_N = 16
// weight rows (one warp's columns); r2 decided per 64 x 32 CTA tile.
#pragma once
#include <cstdint>

namespace patha {

// ---- arithmetic constants (contract)
constexpr int MARGIN_DENSE  = 13;       // path A dense (H1): int64-safe for K <= 8192
constexpr int MARGIN_EXPERT = 20;       // contract v2 (locked)
template <int WP> struct MarginOf { static constexpr int value = (WP == 3) ? MARGIN_DENSE : MARGIN_EXPERT; };
constexpr int SA          = 9;          // E4M3 activation integer units 2^-9
constexpr int SW_DENSE    = 9;          // E4M3 dense weight integer units 2^-9
constexpr int SW_EXPERT   = 1;          // E2M1 expert weight in halves
constexpr int GROUP       = 32;         // scale group (K elements)
constexpr int BASE_BITS   = 7;          // base-128 digits: the low digits balanced in [-64, 63], the top digit the
                                        // carry in the full int8 range [-128, 127] (r3-P; r2 limited it to [-64, 63])
// Digit capacity and fold budgets. Three digits hold |x| <= 64 + 64 * 128 + 127 * 128^2 = 2,089,024 (negative side
// 2,105,408), so an E4M3 integer (<= 229,376) shifted by 3 bits (1,835,008; top digit <= 112) fits and by 4 does not
// (the top binade); four digits hold up to 127 * 128^3 + 1,056,832 = 267,395,136, so a shift of 10 bits (234,881,024;
// top digit <= 112) fits and 11 does not. Fold budgets: activations 3 bits with three planes (Codex's archived-row
// eligibility of the down input is 98.79 percent at 3 bits against 89.96 at 2), 10 bits with four planes; dense
// weights 2 bits (unchanged, the dense chain is not the price target); expert weight 3 bits (12 << 3 = 96, one plane).
// Plane-group bounds with a top digit of 112: expert |Q_s| <= 128 F * 112 * 96 = 1,376,256 F; dense (weight digits
// <= 64, weight top digit <= 56 at budget 2) |Q_2| <= 128 * (112 * 64 + 64 * 64 + 64 * 56) = 1,900,544; both under 2^21 at F = 1 (the model asserts
// 2^21 for every configuration it runs; F = 2 is instantiated for the expert gate/up only, 2^21.4 < 2^31 and the
// combine's int32 pair sums stay exact: |Q_0 + Q_1 << 7| < 2^29).
constexpr int ACT_BUDGET  = 3;          // bits of spread folded for 3-digit activation operands (was 2 in r2)
constexpr int ACT_BUDGET_4 = 10;        // four digits
template <int AP> struct ActBudgetOf { static constexpr int value = (AP == 3) ? ACT_BUDGET : ACT_BUDGET_4; };
constexpr int TOP_DIGIT_MIN = -128, TOP_DIGIT_MAX = 127;
constexpr int DENSE_W_BUDGET  = 2;
constexpr int EXPERT_W_BUDGET = 3;      // 12 << 3 = 96 fits one int8 plane
static_assert((12 << EXPERT_W_BUDGET) <= 127, "the shifted expert weight must fit one int8 plane");
static_assert((229376 << ACT_BUDGET) <= 64 + 64 * 128 + TOP_DIGIT_MAX * 128 * 128, "three digits hold the activation budget");
static_assert((229376 << ACT_BUDGET_4) <= 64 + 64 * 128 + 64 * 128 * 128 + TOP_DIGIT_MAX * 128 * 128 * 128,
              "four digits hold the activation budget");
constexpr int ACT_PLANES  = 3;          // the three-digit activation operand (dense, expert gate/up and down)
constexpr int ACT_PLANES_4 = 4;         // the four-digit activation operand (the data-independent knob for down)

// ---- geometry shared by every configuration
constexpr int BM          = 64;         // token rows per CTA tile
constexpr int STAGE_GROUPS = 4;         // groups per pipeline stage (128 K = one 128-byte plane row)
constexpr int STAGE_K     = STAGE_GROUPS * GROUP;     // 128
constexpr int CHUNKS16    = STAGE_K / 16;             // 8 sixteen-byte pieces per 128-byte row
constexpr int WARP_M      = 32;         // warp sub-tile 32 x 16: two m16 tiles by two n8 tiles
constexpr int WARP_N      = 16;
constexpr int MT          = WARP_M / 16;            // m16 tiles per warp (2)
constexpr int NT          = WARP_N / 8;             // n8 tiles per warp (2)
constexpr int WARPS_M     = 2;                      // warps along M (BM = WARPS_M * WARP_M)
constexpr int DECISION_ROWS = 16;                   // fold decision granularity: one m16 tile x WARP_N columns
constexpr int META_BYTES  = 4;          // per row per chunk: {e_min, eligible, E_row, 0}
constexpr int SMEM_LIMIT_SM89 = 101376; // the per-block opt-in limit (99 KiB)

// The kernel configuration (free parameters; the model carries the same numbers):
//   WP   weight planes: 3 dense (E4M3 digits), 1 expert (E2M1 halves shifted by <= 3 bits)
//   AP   activation planes: 3 (budget 3 bits) or 4 (budget 10 bits)
//   BN   output rows per CTA tile = WARPS_N * 16
//   F    pipeline stages per fold chunk (1 or 2): the fold chunk is 4F groups = 128F K, the eligibility, ea_min /
//        ew_min and the epilogue are per fold chunk; the operand staging, barrier and per-group fallback are per stage
//   S    pipeline stages in shared memory
template <int WP_, int AP_, int BN_, int F_, int S_> struct Cfg {
  static constexpr int WP = WP_, AP = AP_, BN = BN_, F = F_, STAGES = S_;
  static constexpr int WARPS_N = BN / WARP_N;
  static constexpr int WARPS = WARPS_M * WARPS_N;
  static constexpr int THREADS = 32 * WARPS;
  static constexpr int NTP = (WP == 3) ? 1 : 2;      // n8 tiles per pass: dense one (B by ldmatrix.x2), expert the pair
  static constexpr int NPASS = NT / NTP;
  static constexpr int NS = AP + WP - 1;             // plane groups Q_s, s = i + j
  static constexpr int MARGIN = MarginOf<WP>::value;
  static constexpr int ACT_BUDGET_BITS = ActBudgetOf<AP>::value;
  static constexpr int W_BUDGET = (WP == 3) ? DENSE_W_BUDGET : EXPERT_W_BUDGET;
  static constexpr int CHUNK_GROUPS = STAGE_GROUPS * F;      // fold chunk in groups (4 or 8)
  static constexpr int CHUNK_K = STAGE_K * F;
  // per-stage shared memory, three regions (16-byte multiples):
  //   A region (stage x in buffer x % S): activation planes [AP][BM][128], ameta [BM] u32 (the fold chunk's
  //     {ea_min, eligible, EA, 0} per token row), aexp [BM] u32 (the stage's four group exponents per token row)
  //   B region: weight planes [WP][BN][128] written by the expander, bmeta [BN] u32 (the fold chunk's
  //     {ew_min, eligible, EW, 0}), bexpb [BN] u32 (the stage's four group exponents per weight row, copied by the
  //     expander for the per-group path)
  //   W region (codes of stage x in buffer x % S): weight codes [BN][128] (dense) or [BN][64] (expert) by cp.async,
  //     bexp [BN][F] u32 (the fold chunk's 4F group exponents); expanded one stage ahead of use, then overwritten by
  //     the copy of stage x + S
  static constexpr int PLANE_BYTES = BM * STAGE_K;                          // 8192 activation plane
  static constexpr int BPLANE_BYTES = BN * STAGE_K;                         // 4096 (BN 32) / 8192 (BN 64)
  static constexpr int A_REGION = AP * PLANE_BYTES + 2 * BM * 4;            // 25088 (AP 3) / 33280 (AP 4)
  static constexpr int B_REGION = WP * BPLANE_BYTES + 2 * BN * 4;           // 12544 dense / 8704 expert (BN 64)
  static constexpr int WCODE_BYTES = (WP == 3) ? BN * STAGE_K : BN * STAGE_K / 2;
  static constexpr int WPIECES = WCODE_BYTES / BN / 16;                     // 8 dense, 4 expert
  static constexpr int WEXP_BYTES = 4 * F;                                  // the fold chunk's exponents per weight row
  static constexpr int W_REGION = WCODE_BYTES + BN * WEXP_BYTES;            // 4224 dense / 4352, 4608 expert
  static constexpr int STAGE_BYTES = A_REGION + B_REGION + W_REGION;
  static constexpr int SMEM = STAGE_BYTES * STAGES;
  static_assert(BN % WARP_N == 0 && WARPS_N >= 1, "BN must be a multiple of the warp's 16 columns");
  static_assert(F == 1 || F == 2, "fold chunk of one or two stages");
  static_assert(AP == 3 || AP == 4, "three or four activation digit planes");
  static_assert(SMEM <= SMEM_LIMIT_SM89, "stages exceed the sm_89 opt-in limit");
  static_assert(STAGE_BYTES % 16 == 0 && A_REGION % 16 == 0 && B_REGION % 16 == 0 && W_REGION % 16 == 0, "alignment");
  static_assert(THREADS >= 2 * BM && THREADS >= BN, "metadata copies need one thread per row");
  static_assert((AP * BM * CHUNKS16) % THREADS == 0 && (BN * WPIECES) % THREADS == 0, "cp.async pieces per thread");
  static_assert(THREADS == 4 * BN, "the expander maps thread (row = tid / 4, group = tid % 4)");
};

// The instantiations. Dense keeps r2's geometry (not the price target); the expert kernels are r3-P's. r3-P2: the
// gate/up and down projections share CfgExpert3 (fold chunk of one stage), the configuration Codex's r3 build found
// clean (151 registers, zero stack, no LDL/STL); the two-stage fold chunk compiled to 199 registers with 8 bytes of
// stack and stays a modelled, gated knob that the binding does not instantiate.
using CfgDense  = Cfg<3, 3, 32, 1, 2>;   // 2 x 41856 = 83712 B, 4 warps (r2)
using CfgExpert3 = Cfg<1, 3, 64, 1, 2>;  // 2 x 38144 = 76288 B, 8 warps, fold chunk 4 groups (expert gate/up and down)
using CfgGateUp = CfgExpert3;
using CfgDown   = CfgExpert3;
using CfgDown4  = Cfg<1, 4, 64, 1, 2>;   // 2 x 46336 = 92672 B, 8 warps, four activation planes (budget 10): the
                                         // data-independent knob for a down input whose rows exceed 3 bits of spread
using CfgGateUpF2 = Cfg<1, 3, 64, 2, 2>; // 2 x 38400 = 76800 B: the 8-group fold-chunk knob (not instantiated)
static_assert(CfgDense::SMEM == 83712 && CfgExpert3::SMEM == 76288 && CfgDown4::SMEM == 92672 && CfgGateUpF2::SMEM == 76800,
              "shared-memory budget as documented");

// ---- chunking (no split-K: one CTA owns the whole K of its tile; stages ascend)
inline __host__ __device__ int stages_total(int K) { return K / STAGE_K; }   // K % (128 F) == 0 is checked by the binding

// Swizzle of the 16-byte pieces inside one 128-byte plane row: physical = logical ^ (row & 7). With 8 pieces per
// row the xor stays in range; every ldmatrix phase (8 rows at one logical piece) hits 8 distinct bank groups.
inline __host__ __device__ int swz(int row, int piece16) { return piece16 ^ (row & 7); }
// The staged weight-code rows use the same rule with their own piece count (8 dense, 4 expert).
inline __host__ __device__ int swz_w(int row, int piece16, int npieces) { return piece16 ^ (row & (npieces - 1)); }

// ---- digit helpers (balanced base 128; arithmetic shifts). x must satisfy |x| <= 2^20 for three digits, 2^27 for four.
__host__ __device__ inline int digit0(int x) { return ((x + 64) & 127) - 64; }
__host__ __device__ inline int carry(int x, int d) { return (x - d) >> 7; }   // exact: x - d is a multiple of 128

// E4M3 code byte -> unsigned integer in 2^-9 units (N_OF_CODE of contract_v2); NaN codes (0x7F / 0xFF) -> 0.
__host__ __device__ inline int e4m3_mag(unsigned c) {
  const unsigned e = (c >> 3) & 15u, mant = c & 7u;
  if ((c & 127u) == 127u) return 0;
  return e ? static_cast<int>((8u | mant) << (e - 1u)) : static_cast<int>(mant);
}
__host__ __device__ inline bool e4m3_is_nan(unsigned c) { return (c & 127u) == 127u; }

// E2M1 nibble -> halves (W_INT of contract_v2): 0,1,2,3,4,6,8,12 for magnitudes 0..7, sign in bit 3.
__host__ __device__ inline int e2m1_halves(unsigned nib) {
  const unsigned mag = nib & 7u;
  const int m = (mag < 2u) ? static_cast<int>(mag) : static_cast<int>((2u + (mag & 1u)) << ((mag >> 1) - 1u));
  return (nib & 8u) ? -m : m;
}

// r3-P expert expansion by byte permutation (prmt.b32 in its default mode, as inline PTX for the sign masks: CUDA's
// __byte_perm intrinsic masks bit 3 of every selector nibble and must not carry these selectors; r3-P2): four nibbles
// of a 16-bit field become four int8 plane bytes
// (e2m1_halves(nib) << s) in one 32-bit word. The magnitude tables hold the eight magnitudes 0,1,2,3 | 4,6,8,12
// shifted by s (no byte overflows: 12 << 3 = 96); the negative tables are their per-byte two's complements, formed
// as (0x80 - v) ^ 0x80 per byte, exact for v in [0, 127] without a borrow across bytes. The sign mask replicates
// nibble bit 3 across its output byte with prmt's sign-replicate selectors (bit 3 of each selector nibble): for the
// low field of a code word v the signs sit at v bits 3, 7, 11, 15 = the msbs of bytes 0, 1 of (v << 4) and v, so the
// selector is 0x9D8C over {v, v << 4}; for the high field (bits 19, 23, 27, 31) it is 0xBFAE. The model reproduces
// the same byte semantics (gate expert_lut_expander_equals_arithmetic).
constexpr uint32_t LUT_POS0 = 0x03020100u, LUT_POS1 = 0x0C080604u;
constexpr uint32_t LUT_SIGN_SEL_LO = 0x9D8Cu, LUT_SIGN_SEL_HI = 0xBFAEu;
constexpr uint32_t LUT_MAG_MASK = 0x7777u;

// ---- device error flag bits (g_error)
constexpr unsigned ERR_ACC_OVERFLOW   = 1u;    // int64 row accumulator overflow (checked add)
constexpr unsigned ERR_NEG_FOLD_SHIFT = 2u;    // negative chunk shift reached on the fold path (impossible by construction)
constexpr unsigned ERR_FINISH_RANGE   = 4u;    // finish left the normal FP32 range (subnormal or inf)
constexpr unsigned ERR_NAN_ACT        = 8u;    // NaN E4M3 code among the activations (prep kernel)
constexpr unsigned ERR_NAN_WEIGHT     = 16u;   // NaN E4M3 code among the dense weights
constexpr unsigned ERR_DIGIT_RANGE    = 32u;   // top digit outside [-128, 127] (would need a shift above budget)
constexpr unsigned ERR_EXPERT_PLANE   = 64u;   // shifted expert weight outside int8 (unreachable: 12 << 3 = 96; retained bit)
constexpr unsigned ERR_META           = 128u;  // a group exponent above the row maximum given as EA / EW, or a term
                                               // shift above the margin: stale exponent metadata (L5)

}  // namespace patha
