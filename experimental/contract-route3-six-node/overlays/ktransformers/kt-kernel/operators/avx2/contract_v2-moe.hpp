/**
 * @Description  : AVX2 contract v2 MoE backend (DeepSeek-V4.1-Flash-1017 shared E4M3 x E2M1 contract).
 *
 * Compiled only with KT_ENABLE_CONTRACT_V2 and selected only by KT_MXFP4_BACKEND=contract_v2. Every routed
 * expert is computed exactly as contract-v2-r1/contract_v2.py defines it (bit-identical to the FP64 oracle on
 * the archived captures, and to the contract v2 Triton kernels on SM89):
 *   BF16 rows -> E4M3 groups of 32 (digits hi/lo) -> gate/up as exact integer dots, exact int64 accumulation at
 *   base EA+EW-MARGIN, one RNE to FP32, exact power of two, BF16 -> table SiLU(gate) * clamp(up) * routing weight
 *   -> BF16 -> E4M3 -> down projection, exact int64 -> one RNE -> BF16 per expert.
 * The routing weight is applied inside the activation (before the down quantisation), as the reference does, so
 * the per-expert outputs are already weighted and the combine is a plain sum.
 *
 * Parts (one per NUMA pool) hold intermediate-dimension slices as the MXFP4 backend does, so the down projection
 * is K-partitioned: each part keeps the int64 partial and its own base, and the TP wrapper merges the parts at
 * the global base (max row exponent, max weight exponent per output row) before the single RNE. The merge equals
 * the single-part contract whenever no term lies in the truncation regime (more than MARGIN = 20 bits below the
 * row maximum); the model's data has spreads of at most ~8 bits. Low bits lost in the merge shifts are counted
 * (kt_contract_v2_truncation_events()) so the condition is verified, not assumed.
 *
 * Outputs: forward() writes the routed sum (ascending expert id, FP32) as BF16 [qlen, hidden];
 * forward_routes_binding() writes the per-route BF16 values [qlen, topk, hidden] (weights already applied) for a
 * GPU-side combine with the shared expert and the GPU experts in FP32, one BF16 cast.
 */
#ifndef CPUINFER_OPERATOR_AVX2_CONTRACT_V2_MOE_H
#define CPUINFER_OPERATOR_AVX2_CONTRACT_V2_MOE_H
#ifdef KT_ENABLE_CONTRACT_V2

#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "moe_base.hpp"
#include "../moe-tp.hpp"

// The surrounding KT translation unit uses -ffast-math. Keep the contract's
// ordered FP32 operations local to these definitions; do not change old backends.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#else
#error "Contract v2 currently requires the validated GCC strict-function build"
#endif

#include "contract_v2_tables.inc"   // declares namespace kt_contract_v2 { kFp8TableBits, kSiluTableBits }
namespace kt_contract_v2 {
// A fast-math shared object's initializer may enable DAZ/FTZ, and worker threads
// inherit that state. Restore the caller's state after this arithmetic scope.
struct ScopedFpContract {
  static constexpr unsigned kDaz = 1u << 6, kFtz = 1u << 15, kRoundingMode = 3u << 13;
  unsigned previous;
  bool changed;
  ScopedFpContract() : previous(_mm_getcsr()), changed(previous & (kDaz | kFtz | kRoundingMode)) {
    if (changed) _mm_setcsr(previous & ~(kDaz | kFtz | kRoundingMode));
  }
  ~ScopedFpContract() { if (changed) _mm_setcsr(previous); }
  ScopedFpContract(const ScopedFpContract&) = delete;
  ScopedFpContract& operator=(const ScopedFpContract&) = delete;
};
static std::atomic<long> g_truncation_events{0};
#define V2_TRUNC_HOOK() (::kt_contract_v2::g_truncation_events.fetch_add(1, std::memory_order_relaxed))
#include "contract_v2_kernels.inc"
#undef V2_TRUNC_HOOK

static inline const float* fp8_table() { return reinterpret_cast<const float*>(kFp8TableBits); }
static inline const float* silu_table() { return reinterpret_cast<const float*>(kSiluTableBits); }
static inline size_t align64(size_t v) { return (v + 63) & ~(size_t)63; }
static inline uint8_t* align64p(void* p) { return (uint8_t*)(((uintptr_t)p + 63) & ~(uintptr_t)63); }
}  // namespace kt_contract_v2

inline long kt_contract_v2_truncation_events() { return kt_contract_v2::g_truncation_events.load(); }

namespace avx2 {

// Weights: FP4 E2M1 nibble-packed [n, k/2] (low nibble first), FP32 group scales [n, k/32] as the MXFP4 backend
// stores them (the TP loader fills b and d); finalize() derives the E8M0 codes and the row exponent bounds.
struct GemmKernelAVX2ContractV2 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}
  static int recommended_nth(int n) { return std::max(1, div_up(n, N_BLOCK)); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }
  static std::string name() { return "AVX2_CONTRACT_V2"; }

  // Activations: the contract's E4M3 digits of each BF16 row plus the folded copies for the fast path.
  struct BufferA {
    size_t max_m = 0, k = 0;
    int G = 0;
    uint8_t* codes = nullptr;   // [m, k]   E4M3 codes (quantizer output, kept for diagnostics)
    int8_t* ea = nullptr;       // [m, G]   group exponents
    int8_t* EA = nullptr;       // [m]      row maximum exponent
    int16_t* hi = nullptr;      // [m, k]   digits n >> 9 (signed)
    int16_t* lo = nullptr;      // [m, k]   digits n & 511 (signed)
    int16_t* hi_f = nullptr;    // [m, k]   folded digits (nibble order)
    int16_t* lo_f = nullptr;    // [m, k]
    int8_t* EAmin = nullptr;    // [m]
    int8_t* SA = nullptr;       // [m]      exponent spread, 127 when the row cannot be folded

    BufferA(size_t m, size_t k_, int /*group_size*/, void* ptr) : max_m(m), k(k_), G((int)(k_ / 32)) {
      if (k % 32 != 0) throw std::runtime_error("contract v2 requires k % 32 == 0");
      set_data(ptr);
    }
    static size_t required_size(size_t m, size_t k, int /*group_size*/) {
      const size_t G = k / 32;
      return 8 * 64 + m * k * (1 + 4 * 2) + m * G + 3 * m;
    }
    void set_data(void* ptr) {
      uint8_t* p = kt_contract_v2::align64p(ptr);
      codes = p; p = kt_contract_v2::align64p(p + max_m * k);
      hi = (int16_t*)p; p = kt_contract_v2::align64p(p + max_m * k * 2);
      lo = (int16_t*)p; p = kt_contract_v2::align64p(p + max_m * k * 2);
      hi_f = (int16_t*)p; p = kt_contract_v2::align64p(p + max_m * k * 2);
      lo_f = (int16_t*)p; p = kt_contract_v2::align64p(p + max_m * k * 2);
      ea = (int8_t*)p; p = kt_contract_v2::align64p(p + max_m * G);
      EA = (int8_t*)p; p = kt_contract_v2::align64p(p + max_m);
      EAmin = (int8_t*)p; p = kt_contract_v2::align64p(p + max_m);
      SA = (int8_t*)p;
    }
    // BF16 rows [r0, r1) of base -> FP32 (exact) -> contract quantizer -> folded digits. Rows are independent,
    // so the MoE class spreads row blocks of every expert over the pool (the quantizer is scalar: ~5 ns/value).
    void from_rows(int r0, int r1, const ggml_bf16_t* base) {
      kt_contract_v2::ScopedFpContract fp_contract;
      if (r1 > (int)max_m || r0 < 0 || r0 > r1) throw std::runtime_error("contract v2 BufferA: row range exceeds max_m");
      thread_local std::vector<float> row;
      if (row.size() < k) row.resize(k);
      for (int r = r0; r < r1; r++) {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(base) + (size_t)r * k;
        size_t j = 0;
        for (; j + 8 <= k; j += 8) {
          __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s + j)));
          _mm256_storeu_ps(row.data() + j, _mm256_castsi256_ps(_mm256_slli_epi32(w, 16)));
        }
        for (; j < k; j++) { uint32_t b = (uint32_t)s[j] << 16; std::memcpy(row.data() + j, &b, 4); }
        kt_contract_v2::v2_quant(row.data(), 1, (int)k, codes + (size_t)r * k, ea + (size_t)r * G, hi + (size_t)r * k,
                                 lo + (size_t)r * k, EA + r);
      }
      if (r1 > r0)
        kt_contract_v2::v2_fold(hi + (size_t)r0 * k, lo + (size_t)r0 * k, ea + (size_t)r0 * G, EA + r0, r1 - r0, (int)k,
                                hi_f + (size_t)r0 * k, lo_f + (size_t)r0 * k, EAmin + r0, SA + r0);
    }
    void from_mat(int m, const ggml_bf16_t* src, int /*ith*/, int /*nth*/) { from_rows(0, m, src); }
  };

  struct BufferB {
    uint8_t* b = nullptr;  // nibble-packed FP4 [n, k/2]; the TP loader may point this at its own arena
    float* d = nullptr;    // FP32 group scales [n, k/32] (exact powers of two)
    int n = 0, k = 0, k_group_size = 0, k_group_count = 0;
    uint8_t* codes = nullptr;  // [n, G] E8M0 codes derived from d
    int8_t* EW = nullptr;      // [n] row maximum weight exponent (code - 127)
    int8_t* EWmin = nullptr;   // [n]
    bool finalized = false;

    BufferB() = default;
    BufferB(int n_, int k_, int k_group_size_, void* ptr) : n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size != 32 || k % 32 != 0) throw std::runtime_error("contract v2 requires group size 32 and k % 32 == 0");
      k_group_count = k / 32;
      uint8_t* p = kt_contract_v2::align64p(ptr);
      b = p; p = kt_contract_v2::align64p(p + (size_t)n * k / 2);
      d = (float*)p; p = kt_contract_v2::align64p(p + (size_t)n * k_group_count * sizeof(float));
      codes = p; p = kt_contract_v2::align64p(p + (size_t)n * k_group_count);
      EW = (int8_t*)p; p = kt_contract_v2::align64p(p + n);
      EWmin = (int8_t*)p;
    }
    static size_t required_size(size_t n, size_t k, int k_group_size) {
      const size_t G = k / k_group_size;
      return 6 * 64 + n * k / 2 + n * G * sizeof(float) + n * G + 2 * n;
    }
    void from_raw_mat(const uint8_t* src, int /*ith*/, int /*nth*/) { std::memcpy(b, src, (size_t)n * k / 2); }
    // E8M0 code of each FP32 scale (the exponent field; the scale is an exact power of two), then the row bounds.
    void finalize() {
      for (size_t i = 0; i < (size_t)n * k_group_count; i++) {
        uint32_t bits; std::memcpy(&bits, d + i, 4);
        if ((bits & 0x7fffff) != 0 || (bits >> 31) != 0) throw std::runtime_error("contract v2: weight scale is not a positive power of two");
        const uint32_t e = bits >> 23;
        if (e < 32 || e > 254) throw std::runtime_error("contract v2: weight scale code outside 32..254");
        codes[i] = (uint8_t)e;
      }
      kt_contract_v2::v2_prepare_weights2(codes, n, k_group_count, EW, EWmin);
      finalized = true;
    }
  };

  // Exact int64 accumulators; to_mat is the contract's finish (one RNE to FP32, exact 2^B, BF16 RNE).
  struct BufferC {
    size_t max_m = 0, n = 0;
    int64_t* acc = nullptr;
    const int8_t* EA = nullptr;  // from the BufferA of the gemm that filled acc
    const int8_t* EW = nullptr;  // from the BufferB
    BufferC(size_t m, size_t n_, void* ptr) : max_m(m), n(n_) { set_data(ptr); }
    static size_t required_size(size_t m, size_t n) { return 64 + m * n * sizeof(int64_t); }
    void set_data(void* ptr) { acc = (int64_t*)kt_contract_v2::align64p(ptr); }
    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      kt_contract_v2::ScopedFpContract fp_contract;
      const auto [c0, c1] = split_range(n, ith, nth);
      uint16_t* out = reinterpret_cast<uint16_t*>(dst);
      for (int r = 0; r < m; r++)
        for (int col = c0; col < c1; col++) {
          const int B = int(EA[r]) + int(EW[col]) - kt_contract_v2::MARGIN;
          const float y = kt_contract_v2::bf(std::ldexp(float(acc[(size_t)r * n + col]), B - 10));
          out[(size_t)r * n + col] = (uint16_t)(kt_contract_v2::bits(y) >> 16);
        }
    }
  };
};

// One column slice [c0, c1) of y = x W^T for m rows: the folded kernel into a contiguous scratch, then into acc.
static void gemm_contract_v2(int m, int n, int k, GemmKernelAVX2ContractV2::BufferA& a, GemmKernelAVX2ContractV2::BufferB& b,
                             GemmKernelAVX2ContractV2::BufferC& c, int ith, int nth) {
  if (!b.finalized) throw std::runtime_error("contract v2: BufferB not finalized (load_weights must call finalize)");
  if (b.k != k || (size_t)b.n != c.n || a.k != (size_t)k) throw std::runtime_error("contract v2: gemm shape mismatch");
  const auto [c0, c1] = GemmKernelAVX2ContractV2::split_range_n(n, ith, nth);
  const int nc = c1 - c0;
  if (nc <= 0 || m <= 0) return;
  const int G = k / 32;
  thread_local std::vector<int64_t> scratch;
  if (scratch.size() < (size_t)m * nc) scratch.resize((size_t)m * nc);
  std::fill(scratch.begin(), scratch.begin() + (size_t)m * nc, 0);
  kt_contract_v2::v2_linear_partial_fold(a.hi, a.lo, a.ea, a.EA, a.hi_f, a.lo_f, a.EAmin, a.SA, b.b + (size_t)c0 * (k / 2),
                                         b.codes + (size_t)c0 * G, b.EW + c0, b.EWmin + c0, scratch.data(), m, nc, k, 0, G);
  for (int r = 0; r < m; r++) std::memcpy(c.acc + (size_t)r * n + c0, scratch.data() + (size_t)r * nc, (size_t)nc * sizeof(int64_t));
  c.EA = a.EA;
  c.EW = b.EW;
}

}  // namespace avx2

template <class T = avx2::GemmKernelAVX2ContractV2>
class AVX2_CONTRACT_V2_MOE_TP : public AVX2_MOE_BASE<T, AVX2_CONTRACT_V2_MOE_TP<T>> {
 public:
  // The allocation base must not instantiate its float-partial forward for this
  // backend, including through the generic TP base's virtual method table.
  using ExactTpMergeOnly = void;
  using Base = AVX2_MOE_BASE<T, AVX2_CONTRACT_V2_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;
  using Base::m_expert_id_map_;
  using Base::m_local_pos_;
  using Base::m_local_input_;
  using Base::m_local_input_ptr_;
  using Base::m_local_gate_output_;
  using Base::m_local_gate_output_ptr_;
  using Base::m_local_up_output_;
  using Base::m_local_up_output_ptr_;
  using Base::m_local_down_output_ptr_;
  using Base::gate_up_ba_pool_;
  using Base::gate_bc_pool_;
  using Base::up_bc_pool_;
  using Base::down_ba_pool_;
  using Base::down_bc_pool_;
  using Base::buffer_a_required_size;
  using Base::buffer_c_required_size;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  // Per (token, slot) results of this part: the down projection's int64 partial row (K-slice of this part), the
  // row exponent of its down input, and the expert id (-1: skipped). Filled by forward_contract, read by the merge.
  // One scratch per NUMA part, shared by every layer's object (layers run one at a time on the CPUInfer queue):
  // a per-object buffer would cost max_len * topk * hidden * 8 bytes per layer and part.
  struct RouteScratch {
    std::vector<int64_t> acc;
    std::vector<int8_t> ea;
    std::vector<int32_t> expert;
    void reserve(size_t routes, size_t hidden) {
      if (acc.size() < routes * hidden) acc.resize(routes * hidden);
      if (ea.size() < routes) { ea.resize(routes); expert.resize(routes, -1); }
    }
  };
  static RouteScratch& route_scratch(int part) {
    static RouteScratch scratch[64];
    if (part < 0 || part >= 64) throw std::runtime_error("contract v2: part index out of range");
    return scratch[part];
  }
  static bool lazy_route_scratch() {
    static const bool enabled = [] {
      const char* value = std::getenv("KT_CONTRACT_V2_LAZY_ROUTE_SCRATCH");
      return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
  }
  int64_t* route_acc_ = nullptr;
  int8_t* route_ea_ = nullptr;
  int32_t* route_expert_ = nullptr;
  std::vector<float> local_weights_;   // routing weight per local row, carved per expert like the base's buffers
  std::vector<float*> local_weight_ptr_;

  AVX2_CONTRACT_V2_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    const size_t routes = (size_t)config_.max_len * config_.num_experts_per_tok;
    auto& s = route_scratch(tp_part_idx);
    if (!lazy_route_scratch()) s.reserve(routes, (size_t)config_.hidden_size);
    route_acc_ = s.acc.data(); route_ea_ = s.ea.data(); route_expert_ = s.expert.data();
    local_weights_.resize(routes + config_.expert_num);
    local_weight_ptr_.resize(config_.expert_num);
    if (tp_part_idx == 0) printf("Created AVX2_CONTRACT_V2_MOE_TP (exact E4M3 x E2M1 contract), max_len %d, topk %d\n", config_.max_len, config_.num_experts_per_tok);
  }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k, config_.quant_config.group_size); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k, config_.quant_config.group_size); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int) {
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_contract_v2(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx], *bb, *bc, ith, nth);
  }
  void do_down_gemm(int expert_idx, int ith, int nth, int) {
    avx2::gemm_contract_v2(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // The contract's activation on one expert's local rows: h = bf((silu_table[gate] * clamp(up, -10, 10)) * weight).
  // Two FP32 multiplies (never contracted into an FMA) and one BF16 RNE, as contract_v2.activate.
  void apply_activation_contract(int expert_idx, int ith, int nth) {
    kt_contract_v2::ScopedFpContract fp_contract;
    const int I = config_.intermediate_size;
    const auto [c0, c1] = T::split_range_n(I, ith, nth);
    const float* silu = kt_contract_v2::silu_table();
    const __m256 lo = _mm256_set1_ps(-10.f), hi = _mm256_set1_ps(10.f);
    for (int row = 0; row < m_local_num_[expert_idx]; row++) {
      uint16_t* gate = reinterpret_cast<uint16_t*>(m_local_gate_output_ptr_[expert_idx]) + (size_t)row * I;
      const uint16_t* up = reinterpret_cast<const uint16_t*>(m_local_up_output_ptr_[expert_idx]) + (size_t)row * I;
      const __m256 w = _mm256_set1_ps(local_weight_ptr_[expert_idx][row]);
      int col = c0;
      for (; col + 8 <= c1; col += 8) {
        const __m256i gbits = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(gate + col)));
        const __m256 s = _mm256_i32gather_ps(silu, gbits, 4);
        const __m256 u = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(up + col))), 16));
        const __m256 uc = _mm256_min_ps(hi, _mm256_max_ps(lo, u));
        const __m256 v = _mm256_mul_ps(_mm256_mul_ps(s, uc), w);
        // BF16 RNE of the FP32 product
        const __m256i vb = _mm256_castps_si256(v);
        const __m256i r = _mm256_add_epi32(_mm256_add_epi32(vb, _mm256_set1_epi32(0x7fff)), _mm256_and_si256(_mm256_srli_epi32(vb, 16), _mm256_set1_epi32(1)));
        const __m256i hi16 = _mm256_srli_epi32(r, 16);
        const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(hi16), _mm256_extracti128_si256(hi16, 1));
        _mm_storeu_si128((__m128i*)(gate + col), packed);
      }
      for (; col < c1; col++) {
        float g, u; uint32_t gb = (uint32_t)gate[col] << 16, ub = (uint32_t)up[col] << 16;
        std::memcpy(&g, &gb, 4); std::memcpy(&u, &ub, 4);
        const float uc = std::min(10.f, std::max(-10.f, u));
        const float v = kt_contract_v2::bf((silu[gate[col]] * uc) * local_weight_ptr_[expert_idx][row]);
        gate[col] = (uint16_t)(kt_contract_v2::bits(v) >> 16);
      }
    }
  }

  // The whole chain for this part. Same bookkeeping and buffer carving as the base's forward_prefill; the
  // activation takes the routing weight, the down projection stays int64 and lands in route_acc_.
  void forward_contract(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (qlen < 1 || qlen > config_.max_len) throw std::runtime_error("contract v2: qlen out of range");
    if (k != config_.num_experts_per_tok) throw std::runtime_error("contract v2: topk mismatch");
    {  // the shared scratch may have grown for a later object: take the current pointers
      auto& s = route_scratch(tp_part_idx);
      s.reserve((size_t)(lazy_route_scratch() ? qlen : config_.max_len) * k, (size_t)config_.hidden_size);
      route_acc_ = s.acc.data(); route_ea_ = s.ea.data(); route_expert_ = s.expert.data();
    }

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++)
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    for (int i = 0; i < config_.expert_num; i++)
      if (m_local_num_[i] > 0) m_expert_id_map_[activated_expert++] = i;

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      local_weight_ptr_[i] = local_weights_.data() + offset;
      offset += m_local_num_[i];
      if (m_local_num_[i] == 0) continue;
      const size_t max_m = m_local_num_[i];
      gate_up_ba_[i]->max_m = max_m; gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.hidden_size)));
      gate_bc_[i]->max_m = max_m; gate_bc_[i]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));
      up_bc_[i]->max_m = max_m; up_bc_[i]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));
      down_ba_[i]->max_m = max_m; down_ba_[i]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.intermediate_size)));
      down_bc_[i]->max_m = max_m; down_bc_[i]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.hidden_size)));
    }
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) { for (int i = 0; i < count; i++) fn(i); }
      else pool->do_work_stealing_job(count, nullptr, fn, nullptr);
    };
    // KT_CONTRACT_V2_TIMING=1: part 0 prints the phase times of its first calls
    static const int timing_calls = [] { const char* v = std::getenv("KT_CONTRACT_V2_TIMING"); return v ? std::atoi(v) * 8 : 0; }();
    static std::atomic<int> timing_left{timing_calls};
    const bool timed = tp_part_idx == 0 && timing_left.load() > 0;
    double tm[8]; int ti = 0;
    auto mark = [&]() { if (timed) tm[ti++] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    mark();

    // Inputs and routing weights to the per-expert local rows
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        const int64_t e = expert_ids[i * k + j];
        if (config_.should_skip_expert(e)) continue;
        memcpy(m_local_input_ptr_[e] + m_local_pos_[i][j] * config_.hidden_size, (const ggml_bf16_t*)input + (size_t)i * config_.hidden_size,
               sizeof(ggml_bf16_t) * config_.hidden_size);
        local_weight_ptr_[e][m_local_pos_[i][j]] = weights[i * k + j];
      }
    });
    if (activated_expert == 0) {
      for (int i = 0; i < qlen * k; i++) route_expert_[i] = -1;
      return;
    }
    mark();
    // Quantise the inputs: row blocks of every active expert spread over the pool (the quantizer is the serial
    // hot spot at decode: 0.17 of 0.46 ms per part when done expert by expert on one thread)
    struct QTask { int e, r0, r1; };
    std::vector<QTask> qtasks;
    auto plan_quant = [&](int block) {
      qtasks.clear();
      for (int t = 0; t < activated_expert; t++) {
        const int e = m_expert_id_map_[t];
        for (int r0 = 0; r0 < m_local_num_[e]; r0 += block) qtasks.push_back({e, r0, std::min(m_local_num_[e], r0 + block)});
      }
    };
    plan_quant(4);
    pool->do_work_stealing_job((int)qtasks.size(), nullptr,
        [this, &qtasks](int t) { const QTask& q = qtasks[t]; gate_up_ba_[q.e]->from_rows(q.r0, q.r1, m_local_input_ptr_[q.e]); }, nullptr);
    mark();
    // Task counts: the base splits every projection into 64-column tasks; at eight parts the down projection's
    // K-slice is 288 wide, so 480 tasks a layer were pure scheduling cost. Give each task at least 64 KiB of weights.
    auto nth_for = [](int n, int k) { return std::max(1, std::min(T::recommended_nth(n), (int)(((size_t)n * k / 2) / 65536))); };
    // Gate and up projections, finished to BF16
    int nth = nth_for(config_.intermediate_size, config_.hidden_size);
    pool->do_work_stealing_job(nth * activated_expert * 2, [](int) { T::config(); },
        [this, nth](int task_id2) {
          const int task_id = task_id2 / 2; const bool do_up = task_id2 % 2;
          const int e = m_expert_id_map_[task_id / nth]; const int ith = task_id % nth;
          do_gate_up_gemm(do_up, e, ith, nth, 0);
          if (do_up) up_bc_[e]->to_mat(m_local_num_[e], m_local_up_output_ptr_[e], ith, nth);
          else gate_bc_[e]->to_mat(m_local_num_[e], m_local_gate_output_ptr_[e], ith, nth);
        }, nullptr);
    mark();
    // Activation with the routing weight (in place in the gate buffer)
    pool->do_work_stealing_job(nth * activated_expert, nullptr,
        [this, nth](int task_id) { apply_activation_contract(m_expert_id_map_[task_id / nth], task_id % nth, nth); }, nullptr);
    mark();
    // Quantise the activations for the down projection (same row-block tasks; k is the part's intermediate slice)
    plan_quant(8);
    pool->do_work_stealing_job((int)qtasks.size(), nullptr,
        [this, &qtasks](int t) { const QTask& q = qtasks[t]; down_ba_[q.e]->from_rows(q.r0, q.r1, m_local_gate_output_ptr_[q.e]); }, nullptr);
    mark();
    // Down projection, exact int64 partials (no finish here: the parts are merged first)
    nth = nth_for(config_.hidden_size, config_.intermediate_size);
    pool->do_work_stealing_job(nth * activated_expert, [](int) { T::config(); },
        [this, nth](int task_id) { do_down_gemm(m_expert_id_map_[task_id / nth], task_id % nth, nth, 0); }, nullptr);
    mark();
    // Per (token, slot): the partial row and its base
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        const size_t r = (size_t)i * k + j; const int64_t e = expert_ids[i * k + j];
        if (config_.should_skip_expert(e)) { route_expert_[r] = -1; continue; }
        const int row = m_local_pos_[i][j];
        std::memcpy(route_acc_ + r * config_.hidden_size, down_bc_[e]->acc + (size_t)row * config_.hidden_size, sizeof(int64_t) * config_.hidden_size);
        route_ea_[r] = down_ba_[e]->EA[row];
        route_expert_[r] = (int32_t)e;
      }
    });
    mark();
    if (timed && ti == 8) {
      timing_left.fetch_sub(1);
      fprintf(stderr, "[contract-v2 part0] qlen %d experts %d | bookkeep+copy %.2f quant %.2f gate/up %.2f act %.2f quant2 %.2f down %.2f scatter %.2f | total %.2f ms\n",
              qlen, activated_expert, tm[1] - tm[0], tm[2] - tm[1], tm[3] - tm[2], tm[4] - tm[3], tm[5] - tm[4], tm[6] - tm[5], tm[7] - tm[6], tm[7] - tm[0]);
    }
  }
};

// TP wrapper: the MXFP4 per-expert slicing for the weights (plus finalize), the exact K-partition merge, and the
// two outputs (routed BF16 sum, or per-route BF16 values through forward_routes_binding).
template <typename K>
class TP_MOE<AVX2_CONTRACT_V2_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_CONTRACT_V2_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_CONTRACT_V2_MOE_TP<K>>>;
  using Part = AVX2_CONTRACT_V2_MOE_TP<K>;
  using Base::Base;

  Part* part(int i) { return static_cast<Part*>(this->tps[i].get()); }

  // Contract BufferB keeps packed weights, FP32 scales and derived exponent
  // metadata in one memfd block. Export that existing allocation, including
  // offsets, instead of duplicating weights into the MXFP4 six-arena layout.
  // The separate name prevents legacy streamers interpreting metadata as scales.
  std::vector<std::vector<std::tuple<int, size_t, size_t, size_t, size_t>>>
  contract_expert_arena_infos() {
    std::vector<std::vector<std::tuple<int, size_t, size_t, size_t, size_t>>> out;
    if (!this->weights_loaded) throw std::runtime_error("contract arena export before load_weights");
    for (int i = 0; i < this->tp_count; ++i) {
      auto* tp = part(i);
      auto infos = tp->expert_arena_info();
      if (infos.size() != 3) throw std::runtime_error("contract arena export requires KT_EXPERT_SHM=1");
      std::vector<std::tuple<int, size_t, size_t, size_t, size_t>> row;
      for (int p = 0; p < 3; ++p) {
        const auto& blocks = p == 0 ? tp->gate_bb_ : (p == 1 ? tp->up_bb_ : tp->down_bb_);
        const auto [fd, size, stride] = infos[p];
        const auto* first = blocks.at(0).get();
        const size_t scales = reinterpret_cast<const uint8_t*>(first->d) - first->b;
        const size_t packed_bytes = (size_t)first->n * first->k / 2;
        const size_t scale_bytes = (size_t)first->n * first->k_group_count * sizeof(float);
        if (fd < 0 || stride % 64 || size != stride * blocks.size() ||
            scales < packed_bytes || scales % 4 || scales + scale_bytes > stride)
          throw std::runtime_error("contract arena export: unsupported BufferB layout");
        for (size_t e = 0; e < blocks.size(); ++e) {
          const auto* b = blocks[e].get();
          if (b->b != first->b + e * stride ||
              reinterpret_cast<const uint8_t*>(b->d) != b->b + scales)
            throw std::runtime_error("contract arena export: nonuniform expert stride");
        }
        row.emplace_back(fd, size, stride, 0, scales);
      }
      out.push_back(std::move(row));
    }
    return out;
  }

  // TP_MOE_Common::warm_up statically dispatches to its floating-point partial
  // path. Exercise this backend's integer projection/merge path instead, within
  // the configured native capacity (max_possible_qlen includes extra padding).
  void warm_up() {
    const auto& c = this->config;
    if (c.max_len < 1 || c.num_experts_per_tok < 1 || c.num_experts_per_tok > 64)
      throw std::runtime_error("contract v2 warm-up: bad shape");
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * c.max_len * c.hidden_size, 0);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * c.max_len * c.hidden_size, 0);
    std::vector<int64_t> ids((size_t)c.max_len * c.num_experts_per_tok);
    std::vector<float> weights(ids.size(), 0.01f);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % c.expert_num;
    forward(c.max_len, c.num_experts_per_tok, ids.data(), weights.data(), input.data(), output.data(), false);
  }

  // The per-expert slices of the MXFP4 TP loader (gate/up rows, down K-columns of this part), copied into the
  // weight blocks the base class already allocated for each BufferB (this backend has no scale-only mode), then
  // the scale codes and row bounds are derived.
  void load_weights() override {
    auto& config = this->config;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.gate_projs.empty()) throw std::runtime_error("contract v2 MoE requires per-expert weight pointers (gate_projs)");
    if (config.quant_config.group_size != 32) throw std::runtime_error("contract v2 MoE requires group size 32");
    const int group_size = 32;
    const int full_interm = config.intermediate_size;
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto* tp = part(i);
      auto& tpc = tp->config_;
      tpc.physical_to_logical_map = config.physical_to_logical_map;
      const int per_tp_interm = tpc.intermediate_size;
      if (per_tp_interm % 32 != 0) throw std::runtime_error("contract v2 TP: the intermediate slice must be a multiple of 32 (one E4M3 group)");
      const size_t gate_up_wt = (size_t)per_tp_interm * tpc.hidden_size / 2;
      pool->get_subpool(i)->do_work_stealing_job(tpc.expert_num, nullptr, [&, i, per_tp_interm, gate_up_wt](int eid) {
        if (tpc.should_skip_expert(eid)) return;
        const uint64_t lid = expert_map(physical_to_logical_map, eid);
        if (lid >= config.gate_projs[0].size() || !config.gate_projs[0][lid] || !config.up_projs[0][lid] || !config.down_projs[0][lid] ||
            !config.gate_scales[0][lid] || !config.up_scales[0][lid] || !config.down_scales[0][lid])
          throw std::runtime_error("contract v2 load_weights: missing weight or scale pointer for expert " + std::to_string(lid));
        const size_t n_byte_off = (size_t)i * per_tp_interm * tpc.hidden_size / 2;
        std::memcpy(tp->gate_bb_[eid]->b, (const uint8_t*)config.gate_projs[0][lid] + n_byte_off, gate_up_wt);
        std::memcpy(tp->up_bb_[eid]->b, (const uint8_t*)config.up_projs[0][lid] + n_byte_off, gate_up_wt);
        const size_t scale_count = (size_t)(tpc.hidden_size / group_size) * per_tp_interm;
        const size_t scale_off = (size_t)i * per_tp_interm * (tpc.hidden_size / group_size);
        convert_or_copy(tp->gate_bb_[eid]->d, (const ggml_bf16_t*)config.gate_scales[0][lid] + scale_off, scale_count);
        convert_or_copy(tp->up_bb_[eid]->d, (const ggml_bf16_t*)config.up_scales[0][lid] + scale_off, scale_count);
        const uint8_t* src_down = (const uint8_t*)config.down_projs[0][lid];
        uint8_t* dst_down = tp->down_bb_[eid]->b;
        for (int row = 0; row < tpc.hidden_size; row++)
          std::memcpy(dst_down + (size_t)row * per_tp_interm / 2, src_down + (size_t)row * full_interm / 2 + (size_t)i * per_tp_interm / 2, per_tp_interm / 2);
        const int full_groups = full_interm / group_size, per_tp_groups = per_tp_interm / group_size;
        const ggml_bf16_t* src_ds = (const ggml_bf16_t*)config.down_scales[0][lid];
        for (int row = 0; row < tpc.hidden_size; row++)
          convert_or_copy(tp->down_bb_[eid]->d + (size_t)row * per_tp_groups, src_ds + (size_t)row * full_groups + (size_t)i * per_tp_groups, per_tp_groups);
        tp->gate_bb_[eid]->finalize(); tp->up_bb_[eid]->finalize(); tp->down_bb_[eid]->finalize();
      }, nullptr);
    });
    this->weights_loaded = true;
  }

  // Exact merge of the parts' partials for one (token, slot) over the output columns [e0, e1): the global base is
  // (max row exponent, max weight exponent per column); each part's partial is shifted (floor) to it. Low bits
  // lost in a shift mean a term of that part sat in the contract's truncation regime: counted, never silent.
  void merge_route(int token, int j, int e0, int e1, float* out_f32) {
    kt_contract_v2::ScopedFpContract fp_contract;
    auto& c = this->config;
    const int parts = this->tp_count;
    if (parts > 64) throw std::runtime_error("contract v2: more than 64 parts");
    const size_t r = (size_t)token * c.num_experts_per_tok + j;
    const int expert = part(0)->route_expert_[r];
    // per-part pointers hoisted out of the column loop (the first version re-derived them per element)
    const int64_t* acc[64]; const int8_t* EW[64]; int base_p[64];
    int EAmax = -128;
    for (int p = 0; p < parts; p++) {
      Part* tp = part(p);
      acc[p] = tp->route_acc_ + r * c.hidden_size;
      EW[p] = tp->down_bb_[expert]->EW;
      base_p[p] = int(tp->route_ea_[r]);
      EAmax = std::max(EAmax, base_p[p]);
    }
    long lost = 0;
    for (int col = e0; col < e1; col++) {
      int EWmax = -128;
      for (int p = 0; p < parts; p++) EWmax = std::max(EWmax, int(EW[p][col]));
      const int top = EAmax + EWmax;
      int64_t sum = 0;
      for (int p = 0; p < parts; p++) {
        const int64_t a = acc[p][col];
        const int d = top - (base_p[p] + int(EW[p][col]));
        if (d > 0) {
          const bool discarded = d >= 64 ? a != 0 : (uint64_t(a) & ((uint64_t(1) << d) - 1)) != 0;
          if (discarded) lost++;
          sum += kt_contract_v2::shr_floor(a, d);
        }
        else sum += a;
      }
      out_f32[col] = kt_contract_v2::bf(std::ldexp(float(sum), top - kt_contract_v2::MARGIN - 10));
    }
    if (lost) kt_contract_v2::g_truncation_events.fetch_add(lost, std::memory_order_relaxed);
  }

  void run_parts(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    this->config.pool->dispense_backend()->do_numa_job([&](int numa_id) { part(numa_id)->forward_contract(qlen, k, expert_ids, weights, input); });
  }

  // Ordinary output: the routed sum in ascending expert id (FP32), rounded once to BF16 [qlen, hidden].
  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output, bool incremental) {
    if (incremental) throw std::runtime_error("contract v2: incremental forward not supported");
    const int qlen = *qlen_ptr;
    run_parts(qlen, k, expert_ids, weights, input);
    auto& c = this->config;
    const int chunk = 256;
    const int chunks = (c.hidden_size + chunk - 1) / chunk;
    auto fn = [&](int task) {
      kt_contract_v2::ScopedFpContract fp_contract;
      const int token = task / chunks, e0 = (task % chunks) * chunk, e1 = std::min(c.hidden_size, e0 + chunk);
      thread_local std::vector<float> accf, y;
      if (y.size() < (size_t)c.hidden_size) { y.resize(c.hidden_size); accf.resize(c.hidden_size); }
      std::fill(accf.begin() + e0, accf.begin() + e1, 0.f);
      std::pair<int, int> order[64]; int n = 0;   // (expert, slot) ascending expert id
      for (int j = 0; j < k; j++) { const int e = part(0)->route_expert_[(size_t)token * k + j]; if (e >= 0) order[n++] = {e, j}; }
      std::sort(order, order + n);
      for (int i = 0; i < n; i++) {
        merge_route(token, order[i].second, e0, e1, y.data());
        for (int col = e0; col < e1; col++) accf[col] += y[col];
      }
      uint16_t* dst = reinterpret_cast<uint16_t*>(output) + (size_t)token * c.hidden_size;
      for (int col = e0; col < e1; col++) dst[col] = (uint16_t)(kt_contract_v2::bits(kt_contract_v2::bf(accf[col])) >> 16);
    };
    if (k > 64) throw std::runtime_error("contract v2: topk > 64");
    const int tasks = qlen * chunks;
    if (tasks <= 2) { for (int t = 0; t < tasks; t++) fn(t); }
    else c.pool->do_work_stealing_job(tasks, nullptr, fn, nullptr);
    if (kt_contract_v2_truncation_events() != 0)
      throw std::runtime_error("contract v2: truncation regime reached; partition invariance is not established");
  }
  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output, bool incremental = false) override {
    int q = qlen; forward(&q, k, expert_ids, weights, input, output, incremental);
  }
  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    forward(qlen_ptr, k, expert_ids, weights, input, output, false);
  }
  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output, bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output, incremental);
  }
  void merge_results(int, void*) override { throw std::runtime_error("contract v2: merge_results is internal to forward"); }

  // Per-route output: BF16 [qlen, topk, hidden], routing weights already applied, zero for skipped routes.
  void forward_routes_binding(intptr_t qlen_ptr, int k, intptr_t ids_ptr, intptr_t weights_ptr, intptr_t input, intptr_t output) {
    const int qlen = *(const int*)qlen_ptr;
    const auto* ids = (const int64_t*)ids_ptr;
    const auto* weights = (const float*)weights_ptr;
    auto& c = this->config;
    if (k != c.num_experts_per_tok || qlen < 1 || qlen > c.max_len) throw std::runtime_error("contract v2 routes: bad shape");
    for (int r = 0; r < qlen * k; ++r) if (ids[r] < -1 || ids[r] >= c.expert_num) throw std::runtime_error("contract v2 route id out of range");
    run_parts(qlen, k, ids, weights, (const void*)input);
    const int chunk = 256, chunks = (c.hidden_size + chunk - 1) / chunk;
    auto fn = [&](int task) {
      kt_contract_v2::ScopedFpContract fp_contract;
      const int token = task / chunks, e0 = (task % chunks) * chunk, e1 = std::min(c.hidden_size, e0 + chunk);
      thread_local std::vector<float> y;
      if (y.size() < (size_t)c.hidden_size) y.resize(c.hidden_size);
      for (int j = 0; j < k; j++) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(output) + ((size_t)token * k + j) * c.hidden_size;
        if (part(0)->route_expert_[(size_t)token * k + j] < 0) { std::memset(dst + e0, 0, sizeof(uint16_t) * (e1 - e0)); continue; }
        merge_route(token, j, e0, e1, y.data());
        for (int col = e0; col < e1; col++) dst[col] = (uint16_t)(kt_contract_v2::bits(y[col]) >> 16);
      }
    };
    const int tasks = qlen * chunks;
    if (tasks <= 2) { for (int t = 0; t < tasks; t++) fn(t); }
    else c.pool->do_work_stealing_job(tasks, nullptr, fn, nullptr);
    if (kt_contract_v2_truncation_events() != 0)
      throw std::runtime_error("contract v2: truncation regime reached; partition invariance is not established");
  }
};

#pragma GCC pop_options
#endif  // KT_ENABLE_CONTRACT_V2
#endif  // CPUINFER_OPERATOR_AVX2_CONTRACT_V2_MOE_H
