/**
 * @Description  : Contract v3 (tiers C and D) CPU MoE backend for kt-kernel.
 *
 * Built only with KT_ENABLE_CONTRACT_V3 and selected only by KT_MXFP4_BACKEND=contract_v3. It IS his AVX2 MXFP4
 * backend for everything but the arithmetic: TP_MOE<AVX2_CONTRACT_V3H_TP<K>> derives from his
 * TP_MOE<AVX2_MXFP4_MOE_TP<K>>, so the loader, the per-NUMA-part slicing (part p holds intermediate columns
 * [p*I/P, (p+1)*I/P) of every expert: gate/up rows and the down projection's K slice), the KT_EXPERT_SHM memfd arenas,
 * expert_arena_infos() and write_weight_scale_to_buffer (his hot lend and streamer read those) are his, unchanged.
 * Each part's BufferB keeps the checkpoint nibbles (b, [n][k/2], low nibble = even k) and FP32 scales (d, [n][G]).
 *
 * The arithmetic is oracle_v3_r3.py (sha256 bff0374c...), through the v3 CPU library (operators/avx2/contract_v3/
 * v3_rm.{h,cpp}, compiled without fast-math):
 *   BF16 token rows -> int16 (C) / int8 (D) per 32 with a power-of-two scale (once per token, shared by all parts)
 *   -> per part, per routed expert: gate/up over the part's columns (exact integer block dots, FP32 chain over all K
 *   blocks ascending, BF16) -> SwiGLU (table SiLU, routing weight) -> BF16 -> quantiser over the part's 32-blocks
 *   -> down projection as raw exact int64 sums on the layer grid bL over the part's blocks, added straight into the
 *   part's per-token int64 sums (exact, order-free) -> the parts' token sums added exactly.
 * forward_partials writes that raw int64 [qlen, hidden] (spec r3: no per-expert finish; the GPU side adds every other
 * device's raw partial and floor(y_s * 2^-bL), then finishes once). forward() finishes the CPU-only sum to BF16
 * (warm-up and tests only).
 *
 * Refuses: KT_MXFP4_SCALE_CODES or the MXFP4 expert cache (the FP32 scales must be in d), a scale that is not an exact
 * power of two (code 6), no tier (KT_CONTRACT_V3_TIER=C|D), no bL (set_bl or KT_CONTRACT_V3_BL), the int64 grid flag.
 */
#ifndef CPUINFER_OPERATOR_AVX2_CONTRACT_V3H_MOE_H
#define CPUINFER_OPERATOR_AVX2_CONTRACT_V3H_MOE_H
#ifdef KT_ENABLE_CONTRACT_V3

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../moe-tp.hpp"
#include "contract_v3/v3_rm.h"
#include "mxfp4-moe.hpp"

extern "C" const uint32_t kt_contract_v3_silu_bits[65536];
extern "C" const char kt_contract_v3_silu_sha256[];

namespace kt_contract_v3h {
inline int tier() {
  static const int t = [] {
    const char* v = std::getenv("KT_CONTRACT_V3_TIER");
    if (!v || !*v) return -1;
    if ((v[0] == 'C' || v[0] == 'c') && v[1] == 0) return (int)V3_TIER_C;
    if ((v[0] == 'D' || v[0] == 'd') && v[1] == 0) return (int)V3_TIER_D;
    return -2;
  }();
  if (t < 0) throw std::runtime_error("contract v3: KT_CONTRACT_V3_TIER must be C or D");
  return t;
}
inline int bl_from_env(int layer) {
  const char* v = std::getenv("KT_CONTRACT_V3_BL");
  if (!v) return INT32_MIN;
  const char* p = v;
  while (*p) {
    char* e;
    const long l = std::strtol(p, &e, 10);
    if (e == p || *e != ':') throw std::runtime_error("contract v3: malformed KT_CONTRACT_V3_BL");
    p = e + 1;
    const long b = std::strtol(p, &e, 10);
    if (e == p) throw std::runtime_error("contract v3: malformed KT_CONTRACT_V3_BL");
    if (l == layer) return (int)b;
    p = e;
    if (*p == ',') p++;
    else if (*p) throw std::runtime_error("contract v3: malformed KT_CONTRACT_V3_BL");
  }
  return INT32_MIN;
}
inline void check(int rc, const char* what) {
  if (rc == 0) return;
  throw std::runtime_error(std::string("contract v3: ") + what + " failed, code " + std::to_string(rc));
}
inline bool env_on(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0;
}
inline const float* silu_table() { return reinterpret_cast<const float*>(kt_contract_v3_silu_bits); }
}  // namespace kt_contract_v3h

// Tag part type: his MXFP4 part, unchanged (it only selects the TP_MOE specialization below).
template <class K = avx2::GemmKernelAVX2MXFP4>
class AVX2_CONTRACT_V3H_TP : public AVX2_MXFP4_MOE_TP<K> {
 public:
  using AVX2_MXFP4_MOE_TP<K>::AVX2_MXFP4_MOE_TP;
};

template <typename K>
class TP_MOE<AVX2_CONTRACT_V3H_TP<K>> : public TP_MOE<AVX2_MXFP4_MOE_TP<K>> {
 public:
  using Base = TP_MOE<AVX2_MXFP4_MOE_TP<K>>;
  using Base::Base;
  int bL_ = INT32_MIN;

  // per part, per expert: panel-order E8M0 codes of gate, up, down (from the part's FP32 scales, once at load)
  struct Codes { std::vector<uint8_t> g, u, d; };
  std::vector<std::vector<Codes>> codes_;
  // Scratch: ONE process-wide set shared by every layer's instance (r3c, 9 Nov: per-instance grow-only scratch × 40
  // layers × 8 parts took about 0.5 GB a layer at qlen 1024, which is what dropped MemAvailable below the 64 GiB guard
  // at the first prefill). Calls are serialised by the mutex and processed in token chunks of at most rows_cap() rows,
  // so the scratch is bounded (about 0.1 GB) whatever qlen is. Rows are independent and every token sum is an exact
  // int64 sum, so chunking cannot change a bit.
  struct PartScratch { std::vector<uint16_t> gate, up; std::vector<uint8_t> qd; std::vector<int32_t> ead; std::vector<int64_t> tok; };
  struct Shared {
    std::mutex mu;
    std::vector<uint8_t> qx_;            // token rows quantised, [rows][H] int16 (sized for C)
    std::vector<int32_t> eax_;           // [rows][H/32]
    std::vector<uint8_t> qxe_;           // gathered per expert, [routes][H] int16
    std::vector<int32_t> eaxe_;          // [routes][H/32]
    std::vector<int> e_off_, e_cnt_, e_tok_;   // per expert row offset / count; per gathered row its token
    std::vector<float> e_w_;             // per gathered row its routing weight
    std::vector<int> active_;
    std::vector<PartScratch> ps_;
  };
  static Shared& sh() { static Shared s; return s; }
  static int rows_cap() {
    static const int v = [] { const char* e = std::getenv("KT_CONTRACT_V3_ROWS"); int r = e ? std::atoi(e) : 256; return r < 1 ? 256 : r; }();
    return v;
  }
  std::vector<uint8_t>& qx_ = sh().qx_;
  std::vector<int32_t>& eax_ = sh().eax_;
  std::vector<uint8_t>& qxe_ = sh().qxe_;
  std::vector<int32_t>& eaxe_ = sh().eaxe_;
  std::vector<int>& e_off_ = sh().e_off_;
  std::vector<int>& e_cnt_ = sh().e_cnt_;
  std::vector<int>& e_tok_ = sh().e_tok_;
  std::vector<float>& e_w_ = sh().e_w_;
  std::vector<int>& active_ = sh().active_;
  std::vector<PartScratch>& ps_ = sh().ps_;

  void set_bl(int bL) {
    if (bL < -126 || bL > 127) throw std::runtime_error("contract v3: bL out of range");
    bL_ = bL;
  }
  int get_bl() {
    if (bL_ == INT32_MIN) bL_ = kt_contract_v3h::bl_from_env(this->config.layer_idx);
    return bL_;
  }
  int contract_v3_tier() { return kt_contract_v3h::tier(); }
  std::string contract_v3_silu_sha256() { return std::string(kt_contract_v3_silu_sha256); }
  std::string contract_v3_spec() { return "r3"; }

  auto* part(int i) { return this->tps[i].get(); }   // AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>*: public BufferBs

  void load_weights() override {
    if (kt_contract_v3h::env_on("KT_MXFP4_SCALE_CODES"))
      throw std::runtime_error("contract v3: KT_MXFP4_SCALE_CODES must be off (the FP32 scales must stay in BufferB::d)");
    if (AVX2_MXFP4_MOE_TP<K>::kt_cache_mode())
      throw std::runtime_error("contract v3: the MXFP4 expert cache must be off");
    (void)kt_contract_v3h::tier();
    Base::load_weights();
    const int E = this->config.expert_num;
    codes_.assign(this->tp_count, std::vector<Codes>(E));
    std::atomic<int> err{0};
    for (int p = 0; p < this->tp_count; p++) {
      auto* tp = part(p);
      for (int e = 0; e < E; e++) {
        if (!tp->gate_bb_[e] || !tp->gate_bb_[e]->b) continue;   // not held by this pool (GPU-only expert)
        auto conv = [&](const auto& bb, std::vector<uint8_t>& out) {
          if (!bb->b || !bb->d || bb->k_group_size != 32 || bb->n % 8) { int z = 0; err.compare_exchange_strong(z, V3_ERR_SHAPE); return; }
          out.resize((size_t)bb->n * bb->k_group_count);
          const int rc = v3_rm_scales_to_codes(bb->d, bb->n, bb->k_group_count, out.data());
          if (rc) { int z = 0; err.compare_exchange_strong(z, rc); }
        };
        conv(tp->gate_bb_[e], codes_[p][e].g);
        conv(tp->up_bb_[e], codes_[p][e].u);
        conv(tp->down_bb_[e], codes_[p][e].d);
      }
    }
    kt_contract_v3h::check(err.load(), "weight scale conversion");
  }

  static v3_rm_mat view(const auto& bb, const std::vector<uint8_t>& codes, int n0 = 0, int n1 = -1) {
    // rows [n0, n1) of the matrix (n0, n1 multiples of 8) as their own matrix
    if (n1 < 0) n1 = bb->n;
    v3_rm_mat m;
    m.N = n1 - n0; m.K = bb->k; m.G = bb->k_group_count;
    m.row_bytes = (size_t)bb->k / 2;
    m.b = bb->b + (size_t)n0 * m.row_bytes;
    m.s = codes.data() + (size_t)(n0 / 8) * m.G * 8;
    return m;
  }

  // grow-only scratch for qlen tokens (never sized from max_len: his chunked prefill length is far larger than what the
  // CPU pool is handed once the streamer takes long forwards)
  void ensure_scratch(int qlen) {
    const auto& c = this->config;
    const size_t routes = (size_t)qlen * c.num_experts_per_tok, H = c.hidden_size, G = H / 32;
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(qx_, (size_t)qlen * H * 2); grow(eax_, (size_t)qlen * G);
    grow(qxe_, routes * H * 2); grow(eaxe_, routes * G);
    if (e_off_.size() < (size_t)c.expert_num) { e_off_.assign(c.expert_num, 0); e_cnt_.assign(c.expert_num, 0); }
    grow(e_tok_, routes); grow(e_w_, routes);
    if (ps_.size() != (size_t)this->tp_count) ps_.assign(this->tp_count, {});
    for (int p = 0; p < this->tp_count; p++) {
      const size_t Ip = this->tp_configs[p].intermediate_size;
      grow(ps_[p].gate, routes * Ip); grow(ps_[p].up, routes * Ip);
      grow(ps_[p].qd, routes * Ip * 2); grow(ps_[p].ead, routes * (Ip / 32));
      grow(ps_[p].tok, (size_t)qlen * H);
    }
  }

  // raw int64 [qlen][hidden] of this device's routes (ids < 0 or masked GPU experts skipped). Output overwritten.
  void forward_partials_binding(intptr_t qlen_ptr, int k, intptr_t ids_ptr, intptr_t weights_ptr, intptr_t input, intptr_t output) {
    const int qlen = *(const int*)qlen_ptr;
    auto& c = this->config;
    if (!this->weights_loaded) throw std::runtime_error("contract v3: not loaded");
    if (k != c.num_experts_per_tok || qlen < 1 || qlen > c.max_len) throw std::runtime_error("contract v3 partials: bad shape");
    std::lock_guard<std::mutex> lock(sh().mu);
    const int H = c.hidden_size, R = rows_cap();
    for (int t0 = 0; t0 < qlen; t0 += R) {
      const int n = std::min(R, qlen - t0);
      partials_chunk(n, k, (const int64_t*)ids_ptr + (size_t)t0 * k, (const float*)weights_ptr + (size_t)t0 * k,
                     reinterpret_cast<const uint16_t*>(input) + (size_t)t0 * H, reinterpret_cast<int64_t*>(output) + (size_t)t0 * H);
    }
  }

  void partials_chunk(const int qlen, int k, const int64_t* ids, const float* w, const uint16_t* input, int64_t* out) {
    auto& c = this->config;
    const int tier = kt_contract_v3h::tier();
    const int bL = get_bl();
    if (bL == INT32_MIN) throw std::runtime_error("contract v3: no bL for layer " + std::to_string(c.layer_idx));
    ensure_scratch(qlen);
    const int H = c.hidden_size, G = H / 32, E = c.expert_num;
    const size_t qb = tier == V3_TIER_C ? 2 : 1;
    // 1. routes per expert (skipped: id < 0, or masked as a GPU expert)
    std::fill(e_cnt_.begin(), e_cnt_.end(), 0);
    for (int r = 0; r < qlen * k; r++) {
      const int64_t e = ids[r];
      if (e < -1 || e >= E) throw std::runtime_error("contract v3 route id out of range");
      if (e >= 0 && !c.should_skip_expert(e)) e_cnt_[e]++;
    }
    active_.clear();
    int off = 0;
    for (int e = 0; e < E; e++) { e_off_[e] = off; off += e_cnt_[e]; if (e_cnt_[e]) active_.push_back(e); e_cnt_[e] = 0; }
    for (int r = 0; r < qlen * k; r++) {
      const int64_t e = ids[r];
      if (e < 0 || c.should_skip_expert(e)) continue;
      const int row = e_off_[e] + e_cnt_[e]++;
      e_tok_[row] = r / k; e_w_[row] = w[r];
    }
    std::memset(out, 0, sizeof(int64_t) * (size_t)qlen * H);
    if (active_.empty()) return;
    std::atomic<int> err{0};
    auto rec = [&err](int rc) { if (rc) { int z = 0; err.compare_exchange_strong(z, rc); } };
    // 2. quantise each token row once, then gather the rows per expert (int16/int8 copies)
    c.pool->do_work_stealing_job(qlen, nullptr, [&](int t) {
      rec(v3_quantise_bf16(input + (size_t)t * H, 1, H, H, tier,
                           qx_.data() + (size_t)t * H * qb, H, eax_.data() + (size_t)t * G, G));
    }, nullptr);
    kt_contract_v3h::check(err.load(), "quantiser");
    for (int row = 0; row < off; row++) {
      const int t = e_tok_[row];
      std::memcpy(qxe_.data() + (size_t)row * H * qb, qx_.data() + (size_t)t * H * qb, (size_t)H * qb);
      std::memcpy(eaxe_.data() + (size_t)row * G, eax_.data() + (size_t)t * G, sizeof(int32_t) * G);
    }
    // 3. every part: its gate/up columns, SwiGLU + quantiser on its blocks, raw int64 down into its token sums
    std::vector<std::exception_ptr> errs(this->tp_count);
    c.pool->dispense_backend()->do_numa_job([&](int p) {
      try { part_forward(p, qlen, tier, bL, off); } catch (...) { errs[p] = std::current_exception(); }
    });
    for (auto& e : errs)
      if (e) std::rethrow_exception(e);
    // 4. exact sum of the parts' token sums (any order, wrapping). No range check here: this is a device partial; r3h item 2
    //    checks |S| >= 2^62 only on the final layer sum (the GPU finish, flag 4). 
    const int chunk = 256, chunks = (H + chunk - 1) / chunk;
    c.pool->do_work_stealing_job(qlen * chunks, nullptr, [&](int task) {
      const int t = task / chunks, c0 = (task % chunks) * chunk, c1 = std::min(H, c0 + chunk);
      int64_t* o = out + (size_t)t * H;
      for (int p = 0; p < this->tp_count; p++) {
        const int64_t* a = ps_[p].tok.data() + (size_t)t * H;
        for (int col = c0; col < c1; col++) {
          o[col] = (int64_t)((uint64_t)o[col] + (uint64_t)a[col]);   // wrapping (hardening #8); no partial range check (F12)
        }
      }
    }, nullptr);
    kt_contract_v3h::check(err.load(), "part sum");
  }

  void part_forward(int p, int qlen, int tier, int bL, int nrows) {
    auto* tp = part(p);
    auto pool = this->config.pool->get_subpool(p);
    auto& s = ps_[p];
    const int H = this->config.hidden_size, Ip = this->tp_configs[p].intermediate_size, Gp = Ip / 32, G = H / 32;
    const size_t qb = tier == V3_TIER_C ? 2 : 1;
    if (Ip % 32 || H % 8) throw std::runtime_error("contract v3: part slice must be a multiple of 32");
    std::memset(s.tok.data(), 0, sizeof(int64_t) * (size_t)qlen * H);
    std::atomic<int> err{0};
    auto rec = [&err](int rc) { if (rc) { int z = 0; err.compare_exchange_strong(z, rc); } };
    const int na = (int)active_.size();
    for (int e : active_)
      if (!tp->gate_bb_[e] || !tp->gate_bb_[e]->b) throw std::runtime_error("contract v3: routed expert not held by the CPU pool");
    // gate / up: column ranges of whole 8-panels
    const int panels_i = Ip / 8;
    const int nth_gu = std::max(1, std::min(panels_i, (int)(((size_t)Ip * H / 2) / 65536)));
    pool->do_work_stealing_job(nth_gu * na * 2, nullptr, [&](int task) {
      const bool up = task & 1; const int t2 = task >> 1;
      const int e = active_[t2 / nth_gu], ith = t2 % nth_gu;
      const int c0 = 8 * (panels_i * ith / nth_gu), c1 = 8 * (panels_i * (ith + 1) / nth_gu);
      if (c1 <= c0) return;
      const v3_rm_mat m = view(up ? tp->up_bb_[e] : tp->gate_bb_[e], up ? codes_[p][e].u : codes_[p][e].g);
      const size_t r0 = e_off_[e];
      rec(v3_rm_gemm_bf16(&m, tier, qxe_.data() + r0 * H * qb, H, eaxe_.data() + r0 * G, G, e_cnt_[e], c0, c1,
                          (up ? s.up.data() : s.gate.data()) + r0 * Ip, Ip));
    }, nullptr);
    kt_contract_v3h::check(err.load(), "gate/up");
    // SwiGLU (table SiLU, routing weight) + down-input quantiser over the part's blocks
    const int nth_act = std::min(Gp, 3);
    pool->do_work_stealing_job(nth_act * na, nullptr, [&](int task) {
      const int e = active_[task / nth_act], ith = task % nth_act;
      const int g0 = Gp * ith / nth_act, g1 = Gp * (ith + 1) / nth_act;
      const size_t r0 = e_off_[e];
      rec(v3_swiglu_quant(s.gate.data() + r0 * Ip, s.up.data() + r0 * Ip, Ip, e_cnt_[e], e_w_.data() + r0,
                          kt_contract_v3h::silu_table(), g0, g1, tier, nullptr, 0, s.qd.data() + r0 * Ip * qb, Ip,
                          s.ead.data() + r0 * Gp, Gp));
    }, nullptr);
    kt_contract_v3h::check(err.load(), "swiglu");
    // down: per output column range, every active expert in turn (no write races), its rows' raw int64 sums added to
    // their tokens' sums
    const int panels_h = H / 8;
    const int nth_dn = std::max(1, std::min(panels_h, (int)(((size_t)H * Ip / 2) / 65536) * 4));
    pool->do_work_stealing_job(nth_dn, nullptr, [&](int ith) {
      const int c0 = 8 * (panels_h * ith / nth_dn), c1 = 8 * (panels_h * (ith + 1) / nth_dn);
      if (c1 <= c0) return;
      const int wd = c1 - c0;
      thread_local std::vector<int64_t> tmp;
      for (int e : active_) {
        const int m = e_cnt_[e];
        if (tmp.size() < (size_t)m * wd) tmp.resize((size_t)m * wd);
        std::memset(tmp.data(), 0, sizeof(int64_t) * (size_t)m * wd);
        const v3_rm_mat mw = view(tp->down_bb_[e], codes_[p][e].d, c0, c1);
        const size_t r0 = e_off_[e];
        const int rc = v3_rm_down_partial(&mw, tier, s.qd.data() + r0 * Ip * qb, Ip, s.ead.data() + r0 * Gp, Gp, m, bL,
                                          nullptr, 0, 0, wd, tmp.data(), wd);
        if (rc) { rec(rc); return; }
        for (int r = 0; r < m; r++) {
          int64_t* dst = s.tok.data() + (size_t)e_tok_[r0 + r] * H + c0;
          const int64_t* src = tmp.data() + (size_t)r * wd;
          for (int j = 0; j < wd; j++) dst[j] = (int64_t)((uint64_t)dst[j] + (uint64_t)src[j]);   // wrapping (#8)
        }
      }
    }, nullptr);
    kt_contract_v3h::check(err.load(), "down");
  }

  // BF16 of the CPU-only r3 sum (warm-up and tests; the served path uses forward_partials)
  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output, bool incremental) {
    if (incremental) throw std::runtime_error("contract v3: incremental forward not supported");
    const int qlen = *qlen_ptr;
    auto& c = this->config;
    std::vector<int64_t> S((size_t)qlen * c.hidden_size);
    forward_partials_binding((intptr_t)qlen_ptr, k, (intptr_t)expert_ids, (intptr_t)weights, (intptr_t)input, (intptr_t)S.data());
    kt_contract_v3h::check(v3_finish_rows(S.data(), c.hidden_size, qlen, 0, c.hidden_size, get_bl(), (uint16_t*)output,
                                          c.hidden_size, nullptr, 0),
                           "finish");
  }
  // the CPUInfer forward_task path (bind_moe_module binds &MoeClass::forward_binding): v3 too, never his MXFP4 arithmetic
  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output, incremental);
  }
  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output, bool incremental = false) override {
    int q = qlen;
    forward(&q, k, expert_ids, weights, input, output, incremental);
  }
};

#endif  // KT_ENABLE_CONTRACT_V3
#endif  // CPUINFER_OPERATOR_AVX2_CONTRACT_V3H_MOE_H
