/**
 * @Description  : AVX2 MoE base class (ported from amx/moe_base.hpp)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * All AVX512 intrinsics (__m512, _mm512_*) replaced with AVX2 (__m256, _mm256_*).
 * AMX tile configuration calls (T::config()) are kept but are no-ops.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_MOE_BASE_H
#define CPUINFER_OPERATOR_AVX2_MOE_BASE_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/mman.h>
#include <tuple>
#include <unistd.h>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "avx2_bf16_gemm.hpp"
#include "expert_arena_spill.hpp"
#include "avx2_bf16_utils.hpp"
#include "llama.cpp/ggml.h"

// operators/avx2/strict_activation.cpp: act_fn compiled without fast-math (IEEE division).
__m256 kt_strict_act_fn(__m256 gate, __m256 up, float swiglu_limit, float swiglu_alpha);

// KT_DECODE_PHASES=<calls>: see native-ubuntu/patches/kt-decode-phase-timing.py
namespace kt_decode_phases {
inline int every() {
  static const int n = [] {
    const char* e = std::getenv("KT_DECODE_PHASES");
    return e ? std::atoi(e) : 0;
  }();
  return n;
}
struct Part {
  double t[4] = {0, 0, 0, 0};
  long calls = 0, experts = 0;
};
inline Part& part(int i) {
  static Part parts[16];
  return parts[i & 15];
}
inline double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
}  // namespace kt_decode_phases

template <class T, class Derived>
class AVX2_MOE_BASE {
 public:
  int tp_part_idx = 0;

  ggml_bf16_t* m_local_input_ = nullptr;
  ggml_bf16_t* m_local_gate_output_ = nullptr;
  ggml_bf16_t* m_local_up_output_ = nullptr;
  ggml_bf16_t* m_local_down_output_ = nullptr;

  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<ggml_bf16_t*> m_local_input_ptr_;
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  std::vector<void*> owned_aligned_allocs_;
  // KT_EXPERT_SHM=1: expert weight blocks live in memfd-backed shared
  // mappings, one per projection (gate, up, down), expert i at base + i *
  // stride, so a GPU rank in another process can map the same pages and DMA
  // straight from the resident copy (streamed prefill, 09-14).
  struct ExpertArena {
    int fd = -1;
    uint8_t* base = nullptr;
    size_t size = 0, stride = 0;
  };
  std::vector<ExpertArena> expert_arenas_;
  static bool expert_shm_enabled() {
    static const bool on = getenv("KT_EXPERT_SHM") != nullptr && getenv("KT_EXPERT_SHM")[0] == '1';
    return on;
  }
  static ExpertArena make_expert_arena(size_t block, size_t count, const char* name) {
    ExpertArena a;
    a.stride = (block + 63) & ~63ULL;
    a.size = a.stride * count;
    a.fd = memfd_create(name, 0);
    if (a.fd < 0) throw std::runtime_error("memfd_create failed for expert arena");
    if (ftruncate(a.fd, (off_t)a.size) != 0) throw std::runtime_error("ftruncate failed for expert arena");
    void* m = mmap(nullptr, a.size, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
    if (m == MAP_FAILED) throw std::runtime_error("mmap failed for expert arena");
    try {
      kt_expert_arena_spill(m, a.size);
    } catch (...) {
      munmap(m, a.size);
      close(a.fd);
      throw;
    }
    // 2 MB pages when /sys/kernel/mm/transparent_hugepage/shmem_enabled
    // allows it (advise/always): cudaHostRegister of a 4 KB-paged 4.2 GB
    // arena took 3-38 s per layer through the IOMMU (09-13); the DMA engine
    // also prefers them.
    madvise(m, a.size, MADV_HUGEPAGE);
    a.base = (uint8_t*)m;
    return a;
  }
  static void free_expert_arena(ExpertArena& a) {
    if (a.base) munmap(a.base, a.size);
    if (a.fd >= 0) close(a.fd);
    a.base = nullptr;
    a.fd = -1;
  }

  size_t pool_count_ = 0;
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;

  GeneralMOEConfig config_;
  using input_t = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AVX2_MOE_BASE(GeneralMOEConfig config, int tp_part_idx_) : tp_part_idx(tp_part_idx_), config_(config) {
    init();
    derived()->derived_init();
  }

  void init() {
    if (config_.load && config_.path == "") {
      config_.load = false;
    }

    MemoryRequest mem_requests;
    mem_requests.append_pointer(
        &m_local_input_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                         config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.hidden_size);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    // KT_EXPERT_SHM=1: per-expert loaders keep the packed weights in the TP
    // class's per-part buffers, so the base block only ever holds the fp32
    // group scales; allocate scale-only arenas and scale-only BufferBs.
    const bool use_expert_shm = expert_shm_enabled() && kHasScaleOnlyBufferB;
    // Backends whose BufferB is the plain weight matrix (BF16): the whole
    // BufferB goes into the arenas, so the GPU can DMA it as is.
    const bool use_full_shm = expert_shm_enabled() && !kHasScaleOnlyBufferB;
    if (use_expert_shm) {
      expert_arenas_.push_back(make_expert_arena(
          buffer_b_scale_size(config_.intermediate_size, config_.hidden_size), config_.expert_num, "kt_gate_s"));
      expert_arenas_.push_back(make_expert_arena(
          buffer_b_scale_size(config_.intermediate_size, config_.hidden_size), config_.expert_num, "kt_up_s"));
      expert_arenas_.push_back(make_expert_arena(
          buffer_b_scale_size(config_.hidden_size, config_.intermediate_size), config_.expert_num, "kt_down_s"));
    } else if (use_full_shm) {
      expert_arenas_.push_back(make_expert_arena(
          buffer_b_required_size(config_.intermediate_size, config_.hidden_size), config_.expert_num, "kt_gate_w"));
      expert_arenas_.push_back(make_expert_arena(
          buffer_b_required_size(config_.intermediate_size, config_.hidden_size), config_.expert_num, "kt_up_w"));
      expert_arenas_.push_back(make_expert_arena(
          buffer_b_required_size(config_.hidden_size, config_.intermediate_size), config_.expert_num, "kt_down_w"));
    }

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(make_buffer_a(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(make_buffer_a(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(make_buffer_c(config_.max_len, config_.hidden_size, nullptr));

      void *gate_bb_ptr, *up_bb_ptr, *down_bb_ptr;
      if (use_expert_shm || use_full_shm) {
        gate_bb_ptr = expert_arenas_[0].base + i * expert_arenas_[0].stride;
        up_bb_ptr = expert_arenas_[1].base + i * expert_arenas_[1].stride;
        down_bb_ptr = expert_arenas_[2].base + i * expert_arenas_[2].stride;
      } else {
        gate_bb_ptr = std::aligned_alloc(
            64, (buffer_b_required_size(config_.intermediate_size, config_.hidden_size) + 63) & ~63ULL);
        if (!gate_bb_ptr) throw std::runtime_error("aligned_alloc failed for gate BufferB");
        owned_aligned_allocs_.push_back(gate_bb_ptr);
        up_bb_ptr = std::aligned_alloc(
            64, (buffer_b_required_size(config_.intermediate_size, config_.hidden_size) + 63) & ~63ULL);
        if (!up_bb_ptr) throw std::runtime_error("aligned_alloc failed for up BufferB");
        owned_aligned_allocs_.push_back(up_bb_ptr);
        down_bb_ptr = std::aligned_alloc(
            64, (buffer_b_required_size(config_.hidden_size, config_.intermediate_size) + 63) & ~63ULL);
        if (!down_bb_ptr) throw std::runtime_error("aligned_alloc failed for down BufferB");
        owned_aligned_allocs_.push_back(down_bb_ptr);
      }
      if (use_expert_shm) {
        gate_bb_.push_back(make_buffer_b_scale_only(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));
        up_bb_.push_back(make_buffer_b_scale_only(config_.intermediate_size, config_.hidden_size, up_bb_ptr));
        down_bb_.push_back(make_buffer_b_scale_only(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
      } else {
        gate_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));
        up_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, up_bb_ptr));
        down_bb_.push_back(make_buffer_b(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
      }
    }

    pool_count_ = config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::M_STEP;

    gate_up_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;
    gate_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    up_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  virtual ~AVX2_MOE_BASE() {
    shared_mem_buffer_numa.dealloc(tp_part_idx, this);
    for (void* p : owned_aligned_allocs_) std::free(p);
    for (auto& a : expert_arenas_) free_expert_arena(a);
  }

  // (fd, size, stride) of the gate, up and down arenas; empty unless
  // KT_EXPERT_SHM=1. Within a block: packed weights first, fp32 group scales
  // at n * k / 2 (BufferB's layout).
  std::vector<std::tuple<int, size_t, size_t>> expert_arena_info() const {
    std::vector<std::tuple<int, size_t, size_t>> out;
    for (auto& a : expert_arenas_) out.emplace_back(a.fd, a.size, a.stride);
    return out;
  }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if constexpr (requires { typename Derived::ExactTpMergeOnly; }) {
      throw std::runtime_error("exact-merge backend cannot use generic float-partial forward");
    } else {
      if (qlen > 1) {
        forward_prefill(qlen, k, expert_ids, weights, input, output);
      } else {
        forward_decode(k, expert_ids, weights, input, output);
      }
    }
  }

  template <typename... Args>
  void load_weights(Args&&... args) {
    derived()->load_weights(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void write_weights_to_buffer(Args&&... args) const {
    derived_const()->write_weights_to_buffer(std::forward<Args>(args)...);
  }

  static bool stage_timing_enabled() {
    static const bool on = getenv("KT_PREFILL_STAGE_TIMING") != nullptr;
    return on;
  }
  static double stage_now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const bool st_on = stage_timing_enabled();
    double st_t[8]; int st_i = 0;
    auto st_mark = [&]() { if (st_on) st_t[st_i++] = stage_now_ms(); };
    st_mark();

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }
    const bool fused_prefill = activated_expert > 0 && has_fused_prefill();

    // Assign pool memory to buffers
    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) continue;

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr =
          (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.hidden_size)));

      if (!fused_prefill) {
        gate_bc_[i]->max_m = max_m;
        gate_bc_[i]->set_data(gate_bc_pool_ptr);
        gate_bc_pool_ptr =
            (void*)((uintptr_t)gate_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

        up_bc_[i]->max_m = max_m;
        up_bc_[i]->set_data(up_bc_pool_ptr);
        up_bc_pool_ptr =
            (void*)((uintptr_t)up_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));
      }

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr =
          (void*)((uintptr_t)down_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.intermediate_size)));

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr =
          (void*)((uintptr_t)down_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.hidden_size)));
    }

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) fn(i);
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    st_mark();  // 1: routing bookkeeping + buffer carve
    // Copy input to per-expert buffers
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

    st_mark();  // 2: copy_input
    // Pack input into BufferA (trivial memcpy for AVX2)
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

    st_mark();  // 3: pack
    if constexpr (has_fused_prefill()) {
      if (activated_expert > 0) run_fused_prefill(activated_expert, qlen);
    } else {
      // Gate + Up GEMM
      int nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * activated_expert * 2, [](int) { T::config(); },
          [this, nth, qlen](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
            if (fp32_path()) return;  // KT_MOE_FP32_INTERMEDIATES: the FP32 result stays in BufferC
            if (do_up) {
              up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
            } else {
              gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
            }
          },
          nullptr);

      st_mark();  // 4: gate/up gemm
      // Activation: SiLU(gate) * up — AVX2 version (8 elements at a time)
      if (fp32_path())
        apply_activation_fp32(activated_expert, nth, qlen);
      else
        apply_activation(activated_expert, nth, qlen);
      st_mark();  // 5: activation

      // Pack activation output into BufferA for down projection
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [this](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            if constexpr (kHasF32Input) {
              if (fp32_path()) {
                down_ba_[expert_idx]->from_mat_f32(m_local_num_[expert_idx], gate_bc_[expert_idx]->data, 0, 1);
                return;
              }
            }
            down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
          },
          nullptr);

      st_mark();  // 6: pack down
      // Down GEMM
      nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * activated_expert, [](int) { T::config(); },
          [this, nth, qlen](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            derived()->do_down_gemm(expert_idx, ith, nth, qlen);
            if (fp32_path()) return;  // KT_MOE_FP32_INTERMEDIATES: FP32 partial stays in BufferC
            down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
          },
          nullptr);
    }

    st_mark();  // 7: down gemm
    // Weighted sum of expert outputs — AVX2 version (16 BF16 = 2x8 FP32 at a time)
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 16) {
            __m256 x0 = _mm256_setzero_ps();
            __m256 x1 = _mm256_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
              __m256 weight = _mm256_set1_ps(weights[i * k + j]);
              __m256 d0, d1;
              if (fp32_path() && !has_fused_prefill()) {
                const float* src = down_bc_[expert_ids[i * k + j]]->data + (size_t)m_local_pos_[i][j] * config_.hidden_size + e;
                d0 = _mm256_loadu_ps(src);
                d1 = _mm256_loadu_ps(src + 8);
              } else {
                avx2::load_16xbf16_to_2x8xfp32(
                    m_local_down_output_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size + e, &d0,
                    &d1);
              }
              x0 = _mm256_fmadd_ps(d0, weight, x0);
              x1 = _mm256_fmadd_ps(d1, weight, x1);
            }
            auto f32out = (__m256*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);
    if (st_on) {
      st_t[st_i++] = stage_now_ms();
      // rows per expert: total routed rows on this part / activated experts
      size_t rows = 0; for (int i = 0; i < config_.expert_num; i++) rows += m_local_num_[i];
      fprintf(stderr, "[kt-prefill-stage] tp%d layer%d qlen=%d experts=%d rows=%zu | bookkeep %.1f copy %.1f pack %.1f gateup %.1f act %.1f packdown %.1f down %.1f sum %.1f | total %.1f ms\n",
              tp_part_idx, config_.layer_idx, qlen, activated_expert, rows,
              st_t[1]-st_t[0], st_t[2]-st_t[1], st_t[3]-st_t[2], st_t[4]-st_t[3], st_t[5]-st_t[4], st_t[6]-st_t[5], st_t[7]-st_t[6], st_t[8]-st_t[7], st_t[8]-st_t[0]);
    }
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const bool kt_timed = kt_decode_phases::every() > 0;
    std::chrono::steady_clock::time_point kt_t0, kt_t1, kt_t2, kt_t3;
    if (kt_timed) kt_t0 = std::chrono::steady_clock::now();

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) continue;
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    // Assign pool memory for decode
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    const bool gate_up_fused = activated_expert > 0 && has_fused_gate_up_decode();

    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      if (!gate_up_fused) {
        gate_bc_[expert_idx]->max_m = max_m;
        gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
        gate_bc_pool_ptr =
            (void*)((uintptr_t)gate_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));

        up_bc_[expert_idx]->max_m = max_m;
        up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
        up_bc_pool_ptr =
            (void*)((uintptr_t)up_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.intermediate_size)));
      }

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr =
          (void*)((uintptr_t)down_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.intermediate_size)));

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr =
          (void*)((uintptr_t)down_bc_pool_ptr + align64(buffer_c_required_size(max_m, config_.hidden_size)));
    }

    // Pack input into BufferA for each activated expert
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr =
          (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(buffer_a_required_size(max_m, config_.hidden_size)));
      gate_up_ba_[expert_idx]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);
    }

    if (kt_timed) kt_t1 = std::chrono::steady_clock::now();
    // Gate + Up GEMM
    int nth = T::recommended_nth(config_.intermediate_size);
    if constexpr (has_fused_gate_up_decode()) {
      if (activated_expert > 0) run_fused_gate_up_decode(activated_expert, qlen);
    } else {
      pool->do_work_stealing_job(
          nth * activated_expert * 2, [](int) { T::config(); },
          [this, nth, qlen](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
            if (fp32_path()) return;  // KT_MOE_FP32_INTERMEDIATES: the FP32 result stays in BufferC
            if (do_up) {
              up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
            } else {
              gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
            }
          },
          nullptr);

      // Activation
      if (fp32_path())
        apply_activation_fp32(activated_expert, nth, qlen);
      else
        apply_activation(activated_expert, nth, qlen);

      // Pack for down projection
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [this, qlen](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            if constexpr (kHasF32Input) {
              if (fp32_path()) {
                down_ba_[expert_idx]->from_mat_f32(qlen, gate_bc_[expert_idx]->data, 0, 1);
                return;
              }
            }
            down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
          },
          nullptr);
    }

    if (kt_timed) kt_t2 = std::chrono::steady_clock::now();
    // Down GEMM
    nth = T::recommended_nth(config_.hidden_size);
    if constexpr (has_fused_down_decode()) {
      if (activated_expert > 0) run_fused_down_decode(activated_expert, qlen);
    } else {
      pool->do_work_stealing_job(
          nth * activated_expert, [](int) { T::config(); },
          [this, nth, qlen](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            derived()->do_down_gemm(expert_idx, ith, nth, qlen);
            if (fp32_path()) return;  // KT_MOE_FP32_INTERMEDIATES: FP32 partial stays in BufferC
            down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
          },
          nullptr);
    }

    if (kt_timed) kt_t3 = std::chrono::steady_clock::now();
    // Weighted sum — AVX2 (16 BF16 at a time)
    for (int e = 0; e < config_.hidden_size; e += 16) {
      __m256 x0 = _mm256_setzero_ps();
      __m256 x1 = _mm256_setzero_ps();
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[j])) continue;
        __m256 weight = _mm256_set1_ps(weights[j]);
        __m256 d0, d1;
        if (fp32_path() && !gate_up_fused) {
          const float* src = down_bc_[expert_ids[j]]->data + (size_t)m_local_pos_[0][j] * config_.hidden_size + e;
          d0 = _mm256_loadu_ps(src);
          d1 = _mm256_loadu_ps(src + 8);
        } else {
          avx2::load_16xbf16_to_2x8xfp32(
              m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size + e, &d0, &d1);
        }
        x0 = _mm256_fmadd_ps(d0, weight, x0);
        x1 = _mm256_fmadd_ps(d1, weight, x1);
      }
      auto f32out = (__m256*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }
    if (kt_timed) {
      auto kt_t4 = std::chrono::steady_clock::now();
      auto& p = kt_decode_phases::part(tp_part_idx);
      p.t[0] += kt_decode_phases::us(kt_t0, kt_t1);
      p.t[1] += kt_decode_phases::us(kt_t1, kt_t2);
      p.t[2] += kt_decode_phases::us(kt_t2, kt_t3);
      p.t[3] += kt_decode_phases::us(kt_t3, kt_t4);
      p.experts += activated_expert;
      if (++p.calls == kt_decode_phases::every()) {
        std::printf("[kt-decode] part %d, %ld calls: prep %.1f us, gate+up %.1f us, down %.1f us, "
                    "wsum %.1f us, %.2f experts/call\n",
                    tp_part_idx, p.calls, p.t[0] / p.calls, p.t[1] / p.calls, p.t[2] / p.calls,
                    p.t[3] / p.calls, (double)p.experts / p.calls);
        std::fflush(stdout);
        p = kt_decode_phases::Part();
      }
    }
  }

 protected:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived_const() const { return static_cast<const Derived*>(this); }

  static constexpr bool has_fused_prefill() {
    return requires(Derived& backend) { backend.fused_prefill_experts(0, 0); };
  }

  static constexpr bool has_fused_gate_up_decode() {
    return requires(Derived& backend) { backend.decode_gate_up_activation(0, 0); };
  }

  static constexpr bool has_fused_down_decode() {
    return requires(Derived& backend) { backend.decode_down_projection(0, 0); };
  }

  void run_fused_prefill(int activated_experts, int qlen) {
    if constexpr (has_fused_prefill()) derived()->fused_prefill_experts(activated_experts, qlen);
  }

  void run_fused_gate_up_decode(int activated_experts, int qlen) {
    if constexpr (has_fused_gate_up_decode()) derived()->decode_gate_up_activation(activated_experts, qlen);
  }

  void run_fused_down_decode(int activated_experts, int qlen) {
    if constexpr (has_fused_down_decode()) derived()->decode_down_projection(activated_experts, qlen);
  }

  void derived_init() {}

  // Buffer creation/size delegation (CRTP)
  size_t buffer_a_required_size(size_t m, size_t k) const { return derived_const()->buffer_a_required_size_impl(m, k); }
  size_t buffer_b_required_size(size_t n, size_t k) const { return derived_const()->buffer_b_required_size_impl(n, k); }
  size_t buffer_c_required_size(size_t m, size_t n) const { return derived_const()->buffer_c_required_size_impl(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a(size_t m, size_t k, void* data) const {
    return derived_const()->make_buffer_a_impl(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b(size_t n, size_t k, void* data) const {
    return derived_const()->make_buffer_b_impl(n, k, data);
  }
  // Backends whose BufferB can be scale-only (MXFP4, RawInt4) can keep the
  // packed weights elsewhere; the KT_EXPERT_SHM arenas rely on that.
  static constexpr bool kHasScaleOnlyBufferB =
      requires(const Derived& d) {
        T::BufferB::required_size_scale_only(size_t{}, size_t{}, int{});
        d.make_buffer_b_scale_only_impl(size_t{}, size_t{}, (void*)nullptr);
      };
  std::shared_ptr<typename T::BufferB> make_buffer_b_scale_only(size_t n, size_t k, void* scales) const {
    if constexpr (kHasScaleOnlyBufferB) {
      return derived_const()->make_buffer_b_scale_only_impl(n, k, scales);
    } else {
      throw std::runtime_error("KT_EXPERT_SHM needs a backend with scale-only BufferB");
    }
  }
  size_t buffer_b_scale_size(size_t n, size_t k) const {
    if constexpr (kHasScaleOnlyBufferB) {
      return T::BufferB::required_size_scale_only(n, k, config_.quant_config.group_size);
    } else {
      return 0;
    }
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c(size_t m, size_t n, void* data) const {
    return derived_const()->make_buffer_c_impl(m, n, data);
  }

  // KT_MOE_FP32_INTERMEDIATES=1 (default off): the gate/up outputs, the activation
  // and the down partials stay FP32 where the kernels already produce them; the
  // only BF16 rounding left is the merged output (merge_results). Only backends whose
  // BufferA accepts FP32 rows (from_mat_f32) take the path; all others are unchanged.
  static bool fp32_intermediates() {
    static const bool on = [] {
      const char* value = std::getenv("KT_MOE_FP32_INTERMEDIATES");
      return value && std::string(value) == "1";
    }();
    return on;
  }
  static constexpr bool kHasF32Input =
      requires(typename T::BufferA& a, const float* rows) { a.from_mat_f32(1, rows, 0, 1); };
  bool fp32_path() const {
    if constexpr (kHasF32Input) return fp32_intermediates() &&
        config_.quant_config.group_size == 32 &&
        config_.activation_type == MOE_ACTIVATION_SILU &&
        !has_fused_prefill() && !has_fused_gate_up_decode() && !has_fused_down_decode();
    else return false;
  }

  // FP32 in, FP32 out, in place over the gate GEMM output. kt_strict_act_fn lives in
  // operators/avx2/strict_activation.cpp, compiled without fast-math (IEEE division,
  // no reciprocal rewrite), so the CPU activation matches the GPU's div.rn.
  void apply_activation_fp32(int activated_expert, int nth, int qlen) {
    ASSERT_RELEASE(config_.activation_type == MOE_ACTIVATION_SILU, "KT_MOE_FP32_INTERMEDIATES supports SiLU only");
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const float limit = config_.swiglu_limit, alpha = config_.swiglu_alpha;
    auto fn = [this, nth, limit, alpha](int task_id) {
      const int expert_idx = m_expert_id_map_[task_id / nth];
      const auto [begin, end] = T::split_range_n(config_.intermediate_size, task_id % nth, nth);
      const size_t stride = gate_bc_[expert_idx]->n;
      for (int row = 0; row < m_local_num_[expert_idx]; ++row) {
        float* gate = gate_bc_[expert_idx]->data + (size_t)row * stride;
        const float* up = up_bc_[expert_idx]->data + (size_t)row * stride;
        int col = begin;
        for (; col + 8 <= end; col += 8)
          _mm256_storeu_ps(gate + col, kt_strict_act_fn(_mm256_loadu_ps(gate + col), _mm256_loadu_ps(up + col), limit, alpha));
        if (col < end) {
          // Zero padding keeps every lane on the vector arithmetic and inside the task's range.
          alignas(32) float g[8]{}, u[8]{}, out[8]{};
          const int count = end - col;
          for (int lane = 0; lane < count; ++lane) { g[lane] = gate[col + lane]; u[lane] = up[col + lane]; }
          _mm256_store_ps(out, kt_strict_act_fn(_mm256_load_ps(g), _mm256_load_ps(u), limit, alpha));
          for (int lane = 0; lane < count; ++lane) gate[col + lane] = out[lane];
        }
      }
    };
    if (activated_expert == 0) return;
    if (qlen < 10) {
      for (int task = 0; task < nth * activated_expert; ++task) fn(task);
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }

  // Routed-expert activation — AVX2: process 8 BF16 elements at a time.
  void apply_activation(int activated_expert, int nth, int qlen) {
    ASSERT_RELEASE(
        config_.activation_type == MOE_ACTIVATION_SILU ||
            config_.activation_type == MOE_ACTIVATION_GELU_TANH,
        "Unsupported AVX2 MoE activation_type");
    const bool use_gelu_tanh = config_.activation_type == MOE_ACTIVATION_GELU_TANH;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, nth, use_gelu_tanh](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      const float swiglu_limit = config_.swiglu_limit;
      const float swiglu_alpha = config_.swiglu_alpha;
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          __m256 gate_val = avx2::load_bf16_to_fp32(gate_ptr + j);
          __m256 up_val = avx2::load_bf16_to_fp32(up_ptr + j);
          __m256 result = use_gelu_tanh
                              ? avx2::gelu_tanh_fn(gate_val, up_val)
                              : avx2::act_fn(gate_val, up_val, swiglu_limit, swiglu_alpha);
          avx2::store_fp32_to_bf16(gate_ptr + j, result);
        }
        // Scalar tail — mirror the selected vectorized activation exactly.
        for (; j < n_end; j++) {
          float g = GGML_BF16_TO_FP32(gate_ptr[j]);
          float u = GGML_BF16_TO_FP32(up_ptr[j]);
          if (use_gelu_tanh) {
            constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
            constexpr float kCubicCoeff = 0.044715f;
            const float inner = kSqrtTwoOverPi * (g + kCubicCoeff * g * g * g);
            const float gelu = 0.5f * g * (1.0f + std::tanh(inner));
            gate_ptr[j] = GGML_FP32_TO_BF16(gelu * u);
          } else if (swiglu_alpha > 0.0f) {
            if (swiglu_limit > 0.0f) {
              g = std::min(std::max(g, -swiglu_limit), swiglu_limit);
              u = std::min(std::max(u, -swiglu_limit), swiglu_limit);
            }
            float sigmoid_ga = 1.0f / (1.0f + expf(-g * swiglu_alpha));
            gate_ptr[j] = GGML_FP32_TO_BF16(g * sigmoid_ga * (u + 1.0f));
          } else {
            if (swiglu_limit > 0.0f) {
              g = std::min(g, swiglu_limit);
              u = std::min(std::max(u, -swiglu_limit), swiglu_limit);
            }
            float sigmoid_g = 1.0f / (1.0f + expf(-g));
            gate_ptr[j] = GGML_FP32_TO_BF16(g * sigmoid_g * u);
          }
        }
      }
    };

    if (activated_expert == 0) return;

    if (qlen < 10) {
      for (int task_id = 0; task_id < nth * activated_expert; task_id++) fn(task_id);
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AVX2_MOE_BASE derived classes
// ============================================================================

template <class T, class Derived>
class TP_MOE<AVX2_MOE_BASE<T, Derived>> : public TP_MOE_Common<AVX2_MOE_BASE<T, Derived>, Derived> {
 public:
  using TP_MOE_Common<AVX2_MOE_BASE<T, Derived>, Derived>::TP_MOE_Common;

  void load_weights() override { throw std::runtime_error("Not Implemented"); }

  void write_weight_scale_to_buffer(int gpu_tp_count, int gpu_experts_num,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    throw std::runtime_error("Not Implemented");
  }

  bool supports_fp32_output() const override {
    // In this main-based tree only the stripped MXFP4 BufferA has this API.
    return requires(typename T::BufferA& a, const float* rows) {
      a.from_mat_f32(1, rows, 0, 1);
    };
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental && config.output_fp32) {
        const float* previous = static_cast<const float*>(output) + token_nth * config.hidden_size;
        for (int e = 0; e < config.hidden_size; e += 8) {
          _mm256_storeu_ps(merge_to + e, _mm256_add_ps(
              _mm256_loadu_ps(merge_to + e), _mm256_loadu_ps(previous + e)));
        }
      } else if (incremental) {
        // Convert BF16 output to FP32 and add — AVX2 (16 BF16 at a time)
        for (int e = 0; e < config.hidden_size; e += 16) {
          __m256 x0, x1;
          avx2::load_16xbf16_to_2x8xfp32((ggml_bf16_t*)output + token_nth * config.hidden_size + e, &x0, &x1);
          *((__m256*)(merge_to + e)) = _mm256_add_ps(*((__m256*)(merge_to + e)), x0);
          *((__m256*)(merge_to + e + 8)) = _mm256_add_ps(*((__m256*)(merge_to + e + 8)), x1);
        }
      }
      // Sum across TP parts
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 8) {
          *((__m256*)(merge_to + e)) = _mm256_add_ps(*((__m256*)(merge_to + e)), *((__m256*)(merge_from + e)));
        }
      }
      if (config.output_fp32) {
        std::memcpy(static_cast<float*>(output) + token_nth * config.hidden_size,
                    merge_to, sizeof(float) * config.hidden_size);
        return;
      }
      // Convert FP32 -> BF16 output
      for (int e = 0; e < config.hidden_size; e += 16) {
        __m256 x0 = *(__m256*)(merge_to + e);
        __m256 x1 = *(__m256*)(merge_to + e + 8);
        avx2::store_2x8xfp32_to_16xbf16(&x0, &x1, (ggml_bf16_t*)output + token_nth * config.hidden_size + e);
      }
    };

    auto pool = config.pool;
    if (qlen < 10) {
      for (int i = 0; i < qlen; i++) merge_fn(i);
    } else {
      pool->do_work_stealing_job(qlen, nullptr, merge_fn, nullptr);
    }
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AVX2_MOE_BASE_H
