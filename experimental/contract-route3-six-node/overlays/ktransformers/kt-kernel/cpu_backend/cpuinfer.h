/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <dlfcn.h>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#endif

#include "./vendors/vendor.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    memops_init();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    memops_init();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    memops_init();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    if (mpoller_.joinable()) {
      mstop_.store(true);
      mpoller_.join();
    }
    delete backend_;
    delete task_queue_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    if (mflags_ != nullptr) {
      const int idx = memop_arm((cudaStream_t)user_cuda_stream, func, args, false, 0);
      write32_((void*)user_cuda_stream, req_dev(idx), 1, 0);
      return;
    }
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    sync_(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA)
    if (mflags_ != nullptr) {
      const int idx = memop_arm((cudaStream_t)user_cuda_stream, nullptr, nullptr, true, allow_n_pending);
      write32_((void*)user_cuda_stream, req_dev(idx), 1, 0);
      wait32_((void*)user_cuda_stream, done_dev(idx), 1, 0x1 /* CU_STREAM_WAIT_VALUE_EQ */);
      write32_((void*)user_cuda_stream, done_dev(idx), 0, 0);
      return;
    }
    SyncArgs* args = new SyncArgs{this, allow_n_pending};
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)args);
#endif
  }
#endif
 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;

 private:
  // ---- KT_STREAM_MEMOPS (native-ubuntu/patches/kt-stream-memops.py) ----
  struct MemopEntry {
    void (*func)(void*);
    void* args;
    size_t allow;
    int idx;
    bool sync;
    bool permanent;
  };
  typedef int (*StreamValueFn)(void*, unsigned long long, unsigned int, unsigned int);
  static constexpr int kRing = 4096, kPerm = 4096, kStride = 16;  // 16 x u32 = one cache line
  StreamValueFn write32_ = nullptr, wait32_ = nullptr;
  volatile uint32_t* mflags_ = nullptr;
  uint32_t* mflags_dev_ = nullptr;
  std::atomic<int> ring_next_{0}, perm_next_{0};
  std::mutex mmtx_;
  std::vector<MemopEntry> mentries_;
  std::atomic<uint64_t> mversion_{0};
  std::thread mpoller_;
  std::atomic<bool> mstop_{false};

  volatile uint32_t* req_host(int i) { return mflags_ + (size_t)i * 2 * kStride; }
  volatile uint32_t* done_host(int i) { return mflags_ + (size_t)i * 2 * kStride + kStride; }
  unsigned long long req_dev(int i) { return (unsigned long long)(mflags_dev_ + (size_t)i * 2 * kStride); }
  unsigned long long done_dev(int i) {
    return (unsigned long long)(mflags_dev_ + (size_t)i * 2 * kStride + kStride);
  }

  void memops_init() {
#if defined(KTRANSFORMERS_USE_CUDA)
    const char* on = getenv("KT_STREAM_MEMOPS");
    if (on == nullptr || on[0] != '1') return;
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h != nullptr) {
      write32_ = (StreamValueFn)dlsym(h, "cuStreamWriteValue32_v2");
      wait32_ = (StreamValueFn)dlsym(h, "cuStreamWaitValue32_v2");
    }
    const size_t bytes = (size_t)(kRing + kPerm) * 2 * kStride * sizeof(uint32_t);
    void* host = nullptr;
    void* dev = nullptr;
    if (write32_ == nullptr || wait32_ == nullptr ||
        cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) {
      printf("[kt-memops] unavailable, keeping host functions\n");
      return;
    }
    std::memset(host, 0, bytes);
    mflags_dev_ = (uint32_t*)dev;
    mflags_ = (volatile uint32_t*)host;
    mpoller_ = std::thread(&CPUInfer::memop_poller, this);
    printf("[kt-memops] CPUInfer[0x%lx]: stream memops on, %d ring + %d graph slots\n", (intptr_t)this, kRing,
           kPerm);
#endif
  }

  int memop_arm(cudaStream_t st, void (*func)(void*), void* args, bool sync, size_t allow) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &cs);
    const bool perm = cs == cudaStreamCaptureStatusActive;
    int idx;
    if (perm) {
      const int n = perm_next_.fetch_add(1);
      if (n >= kPerm) throw std::runtime_error("KT_STREAM_MEMOPS: out of graph slots");
      idx = kRing + n;
    } else {
      idx = ring_next_.fetch_add(1) % kRing;
    }
    std::lock_guard<std::mutex> g(mmtx_);
    mentries_.push_back(MemopEntry{func, args, allow, idx, sync, perm});
    mversion_.fetch_add(1, std::memory_order_release);
    return idx;
  }

  void memop_poller() {
    // KT_MEMOP_POLL_HINT=1 (opt-in): scan from the entry after the last fired one.
    const char* kt_ph = getenv("KT_MEMOP_POLL_HINT");
    const bool poll_hint = kt_ph != nullptr && kt_ph[0] == '1';
    size_t hint = 0;
    std::vector<MemopEntry> snap;
    uint64_t seen = ~0ull;
    while (!mstop_.load(std::memory_order_relaxed)) {
      const uint64_t v = mversion_.load(std::memory_order_acquire);
      if (v != seen) {
        std::lock_guard<std::mutex> g(mmtx_);
        snap = mentries_;
        seen = mversion_.load(std::memory_order_acquire);
      }
      std::vector<int> retire;
      if (!poll_hint) {
        // Arming order is stream order, so within one scan a submit is handled
        // before the sync that follows it.
        for (const MemopEntry& e : snap) {
          if (*req_host(e.idx) != 1) continue;
          *req_host(e.idx) = 0;
          if (!e.sync) {
            e.func(e.args);
          } else {
            task_queue_->sync(e.allow);
            std::atomic_thread_fence(std::memory_order_release);
            *done_host(e.idx) = 1;
          }
          if (!e.permanent) retire.push_back(e.idx);
        }
      } else {
        // Entries fire in arming order: look at the one after the last fired entry first
        // and again every 32 entries, so a request is seen within ~32 cache-line reads
        // instead of up to one full scan of every graph slot. The rotated scan can reach
        // an entry before older ones of its run (the entries armed since the previous
        // sync); those are handled first, oldest first, so stream order holds.
        const size_t n = snap.size();
        if (hint >= n) hint = 0;
        auto handle = [&](size_t i) {
          const MemopEntry& e = snap[i];
          *req_host(e.idx) = 0;
          if (!e.sync) {
            e.func(e.args);
          } else {
            task_queue_->sync(e.allow);
            std::atomic_thread_fence(std::memory_order_release);
            *done_host(e.idx) = 1;
          }
          if (!e.permanent) retire.push_back(e.idx);
          hint = (i + 1) % n;
        };
        auto fire = [&](size_t i) {
          if (*req_host(snap[i].idx) != 1) return;
          size_t j = i;
          while (j > 0 && !snap[j - 1].sync) --j;
          for (; j < i; ++j) {
            if (*req_host(snap[j].idx) == 1) handle(j);
          }
          handle(i);
        };
        const size_t start = hint;
        for (size_t k = 0; k < n; ++k) {
          fire((start + k) % n);
          if ((k & 31) == 31) fire(hint);
        }
      }
      __builtin_ia32_pause();
      if (!retire.empty()) {
        std::lock_guard<std::mutex> g(mmtx_);
        for (int idx : retire) {
          for (size_t i = 0; i < mentries_.size(); i++) {
            if (!mentries_[i].permanent && mentries_[i].idx == idx) {
              mentries_.erase(mentries_.begin() + i);
              break;
            }
          }
        }
        mversion_.fetch_add(1, std::memory_order_release);
      }
    }
  }
};

#endif
