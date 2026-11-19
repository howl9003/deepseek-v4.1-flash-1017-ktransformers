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
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <string>
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
#include "memop_detect.h"
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
    if (mflusher_.joinable()) mflusher_.join();   // KT_MEMOP_DETECT: stops on mstop_ too
    if (mdet_ != nullptr) {
      mdet_->flush(mdet_fd_, true);
      if (mdet_fd_ >= 0) close(mdet_fd_);
      delete mdet_;
    }
    if (mpaired_ != nullptr) {   // KT_MEMOP_PAIRED: the poller has stopped
      printf("[kt-memops] CPUInfer[0x%lx]: KT_MEMOP_PAIRED=1: %llu syncs were left for a later pass\n", (intptr_t)this,
             (unsigned long long)mpaired_->deferred());
      delete mpaired_;
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
    int pair;    // KT_MEMOP_DETECT or KT_MEMOP_PAIRED: a sync's submit slot on the same stream, else -1
    int graph;   // KT_MEMOP_DETECT: the graph ordinal on its first sync (the graph's step marker), else -1
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
  kt_memop_detect::Detector* mdet_ = nullptr;   // KT_MEMOP_DETECT=1 only (memop_detect.h)
  int mdet_fd_ = -1;
  std::thread mflusher_;
  kt_memop_detect::Paired* mpaired_ = nullptr;   // KT_MEMOP_PAIRED=1 only (memop_detect.h): the fix

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
    // pathb-cpu-r1 (13 Nov, ZEN3_OPTIMISATION_r1.md section 3.2 step 6): KT_MEMOP_FLAGS_NODE=<node> makes the calling
    // thread's memory policy MPOL_BIND {node} for the cudaHostAlloc below only, then restores it, so that the flag pages
    // can land on the GPU's node (GPU0: node 1) instead of the rank's interleave:0-7. Whether the driver's pinned pages
    // follow the policy is not known in advance: the pages' nodes are read back (move_pages) and printed. Unset: as before.
    const int flags_node = []() {
      const char* e = getenv("KT_MEMOP_FLAGS_NODE");
      return (e && *e) ? atoi(e) : -1;
    }();
    unsigned long saved_mask[16] = {0};
    int saved_mode = 0;
    bool policy_set = false;
    if (flags_node >= 0 && flags_node < 64 &&
        syscall(SYS_get_mempolicy, &saved_mode, saved_mask, (unsigned long)(sizeof(saved_mask) * 8), nullptr, 0ul) == 0) {
      unsigned long want = 1ul << flags_node;
      policy_set = syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, &want, 64ul) == 0;
    }
    const bool alloc_ok = write32_ != nullptr && wait32_ != nullptr &&
                          cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
    if (alloc_ok && policy_set) std::memset(host, 0, bytes);   // first touch under the bound policy as well
    if (policy_set) {
      const int rc = (saved_mode & 0xff) == 0 ? (int)syscall(SYS_set_mempolicy, 0, nullptr, 0ul)
                                              : (int)syscall(SYS_set_mempolicy, saved_mode, saved_mask,
                                                             (unsigned long)(sizeof(saved_mask) * 8));
      if (rc != 0) printf("[kt-memops] KT_MEMOP_FLAGS_NODE: could not restore the thread's memory policy\n");
    }
    if (flags_node >= 0) {
      std::string where;
      if (alloc_ok) {   // up to 16 sampled 4 KiB pages: their nodes (negative: the kernel's errno for that page)
        const size_t pg = 4096, np = bytes / pg, n = np < 16 ? np : 16;
        void* addr[16];
        int st[16];
        for (size_t i = 0; i < n; i++) {
          addr[i] = (char*)host + (np * i / n) * pg;
          st[i] = -1000;
        }
        const long r = syscall(SYS_move_pages, 0, (unsigned long)n, addr, nullptr, st, 0);
        where = r == 0 ? "" : " (move_pages failed)";
        for (size_t i = 0; r == 0 && i < n; i++) where += " " + std::to_string(st[i]);
      }
      printf("[kt-memops] KT_MEMOP_FLAGS_NODE=%d: policy %s; flag pages' nodes:%s\n", flags_node,
             policy_set ? "bound for the allocation" : "NOT set", where.c_str());
      fflush(stdout);
    }
    if (!alloc_ok || cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) {
      printf("[kt-memops] unavailable, keeping host functions\n");
      return;
    }
    std::memset(host, 0, bytes);
    mflags_dev_ = (uint32_t*)dev;
    mflags_ = (volatile uint32_t*)host;
    if (kt_memop_detect::paired_from_env()) {   // KT_MEMOP_PAIRED=1: set before the poller starts and before any arm
      mpaired_ = new kt_memop_detect::Paired(kRing + kPerm);
      printf("[kt-memops] KT_MEMOP_PAIRED=1: a stream sync is served only after its own submit\n");
    }
    if (kt_memop_detect::on_from_env()) {   // KT_MEMOP_DETECT=1: set before the poller starts and before any arm
      mdet_ = new kt_memop_detect::Detector(kRing + kPerm);
      mdet_fd_ = kt_memop_detect::open_log(kRing, kPerm, mpaired_ != nullptr);
      mflusher_ = std::thread(&CPUInfer::memop_flusher, this);
    }
    mpoller_ = std::thread(mdet_ != nullptr      ? &CPUInfer::memop_poller_detect
                           : mpaired_ != nullptr ? &CPUInfer::memop_poller_paired
                                                 : &CPUInfer::memop_poller,
                           this);
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
    int pair = -1, graph = -1;
    if (mpaired_ != nullptr)   // KT_MEMOP_PAIRED: pair the sync with its submit; a ring slot's count restarts
      pair = mpaired_->arm((const void*)st, idx, sync, perm);
    if (mdet_ != nullptr) {   // KT_MEMOP_DETECT: pair the sync with its submit; the first sync of a graph marks its steps
      unsigned long long cid = 0;
      cudaStreamCaptureStatus cs2 = cudaStreamCaptureStatusNone;
      if (perm && cudaStreamGetCaptureInfo(st, &cs2, &cid) != cudaSuccess) cid = 0;
      int dpair = -1;
      mdet_->arm((const void*)st, idx, sync, perm, (uint64_t)cid, &dpair, &graph);
      if (mpaired_ != nullptr && dpair != pair)   // the same rule on the same calls: never expected
        throw std::runtime_error("KT_MEMOP_PAIRED: the detector paired a stream sync differently");
      pair = dpair;
    }
    mentries_.push_back(MemopEntry{func, args, allow, idx, sync, perm, pair, graph});
    mversion_.fetch_add(1, std::memory_order_release);
    return idx;
  }

  // KT_MEMOP_DETECT=1: memop_poller with the detector's bookkeeping (memop_detect.h scan_detect). The same entries are
  // served in the same order with the same calls; the detector only counts, and logs from its own flusher thread. With
  // KT_MEMOP_PAIRED=1 too, scan_detect applies the pairing rule (as memop_poller_paired) and records each deferral.
  void memop_poller_detect() {
    memop_pin_from_env();
    std::vector<MemopEntry> snap;
    uint64_t seen = ~0ull;
    auto req = [this](int i) { return req_host(i); };
    auto done = [this](int i) { return done_host(i); };
    auto full_sync = [this](size_t allow) { task_queue_->sync(allow); };
    auto pending = [this]() { return task_queue_->pending_now(); };
    auto nothing = [](const MemopEntry&, bool) {};
    while (!mstop_.load(std::memory_order_relaxed)) {
      const uint64_t v = mversion_.load(std::memory_order_acquire);
      if (v != seen) {
        std::lock_guard<std::mutex> g(mmtx_);
        snap = mentries_;
        seen = mversion_.load(std::memory_order_acquire);
      }
      std::vector<int> retire;
      kt_memop_detect::scan_detect(snap, *mdet_, mpaired_, req, done, full_sync, pending, retire, nothing);
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

  // KT_MEMOP_PAIRED=1 without KT_MEMOP_DETECT: memop_poller with the pairing rule (memop_detect.h scan_paired). A raised
  // sync whose own submit has not been served since the sync's previous service is left raised for the next pass, which
  // serves the submit first; the poller never waits for a submit. Otherwise the same entries, order and calls.
  void memop_poller_paired() {
    memop_pin_from_env();
    std::vector<MemopEntry> snap;
    uint64_t seen = ~0ull;
    auto req = [this](int i) { return req_host(i); };
    auto done = [this](int i) { return done_host(i); };
    auto full_sync = [this](size_t allow) { task_queue_->sync(allow); };
    auto nothing = [](const MemopEntry&, bool) {};
    while (!mstop_.load(std::memory_order_relaxed)) {
      const uint64_t v = mversion_.load(std::memory_order_acquire);
      if (v != seen) {
        std::lock_guard<std::mutex> g(mmtx_);
        snap = mentries_;
        seen = mversion_.load(std::memory_order_acquire);
      }
      std::vector<int> retire;
      kt_memop_detect::scan_paired(snap, *mpaired_, req, done, full_sync, retire, nothing);
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

  // KT_MEMOP_DETECT=1: writes the detector's new records once a second; the poller never does I/O.
  void memop_flusher() {
    while (!mstop_.load(std::memory_order_relaxed)) {
      for (int i = 0; i < 10 && !mstop_.load(std::memory_order_relaxed); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      mdet_->flush(mdet_fd_, false);
    }
  }

  // concept-r1 CPU smoothing (12 Nov): KT_MEMOP_POLLER_CPU=<cpu> pins the poller thread. Unset: unpinned, as before.
  // pathb-cpu-r1 (13 Nov): called by all three poller variants (plain, KT_MEMOP_DETECT, KT_MEMOP_PAIRED); before, only
  // the plain poller read the knob, so a paired arm's poller stayed unpinned.
  static void memop_pin_from_env() {
    if (const char* e = getenv("KT_MEMOP_POLLER_CPU"); e && *e) {
      const int cpu = atoi(e);
      if (cpu >= 0 && cpu < CPU_SETSIZE) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        printf("[kt-pin] stream-memop poller pinned to cpu %d (rc %d)\n", cpu, rc);
        fflush(stdout);
      }
    }
  }

  void memop_poller() {
    memop_pin_from_env();
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
