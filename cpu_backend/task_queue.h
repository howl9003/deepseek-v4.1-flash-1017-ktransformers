/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-16 10:43:18
 * @Version   : 1.0.0
 * @LastEditors : chenht
 * @LastEditTime : 2024-10-09 11:08:07
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// concept-r1 pre-read (PREREAD_r1.md, 13 Nov 2026). Two process-wide words, no TaskQueue member added:
//   kt_tq_idle_hook      null unless the contract v3 backend sets it (KT_PREREAD=1, or its test binding). When set, the
//                        worker calls it after a task has finished and its completion is published (pending decremented,
//                        waiters notified) and nothing is queued. `pending` is the queue's own count: the hook must
//                        return as soon as it reads non-zero. It is called on the worker thread, so the next task cannot
//                        start before it returns. Null: nothing is called and the worker behaves as before.
//   kt_tq_last_qwait_ns  with KT_TASKQUEUE_TIMING set: the queue wait (enqueue to start) of the task the worker started
//                        last, in ns. Only read by tests.
typedef void (*kt_tq_idle_fn)(const std::atomic<size_t>* pending);
extern std::atomic<kt_tq_idle_fn> kt_tq_idle_hook;
extern std::atomic<uint64_t> kt_tq_last_qwait_ns;

class TaskQueue {
 public:
  TaskQueue();
  ~TaskQueue();

  void enqueue(std::function<void()>);

  void sync(size_t allow_n_pending);

  // KT_MEMOP_DETECT (memop_detect.h): tasks enqueued and not finished, as sync() reads it. Read-only.
  size_t pending_now() const { return pending.load(std::memory_order_acquire); }

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    uint64_t t_enq = 0;
    Node() : task(nullptr), next(nullptr) {}
    Node(const std::function<void()>& t) : task(t), next(nullptr) {}
  };

  // KT_TASKQUEUE_TIMING=<syncs>: every that many sync() calls, print how long
  // tasks waited in the queue, how long they ran, and how long sync() blocked,
  // to stderr. One sync per MoE layer in decode, so 400 = ten tokens.
  int timing_every = 0;
  // KT_TASKQUEUE_SPIN_US: spin this long before a condition-variable wait.
  int spin_us = 0;
  std::atomic<uint64_t> n_tasks{0}, q_ns{0}, run_ns{0}, n_syncs{0}, sync_ns{0}, sync_zero{0};
  static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  std::atomic<Node*> head;
  std::atomic<Node*> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;
  std::mutex mtx;
  std::condition_variable cv;
  std::exception_ptr first_exception;

  void worker();
};

#endif
