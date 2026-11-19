/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <cstdlib>
#include <cstdio>

// concept-r1 pre-read (PREREAD_r1.md): see task_queue.h. Null until the contract v3 backend arms its pre-read.
std::atomic<kt_tq_idle_fn> kt_tq_idle_hook{nullptr};
std::atomic<uint64_t> kt_tq_last_qwait_ns{0};

TaskQueue::TaskQueue() : done(false), pending(0) {
  if (const char* e = getenv("KT_TASKQUEUE_TIMING")) timing_every = atoi(e);
  if (const char* e = getenv("KT_TASKQUEUE_SPIN_US")) spin_us = atoi(e);
  if (const char* e = getenv("KT_MEMOP_SYNC_PER_TASK")) count_tasks = e[0] == '1' && e[1] == '\0';
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  if (count_tasks) n_enq.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  if (timing_every) node->t_enq = now_ns();
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  std::exception_ptr task_exception;
  const uint64_t t0 = timing_every ? now_ns() : 0;
  if (timing_every && pending.load(std::memory_order_acquire) <= allow_n_pending)
    sync_zero.fetch_add(1, std::memory_order_relaxed);
  if (spin_us > 0) {
    const uint64_t until = now_ns() + uint64_t(spin_us) * 1000;
    while (pending.load(std::memory_order_acquire) > allow_n_pending && now_ns() < until) {
    }
  }
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (timing_every) {
    sync_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
    const uint64_t n = n_syncs.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % uint64_t(timing_every) == 0) {
      const uint64_t nt = n_tasks.exchange(0), qn = q_ns.exchange(0), rn = run_ns.exchange(0);
      const uint64_t sn = sync_ns.exchange(0), sz = sync_zero.exchange(0);
      fprintf(stderr,
              "[kt-timing] %d syncs: tasks=%llu queue_wait=%.0f us/task run=%.0f us/task "
              "(run total %.1f ms) | sync_wait=%.0f us/sync (total %.1f ms), %llu syncs found nothing pending\n",
              timing_every, (unsigned long long)nt, nt ? qn / 1e3 / nt : 0.0, nt ? rn / 1e3 / nt : 0.0,
              rn / 1e6, sn / 1e3 / timing_every, sn / 1e6, (unsigned long long)sz);
      fflush(stderr);
    }
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

// concept-r1 CPU smoothing (12 Nov): KT_TASKQUEUE_CPU=<cpu> pins the worker thread (the thread every CPU expert call
// runs on, and worker 0 of the whole-pool jobs) to that CPU. Unset: unpinned, as before.
static void kt_pin_from_env(const char* name, const char* what) {
  const char* e = getenv(name);
  if (!e || !*e) return;
  const int cpu = atoi(e);
  if (cpu < 0 || cpu >= CPU_SETSIZE) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  printf("[kt-pin] %s pinned to cpu %d (%s=%s, rc %d)\n", what, cpu, name, e, rc);
  fflush(stdout);
}

void TaskQueue::worker() {
  kt_pin_from_env("KT_TASKQUEUE_CPU", "task-queue worker");
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        const uint64_t ts = timing_every ? now_ns() : 0;
        if (timing_every) kt_tq_last_qwait_ns.store(ts - next->t_enq, std::memory_order_relaxed);   // pre-read tests
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
        if (timing_every) {
          const uint64_t te = now_ns();
          n_tasks.fetch_add(1, std::memory_order_relaxed);
          q_ns.fetch_add(ts - next->t_enq, std::memory_order_relaxed);
          run_ns.fetch_add(te - ts, std::memory_order_relaxed);
        }
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
        }
        if (count_tasks) n_fin.fetch_add(1, std::memory_order_acq_rel);   // after the exception is stored
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();
      // concept-r1 pre-read (PREREAD_r1.md): the task's completion is published; with nothing queued, the idle wait
      // goes to the hook, which returns as soon as `pending` is non-zero. Unset (the default): nothing runs here.
      if (const kt_tq_idle_fn hook = kt_tq_idle_hook.load(std::memory_order_acquire)) {
        if (curr->next.load(std::memory_order_acquire) == nullptr) {
          try {
            hook(&pending);
          } catch (...) {
          }
        }
      }
    } else {
      if (spin_us > 0) {
        const uint64_t until = now_ns() + uint64_t(spin_us) * 1000;
        while (curr->next.load(std::memory_order_acquire) == nullptr &&
               !done.load(std::memory_order_acquire) && now_ns() < until) {
        }
        if (curr->next.load(std::memory_order_acquire) != nullptr) continue;
      }
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}

// Path B stagger r2 (KT_MEMOP_SYNC_PER_TASK=1, memop_per_task.h): the two pieces of sync() a per-task wait still needs.
void TaskQueue::rethrow_if_failed() {
  std::exception_ptr task_exception;
  {
    std::lock_guard<std::mutex> lock(mtx);
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::note_sync(uint64_t waited_ns) {
  if (!timing_every) return;
  sync_ns.fetch_add(waited_ns, std::memory_order_relaxed);
  const uint64_t n = n_syncs.fetch_add(1, std::memory_order_relaxed) + 1;
  if (n % uint64_t(timing_every) == 0) {
    const uint64_t nt = n_tasks.exchange(0), qn = q_ns.exchange(0), rn = run_ns.exchange(0);
    const uint64_t sn = sync_ns.exchange(0);
    fprintf(stderr,
            "[kt-timing] %d syncs (per task): tasks=%llu queue_wait=%.0f us/task run=%.0f us/task "
            "(run total %.1f ms) | sync_wait=%.0f us/sync (total %.1f ms)\n",
            timing_every, (unsigned long long)nt, nt ? qn / 1e3 / nt : 0.0, nt ? rn / 1e3 / nt : 0.0,
            rn / 1e6, sn / 1e3 / timing_every, sn / 1e6);
    fflush(stderr);
  }
}
