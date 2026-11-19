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

TaskQueue::TaskQueue() : done(false), pending(0) {
  if (const char* e = getenv("KT_TASKQUEUE_TIMING")) timing_every = atoi(e);
  if (const char* e = getenv("KT_TASKQUEUE_SPIN_US")) spin_us = atoi(e);
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

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        const uint64_t ts = timing_every ? now_ns() : 0;
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
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      cv.notify_all();
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
