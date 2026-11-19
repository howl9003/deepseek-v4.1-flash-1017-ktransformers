/**
 * Path B stagger r2 (concept-r1/STAGGER_DESIGN_r1.md section 9.2, STAGGER_PREREQ_r1.md section 5.1): KT_MEMOP_SYNC_PER_TASK=1
 * makes a stream-memop sync wait for the task of its own submit (the submit armed last before it on the same stream),
 * not for every pending task. Two decode lanes on two CUDA streams then stop waiting for each other's CPU task: a
 * two-stream surrogate measured near-serial pairs under the all-tasks rule (TaskQueue::sync) and pairs at the CPU floor
 * under this one.
 *
 * Built on KT_MEMOP_PAIRED's rule (memop_detect.h, class Paired; SEAM_FUSION_r1.md section 11), which it keeps verbatim:
 * the same pairing at arm time, the same per-slot service counts (a ring slot's count restarts at its arm), the same
 * deferral (a raised sync whose own submit has not been served since the sync's previous service is left raised for a
 * later pass, never waited on in place: the poller is the only thread that serves submits). Only what follows a passed
 * deferral test differs: KT_MEMOP_PAIRED calls TaskQueue::sync(0), which blocks the poller until NO task is pending;
 * this rule records the sync with the task count its submit left (the ticket) and raises the done flag once that many
 * tasks have finished, without blocking. One worker runs the tasks first in first out, so finished() >= ticket means
 * this submit's task and every task enqueued before it have ended. A later enqueue by any thread only raises a later
 * ticket, never this one. So KT_MEMOP_SYNC_PER_TASK=1 closes the poller race of SEAM_FUSION_r1.md section 9 by the same
 * test KT_MEMOP_PAIRED uses, and implies it: cpuinfer.h makes a Paired whenever either switch is set.
 *
 * Kept on today's path (TaskQueue::sync, blocking): an unpaired sync, and a sync with allow > 0 (his deferred experts).
 * v3 issues neither (v3_his.cpu_sync: sync_with_cuda_stream(stream, 0) right after its submit on the same stream).
 *
 * No CUDA in this file: CPUInfer::memop_poller_per_task (cpuinfer.h) drives scan_per_task with the memop flag page, and
 * the host-only test (tests/test_pertask.cpp) drives the same scan_per_task with flags of its own.
 **/
#ifndef CPUINFER_MEMOP_PER_TASK_H
#define CPUINFER_MEMOP_PER_TASK_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "memop_detect.h"

namespace kt_memop {

inline bool per_task_from_env() {
  const char* e = std::getenv("KT_MEMOP_SYNC_PER_TASK");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

// A sync takes the per-task wait when it is paired and allows nothing pending; otherwise today's TaskQueue::sync.
inline bool per_task_takes(int pair, size_t allow) { return pair >= 0 && allow == 0; }

// Poller side, one thread. Q is TaskQueue: enqueued() / finished() count tasks (one worker, first in first out),
// rethrow_if_failed(), timing_on(), note_sync().
template <class Q>
class Waits {
 public:
  Waits(Q* q, int slots) : q_(q), ticket_(slots, 0) {}

  // A submit entry was served: its host function has enqueued the task, so the count now covers it.
  void submitted(int idx) { ticket_[idx] = q_->enqueued(); }

  // A paired sync entry was served after its own submit (Paired::defer said no): wait for that submit's ticket.
  void wait(int idx, int pair, uint64_t now_ns) {
    waits_.push_back(W{idx, ticket_[pair], now_ns});
    n_waits_++;
  }

  size_t waiting() const { return waits_.size(); }
  uint64_t waits() const { return n_waits_; }
  uint64_t early() const { return n_early_; }   // released while a later task was still pending (decoupled)

  // Sets the done flag of every waiting sync whose ticket has finished.
  template <class Done>
  void raise(Done&& done, uint64_t now_ns) {
    if (waits_.empty()) return;
    const uint64_t fin = q_->finished();
    for (size_t i = 0; i < waits_.size();) {
      const W w = waits_[i];
      if (fin < w.ticket) {
        i++;
        continue;
      }
      waits_[i] = waits_.back();
      waits_.pop_back();
      q_->rethrow_if_failed();   // as TaskQueue::sync: a task's exception surfaces in the poller
      if (q_->enqueued() > fin) n_early_++;
      if (q_->timing_on()) q_->note_sync(now_ns - w.t0);
      done(w.idx);
    }
  }

 private:
  struct W {
    int idx;
    uint64_t ticket, t0;
  };
  Q* q_;
  std::vector<uint64_t> ticket_;   // by submit slot: enqueued() right after its last service
  std::vector<W> waits_;
  uint64_t n_waits_ = 0, n_early_ = 0;
};

// One scan of the armed entries in arming order: scan_paired (memop_detect.h) with the per-task wait in place of
// TaskQueue::sync for a paired sync with allow 0. E has func, args, allow, idx, sync, permanent, pair. req(i) / done(i) are
// the flag words of entry i; full_sync(allow) is today's rule (TaskQueue::sync). after_check(e, raised) runs right after
// a slot's flag is read (the server passes a no-op; the host-only test uses it to hold the poller and force the race).
template <class E, class Q, class Req, class Done, class FullSync, class AfterCheck>
void scan_per_task(const std::vector<E>& snap, kt_memop_detect::Paired& paired, Waits<Q>& w, Req&& req, Done&& done,
                   FullSync&& full_sync, std::vector<int>& retire, uint64_t now_ns, AfterCheck&& after_check) {
  for (const E& e : snap) {
    const bool raised = *req(e.idx) == 1;
    after_check(e, raised);
    if (!raised) continue;
    if (e.sync && paired.defer(e.idx, e.pair, e.allow)) continue;   // its submit first, in a later pass
    *req(e.idx) = 0;
    if (!e.sync) {
      e.func(e.args);
      w.submitted(e.idx);
    } else if (per_task_takes(e.pair, e.allow)) {
      w.wait(e.idx, e.pair, now_ns);   // its own submit was served: wait for that task only, without blocking
    } else {
      full_sync(e.allow);
      std::atomic_thread_fence(std::memory_order_release);
      *done(e.idx) = 1;
    }
    paired.served(e.idx);
    if (!e.permanent) retire.push_back(e.idx);
  }
  w.raise(
      [&](int idx) {
        std::atomic_thread_fence(std::memory_order_release);
        *done(idx) = 1;
      },
      now_ns);
}

}  // namespace kt_memop

#endif
