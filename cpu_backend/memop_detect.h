/**
 * Path B (concept-r1/SEAM_FUSION_r1.md section 9): KT_MEMOP_DETECT=1 records every time the stream-memop poller serves a
 * sync out of order. The default poller walks the armed flag slots in arming order and serves each raised one; a sync is
 * served by TaskQueue::sync, which returns at once when nothing is pending. If the poller passes a submit's slot while
 * its flag is still 0 and is then held long enough for the GPU to raise that submit and its sync, it reaches the sync
 * first: the GPU's wait passes before the task ran, and the GPU copies the slot's previous CPU partial.
 *
 * Recorded per served sync (mask bits of an Event):
 *   1  the sync's own submit (the submit armed last before it on the same stream) has not been served since the sync's
 *      previous service;
 *   2  TaskQueue had nothing pending and some submit's flag was raised at that moment.
 * With KT_MEMOP_PAIRED=1 also, per sync left for a later pass (not served): mask 4 (Detector::sync_deferred).
 * Plus, for the first sync of every captured graph (a step marker), the graph's replay count and the time, so events and
 * client batches can be lined up by step. Nothing here touches the GPU, the flags' values or the task order: it counts.
 * The poller's cost per sync is two relaxed loads, one store, and a flag scan only when nothing is pending.
 *
 * Output: <KT_MEMOP_DETECT_DIR, default /tmp>/kt-memop-detect-<pid>.jsonl, JSON lines written by a flusher thread once a
 * second and at shutdown (never by the poller): "start", "arm" (graph slots: idx, pair, graph, sync), "event", "step",
 * "count". No CUDA in this file: CPUInfer::memop_poller_detect (cpuinfer.h) drives scan_detect with the memop flag page,
 * and the host-only test (tests/test_memop_detect.cpp) drives the same scan_detect with flags of its own.
 *
 * KT_MEMOP_PAIRED=1 (SEAM_FUSION_r1.md section 11), the fix, default off: class Paired pairs each sync with the submit
 * armed last on its stream (the detector's rule) and counts each slot's services. When a pass meets a raised sync whose
 * submit has not been served since the sync's previous service (served(submit) <= served(sync), the detector's mask-1
 * test), it leaves the sync's flag raised, counts a deferral and goes on with the pass; the next pass serves the submit
 * (armed earlier, so met first) and then the sync. The poller never waits for the submit: it is the only thread that
 * serves submits. An unpaired sync, or one with allow > 0, keeps today's path. A ring slot's count restarts at its arm.
 * Used by scan_paired (CPUInfer::memop_poller_paired, no detector) and by scan_detect (with the detector, which then
 * records each deferral as an event with mask 4). Unset: no Paired exists, and every loop runs as before.
 **/
#ifndef CPUINFER_MEMOP_DETECT_H
#define CPUINFER_MEMOP_DETECT_H

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kt_memop_detect {

inline bool on_from_env() {
  const char* e = std::getenv("KT_MEMOP_DETECT");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

inline bool paired_from_env() {   // KT_MEMOP_PAIRED=1: the fix (class Paired)
  const char* e = std::getenv("KT_MEMOP_PAIRED");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

// KT_MEMOP_PAIRED=1: a sync is served only after its own submit. Arm side under memop_arm's mutex, in stream order;
// poller side on the poller thread only.
class Paired {
 public:
  explicit Paired(int slots) : served_(slots) {
    for (auto& s : served_) s.store(0, std::memory_order_relaxed);
  }

  // Arm side, before the entry is published: returns a sync's submit slot (the submit armed last before it on the same
  // stream), else -1. A ring slot (perm false) is reused: its count restarts at zero, as the detector's does.
  int arm(const void* stream, int idx, bool sync, bool perm) {
    int pair = -1;
    if (!sync) {
      last_[stream] = idx;
    } else {
      auto it = last_.find(stream);
      if (it != last_.end()) {
        pair = it->second;
        last_.erase(it);
      }
    }
    if (!perm) served_[idx].store(0, std::memory_order_relaxed);
    return pair;
  }

  // Poller side. True: leave this raised sync for a later pass, because its own submit has not been served since the
  // sync's previous service. False for an unpaired sync or allow > 0 (today's path).
  bool defer(int idx, int pair, size_t allow) {
    if (pair < 0 || allow != 0) return false;
    if (served_[pair].load(std::memory_order_relaxed) > served_[idx].load(std::memory_order_relaxed)) return false;
    n_deferred_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  void served(int idx) { served_[idx].fetch_add(1, std::memory_order_relaxed); }   // a submit or sync was served
  uint64_t deferred() const { return n_deferred_.load(std::memory_order_relaxed); }

 private:
  std::vector<std::atomic<uint64_t>> served_;   // by flag slot: services so far (a ring slot: since its last arm)
  std::unordered_map<const void*, int> last_;   // arm side only
  std::atomic<uint64_t> n_deferred_{0};
};

inline uint64_t clock_ns(clockid_t c) {
  timespec ts;
  clock_gettime(c, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

struct Event {
  uint32_t mask;
  int32_t idx, pair, raised;            // the sync's slot, its submit's slot (-1: none), a raised submit's slot (-1: none)
  uint64_t served_sync, served_pair;    // services of the two slots before this one (graph slots: replays of the graph)
  uint64_t pending, n_sync, t_mono, t_real;
};

struct Step {
  int32_t graph;
  uint64_t n, t_real;                   // the marker's service count before this one, CLOCK_REALTIME ns
};

struct Arm {
  int32_t idx, pair, graph, sync;
  uint64_t seq, capture_id, t_real;
};

class Detector {
 public:
  explicit Detector(int slots, size_t ev_cap = 4096, size_t step_cap = (size_t)1 << 17)
      : served_(slots), ev_(ev_cap), step_(step_cap) {
    for (auto& s : served_) s.store(0, std::memory_order_relaxed);
  }

  // Arm side, in stream order (memop_arm calls it under its mutex, before the entry is published). Sets the sync's paired
  // submit slot and, for the first sync armed in a captured graph, that graph's ordinal (the step marker).
  void arm(const void* stream, int idx, bool sync, bool perm, uint64_t capture_id, int* pair_out, int* graph_out) {
    std::lock_guard<std::mutex> g(mu_);
    int pair = -1, graph = -1;
    if (!sync) {
      last_[stream] = idx;
    } else {
      auto it = last_.find(stream);
      if (it != last_.end()) {
        pair = it->second;
        last_.erase(it);
      }
    }
    seq_++;
    if (!perm) {
      served_[idx].store(0, std::memory_order_relaxed);   // a ring slot is reused: count its new use from zero
      n_ring_arms_++;
    } else {
      int gid = -1;
      if (capture_id != 0) {
        auto gi = graphs_.find(capture_id);
        if (gi == graphs_.end()) gi = graphs_.emplace(capture_id, (int)graphs_.size()).first;
        gid = gi->second;
        if (sync && marked_.insert(capture_id).second) graph = gid;
      }
      arms_.push_back(Arm{idx, pair, gid, sync ? 1 : 0, seq_, capture_id, clock_ns(CLOCK_REALTIME)});
    }
    *pair_out = pair;
    *graph_out = graph;
  }

  // Poller side (one thread).
  void submit_served(int idx) { served_[idx].fetch_add(1, std::memory_order_relaxed); }

  // Before the sync of slot idx is served. raised() is called only when nothing is pending; it returns a slot whose
  // submit flag is raised, or -1.
  template <class Raised>
  void sync_serving(int idx, int pair, int graph, uint64_t pending, Raised&& raised) {
    const uint64_t ss = served_[idx].load(std::memory_order_relaxed);
    const uint64_t sp = pair >= 0 ? served_[pair].load(std::memory_order_relaxed) : 0;
    uint32_t mask = 0;
    int r = -1;
    if (pair >= 0 && sp <= ss) mask |= 1;
    if (pending == 0) {
      n_empty_.fetch_add(1, std::memory_order_relaxed);
      r = raised();
      if (r >= 0) mask |= 2;
    }
    const uint64_t ns = n_sync_.fetch_add(1, std::memory_order_relaxed) + 1;
    served_[idx].store(ss + 1, std::memory_order_relaxed);
    if (graph >= 0) {
      const uint64_t w = step_w_.load(std::memory_order_relaxed);
      step_[w % step_.size()] = Step{graph, ss, clock_ns(CLOCK_REALTIME)};
      step_w_.store(w + 1, std::memory_order_release);
    }
    if (mask) {
      if (mask & 1) n_kind1_.fetch_add(1, std::memory_order_relaxed);
      if (mask & 2) n_kind2_.fetch_add(1, std::memory_order_relaxed);
      const uint64_t w = ev_w_.load(std::memory_order_relaxed);
      ev_[w % ev_.size()] = Event{mask, idx, pair, r, ss, sp, pending, ns, clock_ns(CLOCK_MONOTONIC),
                                  clock_ns(CLOCK_REALTIME)};
      ev_w_.store(w + 1, std::memory_order_release);
    }
  }

  // KT_MEMOP_PAIRED=1: a raised sync of slot idx was left for a later pass because its submit (slot pair) has not been
  // served since the sync's previous service. Recorded as an event with mask 4: raised = pair when the submit's flag is
  // raised at this moment, else -1; served_sync = the sync's services so far (a graph slot: its graph's replays, the
  // step); n_sync = syncs served so far. The sync's own count does not move (it was not served).
  void sync_deferred(int idx, int pair, int raised_pair, uint64_t pending) {
    const uint64_t ss = served_[idx].load(std::memory_order_relaxed);
    const uint64_t sp = pair >= 0 ? served_[pair].load(std::memory_order_relaxed) : 0;
    n_kind4_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t w = ev_w_.load(std::memory_order_relaxed);
    ev_[w % ev_.size()] = Event{4u, idx, pair, raised_pair, ss, sp, pending, n_sync_.load(std::memory_order_relaxed),
                                clock_ns(CLOCK_MONOTONIC), clock_ns(CLOCK_REALTIME)};
    ev_w_.store(w + 1, std::memory_order_release);
  }

  uint64_t events() const { return ev_w_.load(std::memory_order_acquire); }
  uint64_t kind1() const { return n_kind1_.load(std::memory_order_relaxed); }
  uint64_t kind2() const { return n_kind2_.load(std::memory_order_relaxed); }
  uint64_t kind4() const { return n_kind4_.load(std::memory_order_relaxed); }
  uint64_t syncs() const { return n_sync_.load(std::memory_order_relaxed); }

  // Flusher side (one thread): everything new since the last call, as JSON lines; a count line when anything moved or
  // when final. Returns the number of lines written.
  int flush(int fd, bool final) {
    if (fd < 0) return 0;
    int lines = 0;
    char b[512];
    {
      std::lock_guard<std::mutex> g(mu_);
      for (; arms_done_ < arms_.size(); arms_done_++) {
        const Arm& a = arms_[arms_done_];
        lines += put(fd, b, snprintf(b, sizeof b,
                                     "{\"k\":\"arm\",\"idx\":%d,\"pair\":%d,\"graph\":%d,\"sync\":%d,\"seq\":%llu,"
                                     "\"capture\":%llu,\"t_real\":%llu}\n",
                                     a.idx, a.pair, a.graph, a.sync, (unsigned long long)a.seq,
                                     (unsigned long long)a.capture_id, (unsigned long long)a.t_real));
      }
    }
    const uint64_t ew = ev_w_.load(std::memory_order_acquire);
    if (ew - ev_r_ > ev_.size()) {
      ev_dropped_ += ew - ev_r_ - ev_.size();
      ev_r_ = ew - ev_.size();
    }
    for (; ev_r_ < ew; ev_r_++) {
      const Event& e = ev_[ev_r_ % ev_.size()];
      lines += put(fd, b, snprintf(b, sizeof b,
                                   "{\"k\":\"event\",\"mask\":%u,\"idx\":%d,\"pair\":%d,\"raised\":%d,\"served_sync\":%llu,"
                                   "\"served_pair\":%llu,\"pending\":%llu,\"n_sync\":%llu,\"t_mono\":%llu,\"t_real\":%llu}\n",
                                   e.mask, e.idx, e.pair, e.raised, (unsigned long long)e.served_sync,
                                   (unsigned long long)e.served_pair, (unsigned long long)e.pending,
                                   (unsigned long long)e.n_sync, (unsigned long long)e.t_mono,
                                   (unsigned long long)e.t_real));
    }
    const uint64_t sw = step_w_.load(std::memory_order_acquire);
    if (sw - step_r_ > step_.size()) {
      step_dropped_ += sw - step_r_ - step_.size();
      step_r_ = sw - step_.size();
    }
    for (; step_r_ < sw; step_r_++) {
      const Step& s = step_[step_r_ % step_.size()];
      lines += put(fd, b, snprintf(b, sizeof b, "{\"k\":\"step\",\"g\":%d,\"n\":%llu,\"t\":%llu}\n", s.graph,
                                   (unsigned long long)s.n, (unsigned long long)s.t_real));
    }
    const uint64_t ns = syncs();
    if (final || ns != last_count_syncs_) {
      last_count_syncs_ = ns;
      lines += put(fd, b, snprintf(b, sizeof b,
                                   "{\"k\":\"count\",\"final\":%d,\"t_real\":%llu,\"syncs\":%llu,\"empty\":%llu,"
                                   "\"events\":%llu,\"kind1\":%llu,\"kind2\":%llu,\"kind4\":%llu,\"ev_dropped\":%llu,"
                                   "\"step_dropped\":%llu,\"ring_arms\":%llu,\"graphs\":%zu}\n",
                                   final ? 1 : 0, (unsigned long long)clock_ns(CLOCK_REALTIME), (unsigned long long)ns,
                                   (unsigned long long)n_empty_.load(std::memory_order_relaxed),
                                   (unsigned long long)ew, (unsigned long long)kind1(), (unsigned long long)kind2(),
                                   (unsigned long long)kind4(), (unsigned long long)ev_dropped_,
                                   (unsigned long long)step_dropped_,
                                   (unsigned long long)n_ring_arms_, graphs_size()));
    }
    return lines;
  }

 private:
  static int put(int fd, const char* b, int n) {
    if (n <= 0) return 0;
    const ssize_t w = ::write(fd, b, (size_t)n < 512 ? (size_t)n : 511);
    return w > 0 ? 1 : 0;
  }
  size_t graphs_size() {
    std::lock_guard<std::mutex> g(mu_);
    return graphs_.size();
  }

  std::vector<std::atomic<uint64_t>> served_;   // by flag slot: services so far (a ring slot: since its last arm)
  std::vector<Event> ev_;
  std::vector<Step> step_;
  std::atomic<uint64_t> ev_w_{0}, step_w_{0}, n_sync_{0}, n_empty_{0}, n_kind1_{0}, n_kind2_{0}, n_kind4_{0};
  // flusher only
  uint64_t ev_r_ = 0, step_r_ = 0, ev_dropped_ = 0, step_dropped_ = 0, last_count_syncs_ = ~0ull;
  size_t arms_done_ = 0;
  // arm side, under mu_
  std::mutex mu_;
  std::unordered_map<const void*, int> last_;
  std::unordered_map<uint64_t, int> graphs_;
  std::unordered_set<uint64_t> marked_;
  std::vector<Arm> arms_;
  uint64_t seq_ = 0, n_ring_arms_ = 0;
};

// <KT_MEMOP_DETECT_DIR or /tmp>/kt-memop-detect-<pid>.jsonl, append; -1 when it cannot be opened (then nothing is written).
// paired: KT_MEMOP_PAIRED=1 is on (recorded in the start line).
inline int open_log(int ring, int perm, bool paired) {
  const char* d = std::getenv("KT_MEMOP_DETECT_DIR");
  std::string dir = (d != nullptr && d[0] != '\0') ? std::string(d) : std::string("/tmp");
  if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) return -1;
  const std::string path = dir + "/kt-memop-detect-" + std::to_string((long)::getpid()) + ".jsonl";
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  if (fd >= 0) {
    char b[256];
    const int n = snprintf(b, sizeof b, "{\"k\":\"start\",\"pid\":%ld,\"ring\":%d,\"perm\":%d,\"paired\":%d,\"t_real\":%llu}\n",
                           (long)::getpid(), ring, perm, paired ? 1 : 0, (unsigned long long)clock_ns(CLOCK_REALTIME));
    if (n > 0) (void)!::write(fd, b, (size_t)n);
  }
  printf("[kt-memops] KT_MEMOP_DETECT=1: out-of-order syncs are counted into %s\n", fd >= 0 ? path.c_str() : "(no file)");
  return fd;
}

// One scan of the armed entries in arming order: memop_poller's loop body, plus the detector. E has func, args, allow,
// idx, sync, permanent, pair, graph. after_check(e, raised) runs right after a slot's flag is read (the server passes a
// no-op; the host-only test uses it to hold the poller and force the race). paired: KT_MEMOP_PAIRED=1's state, or
// nullptr (then the scan is the detector's alone, as before).
template <class E, class Req, class Done, class FullSync, class Pending, class AfterCheck>
void scan_detect(const std::vector<E>& snap, Detector& det, Paired* paired, Req&& req, Done&& done, FullSync&& full_sync,
                 Pending&& pending, std::vector<int>& retire, AfterCheck&& after_check) {
  for (const E& e : snap) {
    const bool raised = *req(e.idx) == 1;
    after_check(e, raised);
    if (!raised) continue;
    if (paired != nullptr && e.sync && paired->defer(e.idx, e.pair, e.allow)) {   // its submit first, in a later pass
      det.sync_deferred(e.idx, e.pair, *req(e.pair) == 1 ? e.pair : -1, (uint64_t)pending());
      continue;                                                                    // flag left raised, not retired
    }
    *req(e.idx) = 0;
    if (!e.sync) {
      e.func(e.args);
      det.submit_served(e.idx);
    } else {
      det.sync_serving(e.idx, e.pair, e.graph, (uint64_t)pending(), [&]() {
        for (const E& o : snap)
          if (!o.sync && *req(o.idx) == 1) return o.idx;
        return -1;
      });
      full_sync(e.allow);
      std::atomic_thread_fence(std::memory_order_release);
      *done(e.idx) = 1;
    }
    if (paired != nullptr) paired->served(e.idx);
    if (!e.permanent) retire.push_back(e.idx);
  }
}

// KT_MEMOP_PAIRED=1 without the detector: memop_poller's loop body (b511cc3 cpuinfer.h:236-246) with the pairing rule.
// The same entries in the same order with the same calls, except that a raised sync whose submit has not been served
// since the sync's previous service is left raised for the next pass. after_check as in scan_detect.
template <class E, class Req, class Done, class FullSync, class AfterCheck>
void scan_paired(const std::vector<E>& snap, Paired& paired, Req&& req, Done&& done, FullSync&& full_sync,
                 std::vector<int>& retire, AfterCheck&& after_check) {
  for (const E& e : snap) {
    const bool raised = *req(e.idx) == 1;
    after_check(e, raised);
    if (!raised) continue;
    if (e.sync && paired.defer(e.idx, e.pair, e.allow)) continue;   // its submit first, in a later pass
    *req(e.idx) = 0;
    if (!e.sync) {
      e.func(e.args);
    } else {
      full_sync(e.allow);
      std::atomic_thread_fence(std::memory_order_release);
      *done(e.idx) = 1;
    }
    paired.served(e.idx);
    if (!e.permanent) retire.push_back(e.idx);
  }
}

}  // namespace kt_memop_detect

#endif
