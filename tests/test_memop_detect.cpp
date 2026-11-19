// Path B poller detector and pairing fix (memop_detect.h), host-only test: no GPU, no CUDA. A "GPU" thread plays the decode
// stream's part of the stream-memop hand-off (raise a layer's submit flag, run its own work, raise the sync flag, wait for
// done, read the CPU partial of the layer's slot), the real TaskQueue (task_queue.cpp) runs the CPU tasks, and a poller
// thread runs memop_detect.h's scan_detect / scan_paired, the code CPUInfer::memop_poller_detect / memop_poller_paired
// run, over flags of its own, with the arming, snapshot and retire steps of CPUInfer (memop_arm, the pollers).
//
// Graph-slot runs (40 layers x STEPS steps, slots 2l / 2l + 1 armed once and permanent, as a captured graph):
//   plain          a copy of CPUInfer::memop_poller's loop body (b511cc3 cpuinfer.h:236-246), no detector: the control
//   detect         scan_detect, no forced delay
//   forced         scan_detect; for every FORCE_EVERY-th hand-off the poller is held right after it read that layer's
//                  submit flag as 0, until the GPU thread has raised the submit AND the sync: the race of
//                  SEAM_FUSION_r1.md section 9
//   fix            scan_paired (KT_MEMOP_PAIRED=1 without the detector), no forced delay
//   fix_forced     scan_paired, held as forced
//   fixdet         scan_detect with the pairing fix (KT_MEMOP_PAIRED=1 and KT_MEMOP_DETECT=1), no forced delay
//   fixdet_forced  scan_detect with the fix, held as forced
// Ring-slot runs (40 layers x RING_STEPS steps; every hand-off arms its submit and sync on the next two of RING ring
// slots, not permanent, retired after service, as eager calls are; RING is odd, so a slot serves as a submit and as a
// sync in turn and its count must restart at every arm):
//   ring_plain, ring_forced, ring_fix, ring_fix_forced, ring_fixdet_forced (the loops above)
// Expect:
//   - every unforced run and every fix run reads every partial as written (the same checksum, no stale read);
//   - detect counts 0 events; forced and ring_forced read a stale partial exactly at the held hand-offs, and their kind-1
//     events name exactly those (graph: the event's served_sync = step, its slot / 2 = layer; ring: n_sync - 1 = hand-off);
//   - with the fix, a held hand-off defers its sync once: the deferral count equals the held count, the detector (when on)
//     counts 0 kind-1 and 0 kind-2 events and one kind-4 event per held hand-off, naming exactly those hand-offs (graph:
//     served_sync = step and slot / 2 = layer; both: n_sync = hand-off index s * 40 + l);
//   - no ring slot is re-armed while still armed (collisions 0) and no run times out.
// Unforced fix runs report their deferrals (0 expected; a non-zero count is a natural race on this host that the fix
// caught, reported, not a failure). A stale read in an unforced run fails the test: a natural race on this host.
// Hand-off time: mean and max microseconds from the GPU raising the sync flag to it seeing done, per run.
// Exit 0 when all of that holds; prints one JSON line either way.
//
//   g++ -O2 -std=c++17 -pthread -Icpu_backend tests/test_memop_detect.cpp cpu_backend/task_queue.cpp -o test_memop_detect
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "memop_detect.h"
#include "task_queue.h"

namespace {

constexpr int LAYERS = 40;
constexpr int STEPS = 300;        // graph-slot runs
constexpr int RING_STEPS = 100;   // ring-slot runs
constexpr int RING = 15;          // ring slots in the ring runs (odd: a slot alternates between submit and sync)
constexpr int FORCE_EVERY = 53;   // prime, so the held layer moves through all 40
constexpr int GPU_US = 100;       // the GPU's own work between the submit and the sync flag (arm P: about 110 us)
constexpr int CPU_US = 250;       // the CPU task (arm P: about 420 us with the queue wait)
constexpr int ATTN_US = 60;       // the GPU's work between a sync and the next layer's submit
constexpr int NSLOT = 2 * LAYERS > RING ? 2 * LAYERS : RING;
constexpr double TIMEOUT_S = 5.0; // a hand-off that waits longer than this ends the run as a failure (no hang)

struct E {                        // CPUInfer::MemopEntry as cpuinfer.h has it
  void (*func)(void*);
  void* args;
  size_t allow;
  int idx;
  bool sync;
  bool permanent;
  int pair;
  int graph;
};

using Clock = std::chrono::steady_clock;

void spin_us(int us) {
  const auto t = Clock::now() + std::chrono::microseconds(us);
  while (Clock::now() < t) {
  }
}

double since_us(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

uint64_t value(int step, int layer) { return 0x9E3779B97F4A7C15ull * (uint64_t)(step * LAYERS + layer + 1); }

uint64_t expect_checksum(int steps) {
  uint64_t x = 0;
  for (int s = 0; s < steps; s++)
    for (int l = 0; l < LAYERS; l++) x = x * 1099511628211ull ^ value(s, l);
  return x;
}

struct World;
struct Args {
  World* w;
  int layer;
};

struct World {
  TaskQueue q;
  volatile uint32_t req[NSLOT] = {};
  volatile uint32_t done[NSLOT] = {};
  volatile uint64_t out[2] = {};
  std::atomic<int> cur_step{0};
  std::atomic<int> hold_idx{-1};       // forced: the submit slot the poller must be held after reading it as 0
  std::atomic<int> holding{0};         // 1: the poller is held; 2: released
  Args args[LAYERS];
  std::mutex mu;                       // as CPUInfer::mmtx_: the armed entries and their version
  std::vector<E> entries;
  std::atomic<uint64_t> version{0};
};

void submit_func(void* p) {            // as the binding's lambda: enqueue the layer's task at once
  Args* a = (Args*)p;
  World* w = a->w;
  const int layer = a->layer, step = w->cur_step.load(std::memory_order_acquire);
  w->q.enqueue([w, layer, step]() {
    spin_us(CPU_US);
    w->out[layer % 2] = value(step, layer);
  });
}

// CPUInfer::memop_poller's loop body (b511cc3 cpuinfer.h:236-246)
void scan_plain(World& w, const std::vector<E>& snap, std::vector<int>& retire) {
  for (const E& e : snap) {
    if (w.req[e.idx] != 1) continue;
    w.req[e.idx] = 0;
    if (!e.sync) {
      e.func(e.args);
    } else {
      w.q.sync(e.allow);
      std::atomic_thread_fence(std::memory_order_release);
      w.done[e.idx] = 1;
    }
    if (!e.permanent) retire.push_back(e.idx);
  }
}

enum Loop { PLAIN, DETECT, PAIRED };

struct Cfg {
  const char* name;
  Loop loop;
  bool fix;      // a Paired exists (KT_MEMOP_PAIRED=1): always with PAIRED, optional with DETECT
  bool forced;
  bool ring;
};

struct Result {
  std::string name;
  bool ring = false, fix = false, forced_run = false;
  int steps = 0;
  uint64_t checksum = 0, stale = 0, forced = 0, kind1 = 0, kind2 = 0, kind4 = 0, events = 0, deferred = 0;
  uint64_t collisions = 0, ring_arms = 0;
  bool pair_ok = true, timeout = false, det = false;
  bool k1_match = true;   // kind-1 events name exactly the stale reads (graph: (step, layer); ring: hand-off)
  bool k4_match = true;   // kind-4 events name exactly the held hand-offs (graph: (step, layer) and hand-off; ring: hand-off)
  double seconds = 0, wait_us = 0, wait_us_max = 0;
};

long long field(const char* line, const char* key) {   // the integer after "key": in a JSON line, or -1
  char k[64];
  snprintf(k, sizeof k, "\"%s\":", key);
  const char* p = strstr(line, k);
  return p == nullptr ? -1 : strtoll(p + strlen(k), nullptr, 10);
}

Result run(const Cfg& c) {
  setenv("KT_TASKQUEUE_SPIN_US", "300", 1);   // as every launch
  World w;
  Result r;
  r.name = c.name;
  r.ring = c.ring;
  r.fix = c.fix;
  r.forced_run = c.forced;
  r.steps = c.ring ? RING_STEPS : STEPS;
  kt_memop_detect::Detector det(NSLOT);
  kt_memop_detect::Paired paired(NSLOT);
  kt_memop_detect::Paired* pp = c.fix ? &paired : nullptr;
  // as CPUInfer::memop_arm (the caller holds w.mu): the fix's pairing first, then the detector's; they must agree
  auto arm = [&](int idx, bool sync, bool perm, Args* a) {
    int pair = -1, graph = -1;
    const int p2 = pp != nullptr ? pp->arm(&w, idx, sync, perm) : -1;
    det.arm(&w, idx, sync, perm, perm ? 7 : 0, &pair, &graph);
    if (pp != nullptr && p2 != pair) r.pair_ok = false;
    w.entries.push_back(E{sync ? nullptr : submit_func, sync ? nullptr : (void*)a, 0, idx, sync, perm, pair, graph});
  };
  for (int l = 0; l < LAYERS; l++) w.args[l] = Args{&w, l};
  if (!c.ring) {   // graph slots: armed once, in capture order, permanent
    std::lock_guard<std::mutex> g(w.mu);
    for (int l = 0; l < LAYERS; l++) {
      arm(2 * l, false, true, &w.args[l]);
      arm(2 * l + 1, true, true, nullptr);
    }
    w.version.fetch_add(1, std::memory_order_release);
  }
  std::atomic<bool> stop{false};
  std::thread poller([&]() {
    auto req = [&](int i) { return &w.req[i]; };
    auto done = [&](int i) { return &w.done[i]; };
    auto full_sync = [&](size_t allow) { w.q.sync(allow); };
    auto pending = [&]() { return w.q.pending_now(); };
    auto hold = [&](const E& e, bool raised) {
      if (!c.forced || raised || e.sync || e.idx != w.hold_idx.load(std::memory_order_acquire)) return;
      int z = 0;
      if (!w.holding.compare_exchange_strong(z, 1)) return;
      while (w.holding.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    };
    std::vector<E> snap;
    uint64_t seen = ~0ull;
    while (!stop.load(std::memory_order_relaxed)) {
      const uint64_t v = w.version.load(std::memory_order_acquire);
      if (v != seen) {
        std::lock_guard<std::mutex> g(w.mu);
        snap = w.entries;
        seen = w.version.load(std::memory_order_acquire);
      }
      std::vector<int> retire;
      if (c.loop == PLAIN) {
        scan_plain(w, snap, retire);
      } else if (c.loop == DETECT) {
        kt_memop_detect::scan_detect(snap, det, pp, req, done, full_sync, pending, retire, hold);
      } else {
        kt_memop_detect::scan_paired(snap, paired, req, done, full_sync, retire, hold);
      }
      if (!retire.empty()) {   // as the pollers: drop the served ring entries, then a new version
        std::lock_guard<std::mutex> g(w.mu);
        for (int idx : retire) {
          for (size_t i = 0; i < w.entries.size(); i++) {
            if (!w.entries[i].permanent && w.entries[i].idx == idx) {
              w.entries.erase(w.entries.begin() + i);
              break;
            }
          }
        }
        w.version.fetch_add(1, std::memory_order_release);
      }
    }
  });
  std::set<std::pair<uint64_t, int>> stale_at, forced_at;   // (step, layer)
  std::set<uint64_t> stale_h, forced_h;                      // hand-off index s * LAYERS + l
  int ring_next = 0;
  double wait_sum = 0;
  uint64_t n_wait = 0;
  const auto t0 = Clock::now();
  for (int s = 0; s < r.steps && !r.timeout; s++) {
    for (int l = 0; l < LAYERS && !r.timeout; l++) {
      w.cur_step.store(s, std::memory_order_release);
      const uint64_t h = (uint64_t)s * LAYERS + l;
      int us = 2 * l, ys = 2 * l + 1;
      if (c.ring) {   // an eager call: the layer's submit and sync on the next two ring slots, published together
        us = ring_next % RING;
        ys = (ring_next + 1) % RING;
        ring_next += 2;
        std::lock_guard<std::mutex> g(w.mu);
        for (const E& e : w.entries)
          if (e.idx == us || e.idx == ys) r.collisions++;
        arm(us, false, false, &w.args[l]);
        arm(ys, true, false, nullptr);
        w.version.fetch_add(1, std::memory_order_release);
      }
      const bool force = c.forced && h % FORCE_EVERY == FORCE_EVERY - 1;
      if (force) {   // hold the poller right after it reads this submit's flag as 0, then raise submit and sync
        w.holding.store(0, std::memory_order_release);
        w.hold_idx.store(us, std::memory_order_release);
        const auto th = Clock::now();
        while (w.holding.load(std::memory_order_acquire) != 1) {
          std::this_thread::yield();
          if (since_us(th) > TIMEOUT_S * 1e6) {
            r.timeout = true;
            break;
          }
        }
        if (r.timeout) {
          w.hold_idx.store(-1, std::memory_order_release);
          break;
        }
        r.forced++;
        forced_at.insert({(uint64_t)s, l});
        forced_h.insert(h);
      }
      w.req[us] = 1;
      spin_us(GPU_US);
      w.req[ys] = 1;
      const auto tw = Clock::now();
      if (force) {
        w.hold_idx.store(-1, std::memory_order_release);
        w.holding.store(2, std::memory_order_release);
      }
      for (uint32_t n = 0; w.done[ys] != 1; n++) {
        if ((n & 4095) == 4095 && since_us(tw) > TIMEOUT_S * 1e6) {
          r.timeout = true;
          break;
        }
      }
      if (r.timeout) break;
      std::atomic_thread_fence(std::memory_order_acquire);
      const double wu = since_us(tw);
      wait_sum += wu;
      n_wait++;
      if (wu > r.wait_us_max) r.wait_us_max = wu;
      w.done[ys] = 0;
      const uint64_t v = w.out[l % 2];
      r.checksum = r.checksum * 1099511628211ull ^ v;
      if (v != value(s, l)) {
        r.stale++;
        stale_at.insert({(uint64_t)s, l});
        stale_h.insert(h);
      }
      spin_us(ATTN_US);
    }
  }
  r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  r.wait_us = n_wait ? wait_sum / (double)n_wait : 0;
  w.holding.store(2, std::memory_order_release);   // never leave the poller held
  w.q.sync(0);
  stop.store(true);
  poller.join();
  r.kind1 = det.kind1();
  r.kind2 = det.kind2();
  r.kind4 = det.kind4();
  r.events = det.events();
  r.deferred = pp != nullptr ? pp->deferred() : 0;
  // the events, read back through the detector's own JSON lines
  std::set<std::pair<uint64_t, int>> k1_at, k4_at;
  std::set<uint64_t> k1_h, k4_h;
  char path[] = "/tmp/kt-memop-detect-test-XXXXXX";
  const int fd = mkstemp(path);
  det.flush(fd, true);
  FILE* f = fd >= 0 ? fdopen(fd, "r") : nullptr;
  if (f != nullptr) {
    rewind(f);
    char line[600];
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "{\"k\":\"event\"", 12) == 0) {
        const long long mask = field(line, "mask"), idx = field(line, "idx"), ss = field(line, "served_sync"),
                        ns = field(line, "n_sync");
        if (mask > 0 && (mask & 1)) {   // sync_serving's n_sync counts this sync: hand-off = n_sync - 1
          k1_at.insert({(uint64_t)ss, (int)(idx / 2)});
          k1_h.insert((uint64_t)(ns - 1));
        }
        if (mask > 0 && (mask & 4)) {   // sync_deferred's n_sync counts the syncs served before: hand-off = n_sync
          k4_at.insert({(uint64_t)ss, (int)(idx / 2)});
          k4_h.insert((uint64_t)ns);
        }
      } else if (strncmp(line, "{\"k\":\"count\"", 12) == 0) {
        r.ring_arms = (uint64_t)field(line, "ring_arms");
      }
    }
    fclose(f);
  } else if (fd >= 0) {
    close(fd);
  }
  unlink(path);
  r.det = c.loop == DETECT;
  r.k1_match = c.ring ? k1_h == stale_h : k1_at == stale_at;
  r.k4_match = c.ring ? k4_h == forced_h : (k4_at == forced_at && k4_h == forced_h);
  return r;
}

std::string json(const Result& r, uint64_t expect) {
  // events_match / deferrals_match: null for the loops without the detector (no events to match)
  const char* em = !r.det ? "null" : r.k1_match ? "true" : "false";
  const char* dm = !r.det || !r.fix ? "null" : r.k4_match ? "true" : "false";
  char b[900];
  snprintf(b, sizeof b,
           "\"%s\":{\"slots\":\"%s\",\"fix\":%s,\"steps\":%d,\"held\":%llu,\"stale\":%llu,\"checksum_ok\":%s,"
           "\"kind1\":%llu,\"kind2\":%llu,\"kind4\":%llu,\"events\":%llu,\"deferred\":%llu,\"events_match\":%s,"
           "\"deferrals_match\":%s,\"pair_ok\":%s,\"collisions\":%llu,\"ring_arms\":%llu,\"timeout\":%s,"
           "\"handoff_us\":%.1f,\"handoff_us_max\":%.1f,\"s\":%.2f}",
           r.name.c_str(), r.ring ? "ring" : "graph", r.fix ? "true" : "false", r.steps, (unsigned long long)r.forced,
           (unsigned long long)r.stale, r.checksum == expect ? "true" : "false", (unsigned long long)r.kind1,
           (unsigned long long)r.kind2, (unsigned long long)r.kind4, (unsigned long long)r.events,
           (unsigned long long)r.deferred, em, dm, r.pair_ok ? "true" : "false", (unsigned long long)r.collisions,
           (unsigned long long)r.ring_arms, r.timeout ? "true" : "false", r.wait_us, r.wait_us_max, r.seconds);
  return b;
}

}  // namespace

int main() {
  const Cfg cfgs[] = {
      {"plain", PLAIN, false, false, false},
      {"detect", DETECT, false, false, false},
      {"forced", DETECT, false, true, false},
      {"fix", PAIRED, true, false, false},
      {"fix_forced", PAIRED, true, true, false},
      {"fixdet", DETECT, true, false, false},
      {"fixdet_forced", DETECT, true, true, false},
      {"ring_plain", PLAIN, false, false, true},
      {"ring_forced", DETECT, false, true, true},
      {"ring_fix", PAIRED, true, false, true},
      {"ring_fix_forced", PAIRED, true, true, true},
      {"ring_fixdet_forced", DETECT, true, true, true},
  };
  const uint64_t eg = expect_checksum(STEPS), er = expect_checksum(RING_STEPS);
  std::string failed, body;
  uint64_t unforced_deferrals = 0;
  for (const Cfg& c : cfgs) {
    const Result r = run(c);
    const uint64_t expect = c.ring ? er : eg;
    const bool clean = r.stale == 0 && r.checksum == expect;
    bool ok = !r.timeout && r.pair_ok && r.collisions == 0;
    if (!c.fix && !c.forced) {           // plain, detect, ring_plain: the controls
      ok = ok && clean && (c.loop != DETECT || (r.kind1 == 0 && r.kind2 == 0 && r.k1_match));
    } else if (!c.fix) {                 // forced, ring_forced: the race without the fix
      ok = ok && r.forced > 0 && r.stale == r.forced && r.kind1 == r.forced && r.k1_match;
    } else if (!c.forced) {              // fix, fixdet, ring_fix: clean; deferrals reported (0 expected)
      ok = ok && clean && (c.loop != DETECT || (r.kind1 == 0 && r.kind2 == 0 && r.kind4 == r.deferred));
      unforced_deferrals += r.deferred;
    } else {                             // fix_forced, fixdet_forced, ring_fix_forced, ring_fixdet_forced
      ok = ok && clean && r.forced > 0 && r.deferred == r.forced &&
           (c.loop != DETECT || (r.kind1 == 0 && r.kind2 == 0 && r.kind4 == r.forced && r.k4_match));
    }
    if (!ok) failed += std::string(failed.empty() ? "" : ",") + "\"" + c.name + "\"";
    body += "," + json(r, expect);
  }
  printf("{\"test\":\"memop_detect\",\"status\":\"%s\",\"layers\":%d,\"steps\":%d,\"ring_steps\":%d,\"ring\":%d,"
         "\"force_every\":%d,\"failed\":[%s],\"unforced_deferrals\":%llu%s}\n",
         failed.empty() ? "PASS" : "FAIL", LAYERS, STEPS, RING_STEPS, RING, FORCE_EVERY, failed.c_str(),
         (unsigned long long)unforced_deferrals, body.c_str());
  return failed.empty() ? 0 : 1;
}
