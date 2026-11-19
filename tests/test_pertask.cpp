// Path B stagger r2 (concept-r1 STAGGER_PREREQ_r1.md section 5.1, STAGED_ARMS_r1.md section 20): host-only test of the
// per-task stream-memop sync (cpu_backend/memop_per_task.h, KT_MEMOP_SYNC_PER_TASK=1) on branch pathb-stagger-r2, with the
// real TaskQueue (cpu_backend/task_queue.cpp) and KT_MEMOP_PAIRED's Paired (memop_detect.h). No CUDA: lane threads play
// the CUDA streams (write request flags, spin on done flags, spin for the dense and partial GPU work), one thread plays
// CPUInfer's poller with the arming, snapshot and retire steps of CPUInfer (memop_arm, the pollers). Three poller loops:
//   today     a copy of cpuinfer.h memop_poller's loop body (TaskQueue::sync for every sync)
//   paired    memop_detect.h scan_paired (KT_MEMOP_PAIRED=1 alone: served after its submit, then TaskQueue::sync)
//   per_task  memop_per_task.h scan_per_task (KT_MEMOP_SYNC_PER_TASK=1), the code CPUInfer::memop_poller_per_task runs
// Runs (two lanes unless named "one_lane"; 40 layers; graph slots armed once and permanent unless named "ring"):
//   today_c*, paired_c350, per_task_c*   the three loops at c = 350 / 250 / 420 us
//   per_task_one_lane_c350               one lane alone
//   per_task_reorder, today_reorder      a lane writes its sync request 200 us before its submit request
//   per_task_unpaired                    syncs armed under another stream key: no pair, today's rule
//   per_task_extra_enqueuer              a third thread enqueues a 50 us task every 300 us
//   per_task_task_throws                 the first task throws: the exception must surface in the poller
//   *_forced2                            two lanes; every FORCE_EVERY-th hand-off of lane 0 holds the poller right after it
//                                        read that submit's flag as 0, until lane 0 has raised its submit AND its sync
//                                        (the race of SEAM_FUSION_r1.md section 9) while lane 1 keeps running
//   *_ring2, *_ring2_forced              the same with ring slots: every hand-off arms its submit and sync on the next two
//                                        of RING slots (odd), published under the arm mutex, retired after service
// A lane counts a violation when its sync's done flag rose before its own task finished (a stale CPU partial in the
// server). Prints one JSON object; the driver (test_pertask.py) checks it.
//   build: g++ -O2 -std=c++17 -pthread -I<cpu_backend> test_pertask.cpp <cpu_backend>/task_queue.cpp -o test_pertask
#include <immintrin.h>
#include <stdlib.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "memop_detect.h"
#include "memop_per_task.h"
#include "task_queue.h"

static uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void spin_ns(uint64_t ns) {
  const uint64_t t0 = now_ns();
  while (now_ns() - t0 < ns) _mm_pause();
}

constexpr int FORCE_EVERY = 53;   // prime: the held layer moves through all 40
constexpr int RING = 63;          // ring slots of the ring runs (odd: a slot serves as a submit and as a sync in turn)

struct Entry {   // CPUInfer::MemopEntry as cpuinfer.h has it (graph unused here)
  void (*func)(void*);
  void* args;
  size_t allow;
  int idx;
  bool sync;
  bool permanent;
  int pair;
  int graph;
};

constexpr int kStride = 16;   // as cpuinfer.h: one cache line per flag
struct Flags {
  std::unique_ptr<uint32_t[]> mem;
  explicit Flags(int n) : mem(new uint32_t[(size_t)n * 2 * kStride]()) {}
  volatile uint32_t* req(int i) { return (volatile uint32_t*)&mem[(size_t)i * 2 * kStride]; }
  volatile uint32_t* done(int i) { return (volatile uint32_t*)&mem[(size_t)i * 2 * kStride + kStride]; }
};

struct TaskArg {
  TaskQueue* q;
  std::atomic<uint64_t>* fin;      // generations finished of this (lane, layer)
  std::atomic<uint64_t>* lane_enq;
  std::atomic<uint64_t>* lane_fin;
  uint64_t c_ns;
  std::atomic<int>* throw_now;
};

static void submit_fn(void* p) {   // a submit entry's host function: enqueue one CPU task, as forward_partials_task's
  TaskArg* a = (TaskArg*)p;
  a->lane_enq->fetch_add(1);
  a->q->enqueue([a]() {
    if (a->throw_now->exchange(0)) {
      a->lane_fin->fetch_add(1);
      throw std::runtime_error("planted task failure");
    }
    spin_ns(a->c_ns);
    a->fin->fetch_add(1, std::memory_order_release);
    a->lane_fin->fetch_add(1);
  });
}

struct Run {
  std::string name, mode;          // mode: today | paired | per_task
  int lanes = 2, layers = 40, steps = 12;
  uint64_t c_ns = 350000, d_ns = 176000, p_ns = 99000;
  bool reorder = false;            // a lane writes its sync request 200 us before its submit request
  bool unpaired = false;           // syncs armed under another stream key: no pair, today's rule
  bool throw_first = false;        // the first task throws
  bool extra_enqueuer = false;     // a third thread enqueues a 50 us task every 300 us
  bool forced = false;             // hold the poller at every FORCE_EVERY-th hand-off of lane 0
  bool ring = false;               // ring slots, armed per hand-off
};

static std::string run_case(const Run& r) {
  if (r.mode == "per_task") setenv("KT_MEMOP_SYNC_PER_TASK", "1", 1);
  else unsetenv("KT_MEMOP_SYNC_PER_TASK");
  setenv("KT_TASKQUEUE_SPIN_US", "300", 1);   // as every launch
  TaskQueue q;
  const int L = r.layers;
  const int n = r.ring ? RING : r.lanes * 2 * L;
  Flags fl(n);
  std::vector<std::atomic<uint64_t>> fin(r.lanes * L);
  for (auto& f : fin) f.store(0);
  std::vector<std::atomic<uint64_t>> lane_enq(r.lanes), lane_fin(r.lanes);
  for (int i = 0; i < r.lanes; i++) lane_enq[i].store(0), lane_fin[i].store(0);
  std::atomic<int> throw_now{r.throw_first ? 1 : 0};
  std::vector<TaskArg> args(r.lanes * L);
  for (int lane = 0; lane < r.lanes; lane++)
    for (int l = 0; l < L; l++)
      args[lane * L + l] = TaskArg{&q, &fin[lane * L + l], &lane_enq[lane], &lane_fin[lane], r.c_ns, &throw_now};
  kt_memop_detect::Paired paired(n);
  std::mutex mu;                   // as CPUInfer::mmtx_: the armed entries, their version, the pairing's arm side
  std::vector<Entry> entries;
  std::atomic<uint64_t> version{0};
  std::vector<int> busy(n, 0);     // ring: a slot is busy from its arm until its lane has seen the hand-off's done
  long collisions = 0;
  int keys[8] = {0};
  std::atomic<int> paired_syncs{0};
  auto arm = [&](int lane, int l, int idx, bool sync, bool perm) {   // the caller holds mu
    const void* ks = &keys[lane];
    const void* key = (sync && r.unpaired) ? (const void*)&keys[4 + lane] : ks;
    const int pair = paired.arm(key, idx, sync, perm);
    if (sync && pair >= 0) paired_syncs++;
    entries.push_back(Entry{sync ? nullptr : &submit_fn, sync ? nullptr : (void*)&args[lane * L + l], 0, idx, sync, perm,
                            pair, -1});
  };
  if (!r.ring) {   // armed as a capture would: each lane's layers in stream order, one lane after the other
    std::lock_guard<std::mutex> g(mu);
    for (int lane = 0; lane < r.lanes; lane++)
      for (int l = 0; l < L; l++) {
        const int sub = lane * 2 * L + 2 * l;
        arm(lane, l, sub, false, true);
        arm(lane, l, sub + 1, true, true);
      }
    version.fetch_add(1, std::memory_order_release);
  }
  std::atomic<bool> stop{false}, stop_extra{false};
  std::atomic<long> releases{0}, released_other_pending{0};
  std::atomic<int> hold_idx{-1}, holding{0};
  std::vector<std::atomic<int>> slot_lane(n);   // ring: the lane that armed a slot last (for the sample)
  for (auto& s : slot_lane) s.store(0);
  std::string exc;
  uint64_t waits = 0, early = 0;
  auto lane_of = [&](int idx) { return r.ring ? slot_lane[idx].load() : idx / (2 * L); };
  auto sample = [&](int idx) {   // at the moment a done flag is raised: had another lane a task enqueued, not finished?
    const int a = lane_of(idx);
    releases++;
    for (int b = 0; b < r.lanes; b++)
      if (b != a && lane_enq[b].load() > lane_fin[b].load()) {
        released_other_pending++;
        break;
      }
  };
  std::thread poller([&] {
    try {
      auto req = [&](int i) { return fl.req(i); };
      auto done = [&](int i) {
        sample(i);
        return fl.done(i);
      };
      auto full_sync = [&](size_t allow) { q.sync(allow); };
      auto hold = [&](const Entry& e, bool raised) {   // the forced race: held after reading a chosen submit flag as 0
        if (!r.forced || raised || e.sync || e.idx != hold_idx.load(std::memory_order_acquire)) return;
        int z = 0;
        if (!holding.compare_exchange_strong(z, 1)) return;
        while (holding.load(std::memory_order_acquire) != 2) std::this_thread::yield();
      };
      kt_memop::Waits<TaskQueue> w(&q, n);
      std::vector<Entry> snap;
      uint64_t seen = ~0ull;
      while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t v = version.load(std::memory_order_acquire);
        if (v != seen) {
          std::lock_guard<std::mutex> g(mu);
          snap = entries;
          seen = version.load(std::memory_order_acquire);
        }
        std::vector<int> retire;
        if (r.mode == "per_task") {
          kt_memop::scan_per_task(snap, paired, w, req, done, full_sync, retire, 0, hold);
        } else if (r.mode == "paired") {
          kt_memop_detect::scan_paired(snap, paired, req, done, full_sync, retire, hold);
        } else {
          for (const Entry& e : snap) {   // cpuinfer.h memop_poller, unchanged (plus the test's hold)
            const bool raised = *fl.req(e.idx) == 1;
            hold(e, raised);
            if (!raised) continue;
            *fl.req(e.idx) = 0;
            if (!e.sync) {
              e.func(e.args);
            } else {
              q.sync(e.allow);
              std::atomic_thread_fence(std::memory_order_release);
              sample(e.idx);
              *fl.done(e.idx) = 1;
            }
            if (!e.permanent) retire.push_back(e.idx);
          }
        }
        _mm_pause();
        if (!retire.empty()) {   // as the pollers: drop the served ring entries, then a new version
          std::lock_guard<std::mutex> g(mu);
          for (int idx : retire)
            for (size_t i = 0; i < entries.size(); i++)
              if (!entries[i].permanent && entries[i].idx == idx) {
                entries.erase(entries.begin() + i);
                break;
              }
          version.fetch_add(1, std::memory_order_release);
        }
      }
      waits = w.waits();
      early = w.early();
    } catch (const std::exception& ex) {
      exc = ex.what();
    }
  });
  std::thread extra;
  if (r.extra_enqueuer)
    extra = std::thread([&] {
      while (!stop_extra.load()) {
        q.enqueue([]() { spin_ns(50000); });
        spin_ns(300000);
      }
    });
  std::atomic<long> viol{0}, tmo{0}, held{0};
  std::atomic<int> ring_next{0};
  std::vector<double> lane_ms(r.lanes, 0.0);
  auto lane_fn = [&](int lane) {
    const uint64_t t0 = now_ns();
    bool dead = false;
    for (int s = 0; s < r.steps && !dead; s++) {
      for (int l = 0; l < L && !dead; l++) {
        spin_ns(r.d_ns);
        int sub = lane * 2 * L + 2 * l, syn = sub + 1;
        if (r.ring) {   // an eager call: the next two ring slots, published together
          std::lock_guard<std::mutex> g(mu);
          const int k = ring_next.fetch_add(2);
          sub = k % RING;
          syn = (k + 1) % RING;
          collisions += busy[sub] + busy[syn];
          busy[sub] = busy[syn] = 1;
          slot_lane[sub].store(lane);
          slot_lane[syn].store(lane);
          arm(lane, l, sub, false, false);
          arm(lane, l, syn, true, false);
          version.fetch_add(1, std::memory_order_release);
        }
        const bool force = r.forced && lane == 0 && (s * L + l) % FORCE_EVERY == FORCE_EVERY - 1;
        if (force) {   // hold the poller right after it reads this submit's flag as 0, then raise submit and sync
          holding.store(0, std::memory_order_release);
          hold_idx.store(sub, std::memory_order_release);
          const uint64_t th = now_ns() + 3000000000ull;
          while (holding.load(std::memory_order_acquire) != 1) {
            if (now_ns() > th) {
              tmo++;
              dead = true;
              break;
            }
            std::this_thread::yield();
          }
          if (dead) {
            hold_idx.store(-1, std::memory_order_release);
            break;
          }
          held++;
        }
        if (!r.reorder) {
          *fl.req(sub) = 1;
          spin_ns(r.p_ns);
          *fl.req(syn) = 1;
        } else {
          *fl.req(syn) = 1;
          spin_ns(200000);
          *fl.req(sub) = 1;
        }
        if (force) {
          hold_idx.store(-1, std::memory_order_release);
          holding.store(2, std::memory_order_release);
        }
        const uint64_t deadline = now_ns() + 3000000000ull;
        while (*fl.done(syn) != 1) {
          if (now_ns() > deadline) {
            tmo++;
            dead = true;
            break;
          }
          _mm_pause();
        }
        if (dead) break;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (fin[lane * L + l].load(std::memory_order_acquire) != (uint64_t)(s + 1)) viol++;   // released before its task
        *fl.done(syn) = 0;
        if (r.ring) {
          std::lock_guard<std::mutex> g(mu);
          busy[sub] = busy[syn] = 0;
        }
      }
    }
    lane_ms[lane] = (now_ns() - t0) / 1e6;
  };
  std::vector<std::thread> lt;
  for (int lane = 0; lane < r.lanes; lane++) lt.emplace_back(lane_fn, lane);
  for (auto& t : lt) t.join();
  holding.store(2, std::memory_order_release);   // never leave the poller held
  stop.store(true);
  poller.join();
  stop_extra.store(true);
  if (extra.joinable()) extra.join();
  double wall = 0;
  for (double v : lane_ms) wall = v > wall ? v : wall;
  const double cu = r.c_ns / 1e3, du = r.d_ns / 1e3, pu = r.p_ns / 1e3;
  const double one = du + (pu > cu ? pu : cu);
  const double floor_layer = (r.lanes * cu > one ? r.lanes * cu : one);
  char buf[1400];
  snprintf(buf, sizeof buf,
           "{\"name\":\"%s\",\"mode\":\"%s\",\"lanes\":%d,\"layers\":%d,\"steps\":%d,\"c_us\":%.0f,\"d_us\":%.0f,\"p_us\":%.0f,"
           "\"reorder\":%s,\"unpaired\":%s,\"extra_enqueuer\":%s,\"throw_first\":%s,\"forced\":%s,\"ring\":%s,"
           "\"paired_syncs\":%d,\"step_ms\":%.3f,\"floor_ms\":%.3f,\"violations\":%ld,\"timeouts\":%ld,\"held\":%ld,"
           "\"deferred\":%llu,\"collisions\":%ld,\"releases\":%ld,\"released_other_pending\":%ld,\"waits\":%llu,"
           "\"early\":%llu,\"enqueued\":%llu,\"finished\":%llu,\"exception\":\"%s\"}",
           r.name.c_str(), r.mode.c_str(), r.lanes, L, r.steps, cu, du, pu, r.reorder ? "true" : "false",
           r.unpaired ? "true" : "false", r.extra_enqueuer ? "true" : "false", r.throw_first ? "true" : "false",
           r.forced ? "true" : "false", r.ring ? "true" : "false", paired_syncs.load(), wall / r.steps,
           floor_layer * L / 1e3, viol.load(), tmo.load(), held.load(), (unsigned long long)paired.deferred(), collisions,
           releases.load(), released_other_pending.load(), (unsigned long long)waits, (unsigned long long)early,
           (unsigned long long)q.enqueued(), (unsigned long long)q.finished(), exc.c_str());
  return buf;
}

static std::string units() {   // the counters, the pairing and the per-task rule's take
  unsetenv("KT_MEMOP_SYNC_PER_TASK");
  uint64_t off_e, off_f, on_e, on_f;
  {
    TaskQueue q;
    for (int i = 0; i < 10; i++) q.enqueue([]() {});
    q.sync(0);
    off_e = q.enqueued(), off_f = q.finished();
  }
  setenv("KT_MEMOP_SYNC_PER_TASK", "1", 1);
  {
    TaskQueue q;
    for (int i = 0; i < 10; i++) q.enqueue([]() {});
    q.sync(0);
    on_e = q.enqueued(), on_f = q.finished();
  }
  const bool env_on = kt_memop::per_task_from_env();
  setenv("KT_MEMOP_SYNC_PER_TASK", "10", 1);
  const bool env_10 = kt_memop::per_task_from_env();
  unsetenv("KT_MEMOP_SYNC_PER_TASK");
  const bool env_unset = kt_memop::per_task_from_env();
  kt_memop_detect::Paired pr(16);
  int a = 0, b = 0;
  const int s1 = pr.arm(&a, 7, false, true), y1 = pr.arm(&a, 8, true, true), y2 = pr.arm(&a, 9, true, true),
            y3 = pr.arm(&b, 10, true, true);
  const bool t_allow1 = kt_memop::per_task_takes(0, 1), t_unpaired = kt_memop::per_task_takes(-1, 0),
             t_paired = kt_memop::per_task_takes(0, 0);
  char buf[640];
  snprintf(buf, sizeof buf,
           "{\"off_enqueued\":%llu,\"off_finished\":%llu,\"on_enqueued\":%llu,\"on_finished\":%llu,\"env_on\":%s,"
           "\"env_10\":%s,\"env_unset\":%s,\"submit_pair\":%d,\"sync_pair\":%d,\"second_sync_pair\":%d,"
           "\"other_stream_pair\":%d,\"takes_allow1\":%s,\"takes_unpaired\":%s,\"takes_paired\":%s}",
           (unsigned long long)off_e, (unsigned long long)off_f, (unsigned long long)on_e, (unsigned long long)on_f,
           env_on ? "true" : "false", env_10 ? "true" : "false", env_unset ? "true" : "false", s1, y1, y2, y3,
           t_allow1 ? "true" : "false", t_unpaired ? "true" : "false", t_paired ? "true" : "false");
  return buf;
}

int main() {
  std::string out = "{\"units\":" + units() + ",\"force_every\":" + std::to_string(FORCE_EVERY) +
                    ",\"ring_slots\":" + std::to_string(RING) + ",\"runs\":[";
  std::vector<Run> runs;
  for (uint64_t c : {350000ull, 250000ull, 420000ull}) {
    for (const char* m : {"today", "per_task"}) {
      Run r;
      r.name = std::string(m) + "_c" + std::to_string(c / 1000);
      r.mode = m;
      r.c_ns = c;
      runs.push_back(r);
    }
  }
  Run pc;
  pc.name = "paired_c350", pc.mode = "paired";
  runs.push_back(pc);
  Run alone;
  alone.name = "per_task_one_lane_c350", alone.mode = "per_task", alone.lanes = 1;
  runs.push_back(alone);
  Run ro;
  ro.name = "per_task_reorder", ro.mode = "per_task", ro.reorder = true, ro.steps = 3;
  runs.push_back(ro);
  Run rt = ro;
  rt.name = "today_reorder", rt.mode = "today";
  runs.push_back(rt);
  Run up;
  up.name = "per_task_unpaired", up.mode = "per_task", up.unpaired = true, up.steps = 4;
  runs.push_back(up);
  Run ex;
  ex.name = "per_task_extra_enqueuer", ex.mode = "per_task", ex.extra_enqueuer = true;
  runs.push_back(ex);
  Run th;
  th.name = "per_task_task_throws", th.mode = "per_task", th.throw_first = true, th.lanes = 1, th.layers = 1, th.steps = 1;
  runs.push_back(th);
  for (const char* m : {"today", "paired", "per_task"}) {   // the forced two-lane race, graph slots then ring slots
    for (int ring = 0; ring < 2; ring++) {
      Run f;
      f.name = std::string(m) + (ring ? "_ring2_forced" : "_forced2");
      f.mode = m, f.forced = true, f.ring = ring != 0, f.steps = 20;
      runs.push_back(f);
    }
  }
  Run rg;
  rg.name = "per_task_ring2", rg.mode = "per_task", rg.ring = true;
  runs.push_back(rg);
  for (size_t i = 0; i < runs.size(); i++) {
    out += (i ? "," : "") + run_case(runs[i]);
    fprintf(stderr, "done %s\n", runs[i].name.c_str());
  }
  out += "]}";
  printf("%s\n", out.c_str());
  return 0;
}
