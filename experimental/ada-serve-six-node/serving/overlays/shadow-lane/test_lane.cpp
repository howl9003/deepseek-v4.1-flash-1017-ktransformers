// Standalone test of the KT shadow lane (shadow-lane-r1, Claude, 4 Nov 2026) on the patched cpu_backend.
// Build: g++ -std=c++20 -O2 -pthread -I<kt-kernel>/cpu_backend test_lane.cpp <kt-kernel>/cpu_backend/worker_pool.cpp
//        <kt-kernel>/cpu_backend/task_queue.cpp -lhwloc -lnuma -o test_lane
// Run:   KT_SHADOW_LANE=1 taskset -c <two spare cores> ./test_lane <numa ids, comma-separated> <threads per part> [rounds]
//        (the threads are unbound and stay on the taskset cores; NUMA ids only set each leader's memory node)
//
// Main work goes through a main TaskQueue, lane work through a lane TaskQueue (as the lane CPUInfer does), both into the
// same WorkerPool. Each task is a forward-like shape: do_numa_job over the parts, three pool jobs per part, then one
// pool-wide job. Checks:
//   1. main outputs equal a single-threaded reference with the lane idle and with the lane busy;
//   2. lane outputs equal their reference;
//   3. lane items do not start while a main task runs (kt_lane::main_active sampled inside every lane item; a race of
//      one item at a main task's start is allowed, so overlap above 0.5% of lane items fails);
//   4. an exception in a lane item reaches the lane caller and the pool keeps working;
//   5. main task wall time with the lane idle and busy (p50 / p99 / max), as information only: pinning and the host's
//      load decide these numbers, not the test.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "task_queue.h"
#include "worker_pool.h"

static uint64_t work(uint64_t seed, int reps) {  // deterministic integer mixing, about reps * 2 ns
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (int r = 0; r < reps; r++) x = (x ^ (x >> 29)) * 0xBF58476D1CE4E5B9ull + (uint64_t)r;
  return x;
}

struct Shape {
  int parts, items, reps;
};

// The pool's parts, built with the unbound constructors: WorkerPool binds its threads to fixed cores of each node, which
// on a serving host are served workers, and a user unit cannot hold threads to a CPU set. Unbound threads inherit the
// caller's taskset. This is WorkerPool minus its constructor (and the three-line lane hookup in WorkerPool::init).
struct TestPool {
  std::vector<std::unique_ptr<InNumaPool>> parts;
  std::unique_ptr<NumaJobDistributor> distributor;
  std::unique_ptr<LaneDistributor> lane;
  TestPool(const std::vector<int>& numa_ids, int threads) {
    for (size_t i = 0; i < numa_ids.size(); i++) parts.emplace_back(new InNumaPool(threads));
    distributor.reset(new NumaJobDistributor(numa_ids));
    lane.reset(new LaneDistributor(numa_ids));
    std::vector<InNumaPool*> raw;
    for (auto& p : parts) raw.push_back(p.get());
    distributor->set_lane(lane.get(), raw);
  }
  NumaJobDistributor* dispense_backend() { return distributor.get(); }
  InNumaPool* get_subpool(int i) { return parts[i].get(); }
  void do_work_stealing_job(int n, std::function<void(int)> init, std::function<void(int)> f, std::function<void(int)> fin) {
    parts[0]->do_work_stealing_job(n, init, f, fin);  // as WorkerPool::do_work_stealing_job
  }
};

// One forward-like task: per part three pool jobs, each item writing its own slot, then one pool-wide job summing.
static void forward(TestPool& pool, const Shape& s, uint64_t round, std::vector<uint64_t>& out, std::atomic<long>* violations) {
  std::vector<uint64_t> stage((size_t)s.parts * s.items * 3);
  pool.dispense_backend()->do_numa_job([&](int part) {
    for (int phase = 0; phase < 3; phase++) {
      pool.get_subpool(part)->do_work_stealing_job(s.items, nullptr, [&, part, phase](int i) {
        if (violations && kt_lane::main_active.load(std::memory_order_acquire) != 0) violations->fetch_add(1);
        const size_t at = ((size_t)part * 3 + phase) * s.items + i;
        const uint64_t prev = phase ? stage[at - s.items] : 0;
        stage[at] = work(round * 1000003ull + at + prev, s.reps);
      }, nullptr);
    }
  });
  out.assign(s.items, 0);
  pool.do_work_stealing_job(s.items, nullptr, [&](int i) {
    if (violations && kt_lane::main_active.load(std::memory_order_acquire) != 0) violations->fetch_add(1);
    uint64_t acc = 0;
    for (int p = 0; p < s.parts; p++) acc ^= stage[((size_t)p * 3 + 2) * s.items + i] + (uint64_t)p;
    out[i] = acc;
  }, nullptr);
}

static std::vector<uint64_t> reference(const Shape& s, uint64_t round) {
  std::vector<uint64_t> stage((size_t)s.parts * s.items * 3), out(s.items, 0);
  for (int part = 0; part < s.parts; part++)
    for (int phase = 0; phase < 3; phase++)
      for (int i = 0; i < s.items; i++) {
        const size_t at = ((size_t)part * 3 + phase) * s.items + i;
        stage[at] = work(round * 1000003ull + at + (phase ? stage[at - s.items] : 0), s.reps);
      }
  for (int i = 0; i < s.items; i++)
    for (int p = 0; p < s.parts; p++) out[i] ^= stage[((size_t)p * 3 + 2) * s.items + i] + (uint64_t)p;
  return out;
}

static void stats(const char* label, std::vector<double> us) {
  std::sort(us.begin(), us.end());
  auto q = [&](double f) { return us[std::min(us.size() - 1, (size_t)(f * us.size()))]; };
  printf("%s: %zu tasks, p50 %.1f us, p99 %.1f us, max %.1f us\n", label, us.size(), q(0.5), q(0.99), us.back());
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <numa ids> <threads per part> [rounds]\n", argv[0]); return 2; }
  if (!kt_lane::enabled()) { fprintf(stderr, "set KT_SHADOW_LANE=1\n"); return 2; }
  std::vector<int> numa_ids;
  for (char* t = strtok(argv[1], ","); t; t = strtok(nullptr, ",")) numa_ids.push_back(atoi(t));
  const int parts = (int)numa_ids.size();
  const int rounds = argc > 3 ? atoi(argv[3]) : 400;
  TestPool pool(numa_ids, atoi(argv[2]));
  TaskQueue main_queue(false), lane_queue(true);
  const Shape main_shape{parts, 64, 400}, lane_shape{parts, 48, 4000};
  int failures = 0;

  // 1 and 5, lane idle
  std::vector<double> idle_us;
  for (int r = 0; r < rounds; r++) {
    std::vector<uint64_t> out;
    const auto t0 = std::chrono::steady_clock::now();
    main_queue.enqueue([&] { forward(pool, main_shape, r, out, nullptr); });
    main_queue.sync(0);
    idle_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    if (out != reference(main_shape, r)) { printf("FAIL main round %d (lane idle)\n", r); failures++; }
    std::this_thread::sleep_for(std::chrono::microseconds(300));  // the GPU part of a layer
  }

  // 1, 2, 3 and 5, lane busy: a lane thread submits forwards back to back while main runs
  std::atomic<bool> stop{false};
  std::atomic<long> violations{0}, lane_done{0}, lane_bad{0};
  std::thread lane_driver([&] {
    for (uint64_t r = 0; !stop.load(); r++) {
      std::vector<uint64_t> out;
      lane_queue.enqueue([&] { forward(pool, lane_shape, 7000000 + r, out, &violations); });
      lane_queue.sync(0);
      if (out != reference(lane_shape, 7000000 + r)) lane_bad.fetch_add(1);
      lane_done.fetch_add(1);
    }
  });
  std::vector<double> busy_us;
  for (int r = 0; r < rounds; r++) {
    std::vector<uint64_t> out;
    const auto t0 = std::chrono::steady_clock::now();
    main_queue.enqueue([&] { forward(pool, main_shape, 100000 + r, out, nullptr); });
    main_queue.sync(0);
    busy_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    if (out != reference(main_shape, 100000 + r)) { printf("FAIL main round %d (lane busy)\n", r); failures++; }
    std::this_thread::sleep_for(std::chrono::microseconds(300));
  }
  stop.store(true);
  lane_driver.join();
  // An item may start just as a main task begins (the check and the start are not one step): the one-item overlap the
  // design allows. Overlap above 0.5% of lane items would mean items start while main runs.
  const long lane_items = lane_done.load() * ((long)lane_shape.parts * 3 * lane_shape.items + lane_shape.items);
  printf("lane forwards %ld, wrong %ld; lane items that saw a main task running: %ld of %ld\n", lane_done.load(),
         lane_bad.load(), violations.load(), lane_items);
  if (lane_bad.load() || lane_done.load() == 0 || violations.load() * 200 > lane_items) failures++;

  // 4: an exception inside a lane item
  bool caught = false;
  lane_queue.enqueue([&] {
    pool.get_subpool(0)->do_work_stealing_job(8, nullptr, [](int i) { if (i == 5) throw std::runtime_error("lane item 5"); }, nullptr);
  });
  try { lane_queue.sync(0); } catch (const std::exception& e) { caught = std::string(e.what()) == "lane item 5"; }
  printf("lane exception reached the lane caller: %s\n", caught ? "yes" : "NO");
  if (!caught) failures++;
  {
    std::vector<uint64_t> out;
    main_queue.enqueue([&] { forward(pool, main_shape, 999, out, nullptr); });
    main_queue.sync(0);
    if (out != reference(main_shape, 999)) { printf("FAIL main after the lane exception\n"); failures++; }
  }

  stats("main, lane idle", idle_us);
  stats("main, lane busy", busy_us);
  printf(failures ? "LANE_TEST_FAIL %d\n" : "LANE_TEST_PASS\n", failures);
  fflush(stdout);
  _exit(failures ? 1 : 0);  // the pool's destructor joins spinning threads; not part of the test
}
