"""Native overlay: the KT shadow lane (shadow-lane-r1, Claude, 4 Nov 2026). Selectors KT_SHADOW_LANE=1 and
KT_SHADOW_LANE_FROM=<KT layer id>, default off. Owner (4 Nov): "the lane is probably the arch we need in the end so
build it".

A low-priority lane for the DSpark shadow draft on the target's own CPU workers:

- A lane CPUInfer (`CPUInfer(main, True)`) shares the main WorkerPool but has its own TaskQueue (thread marked
  kt_lane::t_lane = 1) and its own memop poller. A draft forward submitted through it runs its orchestration there.
- A thread with t_lane set never touches the main job slots. NumaJobDistributor::do_numa_job goes to a
  LaneDistributor (one sleeping leader per part, also marked), and InNumaPool::do_work_stealing_job posts the job to
  a second slot of the pool (LaneJob) and sleeps until it is done.
- Pool threads (workers and the part leaders) run one lane item at a time from their wait loops, only while no main
  TaskQueue task is running (kt_lane::main_active == 0). A main task that starts while an item is in flight waits
  for that item at most (an item is one expert slice, tens of microseconds). If no thread takes an item for 1 ms while
  no main task runs (pool threads asleep after 50 ms idle), the lane leader runs items itself.
- Isolation from main results, after the shared-state audit (4 Nov): objects at or above KT_SHADOW_LANE_FROM take
  their part scratch from a separate SharedMemBufferNuma and their route scratch from a separate static array, and
  truncation events on lane threads or inside lane items go to a lane counter, so the target's fail-closed check never
  sees a draft event and a draft settle never absorbs a target event. Lane items are leaf calls: they never call a job
  API, so the single-job members of the pool and distributor, and thread_local_id, are only ever touched by main.

Off (KT_SHADOW_LANE unset): no lane CPUInfer can be built, no LaneDistributor exists, every lane slot stays empty.
The main path then differs only by one relaxed load of the empty slot per wait-loop pass, two atomic adds per main
TaskQueue task, and a thread_local read when a job is posted or a truncation event is counted. No arithmetic changes.

argv: kt-kernel source dir (patched in place)."""
import hashlib
import sys
from pathlib import Path

PINS = {
    'cpu_backend/worker_pool.h': '319aaf8fb18688e1465a592cee6e84914e3de12d54c10dafe347db0cbbaa9f99',
    'cpu_backend/worker_pool.cpp': '107dd465c3761eceebbcb7708af994020452b136a1a1a3bd2dcbf97a07bf8989',
    'cpu_backend/task_queue.h': 'ddf0967d003159399c967ef885b20261aa690dfd695ef5c10ceaccc06d76f319',
    'cpu_backend/task_queue.cpp': '9080c8ad286235f02ba9b7d4c048f4dcc4e5bf524ebf720785a22d1086f34a0f',
    'cpu_backend/cpuinfer.h': 'd9faa3a8e20aaefffbd6f29abdf60250721c270bf38d90e831265fc3bc46eac8',
    'cpu_backend/shared_mem_buffer.h': 'a6620568a9127516a6052c08ba2b11eb7038f1199b07a349a5b432777497a8ff',
    'operators/avx2/moe_base.hpp': '05b89f50e19de2a92ca0d95c48690173197343ed4f732a44ac1aa7d23c1818e5',
    'operators/avx2/contract_v2-moe.hpp': 'e7047513711ecb141807e3c9df9b788de78fb9a9e6cb68d87af9f1ec2d35c3c7',
    'ext_bindings.cpp': '8cf7c3bf8e59ead56264b924e400301a31cb3c49afa93db28572d675ee283273',
}
root = Path(sys.argv[1])
text = {}
for rel, pin in PINS.items():
    s = (root / rel).read_bytes().decode('utf-8')
    assert hashlib.sha256(s.encode('utf-8')).hexdigest() == pin, rel + ' differs'
    text[rel] = s


def sub(rel, old, new, count=1):
    assert text[rel].count(old) == count, (rel, text[rel].count(old), old[:90])
    text[rel] = text[rel].replace(old, new)


# ---- worker_pool.h -------------------------------------------------------------------------------------------------
H = 'cpu_backend/worker_pool.h'
sub(H, "#include <vector>\n\n// #define PROFILE_BALANCE\n",
    "#include <vector>\n\n// #define PROFILE_BALANCE\n\n"
    "// KT_SHADOW_LANE (shadow-lane-r1): a low-priority lane on the same workers. A thread with kt_lane::t_lane set (the\n"
    "// lane CPUInfer's queue thread, the lane leaders) posts its jobs to a second slot per InNumaPool; pool threads run one\n"
    "// lane item at a time from their wait loops while no main TaskQueue task runs. Off: the slots stay empty.\n"
    "namespace kt_lane {\n"
    "extern thread_local int t_lane;       // 1 on lane orchestration threads\n"
    "extern thread_local int t_item;       // 1 while a pool thread runs a lane item\n"
    "extern std::atomic<int> main_active;  // main TaskQueue tasks running\n"
    "bool enabled();                       // KT_SHADOW_LANE=1\n"
    "int from_layer();                     // KT_SHADOW_LANE_FROM: objects at or above this KT layer id are lane objects; -1 off\n"
    "inline bool lane_layer(int layer_idx) { const int f = from_layer(); return f >= 0 && layer_idx >= f; }\n"
    "}  // namespace kt_lane\n")
sub(H, "  void do_work_stealing_job(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>);\n"
       "  void do_work_stealing_job(int, std::function<void(int)>);\n\n private:\n  int worker_count;\n",
    "  void do_work_stealing_job(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>);\n"
    "  void do_work_stealing_job(int, std::function<void(int)>);\n\n"
    "  // KT_SHADOW_LANE: run one item of the posted lane job, if any and if no main task runs. Pool threads call this\n"
    "  // from their wait loops; returns true when an item ran.\n"
    "  bool lane_run_one(int thread_id);\n\n"
    " private:\n"
    "  struct LaneJob {\n"
    "    std::function<void(int)> init, compute;\n"
    "    int end = 0;\n"
    "    std::atomic<int> curr{0}, done{0};\n"
    "    std::exception_ptr error;\n"
    "    std::mutex error_mtx;\n"
    "  };\n"
    "  std::atomic<LaneJob*> lane_job_{nullptr};\n"
    "  std::atomic<int> lane_users_{0};\n"
    "  std::mutex lane_mtx_;\n"
    "  std::condition_variable lane_cv_;\n"
    "  void lane_job_run(int, std::function<void(int)>, std::function<void(int)>, std::function<void(int)>);\n\n"
    "  int worker_count;\n")
sub(H, "class NumaJobDistributor {\n public:\n",
    "class LaneDistributor;\n\n"
    "class NumaJobDistributor {\n public:\n"
    "  // KT_SHADOW_LANE: jobs from lane threads go to `lane`; part leaders run lane items of `pools[i]` while waiting.\n"
    "  void set_lane(LaneDistributor* lane, const std::vector<InNumaPool*>& pools);\n")
sub(H, "  std::function<void(int)> compute_func;\n  std::vector<std::thread> workers;\n\n  void worker_thread(int);\n};\n",
    "  std::function<void(int)> compute_func;\n  std::vector<std::thread> workers;\n"
    "  LaneDistributor* lane_ = nullptr;\n"
    "  std::atomic<InNumaPool*> lane_pools_[64] = {};\n\n"
    "  void worker_thread(int);\n};\n\n"
    "// KT_SHADOW_LANE: one leader per part for lane jobs. Leaders sleep between jobs (no spinning on worker cores).\n"
    "class LaneDistributor {\n"
    " public:\n"
    "  explicit LaneDistributor(std::vector<int> numa_ids);\n"
    "  ~LaneDistributor();\n"
    "  void do_numa_job(std::function<void(int)>);\n\n"
    " private:\n"
    "  void worker_thread(int);\n"
    "  int numa_count;\n"
    "  std::vector<int> numa_ids;\n"
    "  std::mutex mtx;\n"
    "  std::condition_variable cv_start, cv_done;\n"
    "  std::function<void(int)> compute_func;\n"
    "  uint64_t generation = 0;\n"
    "  int remaining = 0;\n"
    "  bool exiting = false;\n"
    "  std::exception_ptr error;\n"
    "  std::vector<std::thread> workers;\n"
    "};\n")
sub(H, "  std::unique_ptr<NumaJobDistributor> distributor;\n\n  std::vector<std::unique_ptr<InNumaPool>> numa_worker_pools;\n};\n",
    "  std::unique_ptr<NumaJobDistributor> distributor;\n\n  std::vector<std::unique_ptr<InNumaPool>> numa_worker_pools;\n"
    "  std::unique_ptr<LaneDistributor> lane_distributor;  // KT_SHADOW_LANE; destroyed first\n};\n")
sub(H, "#include <cstdio>\n#include <functional>\n", "#include <cstdio>\n#include <cstdint>\n#include <exception>\n#include <functional>\n")

# ---- worker_pool.cpp -----------------------------------------------------------------------------------------------
C = 'cpu_backend/worker_pool.cpp'
sub(C, "thread_local int WorkerPool::thread_local_id = -1;\n",
    "thread_local int WorkerPool::thread_local_id = -1;\n\n"
    "namespace kt_lane {\n"
    "thread_local int t_lane = 0;\n"
    "thread_local int t_item = 0;\n"
    "std::atomic<int> main_active{0};\n"
    "bool enabled() {\n"
    "  static const bool on = [] { const char* v = getenv(\"KT_SHADOW_LANE\"); return v && strcmp(v, \"1\") == 0; }();\n"
    "  return on;\n"
    "}\n"
    "int from_layer() {\n"
    "  static const int from = [] { const char* v = getenv(\"KT_SHADOW_LANE_FROM\"); return enabled() && v && *v ? atoi(v) : -1; }();\n"
    "  return from;\n"
    "}\n"
    "struct ItemScope {\n"
    "  ItemScope() { t_item = 1; }\n"
    "  ~ItemScope() { t_item = 0; }\n"
    "};\n"
    "}  // namespace kt_lane\n")
sub(C, "#include <cstdio>\n#include <stdexcept>\n", "#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <exception>\n#include <stdexcept>\n")
sub(C, "void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,\n"
       "                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {\n",
    "void InNumaPool::do_work_stealing_job(int task_num, std::function<void(int)> init_func,\n"
    "                                      std::function<void(int)> compute_func, std::function<void(int)> finalize_func) {\n"
    "  if (kt_lane::t_lane) {  // KT_SHADOW_LANE: a lane thread never takes the main job slot\n"
    "    lane_job_run(task_num, init_func, compute_func, finalize_func);\n"
    "    return;\n"
    "  }\n")
sub(C, "                                            std::function<void(int)> finalize_func) {\n  init_func_ = init_func;\n",
    "                                            std::function<void(int)> finalize_func) {\n"
    "  if (kt_lane::t_lane) throw std::runtime_error(\"KT_SHADOW_LANE: async pool job from a lane thread\");\n"
    "  init_func_ = init_func;\n")
sub(C, "void InNumaPool::process_tasks(int thread_id) {\n",
    "// KT_SHADOW_LANE: post a job to the lane slot and sleep until pool threads have run every item. A lane item is a leaf\n"
    "// call (it never posts a job), so the main slot, curr_/end_ and thread_local_id stay main's alone.\n"
    "void InNumaPool::lane_job_run(int task_num, std::function<void(int)> init, std::function<void(int)> compute,\n"
    "                              std::function<void(int)> finalize) {\n"
    "  if (finalize != nullptr) throw std::runtime_error(\"KT_SHADOW_LANE: lane jobs take no finalize\");\n"
    "  if (task_num <= 0) return;\n"
    "  LaneJob job;\n"
    "  job.init = init;\n"
    "  job.compute = compute;\n"
    "  job.end = task_num;\n"
    "  lane_job_.store(&job, std::memory_order_seq_cst);\n"
    "  int seen = 0;\n"
    "  while (job.done.load(std::memory_order_acquire) < job.end) {\n"
    "    {\n"
    "      std::unique_lock<std::mutex> lock(lane_mtx_);\n"
    "      lane_cv_.wait_for(lock, std::chrono::milliseconds(1),\n"
    "                        [&] { return job.done.load(std::memory_order_acquire) >= job.end; });\n"
    "    }\n"
    "    const int now = job.done.load(std::memory_order_acquire);\n"
    "    if (now < job.end && now == seen) {  // no item taken for 1 ms: pool threads asleep or main busy\n"
    "      while (lane_run_one(0)) {\n"
    "      }\n"
    "    }\n"
    "    seen = job.done.load(std::memory_order_acquire);\n"
    "  }\n"
    "  lane_job_.store(nullptr, std::memory_order_seq_cst);\n"
    "  while (lane_users_.load(std::memory_order_seq_cst) != 0) _mm_pause();\n"
    "  if (job.error) std::rethrow_exception(job.error);\n"
    "}\n\n"
    "bool InNumaPool::lane_run_one(int thread_id) {\n"
    "  if (lane_job_.load(std::memory_order_relaxed) == nullptr) return false;  // lane idle or off\n"
    "  if (kt_lane::main_active.load(std::memory_order_acquire) != 0) return false;\n"
    "  lane_users_.fetch_add(1, std::memory_order_seq_cst);\n"
    "  LaneJob* job = lane_job_.load(std::memory_order_seq_cst);\n"
    "  bool ran = false;\n"
    "  if (job != nullptr) {\n"
    "    const int i = job->curr.fetch_add(1, std::memory_order_acq_rel);\n"
    "    if (i < job->end) {\n"
    "      {\n"
    "        kt_lane::ItemScope scope;\n"
    "        try {\n"
    "          if (job->init) job->init(thread_id);\n"
    "          job->compute(i);\n"
    "        } catch (...) {\n"
    "          std::lock_guard<std::mutex> g(job->error_mtx);\n"
    "          if (!job->error) job->error = std::current_exception();\n"
    "        }\n"
    "      }\n"
    "      ran = true;\n"
    "      if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 >= job->end) {\n"
    "        std::lock_guard<std::mutex> lock(lane_mtx_);\n"
    "        lane_cv_.notify_all();\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  lane_users_.fetch_sub(1, std::memory_order_seq_cst);\n"
    "  return ran;\n"
    "}\n\n"
    "void InNumaPool::process_tasks(int thread_id) {\n")
sub(C, "    } else if (status == ThreadStatus::WAITING) {\n      for (int p = 0; p < kt_spin_pauses(); p++) {\n"
       "        if (thread_state_[thread_id].status.load(std::memory_order_acquire) != ThreadStatus::WAITING) break;\n",
    "    } else if (status == ThreadStatus::WAITING) {\n"
    "      if (lane_run_one(thread_id)) {  // KT_SHADOW_LANE\n"
    "        start = std::chrono::high_resolution_clock::now();\n"
    "        continue;\n"
    "      }\n"
    "      for (int p = 0; p < kt_spin_pauses(); p++) {\n"
    "        if (thread_state_[thread_id].status.load(std::memory_order_acquire) != ThreadStatus::WAITING) break;\n")
sub(C, "    } else if (stat == ThreadStatus::WAITING) {\n      for (int p = 0; p < kt_spin_pauses(); p++) {\n",
    "    } else if (stat == ThreadStatus::WAITING) {\n"
    "      InNumaPool* lane_pool = lane_pools_[numa_id].load(std::memory_order_relaxed);\n"
    "      if (lane_pool != nullptr && lane_pool->lane_run_one(0)) {  // KT_SHADOW_LANE\n"
    "        start = std::chrono::high_resolution_clock::now();\n"
    "        continue;\n"
    "      }\n"
    "      for (int p = 0; p < kt_spin_pauses(); p++) {\n")
sub(C, "void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {\n  this->compute_func = compute_func;\n",
    "void NumaJobDistributor::do_numa_job(std::function<void(int)> compute_func) {\n"
    "  if (kt_lane::t_lane) {  // KT_SHADOW_LANE\n"
    "    if (lane_ == nullptr) throw std::runtime_error(\"KT_SHADOW_LANE: lane job without a lane distributor\");\n"
    "    lane_->do_numa_job(compute_func);\n"
    "    return;\n"
    "  }\n"
    "  this->compute_func = compute_func;\n", count=2)
sub(C, "void NumaJobDistributor::worker_thread(int numa_id) {\n",
    "void NumaJobDistributor::set_lane(LaneDistributor* lane, const std::vector<InNumaPool*>& pools) {\n"
    "  if ((int)pools.size() != numa_count || numa_count > 64) throw std::runtime_error(\"KT_SHADOW_LANE: pool count mismatch\");\n"
    "  lane_ = lane;\n"
    "  for (int i = 0; i < numa_count; i++) lane_pools_[i].store(pools[i], std::memory_order_release);\n"
    "}\n\n"
    "LaneDistributor::LaneDistributor(std::vector<int> ids) : numa_count((int)ids.size()), numa_ids(ids) {\n"
    "  for (int i = 0; i < numa_count; i++) workers.emplace_back(&LaneDistributor::worker_thread, this, i);\n"
    "}\n\n"
    "LaneDistributor::~LaneDistributor() {\n"
    "  {\n"
    "    std::lock_guard<std::mutex> g(mtx);\n"
    "    exiting = true;\n"
    "  }\n"
    "  cv_start.notify_all();\n"
    "  for (auto& t : workers)\n"
    "    if (t.joinable()) t.join();\n"
    "}\n\n"
    "void LaneDistributor::do_numa_job(std::function<void(int)> f) {\n"
    "  std::unique_lock<std::mutex> lock(mtx);\n"
    "  compute_func = f;\n"
    "  remaining = numa_count;\n"
    "  error = nullptr;\n"
    "  generation++;\n"
    "  cv_start.notify_all();\n"
    "  cv_done.wait(lock, [&] { return remaining == 0; });\n"
    "  compute_func = nullptr;\n"
    "  if (error) {\n"
    "    std::exception_ptr e = error;\n"
    "    error = nullptr;\n"
    "    std::rethrow_exception(e);\n"
    "  }\n"
    "}\n\n"
    "void LaneDistributor::worker_thread(int i) {\n"
    "  kt_lane::t_lane = 1;\n"
    "  set_memory_to_numa(numa_ids[i]);\n"
    "  char name[16];\n"
    "  snprintf(name, sizeof(name), \"lane_%d\", numa_ids[i]);\n"
    "  pthread_setname_np(pthread_self(), name);\n"
    "  uint64_t seen = 0;\n"
    "  std::unique_lock<std::mutex> lock(mtx);\n"
    "  while (true) {\n"
    "    cv_start.wait(lock, [&] { return exiting || generation != seen; });\n"
    "    if (exiting) return;\n"
    "    seen = generation;\n"
    "    std::function<void(int)> f = compute_func;\n"
    "    lock.unlock();\n"
    "    std::exception_ptr e;\n"
    "    try {\n"
    "      f(i);\n"
    "    } catch (...) {\n"
    "      e = std::current_exception();\n"
    "    }\n"
    "    lock.lock();\n"
    "    if (e && !error) error = e;\n"
    "    if (--remaining == 0) cv_done.notify_all();\n"
    "  }\n"
    "}\n\n"
    "void NumaJobDistributor::worker_thread(int numa_id) {\n")
DIST_COMMENT = "  // distributor = std::move(std::unique_ptr<NumaJobDistributor>(new NumaJobDistributor(config.subpool_numa_map)));\n"
sub(C, "  distributor = std::move(std::unique_ptr<NumaJobDistributor>(\n"
       "      new NumaJobDistributor(config.subpool_numa_map, config.subpool_thread_count)));\n" + DIST_COMMENT + "}\n",
    "  distributor = std::move(std::unique_ptr<NumaJobDistributor>(\n"
    "      new NumaJobDistributor(config.subpool_numa_map, config.subpool_thread_count)));\n" + DIST_COMMENT +
    "  if (kt_lane::enabled()) {  // KT_SHADOW_LANE\n"
    "    lane_distributor = std::make_unique<LaneDistributor>(config.subpool_numa_map);\n"
    "    std::vector<InNumaPool*> pools;\n"
    "    for (auto& p : numa_worker_pools) pools.push_back(p.get());\n"
    "    distributor->set_lane(lane_distributor.get(), pools);\n"
    "    printf(\"[kt-lane] WorkerPool[0x%lx]: shadow lane on, %d parts, lane objects from KT layer %d\\n\", (intptr_t)this,\n"
    "           config.subpool_count, kt_lane::from_layer());\n"
    "  }\n}\n")

# ---- task_queue ----------------------------------------------------------------------------------------------------
QH = 'cpu_backend/task_queue.h'
sub(QH, "  TaskQueue();\n", "  // lane = true: the KT_SHADOW_LANE queue (its thread is a lane thread; it never counts as main work).\n"
                          "  explicit TaskQueue(bool lane = false);\n")
sub(QH, "  std::atomic<Node*> head;\n", "  bool lane_ = false;\n  std::atomic<Node*> head;\n")
QC = 'cpu_backend/task_queue.cpp'
sub(QC, '#include "task_queue.h"\n', '#include "task_queue.h"\n\n#include "worker_pool.h"  // kt_lane\n')
sub(QC, "TaskQueue::TaskQueue() : done(false), pending(0) {\n", "TaskQueue::TaskQueue(bool lane) : lane_(lane), done(false), pending(0) {\n")
sub(QC, "void TaskQueue::worker() {\n  Node* curr = head.load(std::memory_order_relaxed);\n",
    "void TaskQueue::worker() {\n"
    "  if (lane_) {\n"
    "    kt_lane::t_lane = 1;\n"
    "    pthread_setname_np(pthread_self(), \"kt_lane_queue\");\n"
    "  }\n"
    "  Node* curr = head.load(std::memory_order_relaxed);\n")
sub(QC, "        try {\n          next->task();\n        } catch (...) {\n          task_exception = std::current_exception();\n        }\n",
    "        if (!lane_) kt_lane::main_active.fetch_add(1, std::memory_order_acq_rel);  // KT_SHADOW_LANE: lane items wait\n"
    "        try {\n          next->task();\n        } catch (...) {\n          task_exception = std::current_exception();\n        }\n"
    "        if (!lane_) kt_lane::main_active.fetch_sub(1, std::memory_order_acq_rel);\n")

# ---- cpuinfer.h ----------------------------------------------------------------------------------------------------
I = 'cpu_backend/cpuinfer.h'
sub(I, '#include "task_queue.h"\n#include "worker_pool.h"\n', '#include <stdexcept>\n\n#include "task_queue.h"\n#include "worker_pool.h"\n')
sub(I, "  ~CPUInfer() {\n", "  // KT_SHADOW_LANE: a lane CPUInfer on `main`'s WorkerPool, with its own queue (a lane thread) and memop poller.\n"
                           "  CPUInfer(CPUInfer* main, bool lane) : shared_backend_(true) {\n"
                           "    if (!lane || !kt_lane::enabled()) throw std::runtime_error(\"KT_SHADOW_LANE: a lane CPUInfer needs KT_SHADOW_LANE=1\");\n"
                           "    printf(\"CPUInfer[0x%lx]: lane on WorkerPool[0x%lx]\\n\", (intptr_t)this, (intptr_t)main->backend_);\n"
                           "    backend_ = main->backend_;\n"
                           "    task_queue_ = new TaskQueue(true);\n"
                           "    memops_init();\n"
                           "  }\n\n"
                           "  ~CPUInfer() {\n")
sub(I, "    delete backend_;\n    delete task_queue_;\n", "    delete task_queue_;  // the lane queue goes first: nothing it runs may outlive the pool\n"
                                                      "    if (!shared_backend_) delete backend_;\n")
sub(I, " public:\n  WorkerPool* backend_;\n  TaskQueue* task_queue_;\n",
    " public:\n  WorkerPool* backend_;\n  TaskQueue* task_queue_;\n  bool shared_backend_ = false;  // KT_SHADOW_LANE: backend_ belongs to the main CPUInfer\n")

# ---- shared_mem_buffer.h / moe_base.hpp ----------------------------------------------------------------------------
S = 'cpu_backend/shared_mem_buffer.h'
sub(S, "static SharedMemBufferNuma shared_mem_buffer_numa;\n",
    "static SharedMemBufferNuma shared_mem_buffer_numa;\n"
    "// KT_SHADOW_LANE: lane objects' part scratch, never the bytes main layers use.\n"
    "static SharedMemBufferNuma shared_mem_buffer_numa_lane;\n")
M = 'operators/avx2/moe_base.hpp'
sub(M, "    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);\n",
    "    (lane_object() ? shared_mem_buffer_numa_lane : shared_mem_buffer_numa).alloc(tp_part_idx, this, mem_requests);\n")
sub(M, "    shared_mem_buffer_numa.dealloc(tp_part_idx, this);\n",
    "    (lane_object() ? shared_mem_buffer_numa_lane : shared_mem_buffer_numa).dealloc(tp_part_idx, this);\n")
sub(M, "  void init() {\n    if (config_.load && config_.path == \"\") {\n",
    "  // KT_SHADOW_LANE: an object at or above KT_SHADOW_LANE_FROM keeps its scratch apart from every main layer's.\n"
    "  bool lane_object() const { return kt_lane::lane_layer(config_.layer_idx); }\n\n"
    "  void init() {\n    if (config_.load && config_.path == \"\") {\n")

# ---- contract_v2-moe.hpp -------------------------------------------------------------------------------------------
V = 'operators/avx2/contract_v2-moe.hpp'
sub(V, "#define V2_TRUNC_HOOK() (::kt_contract_v2::g_truncation_events.fetch_add(1, std::memory_order_relaxed))\n",
    "// KT_SHADOW_LANE: events on lane threads and inside lane items count apart, so the target's check never sees them.\n"
    "static std::atomic<long> g_lane_truncation_events{0};\n"
    "inline std::atomic<long>& trunc_counter() {\n"
    "  return (kt_lane::t_lane || kt_lane::t_item) ? g_lane_truncation_events : g_truncation_events;\n"
    "}\n"
    "#define V2_TRUNC_HOOK() (::kt_contract_v2::trunc_counter().fetch_add(1, std::memory_order_relaxed))\n")
sub(V, "inline long kt_contract_v2_truncation_events() { return kt_contract_v2::g_truncation_events.load(); }\n",
    "inline long kt_contract_v2_truncation_events() { return kt_contract_v2::trunc_counter().load(); }\n")
sub(V, "      kt_contract_v2::g_truncation_events.fetch_sub(now - before, std::memory_order_relaxed);\n",
    "      kt_contract_v2::trunc_counter().fetch_sub(now - before, std::memory_order_relaxed);\n")
sub(V, "    if (lost) kt_contract_v2::g_truncation_events.fetch_add(lost, std::memory_order_relaxed);\n",
    "    if (lost) kt_contract_v2::trunc_counter().fetch_add(lost, std::memory_order_relaxed);\n")
sub(V, "  static RouteScratch& route_scratch(int part) {\n    static RouteScratch scratch[64];\n"
       "    if (part < 0 || part >= 64) throw std::runtime_error(\"contract v2: part index out of range\");\n    return scratch[part];\n",
    "  static RouteScratch& route_scratch(int part, bool lane = false) {  // lane: KT_SHADOW_LANE objects' own scratch\n"
    "    static RouteScratch scratch[64], lane_scratch[64];\n"
    "    if (part < 0 || part >= 64) throw std::runtime_error(\"contract v2: part index out of range\");\n"
    "    return lane ? lane_scratch[part] : scratch[part];\n")
sub(V, "    auto& s = route_scratch(tp_part_idx);\n    if (!lazy_route_scratch()) s.reserve(routes, (size_t)config_.hidden_size);\n",
    "    auto& s = route_scratch(tp_part_idx, this->lane_object());\n    if (!lazy_route_scratch()) s.reserve(routes, (size_t)config_.hidden_size);\n")
sub(V, "      auto& s = route_scratch(tp_part_idx);\n      s.reserve((size_t)(lazy_route_scratch() ? qlen : config_.max_len) * k, (size_t)config_.hidden_size);\n",
    "      auto& s = route_scratch(tp_part_idx, this->lane_object());\n      s.reserve((size_t)(lazy_route_scratch() ? qlen : config_.max_len) * k, (size_t)config_.hidden_size);\n")

# ---- ext_bindings.cpp ----------------------------------------------------------------------------------------------
E = 'ext_bindings.cpp'
sub(E, "  py::class_<CPUInfer>(m, \"CPUInfer\")\n      .def(py::init<int>())\n      .def(py::init<WorkerPoolConfig>())\n",
    "  py::class_<CPUInfer>(m, \"CPUInfer\")\n      .def(py::init<int>())\n      .def(py::init<WorkerPoolConfig>())\n"
    "      // KT_SHADOW_LANE: CPUInfer(main, True), a lane on main's WorkerPool; keeps main alive\n"
    "      .def(py::init<CPUInfer*, bool>(), py::keep_alive<1, 2>())\n")
sub(E, "  m.def(\"contract_v2_truncation_events\", &kt_contract_v2_truncation_events);\n",
    "  m.def(\"contract_v2_truncation_events\", &kt_contract_v2_truncation_events);\n"
    "  m.def(\"shadow_lane_enabled\", &kt_lane::enabled);\n"
    "  m.def(\"shadow_lane_from\", &kt_lane::from_layer);\n"
    "  m.def(\"contract_v2_lane_truncation_events\", [] { return kt_contract_v2::g_lane_truncation_events.load(); });\n")

for rel, s in text.items():
    (root / rel).write_bytes(s.encode('utf-8'))
    print(rel, hashlib.sha256(s.encode('utf-8')).hexdigest())
