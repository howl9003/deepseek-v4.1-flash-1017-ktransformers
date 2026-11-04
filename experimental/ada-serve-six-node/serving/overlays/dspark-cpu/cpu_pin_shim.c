/* Test harness only (LD_PRELOAD for test_draft_forward.py; never part of a launch). kt-kernel's worker pool binds
 * each subpool thread to a NUMA node's CPUs (libnuma numa_bind) and then to one core (hwloc). A CPU test next to the
 * served model must stay on the two cores taskset gave it, so here numa_bind keeps its memory binding and drops the
 * CPU part, and hwloc thread CPU binding is a no-op. Thread scheduling only: no arithmetic, layout or memory policy
 * changes. test_draft_forward.py checks every thread's allowed CPUs after the pool starts and aborts otherwise. */
#include <hwloc.h>
#include <numa.h>
#include <pthread.h>

void numa_bind(struct bitmask *nodes) { numa_set_membind(nodes); }

int hwloc_set_thread_cpubind(hwloc_topology_t topology, pthread_t thread, hwloc_const_cpuset_t set, int flags) {
  (void)topology;
  (void)thread;
  (void)set;
  (void)flags;
  return 0;
}
