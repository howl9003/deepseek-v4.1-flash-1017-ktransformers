#pragma once
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// Experimental, default off. Keep three quarters of each 64 MiB stripe local
// and split the remaining quarter over explicitly selected RAM nodes.
// Each spill extent uses singleton MPOL_BIND: MPOL_INTERLEAVE permits fallback
// outside its mask before reclaiming clean cache, defeating the RAM reservation.
// Only newly created expert memfd arenas call this; worker affinity, scratch
// allocation, weights, tensor layouts and numerical operations are unchanged.
inline void kt_expert_arena_spill(void* base, size_t size) {
  const char* env = std::getenv("KT_EXPERT_ARENA_SPILL_NODES");
  if (!env || !*env) return;
  unsigned long spill = 0;
  const char* p = env;
  while (*p) {
    if (*p < '0' || *p > '9') throw std::runtime_error("invalid expert spill node list");
    char* end = nullptr;
    errno = 0;
    long node = std::strtol(p, &end, 10);
    if (errno || node < 0 || node >= 8 * sizeof(spill) ||
        !numa_bitmask_isbitset(numa_all_nodes_ptr, node) || (spill & (1UL << node)))
      throw std::runtime_error("invalid, duplicate or unavailable expert spill node");
    spill |= 1UL << node;
    if (!*end) break;
    if (*end != ',' || !end[1]) throw std::runtime_error("invalid expert spill node separator");
    p = end + 1;
  }
  // Distributor threads have a fixed memory binding but may execute on a
  // different CPU. Preserve their assigned memory node, never sched_getcpu().
  unsigned long local = 0;
  int mode = -1;
  if (get_mempolicy(&mode, &local, 8 * sizeof(local), nullptr, 0))
    throw std::runtime_error(std::string("expert arena get_mempolicy: ") + std::strerror(errno));
  if (mode != MPOL_BIND || !local || (local & (local - 1)))
    throw std::runtime_error("expert spill requires a single-node thread memory binding");
  if (local & spill)
    throw std::runtime_error("expert spill nodes must exclude the assigned memory node");
  // Default off; six-worker-node topology only. Node 0 carries the host's
  // extra pinned/kernel allocations. Move an additional eighth of its newly
  // created expert arenas across the other compute memory nodes. Keep the
  // existing quarter on nodes 3/7, every byte/layout, and worker affinity.
  const char* rebalance_env = std::getenv("KT_EXPERT_ARENA_REBALANCE_NODE0");
  const bool rebalance = rebalance_env && std::strcmp(rebalance_env, "1") == 0;
  // Value 3: a sixteenth of node 0's stripes to node 3, its own socket's spill node, in place of the eighth across the compute nodes.
  // The eighth left node 0's part 32% slower than the others; with no rebalance its usable memory fell to the guard in a 35K prefill.
  const bool rebalance_socket = rebalance_env && std::strcmp(rebalance_env, "3") == 0;
  if (rebalance || rebalance_socket) {
    if (spill != ((1UL << 3) | (1UL << 7)))
      throw std::runtime_error("node0 arena rebalance requires spill nodes 3,7");
    for (unsigned n : {0U, 1U, 2U, 4U, 5U, 6U})
      if (!numa_bitmask_isbitset(numa_all_nodes_ptr, n))
        throw std::runtime_error("node0 arena rebalance requires the six-node topology");
  }
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  if (!size || reinterpret_cast<uintptr_t>(base) % page)
    throw std::runtime_error("expert spill requires a nonempty page-aligned arena");
  auto bind = [&](char* address, size_t bytes, int policy, const unsigned long* mask) {
    if (mbind(address, bytes, policy, mask, 8 * sizeof(*mask), 0))
      throw std::runtime_error(std::string("expert arena mbind: ") + std::strerror(errno));
  };
  bind(static_cast<char*>(base), size, MPOL_BIND, &local);
  constexpr size_t stripe = 64UL << 20;
  constexpr size_t hugepage = 2UL << 20;
  // Default off, six-worker-node topology only. With every part spilling at the same stripe offset, half of it across the socket,
  // one spilled id sends all 42 workers to a single node. On: a part spills only to its own socket's spill node, each at its own
  // offset, so a stripe position has at most one part on each spill node. Same bytes per node; affinity, layouts and arithmetic unchanged.
  const char* stagger_env = std::getenv("KT_EXPERT_ARENA_SPILL_STAGGER");
  const bool stagger = stagger_env && std::strcmp(stagger_env, "1") == 0;
  size_t stagger_shift = 0;
  unsigned long stagger_node = 0;
  if (stagger) {
    if (spill != ((1UL << 3) | (1UL << 7)))
      throw std::runtime_error("expert spill stagger requires spill nodes 3,7");
    const unsigned me = __builtin_ctzl(local);
    if (me > 6 || me == 3)
      throw std::runtime_error("expert spill stagger requires a compute memory node");
    stagger_node = 1UL << (me < 3 ? 3 : 7);
    stagger_shift = ((me % 4) * stripe / 3 + (me < 3 ? 0 : stripe / 6)) / hugepage * hugepage;
  }
  // An extent `at` bytes into a stripe of the accepted layout; a stagger rotates it inside a full stripe.
  auto place = [&](size_t offset, size_t bytes, size_t at, size_t length, const unsigned long* mask) {
    const size_t shift = bytes == stripe ? stagger_shift : 0;
    const size_t from = (at + bytes - shift) % bytes, first = std::min(length, bytes - from);
    bind(static_cast<char*>(base) + offset + from, first, MPOL_BIND, mask);
    if (first < length) bind(static_cast<char*>(base) + offset, length - first, MPOL_BIND, mask);
  };
  for (size_t offset = 0; offset < size; offset += stripe) {
    const size_t bytes = std::min(stripe, size - offset);
    // Round the split up to a 2 MiB offset to retain large-page opportunities.
    // Small tail ranges may remain local; this never spills more than 25%.
    const size_t local_bytes = ((bytes - bytes / 4 + hugepage - 1) / hugepage) * hugepage;
    if (rebalance && local == 1UL) {
      constexpr unsigned targets[] = {1, 2, 4, 5, 6};
      // Whole 2MiB extents only; leave small tails local. The old spill
      // regions do not overlap this added region, including a partial stripe.
      const size_t extra = (bytes / 8 / hugepage) * hugepage;
      for (size_t i = 0; i < extra / hugepage; ++i) {
        const unsigned n = targets[((offset / stripe) * 4 + i) % 5];
        const unsigned long target = 1UL << n;
        place(offset, bytes, local_bytes - extra + i * hugepage, hugepage, &target);
      }
    }
    if (rebalance_socket && local == 1UL) {
      const size_t extra = (bytes / 16 / hugepage) * hugepage;
      const unsigned long target = 1UL << 3;
      if (extra) place(offset, bytes, local_bytes - extra, extra, &target);
    }
    if (stagger) {
      if (bytes > local_bytes) place(offset, bytes, local_bytes, bytes - local_bytes, &stagger_node);
      continue;
    }
    size_t cursor = local_bytes;
    unsigned remaining_nodes = __builtin_popcountl(spill);
    for (unsigned node = 0; node < 8 * sizeof(spill) && cursor < bytes; ++node) {
      if (!(spill & (1UL << node))) continue;
      const size_t remaining = bytes - cursor;
      const size_t part = std::min(remaining,
        ((remaining / remaining_nodes + hugepage - 1) / hugepage) * hugepage);
      const unsigned long target = 1UL << node;
      bind(static_cast<char*>(base) + offset + cursor, part, MPOL_BIND, &target);
      cursor += part;
      --remaining_nodes;
    }
  }
}
