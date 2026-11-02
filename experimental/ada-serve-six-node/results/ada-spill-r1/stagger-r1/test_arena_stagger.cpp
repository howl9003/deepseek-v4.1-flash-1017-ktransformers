// Layout test of KT_EXPERT_ARENA_SPILL_STAGGER (Claude, 31 Oct 2026). No model, no pages touched: it reads each extent's mempolicy back.
// Selector off must match the accepted header extent for extent; on must keep every node's bytes and put at most one part on a spill node at a position.
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace accepted {
#include ACCEPTED_HEADER
}
namespace candidate {
#include CANDIDATE_HEADER
}

static const size_t HP = 2UL << 20, STRIPE = 64UL << 20, SIZE = STRIPE * 7 + (30UL << 20);
static const unsigned PARTS[] = {0, 1, 2, 4, 5, 6};
static int fails = 0;

static void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

using Layout = std::vector<int>;

template <typename F>
static Layout layout(unsigned node, F call) {
  unsigned long mask = 1UL << node;
  if (set_mempolicy(MPOL_BIND, &mask, 64)) throw std::runtime_error("set_mempolicy");
  int fd = memfd_create("stagger-test", 0);
  if (fd < 0 || ftruncate(fd, SIZE)) throw std::runtime_error("memfd");
  void* p = mmap(nullptr, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) throw std::runtime_error("mmap");
  call(p, SIZE);
  if (set_mempolicy(MPOL_DEFAULT, nullptr, 0)) throw std::runtime_error("reset mempolicy");
  Layout out;
  for (size_t off = 0; off < SIZE; off += HP) {
    int mode = -1;
    unsigned long got = 0;
    if (get_mempolicy(&mode, &got, 64, static_cast<char*>(p) + off, MPOL_F_ADDR)) throw std::runtime_error("get_mempolicy");
    if (mode != MPOL_BIND || !got || (got & (got - 1))) throw std::runtime_error("extent without a single-node bind");
    out.push_back(__builtin_ctzl(got));
  }
  munmap(p, SIZE);
  close(fd);
  return out;
}

static std::map<int, size_t> totals(const std::map<unsigned, Layout>& all, size_t end) {
  std::map<int, size_t> t;
  for (const auto& kv : all)
    for (size_t i = 0; i < end; ++i) ++t[kv.second[i]];
  return t;
}

int main() {
  if (numa_available() < 0) throw std::runtime_error("no numa");
  setenv("KT_EXPERT_ARENA_SPILL_NODES", "3,7", 1);
  setenv("KT_EXPERT_ARENA_REBALANCE_NODE0", "1", 1);
  std::map<unsigned, Layout> base, off, on;
  unsetenv("KT_EXPERT_ARENA_SPILL_STAGGER");
  for (unsigned n : PARTS) {
    base[n] = layout(n, accepted::kt_expert_arena_spill);
    off[n] = layout(n, candidate::kt_expert_arena_spill);
  }
  setenv("KT_EXPERT_ARENA_SPILL_STAGGER", "1", 1);
  for (unsigned n : PARTS) on[n] = layout(n, candidate::kt_expert_arena_spill);
  for (unsigned n : PARTS) check(base[n] == off[n], "selector off: part on node " + std::to_string(n) + " matches the accepted header at every 2 MiB");
  const size_t full = 7 * STRIPE / HP, whole = SIZE / HP;
  for (unsigned n : PARTS)
    check(totals({{n, on[n]}}, full).at(n) == totals({{n, base[n]}}, full).at(n), "part on node " + std::to_string(n) + " keeps its local bytes on full stripes");
  const auto a = totals(base, full), b = totals(on, full);
  for (const auto& kv : a) {
    std::printf("     node %d: accepted %zu MiB, stagger %zu MiB on the full stripes\n", kv.first, kv.second * 2, b.count(kv.first) ? b.at(kv.first) * 2 : 0);
    check(b.count(kv.first) && b.at(kv.first) == kv.second, "node " + std::to_string(kv.first) + " keeps its bytes");
  }
  const auto c = totals(base, whole), d = totals(on, whole);
  for (const auto& kv : c) std::printf("     node %d with the tail: accepted %zu MiB, stagger %zu MiB\n", kv.first, kv.second * 2, d.count(kv.first) ? d.at(kv.first) * 2 : 0);
  for (unsigned n : PARTS) {
    const int same = n < 3 ? 3 : 7, other = n < 3 ? 7 : 3;
    size_t own = 0, cross = 0;
    for (size_t i = 0; i < full; ++i) {
      own += on[n][i] == same;
      cross += on[n][i] == other;
    }
    check(own == 7 * 8 && cross == 0, "part on node " + std::to_string(n) + ": 16 MiB a stripe on its own socket's spill node, none across");
  }
  size_t worst = 0, histogram[7] = {0};
  for (size_t i = 0; i < full; ++i) {
    size_t three = 0, seven = 0;
    for (unsigned n : PARTS) {
      three += on[n][i] == 3;
      seven += on[n][i] == 7;
    }
    worst = std::max(worst, std::max(three, seven));
    ++histogram[three + seven];
  }
  check(worst == 1, "at most one part on a spill node at any position (worst " + std::to_string(worst) + ")");
  std::printf("     share of positions with k parts remote:");
  for (int k = 0; k < 7; ++k) std::printf(" %d:%.3f", k, histogram[k] / double(full));
  std::printf("\n");
  size_t same_base = 0;
  for (size_t i = 0; i < full; ++i) {
    size_t r = 0;
    for (unsigned n : PARTS) r += base[n][i] == 3 || base[n][i] == 7;
    same_base = std::max(same_base, r);
  }
  check(same_base == 6, "the accepted layout has all six parts remote at some position (the fault this removes)");
  setenv("KT_EXPERT_ARENA_SPILL_NODES", "7,3", 1);
  try {
    layout(1, candidate::kt_expert_arena_spill);
    check(true, "spill node order does not matter");
  } catch (const std::exception& e) {
    check(false, std::string("spill node order: ") + e.what());
  }
  setenv("KT_EXPERT_ARENA_SPILL_NODES", "3", 1);
  try {
    layout(1, candidate::kt_expert_arena_spill);
    check(false, "other spill sets are refused");
  } catch (const std::exception&) {
    check(true, "other spill sets are refused");
    set_mempolicy(MPOL_DEFAULT, nullptr, 0);
  }
  std::puts(fails ? "FAILED" : "ALL_PASS");
  return fails ? 1 : 0;
}
