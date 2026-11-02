// Layout test of KT_EXPERT_ARENA_REBALANCE_NODE0=3. No model, no pages touched: it reads each extent's mempolicy back. Value 1 must match the accepted header at every
// 2 MiB; value 3 must change node 0's part only, put a sixteenth of its full stripes on node 3 and nothing on another compute node, and leave every part's bytes bound.
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
  int fd = memfd_create("r16-test", 0);
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

static std::map<int, size_t> count(const Layout& l, size_t end) {
  std::map<int, size_t> t;
  for (size_t i = 0; i < end; ++i) ++t[l[i]];
  return t;
}

int main() {
  if (numa_available() < 0) throw std::runtime_error("no numa");
  setenv("KT_EXPERT_ARENA_SPILL_NODES", "3,7", 1);
  setenv("KT_EXPERT_ARENA_SPILL_STAGGER", "1", 1);
  std::map<unsigned, Layout> base, one, three, zero;
  setenv("KT_EXPERT_ARENA_REBALANCE_NODE0", "1", 1);
  for (unsigned n : PARTS) {
    base[n] = layout(n, accepted::kt_expert_arena_spill);
    one[n] = layout(n, candidate::kt_expert_arena_spill);
  }
  setenv("KT_EXPERT_ARENA_REBALANCE_NODE0", "3", 1);
  for (unsigned n : PARTS) three[n] = layout(n, candidate::kt_expert_arena_spill);
  setenv("KT_EXPERT_ARENA_REBALANCE_NODE0", "0", 1);
  for (unsigned n : PARTS) zero[n] = layout(n, candidate::kt_expert_arena_spill);
  for (unsigned n : PARTS) check(base[n] == one[n], "value 1: part on node " + std::to_string(n) + " matches the accepted header");
  for (unsigned n : PARTS)
    if (n != 0) check(three[n] == zero[n], "value 3: part on node " + std::to_string(n) + " equals no rebalance");
  const size_t full = 7 * STRIPE / HP;
  const auto a = count(three[0], full), z = count(zero[0], full);
  for (const auto& kv : a) std::printf("     node 0's part, full stripes: %zu MiB on node %d (no rebalance %zu)\n", kv.second * 2, kv.first, z.count(kv.first) ? z.at(kv.first) * 2 : 0);
  check(a.size() == 2 && a.count(0) && a.count(3), "value 3: node 0's part stays on nodes 0 and 3");
  check(a.at(3) == z.at(3) + 7 * (STRIPE / 16 / HP), "value 3: a sixteenth more on node 3");
  size_t moved = 0;
  for (size_t i = 0; i < three[0].size(); ++i) moved += three[0][i] != zero[0][i];
  check(moved == 7 * (STRIPE / 16 / HP) + ((30UL << 20) / 16 / HP), "value 3: only those extents moved");
  std::printf(fails ? "FAILED %d\n" : "ALL_PASS\n", fails);
  return fails != 0;
}
