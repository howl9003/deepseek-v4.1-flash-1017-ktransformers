/**
 * @Description  : Path B: contract v3 (spec r3, tiers C and D) CPU MoE backend for 05yuki main's kt-kernel.
 *
 * Built only with KT_ENABLE_CONTRACT_V3 and selected only by KT_MXFP4_BACKEND=contract_v3. It IS his AVX2 MXFP4
 * backend for everything but the arithmetic: TP_MOE<AVX2_CONTRACT_V3H_TP<K>> derives from his
 * TP_MOE<AVX2_MXFP4_MOE_TP<K>>, so the loader, the per-NUMA-part slicing (part p holds intermediate columns
 * [p*I/P, (p+1)*I/P) of every expert: gate/up rows and the down projection's K slice), the KT_EXPERT_SHM memfd arenas,
 * expert_arena_infos() and write_weight_scale_to_buffer (his hot lend and streamer read those) are his, unchanged.
 * Each part's BufferB keeps the checkpoint nibbles (b, [n][k/2], low nibble = even k) and FP32 scales (d, [n][G]).
 *
 * The arithmetic is oracle_v3_r3.py (sha256 bff0374c...), through the v3 CPU library (operators/avx2/contract_v3/
 * v3_rm.{h,cpp}, compiled without fast-math):
 *   BF16 token rows -> int16 (C) / int8 (D) per 32 with a power-of-two scale (once per token, shared by all parts)
 *   -> per part, per routed expert: gate/up over the part's columns (exact integer block dots, FP32 chain over all K
 *   blocks ascending, BF16) -> SwiGLU (table SiLU, routing weight) -> BF16 -> quantiser over the part's 32-blocks
 *   -> down projection as raw exact int64 sums on the layer grid bL over the part's blocks, added straight into the
 *   part's per-token int64 sums (exact, order-free) -> the parts' token sums added exactly.
 * forward_partials writes that raw int64 [qlen, hidden] (spec r3: no per-expert finish; the GPU side adds every other
 * device's raw partial and floor(y_s * 2^-bL), then finishes once). forward() finishes the CPU-only sum to BF16
 * (warm-up and tests only).
 *
 * Byte scales (12 Nov, BYTE_SCALES_r1.md): with KT_ARENA_SCALE_BYTES=1 his loader keeps each group scale's E8M0 byte in
 * the scale arenas instead of the FP32 (mxfp4-moe.hpp), and the codes here are built from those bytes
 * (v3_rm_e8m0_to_codes) instead of from the FP32 (v3_rm_scales_to_codes). The codes are the same bytes either way.
 *
 * Scale codes (12 Nov, RAM_BANDWIDTH_r1.md): each part's E8M0 code tables are one anonymous mapping that the part's own
 * node-bound pool thread allocates, binds (preferred, not strict) to its node and fills, so a part reads its codes
 * from its own node. Before, the calling thread built them and the process's interleave policy spread them over every
 * node. load_weights() samples the pages and prints one [v3-codes] line a layer; KT_V3_CODES_STRICT=1 refuses a spill
 * (a diagnostic, off by default); KT_V3_CODES_LOCAL=0 restores the old placement. The code bytes are the same.
 *
 * The clean layout (12 Nov, RAM_BANDWIDTH_r1.md), KT_V3_CLEAN=1|2|3, off by default, for calls of at most
 * KT_V3_CLEAN_MAXQ rows (default 8: decode; longer calls keep the served layout). The same kernels on the same
 * operands in one dispatch to the parts: each part quantises the token rows itself (no whole-pool quantiser job, no
 * serial gather on the calling thread), runs gate + up + SwiGLU + quantiser as ONE job of (expert, 32-column block)
 * tasks, then the down job, whose tasks also zero their own columns of the part's token sums. The parts' sums are then
 * added by the node-0 pool as before, parts ascending, the first part assigning (the output is not zeroed first):
 *   1  the sum reads the parts' working sums;
 *   2  each down task first copies its columns to a hand-over buffer on the part's node with non-temporal stores, and
 *      the sum reads that: the part's working lines never leave the part's caches;
 *   3  as 2 with the hand-over buffers on the summing pool's node.
 * The output is the same raw int64 [qlen, hidden]: every value is the same exact integer sum, so no bit can change and
 * the hand-off (v3_his.py cpu_submit / cpu_sync, the GPU-side add, the cross-rank sum) is untouched.
 *
 * Node scaling (14 Nov, concept-r1 NODE_SCALING_r1.md section 3, branch pathb-nodescale-r1): parts of different sizes
 * in whole 32-column blocks (KT_TP_PART_BLOCKS, his moe-tp.hpp as edited by build_bs.py; his loader at prefix offsets,
 * mxfp4-moe.hpp), and load_weights_image(): the layer from the block-ordered expert image. A column's arithmetic never
 * depends on which part computes it (gate/up per column over all K blocks, SwiGLU per column, the down-input quantiser
 * per 32-block of the part, down as exact int64 sums added with wrapping adds), so the split changes no bit.
 *
 * The pre-read (13 Nov, concept-r1 PREREAD_r1.md), KT_PREREAD=1, off by default: after a decode call has finished and
 * nothing is queued, the task-queue worker's idle hook (cpu_backend/task_queue.h) has the pool's threads read, part by
 * part on their own node, the weight bytes and scale codes of the experts the CPU ran most often in the next layer's last
 * KT_PREREAD_HIST calls (KT_PREREAD_CAP experts at most); it stops at the next enqueue. It only reads: the call's
 * operands, scratch, kernels and order are unchanged, so no bit can change. Unset: the hook is never installed.
 *
 * pathb-cpu-r1 (13 Nov, concept-r1 ZEN3_OPTIMISATION_r1.md section 3.2), each off unless set (see namespace kt_ccd):
 * KT_V3_CCD=1 splits the clean layout's two jobs statically between each part's two L3 domains (from sysfs) and makes the
 * pre-read read each CCD's share on that CCD; KT_V3_THP_COLLAPSE=1 collapses the written arena extents into 2 MiB pages
 * at load; KT_V3_SUM_PART=<p> runs the part sum on subpool p; KT_V3_RW_RED=2 (v3_rw.cpp) picks the row-wise kernel's
 * microcode-free reduction. Each task still writes only its own outputs and every integer is the same, so no bit changes.
 *
 * Refuses: KT_MXFP4_SCALE_CODES or the MXFP4 expert cache (the FP32 scales must be in d), a scale that is not an exact
 * power of two (code 6), no tier (KT_CONTRACT_V3_TIER=C|D), no bL (set_bl or KT_CONTRACT_V3_BL), the int64 grid flag.
 */
#ifndef CPUINFER_OPERATOR_AVX2_CONTRACT_V3H_MOE_H
#define CPUINFER_OPERATOR_AVX2_CONTRACT_V3H_MOE_H
#ifdef KT_ENABLE_CONTRACT_V3

#include <emmintrin.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../cpu_backend/task_queue.h"   // concept-r1 pre-read: kt_tq_idle_hook, kt_tq_last_qwait_ns
#include "../moe-tp.hpp"
#include "contract_v3/v3_rm.h"
#include "contract_v3/v3_rw.h"  // row-wise kernels behind the v3_rm_gemm_bf16 / v3_rm_down_partial calls (KT_V3_RW=1)
#include "mxfp4-moe.hpp"

extern "C" const uint32_t kt_contract_v3_silu_bits[65536];
extern "C" const char kt_contract_v3_silu_sha256[];

namespace kt_contract_v3h {
inline int tier() {
  static const int t = [] {
    const char* v = std::getenv("KT_CONTRACT_V3_TIER");
    if (!v || !*v) return -1;
    if ((v[0] == 'C' || v[0] == 'c') && v[1] == 0) return (int)V3_TIER_C;
    if ((v[0] == 'D' || v[0] == 'd') && v[1] == 0) return (int)V3_TIER_D;
    return -2;
  }();
  if (t < 0) throw std::runtime_error("contract v3: KT_CONTRACT_V3_TIER must be C or D");
  return t;
}
inline int bl_from_env(int layer) {
  const char* v = std::getenv("KT_CONTRACT_V3_BL");
  if (!v) return INT32_MIN;
  const char* p = v;
  while (*p) {
    char* e;
    const long l = std::strtol(p, &e, 10);
    if (e == p || *e != ':') throw std::runtime_error("contract v3: malformed KT_CONTRACT_V3_BL");
    p = e + 1;
    const long b = std::strtol(p, &e, 10);
    if (e == p) throw std::runtime_error("contract v3: malformed KT_CONTRACT_V3_BL");
    if (l == layer) return (int)b;
    p = e;
    if (*p == ',') p++;
    else if (*p) throw std::runtime_error("contract v3: malformed KT_CONTRACT_V3_BL");
  }
  return INT32_MIN;
}
inline void check(int rc, const char* what) {
  if (rc == 0) return;
  throw std::runtime_error(std::string("contract v3: ") + what + " failed, code " + std::to_string(rc));
}
inline bool env_on(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::strcmp(v, "0") != 0;
}
inline const float* silu_table() { return reinterpret_cast<const float*>(kt_contract_v3_silu_bits); }
inline bool env_off(const char* name) {   // set and "0"
  const char* v = std::getenv(name);
  return v && std::strcmp(v, "0") == 0;
}
inline int this_node() {   // NUMA node of the CPU the calling thread runs on, -1 if unknown
  unsigned cpu = 0, node = 0;
  return syscall(SYS_getcpu, &cpu, &node, nullptr) == 0 ? (int)node : -1;
}
// prefer `node` for the pages of [p, p + len): they fall back to other nodes only when the node is full
inline bool prefer_node(void* p, size_t len, int node) {
  if (node < 0 || node > 62) return false;
  unsigned long mask = 1ul << node;
  return syscall(SYS_mbind, p, len, 1 /* MPOL_PREFERRED */, &mask, 64ul, 0u) == 0;
}
// of up to `want` evenly spaced pages of [p, p + len): how many are on `node`; sampled = 0 if the kernel would not say
inline void pages_on_node(const void* p, size_t len, int node, int want, long& sampled, long& on_node, std::string& elsewhere) {
  sampled = on_node = 0;
  elsewhere.clear();
  const size_t pg = 4096, np = len / pg;
  if (!np || node < 0) return;
  const size_t n = std::min<size_t>(np, (size_t)want);
  std::vector<void*> addr(n);
  std::vector<int> st(n, -1);
  for (size_t i = 0; i < n; i++) addr[i] = (void*)((const char*)p + (np * i / n) * pg);
  if (syscall(SYS_move_pages, 0, (unsigned long)n, addr.data(), nullptr, st.data(), 0) != 0) return;
  int other[64] = {0};
  long unknown = 0;
  for (size_t i = 0; i < n; i++) {
    if (st[i] == node) on_node++;
    else if (st[i] >= 0 && st[i] < 64) other[st[i]]++;
    else unknown++;
  }
  sampled = (long)n;
  for (int i = 0; i < 64; i++)
    if (other[i]) elsewhere += " N" + std::to_string(i) + ":" + std::to_string(other[i]);
  if (unknown) elsewhere += " ?:" + std::to_string(unknown);
}
}  // namespace kt_contract_v3h

// ---- concept-r1 pre-read (PREREAD_r1.md, 13 Nov 2026), KT_PREREAD=1, off by default ----
// In the task-queue worker's idle wait after a decode call (kt_tq_idle_hook, task_queue.h), the pool's own threads read
// the weight bytes of the experts the CPU ran most often in the next layer's last KT_PREREAD_HIST calls, so they are in
// the node's caches when that layer's call comes. Every part reads only its own slice (its node's memory), every thread a
// fixed share; plain loads, rank order, in KT_PREREAD_CHUNK_KB chunks with the queue's pending count checked between
// chunks. Nothing the call reads is written: the call's operands, scratch and code path are unchanged. Nothing is
// allocated: fixed arrays only, and the two pool jobs carry functors that fit std::function's local storage.
namespace kt_preread {
constexpr int kMaxE = 512;       // experts a layer the predictor tracks (more: the pre-read stays off for that layer)
constexpr int kMaxHist = 16;     // calls of history a layer
constexpr int kMaxRoutes = 64;   // distinct CPU experts a call it records (more are dropped from the record)
constexpr int kMaxPlan = 32;     // experts a window
constexpr int kMaxParts = 16;    // parts (subpools)
constexpr int kMaxThreads = 32;  // threads a part
constexpr int kMaxLayers = 256;  // layer indices
constexpr int kLog = 16384;      // window records kept for tests

inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);   // the clock Python's time.monotonic_ns() reads
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
inline int env_int(const char* name, int dflt, int lo, int hi) {
  const char* v = std::getenv(name);
  if (!v || !*v) return dflt;
  const int x = std::atoi(v);
  return x < lo ? lo : x > hi ? hi : x;
}
struct Knobs {
  bool env_on;    // KT_PREREAD=1 (any value but "0")
  int cap;        // KT_PREREAD_CAP: experts a window, the slider (default 12, 0..32)
  int topn;       // KT_PREREAD_TOPN: length of the frequency list (default 12, 1..32)
  int hist;       // KT_PREREAD_HIST: calls of history (default 8, 1..16)
  int maxq;       // KT_PREREAD_MAXQ: calls of more rows neither record history nor start a pre-read (default 8)
  size_t chunk;   // KT_PREREAD_CHUNK_KB * 1024: bytes read between two checks of the pending count (default 8 KiB)
};
inline const Knobs& knobs() {
  static const Knobs k = [] {
    Knobs r;
    const char* v = std::getenv("KT_PREREAD");
    r.env_on = v && *v && std::strcmp(v, "0") != 0;
    r.cap = env_int("KT_PREREAD_CAP", 12, 0, kMaxPlan);
    r.topn = env_int("KT_PREREAD_TOPN", 12, 1, kMaxPlan);
    r.hist = env_int("KT_PREREAD_HIST", 8, 1, kMaxHist);
    r.maxq = env_int("KT_PREREAD_MAXQ", 8, 1, 64);
    r.chunk = (size_t)env_int("KT_PREREAD_CHUNK_KB", 8, 1, 256) * 1024;
    return r;
  }();
  return k;
}
// armed: the call records which layer ran (two stores) and the hook is installed. From KT_PREREAD, or the test binding.
// mode: 0 off (the hook returns at once), 1 history, 2 the test's hinted list (the "actual set" ceiling).
inline std::atomic<bool>& armed() {
  static std::atomic<bool> a{knobs().env_on};
  return a;
}
inline std::atomic<int>& mode() {
  static std::atomic<int> m{knobs().env_on ? 1 : 0};
  return m;
}
// non-queue callers of forward_partials waiting for the shared mutex while a pre-read holds it (tests, warm-up)
inline std::atomic<int>& outside_waiters() {
  static std::atomic<int> n{0};
  return n;
}
inline thread_local bool on_hook_thread = false;
// the one CPU the calling thread may run on, -1 if it may run on several (not pinned)
inline int pinned_cpu() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) != 0 || CPU_COUNT(&set) != 1) return -1;
  for (int c = 0; c < CPU_SETSIZE; c++)
    if (CPU_ISSET(c, &set)) return c;
  return -1;
}

// one thread's account of one window: written only by that thread (own cache line), read by the hook after the join
struct alignas(64) ThreadStat {
  uint64_t bytes = 0, t_in = 0, t_seen = 0, t_out = 0, seq = 0;   // seq: the window it belongs to
  uint32_t experts = 0, stopped = 0;
};
// one window, for tests (contract_v3_preread_log)
struct Rec {
  uint64_t t_start = 0, t_dispatch = 0, t_seen = 0, t_end = 0, bytes = 0;
  int32_t layer = -1, target = -1, planned = 0, experts_min = 0, experts_max = 0, reason = 0;
  int32_t threads = -1;   // -1: the threads' accounts were never summed into this record
};
enum Reason { R_NONE = 0, R_STOPPED = 1, R_DONE = 2, R_STOP_BEFORE = 3 };

// read [p, p + n): one 16-byte load in every 64-byte line, no store anywhere
inline void touch(const uint8_t* p, size_t n) {
  if (!n) return;
  __m128i a = _mm_setzero_si128(), b = a;
  size_t i = 0;
  for (; i + 128 + 16 <= n; i += 128) {
    a = _mm_or_si128(a, _mm_loadu_si128((const __m128i*)(p + i)));
    b = _mm_or_si128(b, _mm_loadu_si128((const __m128i*)(p + i + 64)));
  }
  for (; i + 16 <= n; i += 64) a = _mm_or_si128(a, _mm_loadu_si128((const __m128i*)(p + i)));
  uint8_t t = p[n - 1];   // the last line, whatever the alignment
  if (i < n) t |= p[i];
  a = _mm_or_si128(a, b);
  __asm__ __volatile__("" : : "x"(a), "r"(t) : "memory");
}
}  // namespace kt_preread

// ---- pathb-cpu-r1 (13 Nov 2026, ZEN3_OPTIMISATION_r1.md section 3.2): switches, each off unless set ----
//   KT_V3_CCD=1           the clean layout's two jobs split statically between the two L3 domains (CCDs) of each part,
//                         work stealing only inside a CCD, and the pre-read reads each CCD's share on that CCD. The CPUs
//                         of each CCD come from sysfs (cache index level 3, shared_cpu_list) for the CPU each pool thread
//                         is pinned to; any other layout (a thread not pinned to one CPU, one L3 or more than two in a
//                         part, sysfs unreadable) leaves that part on today's path, with one [kt-ccd] line saying why.
//   KT_V3_THP_COLLAPSE=1  at load, MADV_COLLAPSE on every 2 MiB extent that lies wholly inside the written expert bytes of
//                         a part's weight arenas (and its code region), on the part's own node-bound thread; no root, no
//                         sysctl; the bytes are copied by the kernel, unchanged.
//   KT_V3_SUM_PART=<p>    the clean layout's part sum runs on subpool p (and mode 3's hand-over buffers sit on p's node)
//                         instead of subpool 0; out of range: 0. Same sums, same order.
namespace kt_ccd {
inline std::atomic<int>& on() {   // -1: not read yet
  static std::atomic<int> v{-1};
  return v;
}
inline bool enabled() {
  int v = on().load(std::memory_order_relaxed);
  if (v < 0) {
    const char* e = std::getenv("KT_V3_CCD");
    v = (e && e[0] == '1' && e[1] == 0) ? 1 : 0;
    on().store(v, std::memory_order_relaxed);
  }
  return v == 1;
}
inline bool thp_collapse_on() {
  static const bool v = [] { const char* e = std::getenv("KT_V3_THP_COLLAPSE"); return e && e[0] == '1' && e[1] == 0; }();
  return v;
}
inline int sum_part_env() {
  static const int v = [] { const char* e = std::getenv("KT_V3_SUM_PART"); return (e && *e) ? std::atoi(e) : 0; }();
  return v;
}
// the first share of n items split between two groups of n0 and T - n0 threads: round(n * n0 / T), within [1, n - 1]
inline int share(int n, int n0, int T) {
  int a = (int)((2ll * n * n0 + T) / (2ll * T));
  return a < 1 ? 1 : a > n - 1 ? n - 1 : a;
}
// the L3 of `cpu`: the lowest CPU of its shared_cpu_list (sysfs cache leaf with level 3, type Unified); -1 if unknown
inline int l3_of_cpu(int cpu) {
  char path[160], buf[512];
  for (int idx = 0; idx < 8; idx++) {
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu, idx);
    FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    const bool l3 = std::fgets(buf, sizeof buf, f) && std::atoi(buf) == 3;
    std::fclose(f);
    if (!l3) continue;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/type", cpu, idx);
    if ((f = std::fopen(path, "r")) == nullptr) return -1;
    const bool unified = std::fgets(buf, sizeof buf, f) && std::strncmp(buf, "Unified", 7) == 0;
    std::fclose(f);
    if (!unified) continue;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", cpu, idx);
    if ((f = std::fopen(path, "r")) == nullptr) return -1;
    const bool got = std::fgets(buf, sizeof buf, f) != nullptr;
    std::fclose(f);
    if (!got) return -1;
    int lo = -1;
    for (const char* q = buf; *q;) {   // "a-b,c,d-e": the lowest number
      while (*q && (*q < '0' || *q > '9')) q++;
      if (!*q) break;
      char* end;
      const long x = std::strtol(q, &end, 10);
      if (lo < 0 || x < lo) lo = (int)x;
      q = end;
    }
    return lo;
  }
  return -1;
}
// one part's two-CCD layout (ok = 1), learned once per pool
struct Part {
  int ok = 0;                      // 1: split; 0: today's path
  int T = 0, n[2] = {0, 0};        // threads in all and per CCD (CCD 0 = the L3 of thread 0, the part's distributor)
  int l3[2] = {-1, -1};            // lowest CPU of each L3
  int8_t ccd[kt_preread::kMaxThreads] = {};
  int8_t rank[kt_preread::kMaxThreads] = {};   // the thread's index among its CCD's threads (thread ids ascending)
  char why[160] = {};
};
struct alignas(64) Ctr {             // one CCD's task counter, on its own line
  std::atomic<int> v{0};
};
struct ThpStat {
  long asked = 0, ok = 0, failed = 0, eagain = 0, enomem = 0, einval = 0, other = 0;
};
}  // namespace kt_ccd

// Tag part type: his MXFP4 part, unchanged (it only selects the TP_MOE specialization below).
template <class K = avx2::GemmKernelAVX2MXFP4>
class AVX2_CONTRACT_V3H_TP : public AVX2_MXFP4_MOE_TP<K> {
 public:
  using AVX2_MXFP4_MOE_TP<K>::AVX2_MXFP4_MOE_TP;
};

template <typename K>
class TP_MOE<AVX2_CONTRACT_V3H_TP<K>> : public TP_MOE<AVX2_MXFP4_MOE_TP<K>> {
 public:
  using Base = TP_MOE<AVX2_MXFP4_MOE_TP<K>>;
  using Base::Base;
  int bL_ = INT32_MIN;

  // per part, per expert: panel-order E8M0 codes of gate, up, down (from the part's FP32 scales, once at load). The
  // bytes of one part are one anonymous mapping made and filled on that part's node-bound pool thread (code_regions_).
  struct Codes { const uint8_t *g = nullptr, *u = nullptr, *d = nullptr; };
  struct CodeRegion {
    uint8_t* p = nullptr;
    size_t n = 0;
    int node = -1;          // the node the part's thread ran on
    bool bound = false;     // mbind(MPOL_PREFERRED) took
    long sampled = 0, on_node = 0;
    std::string elsewhere;  // sampled pages on other nodes, " N<node>:<count>"...
  };
  std::vector<std::vector<Codes>> codes_;
  std::vector<CodeRegion> code_regions_;
  void free_codes() {
    for (auto& r : code_regions_)
      if (r.p) munmap(r.p, r.n);
    code_regions_.clear();
    codes_.clear();
  }
  ~TP_MOE() {
    // concept-r1 pre-read: only once the hook was ever installed (KT_PREREAD unset: never); see pr_forget
    if (kt_tq_idle_hook.load(std::memory_order_acquire) != nullptr) pr_forget();
    free_codes();
  }
  // Scratch: ONE process-wide set shared by every layer's instance (r3c, 9 Nov: per-instance grow-only scratch × 40
  // layers × 8 parts took about 0.5 GB a layer at qlen 1024, which is what dropped MemAvailable below the 64 GiB guard
  // at the first prefill). Calls are serialised by the mutex and processed in token chunks of at most rows_cap() rows,
  // so the scratch is bounded (about 0.1 GB) whatever qlen is. Rows are independent and every token sum is an exact
  // int64 sum, so chunking cannot change a bit.
  struct PartScratch { std::vector<uint16_t> gate, up; std::vector<uint8_t> qd; std::vector<int32_t> ead; std::vector<int64_t> tok; };
  // the clean layout's per-part scratch: grown only on the part's own thread
  struct CleanScratch {
    std::vector<uint16_t> gate, up;
    std::vector<uint8_t> qd, qx, qxe;
    std::vector<int32_t> ead, eax, eaxe;
    std::vector<int64_t> tok;
    int64_t* hand = nullptr;             // hand-over buffer: page aligned, hand_words int64, preferring hand_node
    size_t hand_words = 0, hand_bytes = 0;
    int hand_node = -2;
  };
  struct Shared {
    std::mutex mu;
    std::vector<uint8_t> qx_;            // token rows quantised, [rows][H] int16 (sized for C)
    std::vector<int32_t> eax_;           // [rows][H/32]
    std::vector<uint8_t> qxe_;           // gathered per expert, [routes][H] int16
    std::vector<int32_t> eaxe_;          // [routes][H/32]
    std::vector<int> e_off_, e_cnt_, e_tok_;   // per expert row offset / count; per gathered row its token
    std::vector<float> e_w_;             // per gathered row its routing weight
    std::vector<int> active_;
    std::vector<PartScratch> ps_;
    std::vector<CleanScratch> cs_;       // the clean layout's scratch, one per part
    std::vector<int> cs_node_;           // the node each part's distributor thread runs on
    const void* cs_pool_ = nullptr;      // the pool cs_ / cs_node_ were made for
    // concept-r1 pre-read (PREREAD_r1.md). Fixed storage; used under mu by the hook on the task-queue thread, except
    // pr_thr (each pool thread its own line) and pr_ctx (read by the pool threads during the window).
    TP_MOE* pr_last = nullptr;           // the instance whose call ran last (set by the call, consumed by the hook)
    int pr_last_q = 0;                   // its rows
    int pr_prev_layer = -1;              // the layer of the decode call before it
    int16_t pr_next1[kt_preread::kMaxLayers] = {};   // learned successor + 1 of each layer (0: unknown)
    TP_MOE* pr_reg[kt_preread::kMaxLayers] = {};     // the live instance of each layer index
    uint32_t pr_gen = 0;                 // planning scratch
    uint32_t pr_stamp[kt_preread::kMaxE] = {};
    uint16_t pr_cnt[kt_preread::kMaxE] = {};
    int16_t pr_first[kt_preread::kMaxE] = {};
    int16_t pr_cand[kt_preread::kMaxE] = {};
    int pr_plan[kt_preread::kMaxPlan] = {};
    struct PrCtx {
      TP_MOE* self = nullptr;
      const std::atomic<size_t>* pending = nullptr;
      int n = 0;
      size_t chunk = 0;
      uint64_t seq = 0;
    } pr_ctx;
    uint64_t pr_seq = 0;
    uint64_t pr_agg_seq = 0, pr_agg_idx = 0;   // the window whose thread accounts are not yet in its record
    int pr_agg_parts = 0;
    int pr_agg_T[kt_preread::kMaxParts] = {};
    const void* pr_layout_pool = nullptr;   // the pool the layout check (pr_layout) last ran for
    int pr_layout_ok = -1;                  // -1 not checked, 0 refused, 1 ok
    int pr_lay_cpu[kt_preread::kMaxParts][kt_preread::kMaxThreads] = {};
    kt_preread::ThreadStat pr_thr[kt_preread::kMaxParts][kt_preread::kMaxThreads];
    kt_preread::Rec pr_log[kt_preread::kLog];
    uint64_t pr_log_n = 0;               // windows recorded since the last contract_v3_preread_log()
    // pathb-cpu-r1 KT_V3_CCD: each part's two-CCD layout, learned once per pool (ensure_ccd, under mu), and two task
    // counters a part (reset by the part's distributor before each job)
    const void* ccd_pool_ = nullptr;
    kt_ccd::Part ccd_[kt_preread::kMaxParts];
    kt_ccd::Ctr ccd_ctr[kt_preread::kMaxParts][2];
    // pathb-cpu-r1 KT_V3_THP_COLLAPSE: totals over every layer loaded so far
    kt_ccd::ThpStat thp_total;
    double thp_seconds = 0;
  };
  static Shared& sh() { static Shared s; return s; }
  static int rows_cap() {
    static const int v = [] { const char* e = std::getenv("KT_CONTRACT_V3_ROWS"); int r = e ? std::atoi(e) : 256; return r < 1 ? 256 : r; }();
    return v;
  }
  std::vector<uint8_t>& qx_ = sh().qx_;
  std::vector<int32_t>& eax_ = sh().eax_;
  std::vector<uint8_t>& qxe_ = sh().qxe_;
  std::vector<int32_t>& eaxe_ = sh().eaxe_;
  std::vector<int>& e_off_ = sh().e_off_;
  std::vector<int>& e_cnt_ = sh().e_cnt_;
  std::vector<int>& e_tok_ = sh().e_tok_;
  std::vector<float>& e_w_ = sh().e_w_;
  std::vector<int>& active_ = sh().active_;
  std::vector<PartScratch>& ps_ = sh().ps_;
  std::vector<CleanScratch>& cs_ = sh().cs_;
  std::vector<int>& cs_node_ = sh().cs_node_;

  // KT_V3_CLEAN (0 served layout, 1 / 2 / 3 clean layout, see the head of this file); set_clean overrides it per instance
  int clean_ = -1;
  static int clean_env() {
    static const int v = [] { const char* e = std::getenv("KT_V3_CLEAN"); const int m = e ? std::atoi(e) : 0; return m < 0 || m > 3 ? 0 : m; }();
    return v;
  }
  static int clean_maxq() {
    static const int v = [] { const char* e = std::getenv("KT_V3_CLEAN_MAXQ"); const int r = e ? std::atoi(e) : 8; return r < 1 ? 8 : r; }();
    return v;
  }
  void contract_v3_set_clean(int mode) {
    if (mode < -1 || mode > 3) throw std::runtime_error("contract v3: clean mode must be -1 (environment), 0, 1, 2 or 3");
    clean_ = mode;
  }
  int contract_v3_clean() { return clean_ >= 0 ? clean_ : clean_env(); }

  void set_bl(int bL) {
    if (bL < -126 || bL > 127) throw std::runtime_error("contract v3: bL out of range");
    bL_ = bL;
  }
  int get_bl() {
    if (bL_ == INT32_MIN) bL_ = kt_contract_v3h::bl_from_env(this->config.layer_idx);
    return bL_;
  }
  int contract_v3_tier() { return kt_contract_v3h::tier(); }
  bool arena_scale_bytes_reader() const override { return true; }   // this backend reads byte arenas
  bool contract_v3_arena_scale_bytes() { return this->arena_scale_bytes(); }
  // sha256-free fingerprint of every code table (FNV-1a 64 over parts, experts, gate / up / down), for tests: equal
  // between FP32 arenas and byte arenas if and only if the codes are the same bytes. pathb-cpu-r1: the tables are
  // pointers into the part's code region (cpusmooth); each is hashed over its n * G bytes, as the vectors of 88b2f54 were
  // (a GPU-only expert has no table and adds nothing), so the value equals the byte-scales module's for the same codes.
  std::string contract_v3_codes_fnv() {
    uint64_t h = 1469598103934665603ull;
    const int E = this->config.expert_num;
    for (int p = 0; p < (int)codes_.size() && p < this->tp_count; p++) {
      auto* tp = part(p);
      for (int e = 0; e < E && e < (int)codes_[p].size(); e++) {
        const Codes& c = codes_[p][e];
        if (!c.g) continue;
        const uint8_t* t[3] = {c.g, c.u, c.d};
        const size_t n[3] = {(size_t)tp->gate_bb_[e]->n * tp->gate_bb_[e]->k_group_count,
                             (size_t)tp->up_bb_[e]->n * tp->up_bb_[e]->k_group_count,
                             (size_t)tp->down_bb_[e]->n * tp->down_bb_[e]->k_group_count};
        for (int s = 0; s < 3; s++)
          for (size_t i = 0; i < n[s]; i++) { h ^= t[s][i]; h *= 1099511628211ull; }
      }
    }
    return std::to_string(h);
  }
  std::string contract_v3_silu_sha256() { return std::string(kt_contract_v3_silu_sha256); }
  std::string contract_v3_spec() { return "r3"; }

  auto* part(int i) { return this->tps[i].get(); }   // AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>*: public BufferBs

  void load_weights() override { v3_load(false); }

  // pathb-nodescale (NODE_SCALING_r1.md section 3, 14 Nov): the same load from the block-ordered expert image
  // (image_prepare first; mxfp4-moe.hpp load_weights_image_parts and kt_image.hpp), then the same code tables,
  // report, collapse and pre-read install as load_weights(). Never throws: image_status() is "ok" or the reason, and on
  // any failure weights_loaded stays false, so the caller loads the layer through load_weights() with a new object.
  void load_weights_image() {
    try {
      v3_load(true);
    } catch (const std::exception& ex) {
      this->image_status_ = std::string("image: after the load: ") + ex.what();
      this->weights_loaded = false;
    } catch (...) {
      this->image_status_ = "image: after the load: unknown exception";
      this->weights_loaded = false;
    }
    image_done_.store(1, std::memory_order_release);   // pathb-image-r2: the status is final from here on
  }

  // pathb-image-r2 (concept-r1 STARTUP_CUTS_r1.md section 3): 1 once load_weights_image() has returned, so a caller that
  // queued the image task early (amx.py preload) waits for this layer alone, instead of a CPUInfer sync (which waits for
  // every queued task), and reads image_status() only after the worker has stopped writing it. Reads no arena byte.
  std::atomic<int> image_done_{0};
  int image_done() const { return image_done_.load(std::memory_order_acquire); }

  // pathb-image-r3 (16 Nov): the layer's module MADE inside its image task. r2 made the module on the Python main thread
  // (TP_MOE_Common's constructor: NumaJobDistributor::do_numa_job, moe-tp.hpp) while an earlier layer's image task ran
  // its own do_numa_job calls on the CPUInfer worker (load_weights_image_inner, build_codes, thp_collapse); do_numa_job
  // stores one shared compute_func and spins on shared per-node flags with no lock (worker_pool.cpp), so two callers
  // can run each other's function, or return early. Here the constructor, image_prepare and load_weights_image run in
  // ONE task, so every do_numa_job of a preloaded load runs on the worker thread, in queue order. Nothing in the
  // constructor needs the main thread: no CUDA, no Python; its per-part work runs on the node-bound distributor threads
  // whoever calls it (USE_NUMA_JOB_DIRECT_WORK is not defined), its shared buffers are mutex-guarded (shared_mem_buffer).
  // The statements of the load are r2's own (the constructor, then image_prepare, then load_weights_image with the map
  // set as img_task sets it), so the arenas are byte for byte r2's. run() never throws: status() is "ok" or the reason,
  // and on any failure the module is dropped on the worker before done() reads 1. ext_bindings.cpp (build_bs.py) binds
  // it as AVX2ContractV3_ImageBuild with the CPUInfer task "task()".
  struct ImageBuild {
    GeneralMOEConfig cfg;                       // a copy: the Python MOEConfig may go; the GPU mask it points at may not
    std::string path, digests;
    uint64_t file_bytes = 0;
    std::vector<int64_t> geometry;
    std::vector<std::vector<int>> proven_rows;  // the image's proven part rows ...
    bool any_rows = true;                       // ... or any layout (the proof names none)
    uintptr_t map = 0;                          // physical -> logical, uint64 per expert, owned by the caller
    std::shared_ptr<TP_MOE> moe;
    std::string status_ = "queued";
    std::vector<int> rows_;
    double make_seconds_ = 0, total_seconds_ = 0;
    int64_t tid_ = -1;                          // the thread run() ran on (the CPUInfer worker): imgid2's mechanism check
    std::atomic<int> submitted_{0};
    std::atomic<int> done_{0};

    ImageBuild(const GeneralMOEConfig& c, const std::string& p, uint64_t nbytes, const std::vector<int64_t>& geom,
               const std::string& dg, const std::vector<std::vector<int>>& rows, bool any, uintptr_t m)
        : cfg(c), path(p), digests(dg), file_bytes(nbytes), geometry(geom), proven_rows(rows), any_rows(any), map(m) {}

    void run() {   // on the CPUInfer worker
      const auto t0 = std::chrono::steady_clock::now();
      tid_ = (int64_t)::syscall(SYS_gettid);
      std::shared_ptr<TP_MOE> m;
      std::string st;
      try {
        m = std::make_shared<TP_MOE>(cfg);   // r2: ctx["backend_cls"](cfg) on the main thread
        make_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rows_ = m->contract_v3_part_rows();
        bool proven = any_rows;
        for (const auto& r : proven_rows) proven = proven || r == rows_;
        if (!proven) {
          std::string a, b;
          for (int v : rows_) a += (a.empty() ? "" : ", ") + std::to_string(v);
          for (const auto& r : proven_rows) {
            std::string s;
            for (int v : r) s += (s.empty() ? "" : ", ") + std::to_string(v);
            b += (b.empty() ? "[" : ", [") + s + "]";
          }
          st = "part rows [" + a + "] are not among the image's proven part rows [" + b + "]";
        } else {
          st = m->image_prepare(path, file_bytes, geometry, digests);
          if (st == "ok") {
            if (map) m->config.physical_to_logical_map = reinterpret_cast<void*>(map);   // as img_task
            m->load_weights_image();
            st = m->image_status();
          }
        }
      } catch (const std::exception& ex) {
        st = std::string("image: the module: ") + ex.what();
      } catch (...) {
        st = "image: the module: unknown exception";
      }
      if (st != "ok") m.reset();   // dropped here, on the worker, before done
      moe = std::move(m);
      status_ = st;
      total_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      done_.store(1, std::memory_order_release);   // everything above is final from here on
    }
    int done() const { return done_.load(std::memory_order_acquire); }
    std::string status() const { return done() ? status_ : std::string("queued"); }
    std::vector<int> rows() const { return done() ? rows_ : std::vector<int>{}; }
    std::tuple<double, double> seconds() const { return done() ? std::make_tuple(make_seconds_, total_seconds_)
                                                               : std::make_tuple(-1.0, -1.0); }
    int64_t tid() const { return done() ? tid_ : (int64_t)-1; }
    std::shared_ptr<TP_MOE> take() {   // the module once, after done and only when status is "ok"; else None
      if (!done()) throw std::runtime_error("image build: take() before the task has run");
      return std::move(moe);
    }
  };

  void v3_load(bool from_image) {
    if (kt_contract_v3h::env_on("KT_MXFP4_SCALE_CODES"))
      throw std::runtime_error("contract v3: KT_MXFP4_SCALE_CODES must be off (the FP32 scales must stay in BufferB::d)");
    if (AVX2_MXFP4_MOE_TP<K>::kt_cache_mode())
      throw std::runtime_error("contract v3: the MXFP4 expert cache must be off");
    (void)kt_contract_v3h::tier();
    // concept-r1 pre-read: a load uses the pool, so it waits for (and stops) a pre-read running on another thread
    std::unique_lock<std::mutex> pr_lk;
    if (kt_preread::armed().load(std::memory_order_acquire)) {
      const bool outside = !kt_preread::on_hook_thread;
      if (outside) kt_preread::outside_waiters().fetch_add(1, std::memory_order_acq_rel);
      pr_lk = std::unique_lock<std::mutex>(sh().mu);
      if (outside) kt_preread::outside_waiters().fetch_sub(1, std::memory_order_acq_rel);
    }
    if (!from_image) {
      Base::load_weights();
    } else if (!Base::load_weights_image_parts()) {
      return;   // image_status() says why; nothing else is built
    }
    const int E = this->config.expert_num;
    free_codes();
    codes_.assign(this->tp_count, std::vector<Codes>(E));
    code_regions_.assign(this->tp_count, CodeRegion{});
    std::atomic<int> err{0};
    const bool local = !kt_contract_v3h::env_off("KT_V3_CODES_LOCAL");
    if (local) {   // every part builds its own tables on its own node-bound thread
      std::vector<std::exception_ptr> errs(this->tp_count);
      this->config.pool->dispense_backend()->do_numa_job([&](int p) {
        try { build_codes(p, true, err); } catch (...) { errs[p] = std::current_exception(); }
      });
      for (auto& e : errs)
        if (e) std::rethrow_exception(e);
    } else {       // the placement before 12 Nov: the calling thread builds all of them
      for (int p = 0; p < this->tp_count; p++) build_codes(p, false, err);
    }
    kt_contract_v3h::check(err.load(), "weight scale conversion");
    report_codes(local);
    if (kt_ccd::thp_collapse_on()) thp_collapse();   // pathb-cpu-r1 KT_V3_THP_COLLAPSE
    if (kt_preread::armed().load(std::memory_order_acquire)) pr_install();
  }

  // pathb-cpu-r1 KT_V3_THP_COLLAPSE: on each part's own thread, MADV_COLLAPSE every 2 MiB extent lying wholly inside the
  // written bytes of the part's gate, up and down arenas (runs of held experts whose nibbles are contiguous) and inside
  // its code region. A GPU-only expert's bytes were never written: no extent touches them, so no page is added. The
  // kernel copies the 4 KiB pages into a huge page of the node most of them sit on; the bytes are unchanged.
  void thp_collapse() {
    const uint64_t t0 = kt_preread::now_ns();
    std::vector<kt_ccd::ThpStat> st(this->tp_count);
    this->config.pool->dispense_backend()->do_numa_job([&](int p) { thp_collapse_part(p, st[p]); });
    kt_ccd::ThpStat a;
    for (const auto& s : st) {
      a.asked += s.asked; a.ok += s.ok; a.failed += s.failed;
      a.eagain += s.eagain; a.enomem += s.enomem; a.einval += s.einval; a.other += s.other;
    }
    const double secs = (kt_preread::now_ns() - t0) / 1e9;
    auto& T = sh().thp_total;
    T.asked += a.asked; T.ok += a.ok; T.failed += a.failed;
    T.eagain += a.eagain; T.enomem += a.enomem; T.einval += a.einval; T.other += a.other;
    sh().thp_seconds += secs;
    printf("[v3-thp] layer %d: MADV_COLLAPSE of 2 MiB extents: asked %ld, ok %ld, failed %ld (EAGAIN %ld, ENOMEM %ld, "
           "EINVAL %ld, other %ld), %.2f s\n", this->config.layer_idx, a.asked, a.ok, a.failed, a.eagain, a.enomem,
           a.einval, a.other, secs);
    fflush(stdout);
  }
  void thp_collapse_part(int p, kt_ccd::ThpStat& st) {
    auto* tp = part(p);
    const int E = this->config.expert_num;
    const uintptr_t HP = (uintptr_t)2 << 20;
    auto range = [&](const uint8_t* lo, const uint8_t* hi) {   // every 2 MiB extent wholly inside [lo, hi)
      const uintptr_t a = ((uintptr_t)lo + HP - 1) & ~(HP - 1), b = (uintptr_t)hi & ~(HP - 1);
      for (uintptr_t x = a; x + HP <= b; x += HP) {
        st.asked++;
        if (madvise((void*)x, (size_t)HP, 25 /* MADV_COLLAPSE, Linux 6.1+ */) == 0) {
          st.ok++;
        } else {
          st.failed++;
          const int e = errno;
          if (e == EAGAIN) st.eagain++;
          else if (e == ENOMEM) st.enomem++;
          else if (e == EINVAL) st.einval++;
          else st.other++;
        }
      }
    };
    auto runs = [&](const auto& vec) {   // one of gate, up, down: runs of held experts whose bytes follow each other
      const uint8_t *lo = nullptr, *hi = nullptr;
      for (int e = 0; e < E && e < (int)vec.size(); e++) {
        const auto& bb = vec[e];
        const bool held = tp->gate_bb_[e] && tp->gate_bb_[e]->b && bb && bb->b;
        if (!held) {
          if (lo) range(lo, hi);
          lo = hi = nullptr;
          continue;
        }
        const uint8_t* b = bb->b;
        const uint8_t* end = b + (size_t)bb->n * bb->k / 2;
        if (lo && hi == b) {
          hi = end;
        } else {
          if (lo) range(lo, hi);
          lo = b;
          hi = end;
        }
      }
      if (lo) range(lo, hi);
    };
    runs(tp->gate_bb_);
    runs(tp->up_bb_);
    runs(tp->down_bb_);
    if (p < (int)code_regions_.size() && code_regions_[p].p) range(code_regions_[p].p, code_regions_[p].p + code_regions_[p].n);
  }
  std::string contract_v3_thp() {   // tests: the totals of every layer loaded so far
    const auto& T = sh().thp_total;
    char b[256];
    snprintf(b, sizeof b, "on=%d asked=%ld ok=%ld failed=%ld eagain=%ld enomem=%ld einval=%ld other=%ld seconds=%.2f",
             (int)kt_ccd::thp_collapse_on(), T.asked, T.ok, T.failed, T.eagain, T.enomem, T.einval, T.other,
             sh().thp_seconds);
    return std::string(b);
  }

  // part p's code tables: one mapping, the experts ascending, gate then up then down, each table 64-byte aligned
  void build_codes(int p, bool local, std::atomic<int>& err) {
    auto* tp = part(p);
    const int E = this->config.expert_num;
    auto fail = [&err](int rc) { int z = 0; err.compare_exchange_strong(z, rc); };
    auto held = [&](int e) { return tp->gate_bb_[e] && tp->gate_bb_[e]->b; };   // else a GPU-only expert
    // KT_ARENA_SCALE_BYTES: a byte-arena BufferB has its E8M0 bytes at `code` and no FP32 `d` (88b2f54)
    auto ok = [](const auto& bb) {
      return bb && bb->b && (bb->arena_bytes ? bb->code != nullptr : bb->d != nullptr) && bb->k_group_size == 32 &&
             bb->n % 8 == 0;
    };
    auto bytes = [](const auto& bb) { return ((size_t)bb->n * bb->k_group_count + 63) / 64 * 64; };
    size_t total = 0;
    for (int e = 0; e < E; e++) {
      if (!held(e)) continue;
      if (!ok(tp->gate_bb_[e]) || !ok(tp->up_bb_[e]) || !ok(tp->down_bb_[e])) { fail(V3_ERR_SHAPE); return; }
      total += bytes(tp->gate_bb_[e]) + bytes(tp->up_bb_[e]) + bytes(tp->down_bb_[e]);
    }
    if (!total) return;
    auto& r = code_regions_[p];
    r.n = (total + 4095) / 4096 * 4096;
    void* m = mmap(nullptr, r.n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { r.n = 0; throw std::runtime_error("contract v3: no memory for the scale-code tables"); }
    r.p = (uint8_t*)m;
    if (local) {
      r.node = kt_contract_v3h::this_node();
      r.bound = kt_contract_v3h::prefer_node(r.p, r.n, r.node);
    }
    uint8_t* at = r.p;
    for (int e = 0; e < E; e++) {
      if (!held(e)) continue;
      auto conv = [&](const auto& bb, const uint8_t*& out) {
        const int rc = bb->arena_bytes ? v3_rm_e8m0_to_codes(bb->code, bb->n, bb->k_group_count, at)
                                       : v3_rm_scales_to_codes(bb->d, bb->n, bb->k_group_count, at);
        if (rc) fail(rc);
        out = at;
        at += bytes(bb);
      };
      conv(tp->gate_bb_[e], codes_[p][e].g);
      conv(tp->up_bb_[e], codes_[p][e].u);
      conv(tp->down_bb_[e], codes_[p][e].d);
    }
    if (local) kt_contract_v3h::pages_on_node(r.p, r.n, r.node, 512, r.sampled, r.on_node, r.elsewhere);
  }

  // one line a layer: where each part's code pages are. A spill is reported, and refused only with KT_V3_CODES_STRICT=1.
  void report_codes(bool local) {
    size_t bytes = 0;
    bool spill = false;
    std::string shares;
    for (int p = 0; p < this->tp_count; p++) {
      const auto& r = code_regions_[p];
      bytes += r.n;
      const long pm = r.sampled ? r.on_node * 1000 / r.sampled : -1;
      shares += " " + std::to_string(pm);
      if (r.sampled && r.on_node != r.sampled) spill = true;
    }
    if (!local) {
      printf("[v3-codes] layer %d: %d parts, %.1f MiB of scale codes built on the calling thread (KT_V3_CODES_LOCAL=0)\n",
             this->config.layer_idx, this->tp_count, bytes / 1048576.0);
      fflush(stdout);
      return;
    }
    printf("[v3-codes] layer %d: %d parts, %.1f MiB of scale codes; per mille of sampled pages on the part's own node:%s%s\n",
           this->config.layer_idx, this->tp_count, bytes / 1048576.0, shares.c_str(), spill ? "  SPILL" : "");
    if (spill)
      for (int p = 0; p < this->tp_count; p++) {
        const auto& r = code_regions_[p];
        if (r.sampled && r.on_node != r.sampled)
          printf("[v3-codes] WARNING layer %d part %d (node %d, bound %d): %ld of %ld sampled pages elsewhere:%s\n",
                 this->config.layer_idx, p, r.node, (int)r.bound, r.sampled - r.on_node, r.sampled, r.elsewhere.c_str());
      }
    fflush(stdout);
    if (spill && kt_contract_v3h::env_on("KT_V3_CODES_STRICT"))
      throw std::runtime_error("contract v3: scale-code pages off the part's node (KT_V3_CODES_STRICT=1), layer " +
                               std::to_string(this->config.layer_idx));
  }
  // "node:on_node/sampled" per part, for tests ("-1:0/0" = not placed by the part, or the kernel would not say)
  std::string contract_v3_codes_placement() {
    std::string s;
    for (size_t p = 0; p < code_regions_.size(); p++) {
      const auto& r = code_regions_[p];
      s += (p ? ";" : "") + std::to_string(r.node) + ":" + std::to_string(r.on_node) + "/" + std::to_string(r.sampled);
    }
    return s;
  }

  static v3_rm_mat view(const auto& bb, const uint8_t* codes, int n0 = 0, int n1 = -1) {
    // rows [n0, n1) of the matrix (n0, n1 multiples of 8) as their own matrix
    if (n1 < 0) n1 = bb->n;
    v3_rm_mat m;
    m.N = n1 - n0; m.K = bb->k; m.G = bb->k_group_count;
    m.row_bytes = (size_t)bb->k / 2;
    m.b = bb->b + (size_t)n0 * m.row_bytes;
    m.s = codes + (size_t)(n0 / 8) * m.G * 8;
    return m;
  }

  // grow-only scratch for qlen tokens (never sized from max_len: his chunked prefill length is far larger than what the
  // CPU pool is handed once the streamer takes long forwards)
  void ensure_scratch(int qlen) {
    const auto& c = this->config;
    const size_t routes = (size_t)qlen * c.num_experts_per_tok, H = c.hidden_size, G = H / 32;
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(qx_, (size_t)qlen * H * 2); grow(eax_, (size_t)qlen * G);
    grow(qxe_, routes * H * 2); grow(eaxe_, routes * G);
    if (e_off_.size() < (size_t)c.expert_num) { e_off_.assign(c.expert_num, 0); e_cnt_.assign(c.expert_num, 0); }
    grow(e_tok_, routes); grow(e_w_, routes);
    if (ps_.size() != (size_t)this->tp_count) ps_.assign(this->tp_count, {});
    for (int p = 0; p < this->tp_count; p++) {
      const size_t Ip = this->tp_configs[p].intermediate_size;
      grow(ps_[p].gate, routes * Ip); grow(ps_[p].up, routes * Ip);
      grow(ps_[p].qd, routes * Ip * 2); grow(ps_[p].ead, routes * (Ip / 32));
      grow(ps_[p].tok, (size_t)qlen * H);
    }
  }

  // raw int64 [qlen][hidden] of this device's routes (ids < 0 or masked GPU experts skipped). Output overwritten.
  void forward_partials_binding(intptr_t qlen_ptr, int k, intptr_t ids_ptr, intptr_t weights_ptr, intptr_t input, intptr_t output) {
    const int qlen = *(const int*)qlen_ptr;
    auto& c = this->config;
    if (!this->weights_loaded) throw std::runtime_error("contract v3: not loaded");
    if (k != c.num_experts_per_tok || qlen < 1 || qlen > c.max_len) throw std::runtime_error("contract v3 partials: bad shape");
    // concept-r1 pre-read, armed only: a caller off the task-queue thread stops a running pre-read before it waits
    const bool pr = kt_preread::armed().load(std::memory_order_relaxed);
    const bool pr_outside = pr && !kt_preread::on_hook_thread;
    if (pr_outside) kt_preread::outside_waiters().fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lock(sh().mu);
    if (pr_outside) kt_preread::outside_waiters().fetch_sub(1, std::memory_order_acq_rel);
    const int H = c.hidden_size, R = rows_cap();
    for (int t0 = 0; t0 < qlen; t0 += R) {
      const int n = std::min(R, qlen - t0);
      partials_chunk(n, k, (const int64_t*)ids_ptr + (size_t)t0 * k, (const float*)weights_ptr + (size_t)t0 * k,
                     reinterpret_cast<const uint16_t*>(input) + (size_t)t0 * H, reinterpret_cast<int64_t*>(output) + (size_t)t0 * H);
    }
    if (pr) {   // which layer ran, for the hook (it reads this call's active_ / e_cnt_ after the task has finished)
      sh().pr_last = this;
      sh().pr_last_q = qlen;
    }
  }

  void partials_chunk(const int qlen, int k, const int64_t* ids, const float* w, const uint16_t* input, int64_t* out) {
    if (const int mode = contract_v3_clean(); mode && qlen <= clean_maxq()) {
      partials_clean(qlen, ids, w, input, out, mode);
      return;
    }
    auto& c = this->config;
    const int tier = kt_contract_v3h::tier();
    const int bL = get_bl();
    if (bL == INT32_MIN) throw std::runtime_error("contract v3: no bL for layer " + std::to_string(c.layer_idx));
    ensure_scratch(qlen);
    const int H = c.hidden_size, G = H / 32, E = c.expert_num;
    const size_t qb = tier == V3_TIER_C ? 2 : 1;
    // 1. routes per expert (skipped: id < 0, or masked as a GPU expert)
    std::fill(e_cnt_.begin(), e_cnt_.end(), 0);
    for (int r = 0; r < qlen * k; r++) {
      const int64_t e = ids[r];
      if (e < -1 || e >= E) throw std::runtime_error("contract v3 route id out of range");
      if (e >= 0 && !c.should_skip_expert(e)) e_cnt_[e]++;
    }
    active_.clear();
    int off = 0;
    for (int e = 0; e < E; e++) { e_off_[e] = off; off += e_cnt_[e]; if (e_cnt_[e]) active_.push_back(e); e_cnt_[e] = 0; }
    for (int r = 0; r < qlen * k; r++) {
      const int64_t e = ids[r];
      if (e < 0 || c.should_skip_expert(e)) continue;
      const int row = e_off_[e] + e_cnt_[e]++;
      e_tok_[row] = r / k; e_w_[row] = w[r];
    }
    std::memset(out, 0, sizeof(int64_t) * (size_t)qlen * H);
    if (active_.empty()) return;
    std::atomic<int> err{0};
    auto rec = [&err](int rc) { if (rc) { int z = 0; err.compare_exchange_strong(z, rc); } };
    // 2. quantise each token row once, then gather the rows per expert (int16/int8 copies)
    c.pool->do_work_stealing_job(qlen, nullptr, [&](int t) {
      rec(v3_quantise_bf16(input + (size_t)t * H, 1, H, H, tier,
                           qx_.data() + (size_t)t * H * qb, H, eax_.data() + (size_t)t * G, G));
    }, nullptr);
    kt_contract_v3h::check(err.load(), "quantiser");
    for (int row = 0; row < off; row++) {
      const int t = e_tok_[row];
      std::memcpy(qxe_.data() + (size_t)row * H * qb, qx_.data() + (size_t)t * H * qb, (size_t)H * qb);
      std::memcpy(eaxe_.data() + (size_t)row * G, eax_.data() + (size_t)t * G, sizeof(int32_t) * G);
    }
    // 3. every part: its gate/up columns, SwiGLU + quantiser on its blocks, raw int64 down into its token sums
    std::vector<std::exception_ptr> errs(this->tp_count);
    c.pool->dispense_backend()->do_numa_job([&](int p) {
      try { part_forward(p, qlen, tier, bL, off); } catch (...) { errs[p] = std::current_exception(); }
    });
    for (auto& e : errs)
      if (e) std::rethrow_exception(e);
    // 4. exact sum of the parts' token sums (any order, wrapping). No range check here: this is a device partial; r3h item 2
    //    checks |S| >= 2^62 only on the final layer sum (the GPU finish, flag 4). Path B freeze, CODE_AUDIT_r1 F12.
    const int chunk = 256, chunks = (H + chunk - 1) / chunk;
    c.pool->do_work_stealing_job(qlen * chunks, nullptr, [&](int task) {
      const int t = task / chunks, c0 = (task % chunks) * chunk, c1 = std::min(H, c0 + chunk);
      int64_t* o = out + (size_t)t * H;
      for (int p = 0; p < this->tp_count; p++) {
        const int64_t* a = ps_[p].tok.data() + (size_t)t * H;
        for (int col = c0; col < c1; col++) {
          o[col] = (int64_t)((uint64_t)o[col] + (uint64_t)a[col]);   // wrapping (hardening #8); no partial range check (F12)
        }
      }
    }, nullptr);
    kt_contract_v3h::check(err.load(), "part sum");
  }

  void part_forward(int p, int qlen, int tier, int bL, int nrows) {
    auto* tp = part(p);
    auto pool = this->config.pool->get_subpool(p);
    auto& s = ps_[p];
    const int H = this->config.hidden_size, Ip = this->tp_configs[p].intermediate_size, Gp = Ip / 32, G = H / 32;
    const size_t qb = tier == V3_TIER_C ? 2 : 1;
    if (Ip % 32 || H % 8) throw std::runtime_error("contract v3: part slice must be a multiple of 32");
    std::memset(s.tok.data(), 0, sizeof(int64_t) * (size_t)qlen * H);
    std::atomic<int> err{0};
    auto rec = [&err](int rc) { if (rc) { int z = 0; err.compare_exchange_strong(z, rc); } };
    const int na = (int)active_.size();
    for (int e : active_)
      if (!tp->gate_bb_[e] || !tp->gate_bb_[e]->b) throw std::runtime_error("contract v3: routed expert not held by the CPU pool");
    // gate / up: column ranges of whole 8-panels
    const int panels_i = Ip / 8;
    const int nth_gu = std::max(1, std::min(panels_i, (int)(((size_t)Ip * H / 2) / 65536)));
    pool->do_work_stealing_job(nth_gu * na * 2, nullptr, [&](int task) {
      const bool up = task & 1; const int t2 = task >> 1;
      const int e = active_[t2 / nth_gu], ith = t2 % nth_gu;
      const int c0 = 8 * (panels_i * ith / nth_gu), c1 = 8 * (panels_i * (ith + 1) / nth_gu);
      if (c1 <= c0) return;
      const v3_rm_mat m = view(up ? tp->up_bb_[e] : tp->gate_bb_[e], up ? codes_[p][e].u : codes_[p][e].g);
      const size_t r0 = e_off_[e];
      rec(v3_rm_gemm_bf16(&m, tier, qxe_.data() + r0 * H * qb, H, eaxe_.data() + r0 * G, G, e_cnt_[e], c0, c1,
                          (up ? s.up.data() : s.gate.data()) + r0 * Ip, Ip));
    }, nullptr);
    kt_contract_v3h::check(err.load(), "gate/up");
    // SwiGLU (table SiLU, routing weight) + down-input quantiser over the part's blocks
    const int nth_act = std::min(Gp, 3);
    pool->do_work_stealing_job(nth_act * na, nullptr, [&](int task) {
      const int e = active_[task / nth_act], ith = task % nth_act;
      const int g0 = Gp * ith / nth_act, g1 = Gp * (ith + 1) / nth_act;
      const size_t r0 = e_off_[e];
      rec(v3_swiglu_quant(s.gate.data() + r0 * Ip, s.up.data() + r0 * Ip, Ip, e_cnt_[e], e_w_.data() + r0,
                          kt_contract_v3h::silu_table(), g0, g1, tier, nullptr, 0, s.qd.data() + r0 * Ip * qb, Ip,
                          s.ead.data() + r0 * Gp, Gp));
    }, nullptr);
    kt_contract_v3h::check(err.load(), "swiglu");
    // down: per output column range, every active expert in turn (no write races), its rows' raw int64 sums added to
    // their tokens' sums
    const int panels_h = H / 8;
    const int nth_dn = std::max(1, std::min(panels_h, (int)(((size_t)H * Ip / 2) / 65536) * 4));
    pool->do_work_stealing_job(nth_dn, nullptr, [&](int ith) {
      const int c0 = 8 * (panels_h * ith / nth_dn), c1 = 8 * (panels_h * (ith + 1) / nth_dn);
      if (c1 <= c0) return;
      const int wd = c1 - c0;
      thread_local std::vector<int64_t> tmp;
      for (int e : active_) {
        const int m = e_cnt_[e];
        if (tmp.size() < (size_t)m * wd) tmp.resize((size_t)m * wd);
        std::memset(tmp.data(), 0, sizeof(int64_t) * (size_t)m * wd);
        const v3_rm_mat mw = view(tp->down_bb_[e], codes_[p][e].d, c0, c1);
        const size_t r0 = e_off_[e];
        const int rc = v3_rm_down_partial(&mw, tier, s.qd.data() + r0 * Ip * qb, Ip, s.ead.data() + r0 * Gp, Gp, m, bL,
                                          nullptr, 0, 0, wd, tmp.data(), wd);
        if (rc) { rec(rc); return; }
        for (int r = 0; r < m; r++) {
          int64_t* dst = s.tok.data() + (size_t)e_tok_[r0 + r] * H + c0;
          const int64_t* src = tmp.data() + (size_t)r * wd;
          for (int j = 0; j < wd; j++) dst[j] = (int64_t)((uint64_t)dst[j] + (uint64_t)src[j]);   // wrapping (#8)
        }
      }
    }, nullptr);
    kt_contract_v3h::check(err.load(), "down");
  }

  // ---- the clean layout (KT_V3_CLEAN) ----
  // (re)make the per-part scratch and learn each part's node when the pool or the part count changed
  void ensure_clean() {
    const void* pool = (const void*)this->config.pool;
    if (sh().cs_pool_ == pool && cs_.size() == (size_t)this->tp_count) return;
    for (auto& s : cs_)
      if (s.hand) munmap(s.hand, s.hand_bytes);
    cs_.assign(this->tp_count, CleanScratch{});
    cs_node_.assign(this->tp_count, -1);
    this->config.pool->dispense_backend()->do_numa_job([&](int p) { cs_node_[p] = kt_contract_v3h::this_node(); });
    sh().cs_pool_ = pool;
  }

  // pathb-cpu-r1 KT_V3_CCD: learn each part's two-CCD layout once per pool. Every pool thread reports the one CPU it may
  // run on (pr_lay_part, the pre-read's layout job); a part splits only if every thread is pinned to one CPU, the
  // threads' CPUs fall in exactly two L3 domains (sysfs), and no two threads share a CPU. Called with mu held.
  void ensure_ccd() {
    using namespace kt_preread;
    auto& S = sh();
    const void* pool = (const void*)this->config.pool;
    if (S.ccd_pool_ == pool) return;
    S.ccd_pool_ = pool;
    for (int p = 0; p < kMaxParts; p++)
      for (int t = 0; t < kMaxThreads; t++) S.pr_lay_cpu[p][t] = -2;
    TP_MOE* self = this;
    auto job = [self](int p) { self->pr_lay_part(p); };
    this->config.pool->dispense_backend()->do_numa_job(job);
    std::string line;
    int split = 0;
    for (int p = 0; p < kMaxParts; p++) S.ccd_[p] = kt_ccd::Part{};
    for (int p = 0; p < this->tp_count; p++) {
      if (p >= kMaxParts) { line += " p" + std::to_string(p) + ":too-many-parts"; continue; }
      kt_ccd::Part& cp = S.ccd_[p];
      const int T = this->config.pool->config.subpool_thread_count[p];
      cp.T = T;
      int l3[kMaxThreads];
      bool bad = T < 2 || T > kMaxThreads;
      if (bad) snprintf(cp.why, sizeof cp.why, "%d threads", T);
      uint64_t seen[CPU_SETSIZE / 64] = {};
      for (int t = 0; !bad && t < T; t++) {
        const int c = S.pr_lay_cpu[p][t];
        if (c < 0) { snprintf(cp.why, sizeof cp.why, "thread %d %s", t, c == -2 ? "not reached" : "not pinned to one CPU"); bad = true; break; }
        if (seen[c / 64] & (1ull << (c % 64))) { snprintf(cp.why, sizeof cp.why, "thread %d shares CPU %d", t, c); bad = true; break; }
        seen[c / 64] |= 1ull << (c % 64);
        l3[t] = kt_ccd::l3_of_cpu(c);
        if (l3[t] < 0) { snprintf(cp.why, sizeof cp.why, "no L3 in sysfs for CPU %d", c); bad = true; break; }
      }
      if (!bad) {
        cp.l3[0] = l3[0];
        for (int t = 1; t < T; t++)
          if (l3[t] != cp.l3[0]) {
            if (cp.l3[1] < 0) cp.l3[1] = l3[t];
            else if (l3[t] != cp.l3[1]) { snprintf(cp.why, sizeof cp.why, "more than two L3 domains"); bad = true; break; }
          }
        if (!bad && cp.l3[1] < 0) { snprintf(cp.why, sizeof cp.why, "one L3 domain"); bad = true; }
      }
      if (!bad) {
        for (int t = 0; t < T; t++) {
          const int c = l3[t] == cp.l3[0] ? 0 : 1;
          cp.ccd[t] = (int8_t)c;
          cp.rank[t] = (int8_t)cp.n[c]++;
        }
        cp.ok = 1;
        split++;
        snprintf(cp.why, sizeof cp.why, "L3 %d: %d threads, L3 %d: %d threads", cp.l3[0], cp.n[0], cp.l3[1], cp.n[1]);
      }
      line += " p" + std::to_string(p) + (cp.ok ? ":split(" : ":today(") + cp.why + ")";
    }
    printf("[kt-ccd] KT_V3_CCD: %d of %d parts split between two L3 domains:%s\n", split, this->tp_count, line.c_str());
    fflush(stdout);
  }
  // the part's layout when KT_V3_CCD is on and the part splits, else null (today's path)
  const kt_ccd::Part* ccd_part(int p) const {
    if (!kt_ccd::enabled() || p < 0 || p >= kt_preread::kMaxParts) return nullptr;
    const auto& S = sh();
    if (S.ccd_pool_ != (const void*)this->config.pool || S.ccd_[p].ok != 1) return nullptr;
    return &S.ccd_[p];
  }
  // the down job's task count (as served): one helper for the job and for the pre-read's per-CCD rows
  static int down_tasks(int H, int Ip) {
    const int panels_h = H / 8;
    return std::max(1, std::min(panels_h, (int)(((size_t)H * Ip / 2) / 65536) * 4));
  }
  // KT_V3_SUM_PART: the subpool the part sum runs on (0 unless set and in range)
  int sum_part() const {
    const int p = kt_ccd::sum_part_env();
    return p > 0 && p < this->tp_count ? p : 0;
  }
  std::string contract_v3_sum_part() { return "KT_V3_SUM_PART=" + std::to_string(kt_ccd::sum_part_env()) + " used=" + std::to_string(sum_part()); }
  // tests: -1 back to the environment, 0 off, 1 on (process-wide); the layout is learned at the next clean call
  void contract_v3_set_ccd(int m) {
    if (m < -1 || m > 1) throw std::runtime_error("contract v3: ccd mode must be -1, 0 or 1");
    if (m == -1) {
      const char* e = std::getenv("KT_V3_CCD");
      m = (e && e[0] == '1' && e[1] == 0) ? 1 : 0;
    }
    kt_ccd::on().store(m, std::memory_order_relaxed);
  }
  std::string contract_v3_ccd() {   // tests: read after the calls (no lock: a running pre-read holds mu)
    const auto& S = sh();
    std::string s = "on=" + std::to_string((int)kt_ccd::enabled()) +
                    " learned=" + std::to_string((int)(S.ccd_pool_ == (const void*)this->config.pool));
    for (int p = 0; p < this->tp_count && p < kt_preread::kMaxParts; p++)
      s += " p" + std::to_string(p) + ":" + std::to_string(S.ccd_[p].ok) + ":" + std::to_string(S.ccd_[p].n[0]) + "+" +
           std::to_string(S.ccd_[p].n[1]) + ":" + std::to_string(S.ccd_[p].l3[0]) + "/" + std::to_string(S.ccd_[p].l3[1]);
    return s;
  }
  // tests: the row-wise kernel (on -1 keeps, 0 off, 1 on) and its reduction (red -1 keeps, 1 or 2), process-wide
  void contract_v3_set_rw(int on, int red) {
    if (on >= 0) v3_rw_set_enabled(on);
    if (red >= 1) v3_rw_set_red(red);
  }
  std::string contract_v3_rw() { return "rw=" + std::to_string(v3_rw_enabled()) + " red=" + std::to_string(v3_rw_red()); }
  static void grow_hand(CleanScratch& s, size_t words, int node) {
    if (s.hand_words >= words && s.hand_node == node) return;
    if (s.hand) munmap(s.hand, s.hand_bytes);
    s.hand = nullptr;
    s.hand_words = s.hand_bytes = 0;
    const size_t bytes = (words * sizeof(int64_t) + 4095) / 4096 * 4096;
    void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) throw std::runtime_error("contract v3: no memory for the hand-over buffer");
    kt_contract_v3h::prefer_node(m, bytes, node);
    std::memset(m, 0, bytes);
    s.hand = (int64_t*)m;
    s.hand_words = words;
    s.hand_bytes = bytes;
    s.hand_node = node;
  }

  void partials_clean(const int qlen, const int64_t* ids, const float* w, const uint16_t* input, int64_t* out, int mode) {
    auto& c = this->config;
    const int tier = kt_contract_v3h::tier();
    const int bL = get_bl();
    if (bL == INT32_MIN) throw std::runtime_error("contract v3: no bL for layer " + std::to_string(c.layer_idx));
    const int H = c.hidden_size, E = c.expert_num, k = c.num_experts_per_tok;
    const size_t routes = (size_t)qlen * k;
    if (e_off_.size() < (size_t)E) { e_off_.assign(E, 0); e_cnt_.assign(E, 0); }
    if (e_tok_.size() < routes) e_tok_.resize(routes);
    if (e_w_.size() < routes) e_w_.resize(routes);
    // 1. routes per expert: the served bookkeeping (skipped: id < 0, or masked as a GPU expert)
    std::fill(e_cnt_.begin(), e_cnt_.end(), 0);
    for (int r = 0; r < qlen * k; r++) {
      const int64_t e = ids[r];
      if (e < -1 || e >= E) throw std::runtime_error("contract v3 route id out of range");
      if (e >= 0 && !c.should_skip_expert(e)) e_cnt_[e]++;
    }
    active_.clear();
    int off = 0;
    for (int e = 0; e < E; e++) { e_off_[e] = off; off += e_cnt_[e]; if (e_cnt_[e]) active_.push_back(e); e_cnt_[e] = 0; }
    for (int r = 0; r < qlen * k; r++) {
      const int64_t e = ids[r];
      if (e < 0 || c.should_skip_expert(e)) continue;
      const int row = e_off_[e] + e_cnt_[e]++;
      e_tok_[row] = r / k; e_w_[row] = w[r];
    }
    if (active_.empty()) { std::memset(out, 0, sizeof(int64_t) * (size_t)qlen * H); return; }
    ensure_clean();
    if (kt_ccd::enabled()) ensure_ccd();   // pathb-cpu-r1 KT_V3_CCD: once per pool
    // 2. one dispatch: every part does all of its work
    std::vector<std::exception_ptr> errs(this->tp_count);
    c.pool->dispense_backend()->do_numa_job([&](int p) {
      try { part_clean(p, qlen, tier, bL, off, input, mode); } catch (...) { errs[p] = std::current_exception(); }
    });
    for (auto& e : errs)
      if (e) std::rethrow_exception(e);
    // 3. exact sum of the parts' token sums, parts ascending as served (wrapping; no range check: F12). The first part
    //    assigns, so the output is not zeroed first.
    const bool hand = mode >= 2;
    const int chunk = 256, chunks = (H + chunk - 1) / chunk;
    auto sum_task = [&](int task) {
      const int t = task / chunks, c0 = (task % chunks) * chunk, c1 = std::min(H, c0 + chunk);
      int64_t* o = out + (size_t)t * H;
      const int64_t* a0 = (hand ? cs_[0].hand : cs_[0].tok.data()) + (size_t)t * H;
      for (int col = c0; col < c1; col++) o[col] = a0[col];
      for (int p = 1; p < this->tp_count; p++) {
        const int64_t* a = (hand ? cs_[p].hand : cs_[p].tok.data()) + (size_t)t * H;
        for (int col = c0; col < c1; col++) o[col] = (int64_t)((uint64_t)o[col] + (uint64_t)a[col]);   // wrapping (#8)
      }
    };
    if (const int sp = sum_part(); sp > 0)   // pathb-cpu-r1 KT_V3_SUM_PART: the same tasks on subpool sp
      c.pool->get_subpool(sp)->do_work_stealing_job(qlen * chunks, nullptr, sum_task, nullptr);
    else
      c.pool->do_work_stealing_job(qlen * chunks, nullptr, sum_task, nullptr);
  }

  void part_clean(int p, int qlen, int tier, int bL, int nrows, const uint16_t* input, int mode) {
    auto* tp = part(p);
    auto pool = this->config.pool->get_subpool(p);
    auto& s = cs_[p];
    const int H = this->config.hidden_size, Ip = this->tp_configs[p].intermediate_size, Gp = Ip / 32, G = H / 32;
    const size_t qb = tier == V3_TIER_C ? 2 : 1;
    if (Ip % 32 || H % 8) throw std::runtime_error("contract v3: part slice must be a multiple of 32");
    const int na = (int)active_.size();
    for (int e : active_)
      if (!tp->gate_bb_[e] || !tp->gate_bb_[e]->b) throw std::runtime_error("contract v3: routed expert not held by the CPU pool");
    // grow-only scratch, on this part's own thread
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(s.qx, (size_t)qlen * H * qb); grow(s.eax, (size_t)qlen * G);
    grow(s.qxe, (size_t)nrows * H * qb); grow(s.eaxe, (size_t)nrows * G);
    grow(s.gate, (size_t)nrows * Ip); grow(s.up, (size_t)nrows * Ip);
    grow(s.qd, (size_t)nrows * Ip * qb); grow(s.ead, (size_t)nrows * Gp);
    grow(s.tok, (size_t)qlen * H);
    if (mode >= 2) grow_hand(s, (size_t)qlen * H, mode == 3 ? cs_node_[sum_part()] : cs_node_[p]);   // KT_V3_SUM_PART
    int64_t* const hand = mode >= 2 ? s.hand : nullptr;
    std::atomic<int> err{0};
    auto rec = [&err](int rc) { if (rc) { int z = 0; err.compare_exchange_strong(z, rc); } };
    // the part quantises the token rows itself; only an expert with several rows needs them side by side
    for (int t = 0; t < qlen; t++)
      rec(v3_quantise_bf16(input + (size_t)t * H, 1, H, H, tier, s.qx.data() + (size_t)t * H * qb, H, s.eax.data() + (size_t)t * G, G));
    kt_contract_v3h::check(err.load(), "quantiser");
    for (int e : active_) {
      if (e_cnt_[e] < 2) continue;
      const size_t r0 = e_off_[e];
      for (int r = 0; r < e_cnt_[e]; r++) {
        const int t = e_tok_[r0 + r];
        std::memcpy(s.qxe.data() + (r0 + r) * H * qb, s.qx.data() + (size_t)t * H * qb, (size_t)H * qb);
        std::memcpy(s.eaxe.data() + (r0 + r) * G, s.eax.data() + (size_t)t * G, sizeof(int32_t) * G);
      }
    }
    // pathb-cpu-r1 KT_V3_CCD: the part's two-CCD layout, or null (today's path: one shared counter for all threads)
    const kt_ccd::Part* cp = ccd_part(p);
    kt_ccd::Ctr* ctr = sh().ccd_ctr[p < kt_preread::kMaxParts ? p : 0];
    // a static split: CCD c's threads take the tasks [t0, t0 + nt) of its own list, from its own counter, through
    // `run(i)`; the part's distributor (thread 0, CCD 0) then takes any CCD 1 task left (none unless a thread never ran)
    auto ccd_job = [&](int n0_items, int n1_items, auto&& run0, auto&& run1) {
      ctr[0].v.store(0, std::memory_order_relaxed);
      ctr[1].v.store(0, std::memory_order_relaxed);
      pool->do_work_stealing_job(cp->T, [&](int tid) {
        if (tid < 0 || tid >= cp->T) return;
        if (cp->ccd[tid] == 0) {
          for (int i; (i = ctr[0].v.fetch_add(1, std::memory_order_relaxed)) < n0_items;) run0(i);
        } else {
          for (int i; (i = ctr[1].v.fetch_add(1, std::memory_order_relaxed)) < n1_items;) run1(i);
        }
      }, [](int) {}, nullptr);
      if (ctr[0].v.load(std::memory_order_relaxed) < n0_items)
        for (int i; (i = ctr[0].v.fetch_add(1, std::memory_order_relaxed)) < n0_items;) run0(i);
      if (ctr[1].v.load(std::memory_order_relaxed) < n1_items)
        for (int i; (i = ctr[1].v.fetch_add(1, std::memory_order_relaxed)) < n1_items;) run1(i);
    };
    // gate, up, SwiGLU and the down-input quantiser of one 32-column block of one expert: one task, no join between
    auto gu_one = [&](int ei, int g) {
      const int e = active_[ei];
      const size_t r0 = e_off_[e];
      const int cnt = e_cnt_[e];
      const uint8_t* q = cnt < 2 ? s.qx.data() + (size_t)e_tok_[r0] * H * qb : s.qxe.data() + r0 * H * qb;
      const int32_t* ea = cnt < 2 ? s.eax.data() + (size_t)e_tok_[r0] * G : s.eaxe.data() + r0 * G;
      const v3_rm_mat mg = view(tp->gate_bb_[e], codes_[p][e].g), mu = view(tp->up_bb_[e], codes_[p][e].u);
      int rc = v3_rm_gemm_bf16(&mg, tier, q, H, ea, G, cnt, 32 * g, 32 * g + 32, s.gate.data() + r0 * Ip, Ip);
      if (!rc) rc = v3_rm_gemm_bf16(&mu, tier, q, H, ea, G, cnt, 32 * g, 32 * g + 32, s.up.data() + r0 * Ip, Ip);
      if (!rc)
        rc = v3_swiglu_quant(s.gate.data() + r0 * Ip, s.up.data() + r0 * Ip, Ip, cnt, e_w_.data() + r0, kt_contract_v3h::silu_table(),
                             g, g + 1, tier, nullptr, 0, s.qd.data() + r0 * Ip * qb, Ip, s.ead.data() + r0 * Gp, Gp);
      rec(rc);
    };
    if (cp && Gp >= 2) {   // KT_V3_CCD: CCD 0 the blocks [0, gA) of every expert, CCD 1 the blocks [gA, Gp)
      const int gA = kt_ccd::share(Gp, cp->n[0], cp->T), gB = Gp - gA;
      ccd_job(na * gA, na * gB, [&](int i) { gu_one(i / gA, i % gA); }, [&](int i) { gu_one(i / gB, gA + i % gB); });
    } else {
      pool->do_work_stealing_job(na * Gp, nullptr, [&](int task) { gu_one(task / Gp, task % Gp); }, nullptr);
    }
    kt_contract_v3h::check(err.load(), "gate/up/swiglu");
    // down, as served (per output column range, every active expert in turn); the task zeroes its own columns first and
    // hands them over at the end
    const int panels_h = H / 8;
    const int nth_dn = down_tasks(H, Ip);
    auto dn_one = [&](int ith) {
      const int c0 = 8 * (panels_h * ith / nth_dn), c1 = 8 * (panels_h * (ith + 1) / nth_dn);
      if (c1 <= c0) return;
      const int wd = c1 - c0;
      for (int t = 0; t < qlen; t++) std::memset(s.tok.data() + (size_t)t * H + c0, 0, sizeof(int64_t) * (size_t)wd);
      thread_local std::vector<int64_t> tmp;
      for (int e : active_) {
        const int m = e_cnt_[e];
        if (tmp.size() < (size_t)m * wd) tmp.resize((size_t)m * wd);
        std::memset(tmp.data(), 0, sizeof(int64_t) * (size_t)m * wd);
        const v3_rm_mat mw = view(tp->down_bb_[e], codes_[p][e].d, c0, c1);
        const size_t r0 = e_off_[e];
        const int rc = v3_rm_down_partial(&mw, tier, s.qd.data() + r0 * Ip * qb, Ip, s.ead.data() + r0 * Gp, Gp, m, bL,
                                          nullptr, 0, 0, wd, tmp.data(), wd);
        if (rc) { rec(rc); return; }
        for (int r = 0; r < m; r++) {
          int64_t* dst = s.tok.data() + (size_t)e_tok_[r0 + r] * H + c0;
          const int64_t* src = tmp.data() + (size_t)r * wd;
          for (int j = 0; j < wd; j++) dst[j] = (int64_t)((uint64_t)dst[j] + (uint64_t)src[j]);   // wrapping (#8)
        }
      }
      if (hand) {   // c0 and wd are multiples of 8 words: whole cache lines of the page-aligned hand-over buffer
        for (int t = 0; t < qlen; t++) {
          const int64_t* src = s.tok.data() + (size_t)t * H + c0;
          int64_t* dst = hand + (size_t)t * H + c0;
          for (int j = 0; j < wd; j += 2)
            _mm_stream_si128((__m128i*)(dst + j), _mm_loadu_si128((const __m128i*)(src + j)));
        }
        _mm_sfence();
      }
    };
    if (cp && nth_dn >= 2) {   // KT_V3_CCD: CCD 0 the column ranges [0, tA), CCD 1 the ranges [tA, nth_dn)
      const int tA = kt_ccd::share(nth_dn, cp->n[0], cp->T);
      ccd_job(tA, nth_dn - tA, [&](int i) { dn_one(i); }, [&](int i) { dn_one(tA + i); });
    } else {
      pool->do_work_stealing_job(nth_dn, nullptr, dn_one, nullptr);
    }
    kt_contract_v3h::check(err.load(), "down");
  }

  // ---- concept-r1 pre-read (KT_PREREAD, PREREAD_r1.md; see the namespace kt_preread above) ----
  // per instance: the CPU experts of its last kMaxHist decode calls (written by the hook only), and a test hint
  uint16_t pr_ids_[kt_preread::kMaxHist][kt_preread::kMaxRoutes] = {};
  uint8_t pr_w_[kt_preread::kMaxHist][kt_preread::kMaxRoutes] = {};
  uint8_t pr_n_[kt_preread::kMaxHist] = {};
  int pr_pos_ = 0, pr_filled_ = 0;
  uint16_t pr_hint_[kt_preread::kMaxPlan] = {};
  std::atomic<int> pr_hint_n_{0};

  static void pr_install() {
    kt_tq_idle_fn want = &TP_MOE::pr_hook, cur = kt_tq_idle_hook.load(std::memory_order_acquire);
    if (cur == want) return;
    kt_tq_idle_hook.store(want, std::memory_order_release);
    const auto& k = kt_preread::knobs();
    printf("[kt-preread] armed: mode %d, cap %d experts a window, list %d, history %d calls, rows <= %d, chunk %zu B\n",
           kt_preread::mode().load(), k.cap, k.topn, k.hist, k.maxq, k.chunk);
    fflush(stdout);
  }
  // an instance going away: pre-reads stop for good (mode 0, call-side records off) and a running one is waited for, so
  // nothing reads this instance's memory or uses its pool afterwards. Destroy the instances before their CPUInfer.
  void pr_forget() {
    kt_preread::mode().store(0, std::memory_order_release);
    kt_preread::armed().store(false, std::memory_order_release);
    const bool outside = !kt_preread::on_hook_thread;
    if (outside) kt_preread::outside_waiters().fetch_add(1, std::memory_order_acq_rel);
    {
      std::lock_guard<std::mutex> lk(sh().mu);
      auto& S = sh();
      if (S.pr_last == this) S.pr_last = nullptr;
      for (auto& r : S.pr_reg)
        if (r == this) r = nullptr;
      if (S.pr_ctx.self == this) S.pr_ctx.self = nullptr;
      S.pr_layout_pool = nullptr;   // a later pool may reuse the address: check it again
      S.pr_layout_ok = -1;
    }
    if (outside) kt_preread::outside_waiters().fetch_sub(1, std::memory_order_acq_rel);
  }
  static bool pr_stop(const typename Shared::PrCtx& c) {
    return c.pending->load(std::memory_order_relaxed) != 0 ||
           kt_preread::outside_waiters().load(std::memory_order_relaxed) != 0;
  }
  bool pr_holds(int e) {   // the CPU holds e now (not a GPU slot) and every part has its bytes and codes
    if (e < 0 || e >= this->config.expert_num || this->config.should_skip_expert(e)) return false;
    if ((int)codes_.size() != this->tp_count) return false;
    for (int p = 0; p < this->tp_count; p++) {
      auto* tp = part(p);
      if (!tp->gate_bb_[e] || !tp->gate_bb_[e]->b || !tp->up_bb_[e] || !tp->up_bb_[e]->b || !tp->down_bb_[e] ||
          !tp->down_bb_[e]->b || (int)codes_[p].size() <= e || !codes_[p][e].g || !codes_[p][e].u || !codes_[p][e].d)
        return false;
    }
    return true;
  }
  // the hook: on the task-queue worker, after a task has finished and nothing is queued (task_queue.cpp)
  static void pr_hook(const std::atomic<size_t>* pending) {
    using namespace kt_preread;
    on_hook_thread = true;
    if (!armed().load(std::memory_order_acquire)) return;
    auto& S = sh();
    std::unique_lock<std::mutex> lk(S.mu, std::try_to_lock);
    if (!lk.owns_lock()) return;   // a caller on another thread has the backend
    if (S.pr_agg_seq && pending->load(std::memory_order_relaxed) == 0) pr_aggregate();   // the last window's accounts
    TP_MOE* last = S.pr_last;
    const int q = S.pr_last_q;
    S.pr_last = nullptr;
    if (!last) return;             // the task was not a contract v3 call (or was seen already)
    const int L = last->config.layer_idx;
    if (L < 0 || L >= kMaxLayers) return;
    if (q > knobs().maxq) {        // prefill: no history, and the layer chain restarts
      S.pr_prev_layer = -1;
      return;
    }
    // bookkeeping, in every mode: the layer's instance, the layer order, this call's CPU experts
    S.pr_reg[L] = last;
    if (S.pr_prev_layer >= 0 && S.pr_prev_layer != L) S.pr_next1[S.pr_prev_layer] = (int16_t)(L + 1);
    S.pr_prev_layer = L;
    last->pr_push();
    const int md = mode().load(std::memory_order_acquire);
    if (md == 0) return;
    const int nl = S.pr_next1[L] - 1;
    TP_MOE* tgt = nl >= 0 ? S.pr_reg[nl] : nullptr;
    if (!tgt || tgt->config.pool != last->config.pool || !tgt->weights_loaded) return;
    if (S.pr_layout_pool != (const void*)tgt->config.pool) {   // once per pool; this window is spent on it
      char why[256];
      S.pr_layout_ok = tgt->pr_layout(why, sizeof why) ? 1 : 0;
      S.pr_layout_pool = (const void*)tgt->config.pool;
      printf("[kt-preread] layout %s: %s\n", S.pr_layout_ok ? "ok" : "REFUSED, the pre-read stays off", why);
      fflush(stdout);
      return;
    }
    if (S.pr_layout_ok != 1) return;
    Rec r;
    r.t_start = now_ns();
    r.layer = L;
    r.target = nl;
    const int n = md == 2 ? tgt->pr_plan_hint() : tgt->pr_plan_history();
    r.planned = n;
    const uint64_t seq_before = S.pr_agg_seq;
    if (n > 0 && knobs().cap > 0) {
      tgt->pr_run(n, pending, r);
    } else {
      r.t_end = now_ns();
      r.reason = R_NONE;
    }
    if (S.pr_agg_seq != seq_before) S.pr_agg_idx = S.pr_log_n;   // this window's record awaits its accounts
    S.pr_log[S.pr_log_n++ % kLog] = r;
  }
  // Once per pool, before the first window: the pre-read runs only where it cannot take a CPU from the call. The hook
  // thread (the task-queue worker: it joins the pre-read, then runs the call) must be pinned to one CPU, and every pool
  // thread to one CPU of its own, none of them the hook thread's. Otherwise a thread the call needs can sit behind a
  // pre-read thread on the same CPU for a scheduler slice: the first test (PREREAD_r1.md section 8) ran the task-queue
  // worker, the part's distributor and the test's main thread on one CPU, and a call waited 3.5 ms. One small pool job
  // (each thread reports its own affinity into fixed slots); `why` gets one line for the log.
  bool pr_layout(char* why, size_t nwhy) {
    using namespace kt_preread;
    auto& S = sh();
    const int hook_cpu = pinned_cpu();
    const int P = std::min(this->tp_count, kMaxParts);
    for (int p = 0; p < kMaxParts; p++)
      for (int t = 0; t < kMaxThreads; t++) S.pr_lay_cpu[p][t] = -2;
    TP_MOE* self = this;
    auto job = [self](int p) { self->pr_lay_part(p); };
    static_assert(sizeof(job) <= 16 && std::is_trivially_copyable_v<decltype(job)>, "fits std::function's local storage");
    this->config.pool->dispense_backend()->do_numa_job(job);
    if (hook_cpu < 0) {
      snprintf(why, nwhy, "the task-queue worker is not pinned to one CPU (set KT_TASKQUEUE_CPU to a CPU no pool thread uses)");
      return false;
    }
    uint64_t seen[CPU_SETSIZE / 64] = {};
    seen[hook_cpu / 64] |= 1ull << (hook_cpu % 64);
    int threads = 0;
    for (int p = 0; p < P; p++) {
      const int T = std::min(this->config.pool->config.subpool_thread_count[p], kMaxThreads);
      for (int t = 0; t < T; t++) {
        const int c = S.pr_lay_cpu[p][t];
        if (c < 0) {
          snprintf(why, nwhy, "part %d thread %d is %s (task-queue worker on CPU %d)", p, t,
                   c == -2 ? "not reached by a pool job" : "not pinned to one CPU", hook_cpu);
          return false;
        }
        if (seen[c / 64] & (1ull << (c % 64))) {
          snprintf(why, nwhy, "part %d thread %d shares CPU %d with %s", p, t, c,
                   c == hook_cpu ? "the task-queue worker" : "another pool thread");
          return false;
        }
        seen[c / 64] |= 1ull << (c % 64);
        threads++;
      }
    }
    snprintf(why, nwhy, "task-queue worker on CPU %d, %d pool threads on %d CPUs of their own", hook_cpu, threads, threads);
    return true;
  }
  void pr_lay_part(int p) {   // on part p's distributor: every thread of the part reports the one CPU it may run on
    using namespace kt_preread;
    if (p < 0 || p >= kMaxParts) return;
    const int T = std::min(this->config.pool->config.subpool_thread_count[p], kMaxThreads);
    if (T < 1) return;
    auto body = [p](int tid) {
      if (tid >= 0 && tid < kMaxThreads) sh().pr_lay_cpu[p][tid] = pinned_cpu();
    };
    auto none = [](int) {};
    static_assert(sizeof(body) <= 16 && std::is_trivially_copyable_v<decltype(body)>, "fits std::function's local storage");
    this->config.pool->get_subpool(p)->do_work_stealing_job(T, body, none, nullptr);
  }
  // this instance's last call (active_ = its CPU experts ascending, e_cnt_ = rows a expert) into its history
  void pr_push() {
    using namespace kt_preread;
    const int h = pr_pos_;
    int n = 0;
    for (int e : active_) {
      if (n >= kMaxRoutes) break;
      if (e < 0 || e >= kMaxE) continue;
      pr_ids_[h][n] = (uint16_t)e;
      pr_w_[h][n] = (uint8_t)std::min(e_cnt_[e], 255);
      n++;
    }
    pr_n_[h] = (uint8_t)n;
    pr_pos_ = (h + 1) % kMaxHist;
    if (pr_filled_ < kMaxHist) pr_filled_++;
  }
  // freq<hist> top<n>: the experts most often run in this layer's last `hist` calls (rows counted), ties to the more
  // recent; only experts the CPU holds now. Into sh().pr_plan, rank order. Returns the count.
  int pr_plan_history() {
    using namespace kt_preread;
    auto& S = sh();
    if (this->config.expert_num > kMaxE) return 0;
    const int nh = std::min(knobs().hist, pr_filled_);
    if (++S.pr_gen == 0) {
      std::memset(S.pr_stamp, 0, sizeof S.pr_stamp);
      S.pr_gen = 1;
    }
    const uint32_t g = S.pr_gen;
    int nc = 0, order = 0;
    for (int a = 1; a <= nh; a++) {   // newest call first, so the first appearance ranks recency
      const int h = (pr_pos_ - a + kMaxHist) % kMaxHist;
      for (int i = 0; i < pr_n_[h]; i++) {
        const int e = pr_ids_[h][i];
        if (S.pr_stamp[e] != g) {
          S.pr_stamp[e] = g;
          S.pr_cnt[e] = 0;
          S.pr_first[e] = (int16_t)order++;
          S.pr_cand[nc++] = (int16_t)e;
        }
        S.pr_cnt[e] = (uint16_t)(S.pr_cnt[e] + pr_w_[h][i]);
      }
    }
    const int want = std::min(knobs().topn, knobs().cap);
    int n = 0;
    while (n < want) {
      int best = -1;
      for (int i = 0; i < nc; i++) {
        const int e = S.pr_cand[i];
        if (e < 0) continue;
        if (best < 0) { best = i; continue; }
        const int b = S.pr_cand[best];
        if (S.pr_cnt[e] > S.pr_cnt[b] || (S.pr_cnt[e] == S.pr_cnt[b] && S.pr_first[e] < S.pr_first[b])) best = i;
      }
      if (best < 0) break;
      const int e = S.pr_cand[best];
      S.pr_cand[best] = -1;
      if (pr_holds(e)) S.pr_plan[n++] = e;
    }
    return n;
  }
  int pr_plan_hint() {   // tests: the list the test gave for this layer's next window (consumed)
    using namespace kt_preread;
    auto& S = sh();
    const int m = pr_hint_n_.exchange(0, std::memory_order_acq_rel);
    int n = 0;
    for (int i = 0; i < m && n < knobs().cap; i++)
      if (pr_holds(pr_hint_[i])) S.pr_plan[n++] = pr_hint_[i];
    return n;
  }
  // one window: every part reads its slice of the planned experts on its own threads; returns when all have left
  void pr_run(int n, const std::atomic<size_t>* pending, kt_preread::Rec& r) {
    using namespace kt_preread;
    auto& S = sh();
    S.pr_ctx.self = this;
    S.pr_ctx.pending = pending;
    S.pr_ctx.n = std::min(n, knobs().cap);
    S.pr_ctx.chunk = knobs().chunk;
    S.pr_ctx.seq = ++S.pr_seq;
    const typename Shared::PrCtx* ctx = &S.pr_ctx;
    if (pr_stop(*ctx)) {
      r.t_dispatch = r.t_seen = r.t_end = now_ns();
      r.reason = R_STOP_BEFORE;
      return;
    }
    auto job = [ctx](int p) { ctx->self->pr_part(p, ctx); };
    static_assert(sizeof(job) <= 16 && std::is_trivially_copyable_v<decltype(job)>,
                  "the pre-read job must fit std::function's local storage (no allocation)");
    S.pr_agg_parts = std::min(this->tp_count, kMaxParts);
    for (int p = 0; p < S.pr_agg_parts; p++)
      S.pr_agg_T[p] = std::min(this->config.pool->config.subpool_thread_count[p], kMaxThreads);
    r.t_dispatch = now_ns();
    this->config.pool->dispense_backend()->do_numa_job(job);
    r.t_end = now_ns();   // the threads' accounts are summed later (pr_aggregate), off the next call's path
    r.reason = R_STOPPED;
    S.pr_agg_seq = ctx->seq;
  }
  // the threads' accounts of the last window into its record (at the next window's start, or for the log binding)
  static void pr_aggregate() {
    using namespace kt_preread;
    auto& S = sh();
    if (!S.pr_agg_seq || S.pr_agg_idx >= S.pr_log_n || S.pr_log_n - S.pr_agg_idx > (uint64_t)kLog) {
      S.pr_agg_seq = 0;
      return;
    }
    Rec& r = S.pr_log[S.pr_agg_idx % kLog];
    uint64_t bytes = 0, seen = 0;
    int emin = 1 << 30, emax = 0, threads = 0;
    bool stopped = false;
    for (int p = 0; p < S.pr_agg_parts; p++)
      for (int t = 0; t < S.pr_agg_T[p]; t++) {
        const ThreadStat& st = S.pr_thr[p][t];
        if (st.seq != S.pr_agg_seq) {   // never ran this window (its part saw the stop before dispatching)
          stopped = true;
          emin = 0;
          continue;
        }
        threads++;
        bytes += st.bytes;
        if (st.t_seen && (!seen || st.t_seen < seen)) seen = st.t_seen;
        emin = std::min(emin, (int)st.experts);
        emax = std::max(emax, (int)st.experts);
        stopped |= st.stopped != 0;
      }
    r.bytes = bytes;
    r.t_seen = seen;
    r.experts_min = emin == (1 << 30) ? 0 : emin;
    r.experts_max = emax;
    r.threads = threads;
    r.reason = stopped ? R_STOPPED : R_DONE;
    S.pr_agg_seq = 0;
  }
  // on part p's distributor thread: one job on the part's own threads; each runs pr_thread once (the init function of
  // the job, called with its fixed thread index), so every thread reads a fixed share
  void pr_part(int p, const typename Shared::PrCtx* ctx) {
    using namespace kt_preread;
    if (p < 0 || p >= kMaxParts) return;
    const int T = std::min(this->config.pool->config.subpool_thread_count[p], kMaxThreads);
    if (T < 1) return;
    if (pr_stop(*ctx)) {
      ThreadStat& st = sh().pr_thr[p][0];
      st.bytes = 0;
      st.t_in = st.t_seen = st.t_out = now_ns();
      st.experts = 0;
      st.stopped = 1;
      st.seq = ctx->seq;
      return;
    }
    auto body = [ctx, p](int tid) { ctx->self->pr_thread(p, tid, ctx); };
    auto none = [](int) {};
    static_assert(sizeof(body) <= 16 && std::is_trivially_copyable_v<decltype(body)>,
                  "the pre-read job must fit std::function's local storage (no allocation)");
    this->config.pool->get_subpool(p)->do_work_stealing_job(T, body, none, nullptr);
  }
  // one pool thread: its share of every planned expert of part p, in rank order: gate, up, down nibbles, then the
  // three code tables; plain loads, the pending count checked before every chunk
  void pr_thread(int p, int tid, const typename Shared::PrCtx* ctx) {
    using namespace kt_preread;
    const int T = std::min(this->config.pool->config.subpool_thread_count[p], kMaxThreads);
    if (tid < 0 || tid >= T) return;
    auto& S = sh();
    ThreadStat& st = S.pr_thr[p][tid];
    const uint64_t t_in = now_ns();
    uint64_t bytes = 0, seen = 0;
    uint32_t done = 0, stopped = 0;
    auto* tp = part(p);
    const size_t ch = ctx->chunk;
    // pathb-cpu-r1 KT_V3_CCD: on a split part each thread reads its share of what its own CCD computes (part_clean's
    // shares: gate / up rows [32 g0, 32 g1), down rows [r0, r1), and their codes), split among the CCD's threads in rank
    // order; otherwise (k, nk) = (tid, T) over whole ranges, as before
    const kt_ccd::Part* cp = ccd_part(p);
    const int Ip = this->tp_configs[p].intermediate_size, Hh = this->config.hidden_size, Gp = Ip / 32;
    const int ndn = down_tasks(Hh, Ip);
    const bool by_ccd = cp != nullptr && Gp >= 2 && ndn >= 2 && contract_v3_clean() != 0;
    int k = tid, nk = T;
    size_t g0 = 0, g1 = 0, r0 = 0, r1 = 0;
    if (by_ccd) {
      const int c = cp->ccd[tid];
      k = cp->rank[tid];
      nk = cp->n[c];
      const int gA = kt_ccd::share(Gp, cp->n[0], cp->T), tA = kt_ccd::share(ndn, cp->n[0], cp->T);
      g0 = c ? 32 * (size_t)gA : 0;
      g1 = c ? 32 * (size_t)Gp : 32 * (size_t)gA;
      const int ph = Hh / 8, t0 = c ? tA : 0, t1 = c ? ndn : tA;
      r0 = 8 * (size_t)(ph * t0 / ndn);
      r1 = 8 * (size_t)(ph * t1 / ndn);
    }
    for (int i = 0; i < ctx->n && !stopped; i++) {
      const int e = S.pr_plan[i];
      const auto& g = *tp->gate_bb_[e];
      const auto& u = *tp->up_bb_[e];
      const auto& d = *tp->down_bb_[e];
      const Codes& c = codes_[p][e];
      const uint8_t* base[6] = {g.b, u.b, d.b, c.g, c.u, c.d};
      const size_t len[6] = {(size_t)g.n * g.k / 2, (size_t)u.n * u.k / 2, (size_t)d.n * d.k / 2,
                             (size_t)g.n * g.k_group_count, (size_t)u.n * u.k_group_count, (size_t)d.n * d.k_group_count};
      size_t clo[6] = {0, 0, 0, 0, 0, 0};
      size_t chi[6] = {len[0], len[1], len[2], len[3], len[4], len[5]};
      if (by_ccd) {
        const size_t rl[6][2] = {{g0 * (size_t)(g.k / 2), g1 * (size_t)(g.k / 2)},
                                 {g0 * (size_t)(u.k / 2), g1 * (size_t)(u.k / 2)},
                                 {r0 * (size_t)(d.k / 2), r1 * (size_t)(d.k / 2)},
                                 {g0 * (size_t)g.k_group_count, g1 * (size_t)g.k_group_count},
                                 {g0 * (size_t)u.k_group_count, g1 * (size_t)u.k_group_count},
                                 {r0 * (size_t)d.k_group_count, r1 * (size_t)d.k_group_count}};
        for (int s = 0; s < 6; s++) {
          clo[s] = std::min(rl[s][0], len[s]);
          chi[s] = std::min(rl[s][1], len[s]);
        }
      }
      for (int s = 0; s < 6 && !stopped; s++) {
        const size_t L = chi[s] > clo[s] ? chi[s] - clo[s] : 0;
        const size_t lo = clo[s] + L * k / nk / 64 * 64;
        const size_t hi = k + 1 == nk ? chi[s] : clo[s] + L * (k + 1) / nk / 64 * 64;
        for (size_t o = lo; o < hi; o += ch) {
          if (pr_stop(*ctx)) {
            stopped = 1;
            seen = now_ns();
            break;
          }
          const size_t m = std::min(ch, hi - o);
          touch(base[s] + o, m);
          bytes += m;
        }
      }
      if (!stopped) done++;
    }
    st.bytes = bytes;
    st.t_in = t_in;
    st.t_seen = seen;
    st.t_out = now_ns();
    st.experts = done;
    st.stopped = stopped;
    st.seq = ctx->seq;
  }

  // ---- concept-r1 pre-read: bindings for tests ----
  // -1: back to the environment; 0 off (bookkeeping only); 1 history; 2 the test's hinted lists. Any value but -1 arms.
  void contract_v3_set_preread(int m) {
    if (m < -1 || m > 2) throw std::runtime_error("contract v3: pre-read mode must be -1, 0, 1 or 2");
    if (m == -1) {
      kt_preread::armed().store(kt_preread::knobs().env_on, std::memory_order_release);
      kt_preread::mode().store(kt_preread::knobs().env_on ? 1 : 0, std::memory_order_release);
    } else {
      kt_preread::mode().store(m, std::memory_order_release);
      kt_preread::armed().store(true, std::memory_order_release);
    }
    if (kt_preread::armed().load()) pr_install();
  }
  std::string contract_v3_preread() {
    const auto& k = kt_preread::knobs();
    char b[256];
    snprintf(b, sizeof b, "armed=%d mode=%d hook=%d cap=%d topn=%d hist=%d maxq=%d chunk=%zu env=%d layout=%d",
             (int)kt_preread::armed().load(), kt_preread::mode().load(),
             kt_tq_idle_hook.load() == &TP_MOE::pr_hook ? 1 : 0, k.cap, k.topn, k.hist, k.maxq, k.chunk, (int)k.env_on,
             sh().pr_layout_ok);
    return std::string(b);
  }
  // the windows recorded since the last call, 13 int64 each: t_start, t_dispatch, t_seen, t_end (CLOCK_MONOTONIC ns),
  // bytes, layer, target, planned, experts_min, experts_max, reason (0 none, 1 stopped by a call, 2 done, 3 stopped
  // before dispatch), threads, index. Stops a running pre-read first.
  std::vector<int64_t> contract_v3_preread_log() {
    const bool outside = !kt_preread::on_hook_thread;
    if (outside) kt_preread::outside_waiters().fetch_add(1, std::memory_order_acq_rel);
    std::lock_guard<std::mutex> lk(sh().mu);
    if (outside) kt_preread::outside_waiters().fetch_sub(1, std::memory_order_acq_rel);
    auto& S = sh();
    pr_aggregate();
    const uint64_t n = S.pr_log_n, first = n > (uint64_t)kt_preread::kLog ? n - kt_preread::kLog : 0;
    std::vector<int64_t> out;
    out.reserve((size_t)(n - first) * 13);
    for (uint64_t i = first; i < n; i++) {
      const auto& r = S.pr_log[i % kt_preread::kLog];
      for (int64_t v : {(int64_t)r.t_start, (int64_t)r.t_dispatch, (int64_t)r.t_seen, (int64_t)r.t_end, (int64_t)r.bytes,
                        (int64_t)r.layer, (int64_t)r.target, (int64_t)r.planned, (int64_t)r.experts_min,
                        (int64_t)r.experts_max, (int64_t)r.reason, (int64_t)r.threads, (int64_t)i})
        out.push_back(v);
    }
    S.pr_log_n = 0;
    S.pr_agg_seq = 0;
    return out;
  }
  // tests (mode 2): the experts to read in this layer's next window, int64 ids; consumed by that window
  void contract_v3_preread_hint(intptr_t ids_ptr, int n) {
    const int64_t* ids = (const int64_t*)ids_ptr;
    const int m = std::max(0, std::min(n, kt_preread::kMaxPlan));
    for (int i = 0; i < m; i++) pr_hint_[i] = (uint16_t)std::max<int64_t>(0, std::min<int64_t>(ids[i], 65535));
    pr_hint_n_.store(m, std::memory_order_release);
  }
  int64_t contract_v3_tq_last_qwait_ns() { return (int64_t)kt_tq_last_qwait_ns.load(std::memory_order_relaxed); }

  // BF16 of the CPU-only r3 sum (warm-up and tests; the served path uses forward_partials)
  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output, bool incremental) {
    if (incremental) throw std::runtime_error("contract v3: incremental forward not supported");
    const int qlen = *qlen_ptr;
    auto& c = this->config;
    std::vector<int64_t> S((size_t)qlen * c.hidden_size);
    forward_partials_binding((intptr_t)qlen_ptr, k, (intptr_t)expert_ids, (intptr_t)weights, (intptr_t)input, (intptr_t)S.data());
    kt_contract_v3h::check(v3_finish_rows(S.data(), c.hidden_size, qlen, 0, c.hidden_size, get_bl(), (uint16_t*)output,
                                          c.hidden_size, nullptr, 0),
                           "finish");
  }
  // the CPUInfer forward_task path (bind_moe_module binds &MoeClass::forward_binding): v3 too, never his MXFP4 arithmetic
  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output, incremental);
  }
  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output, bool incremental = false) override {
    int q = qlen;
    forward(&q, k, expert_ids, weights, input, output, incremental);
  }
};

#endif  // KT_ENABLE_CONTRACT_V3
#endif  // CPUINFER_OPERATOR_AVX2_CONTRACT_V3H_MOE_H
