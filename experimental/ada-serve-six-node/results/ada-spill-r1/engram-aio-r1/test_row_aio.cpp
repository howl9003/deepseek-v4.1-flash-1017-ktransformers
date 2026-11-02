// The AIO row path against the thread pool on the real table: same bytes on random batches around every size edge, rows across a page edge, and bad ids. argv: file rows woff soff.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Store;
extern "C" Store *row_store_open(const char *, uint64_t, uint64_t, uint64_t, uint64_t, int);
extern "C" void row_store_range(Store *, uint64_t, uint64_t);
extern "C" void row_store_lookup(void *);
extern "C" void row_store_stats(Store *, uint64_t *);
extern "C" void row_store_close(Store *);
struct Work { Store *store; const int64_t *ids; uint8_t *weights, *scales; uint64_t count; };

int main(int argc, char **argv) {
  const char *path = argv[1];
  const uint64_t rows = strtoull(argv[2], 0, 10), woff = strtoull(argv[3], 0, 10), soff = strtoull(argv[4], 0, 10);
  unsetenv("DSV41_ENGRAM_AIO");
  Store *a = row_store_open(path, rows, woff, soff, 0, 0);
  setenv("DSV41_ENGRAM_AIO", "1", 1);
  Store *b = row_store_open(path, rows, woff, soff, 0, 0);
  if (!a || !b) return 2;
  row_store_range(a, 0, rows);
  row_store_range(b, 0, rows);
  std::mt19937_64 rng(1017);
  long calls = 0, compared = 0, bad = 0, edge = 0;
  for (int it = 0; it < 900; it++) {
    const size_t sizes[] = {2, 3, 24, 24, 24, 48, 255, 256, 257, 300};
    const size_t n = sizes[it % 10];
    std::vector<int64_t> ids(n);
    for (auto &x : ids) x = int64_t(rng() % rows);
    if (it % 5 == 0) {
      // rows that straddle a 4 KiB page, and the file's first and last rows
      for (size_t i = 0; i < n; i += 2) {
        const uint64_t page = (rng() % ((rows * 256 + woff) / 4096 - 2) + 1) * 4096;
        ids[i] = page > woff + 100 ? int64_t((page - woff - 100) / 256) : 0;
        edge++;
      }
      ids[0] = 0;
      ids[n - 1] = int64_t(rows - 1);
    }
    std::vector<uint8_t> wa(n * 256), sa(n * 8), wb(n * 256, 0xee), sb(n * 8, 0xee);
    Work ja{a, ids.data(), wa.data(), sa.data(), n}, jb{b, ids.data(), wb.data(), sb.data(), n};
    row_store_lookup(&ja);
    row_store_lookup(&jb);
    calls++;
    compared += long(n);
    if (std::memcmp(wa.data(), wb.data(), n * 256) || std::memcmp(sa.data(), sb.data(), n * 8)) bad++;
  }
  uint64_t st[11], sb[11];
  row_store_stats(a, st);
  row_store_stats(b, sb);
  printf("ROW_AIO calls %ld rows %ld page edges %ld mismatched calls %ld errors %llu %llu\n", calls, compared, edge, bad, (unsigned long long)st[4], (unsigned long long)sb[4]);
  for (int pass = 0; pass < 2; pass++) {
    Store *s = pass ? b : a;
    std::vector<double> t;
    for (int it = 0; it < 2000; it++) {
      int64_t ids[24];
      uint8_t w[24 * 256], sc[24 * 8];
      for (auto &x : ids) x = int64_t(rng() % rows);
      Work j{s, ids, w, sc, 24};
      auto t0 = std::chrono::steady_clock::now();
      row_store_lookup(&j);
      t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    printf("%s 24 rows: median %.0f us, p90 %.0f, p99 %.0f\n", pass ? "aio    " : "threads", t[1000], t[1800], t[1980]);
  }
  row_store_close(a);
  row_store_close(b);
  return bad ? 1 : 0;
}
