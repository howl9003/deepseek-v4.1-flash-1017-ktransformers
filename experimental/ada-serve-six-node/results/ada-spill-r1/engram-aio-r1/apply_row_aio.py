"""row_store.cpp with DSV41_ENGRAM_AIO: a call's misses in one io_submit, polled by the callback thread. Default off. argv: source, destination."""
import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text(encoding='utf-8')
READ = r'''// A call's misses in one io_submit, polled here: no thread to wake. A row the batch did not bring takes the plain read,
// which counts its own failure. An error reaping leaves reads in flight, so the store stops using this path and its pages for good.
template <class Miss>
void aio_read_rows(Store *s, Miss *m, size_t n) {
  iocb cbs[kAioRows];
  iocb *list[kAioRows];
  uint32_t delta[kAioRows];
  io_event events[kAioRows];
  for (size_t i = 0; i < n; i++) {
    const uint64_t offset = s->weight_offset + uint64_t(m[i].id) * kDim;
    const uint64_t base = offset & ~uint64_t(kPage - 1);
    delta[i] = uint32_t(offset - base);
    std::memset(&cbs[i], 0, sizeof(iocb));
    cbs[i].aio_fildes = uint32_t(s->fd);
    cbs[i].aio_lio_opcode = IOCB_CMD_PREAD;
    cbs[i].aio_buf = reinterpret_cast<uint64_t>(s->aio_pages + i * kAioPages * kPage);
    cbs[i].aio_nbytes = ((delta[i] + kDim + kPage - 1) / kPage) * kPage;
    cbs[i].aio_offset = int64_t(base);
    cbs[i].aio_data = i;
    list[i] = &cbs[i];
  }
  size_t sent = 0;
  while (sent < n) {
    const long r = syscall(SYS_io_submit, s->aio, long(n - sent), list + sent);
    if (r > 0) sent += size_t(r);
    else if (!(r < 0 && (errno == EINTR || errno == EAGAIN))) break;
  }
  timespec zero{0, 0};
  size_t done = 0;
  while (done < sent) {
    const long r = syscall(SYS_io_getevents, s->aio, 0L, long(sent - done), events, &zero);
    if (r < 0) {
      if (errno == EINTR) continue;
      std::fprintf(stderr, "row_store: aio reap failed errno=%d, back to the thread pool\n", errno);
      s->aio = 0;
      break;
    }
    for (long j = 0; j < r; j++) {
      const size_t i = size_t(events[j].data);
      if (events[j].res < int64_t(delta[i] + kDim)) continue;
      std::memcpy(m[i].row, s->aio_pages + i * kAioPages * kPage + delta[i], kDim);
      std::memcpy(m[i].row + kDim, s->scales_ram + (uint64_t(m[i].id) - s->row_lo) * kScales, kScales);
      m[i].ok = true;
      s->reads.fetch_add(1, std::memory_order_relaxed);
    }
    done += size_t(r);
  }
  for (size_t i = 0; i < n; i++)
    if (!m[i].ok) m[i].ok = read_row(s, uint64_t(m[i].id), m[i].row);
}

'''
PAIRS = [
    ('#include <unistd.h>\n\nnamespace {\n', '#include <unistd.h>\n#include <ctime>\n#include <linux/aio_abi.h>\n#include <sys/syscall.h>\n\nnamespace {\n'),
    ('constexpr int kLockShards = 256;\n', 'constexpr int kLockShards = 256;\nconstexpr size_t kAioRows = 256;   // a decode call has 24 rows; larger batches stay on the thread pool\nconstexpr size_t kAioPages = 2;    // a row spans at most two 4 KiB pages\n'),
    ('  std::atomic<uint64_t> warm_rows{0}, warm_ns{0};\n};\n',
     '  std::atomic<uint64_t> warm_rows{0}, warm_ns{0};\n'
     '  // DSV41_ENGRAM_AIO=1: up to kAioRows misses of a call go to the device in one io_submit and the\n'
     '  // callback thread polls them, so a miss wakes no thread. At token start the service CPUs are busy:\n'
     '  // layer 1 took 1.1 ms a lookup where layer 14, later in the token, took 0.35 ms on the same rows (11-01).\n'
     '  aio_context_t aio = 0;\n'
     '  uint8_t *aio_pages = nullptr;\n};\n'),
    ('void scales_free(Store *s) {\n', READ + 'void scales_free(Store *s) {\n'),
    ('  if (!s->resident) io_start(s);\n',
     '  if (!s->resident) io_start(s);\n'
     '  if (const char *e = getenv("DSV41_ENGRAM_AIO"); e && std::strcmp(e, "1") == 0 && !s->resident) {\n'
     '    void *p = nullptr;\n'
     '    if (!posix_memalign(&p, kPage, kAioRows * kAioPages * kPage) && !syscall(SYS_io_setup, long(kAioRows), &s->aio)) {\n'
     '      s->aio_pages = static_cast<uint8_t *>(p);\n'
     '    } else {\n'
     '      std::fprintf(stderr, "row_store_open: aio off errno=%d\\n", errno);\n'
     '      s->aio = 0;\n'
     '      std::free(p);\n'
     '    }\n'
     '  }\n'),
    ('    if (s->resident || misses.size() == 1 || s->io.threads.empty()) {\n',
     '    if (s->aio && s->scales_ram && misses.size() > 1 && misses.size() <= kAioRows) {\n'
     '      aio_read_rows(s, misses.data(), misses.size());\n'
     '    } else if (s->resident || misses.size() == 1 || s->io.threads.empty()) {\n'),
    ('  warm_stop(s);\n  io_stop(s);\n',
     '  warm_stop(s);\n  io_stop(s);\n  if (s->aio) {\n    syscall(SYS_io_destroy, s->aio);\n    std::free(s->aio_pages);\n  }\n'),
]
for old, new in PAIRS:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
Path(sys.argv[2]).write_bytes(s.encode('utf-8'))
