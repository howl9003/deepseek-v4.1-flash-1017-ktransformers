/**
 * @Description  : pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3, 14 Nov): reader of the block-ordered
 * expert image (format pathb-expert-image-v1, concept-r1/trials/queue/staged-ns1/img_format.py) for
 * TP_MOE<AVX2_MXFP4_MOE_TP>::load_weights_image. Self-contained (C++20 + POSIX, no kt-kernel header), so
 * tests/test_kt_image.cpp compiles and checks it alone.
 *
 * Layout of one layer's file: for each projection p (gate, up, down) and block b of 32 intermediate columns, one chunk
 * holding that block of every expert in logical order (gate/up: 32 rows x hidden/2 bytes an expert; down: hidden rows x
 * 16 bytes an expert); the weight chunks first, then the E8M0 scale chunks in the same order (gate/up: 32 rows x
 * hidden/32 bytes; down: hidden rows x 1 byte). These are the bytes his loader puts in the arenas with
 * KT_ARENA_SCALE_BYTES=1: a part holding blocks [b0, b0 + nb) needs exactly chunks b0..b0+nb-1 of each projection and
 * section, and scatter() puts each expert's piece where his loader would (gate/up rows contiguous at (b - b0) * 32
 * rows; down 16 bytes into every row at byte (b - b0) * 16 of the part's Ip/2; down scales one byte a row).
 *
 * Every chunk is read into a staging buffer (O_DIRECT where the file system takes it), its sha256 compared with the
 * manifest's digest, and only then scattered (verify, then copy). Problems are returned as text, never thrown.
 */
#ifndef CPUINFER_OPERATOR_AVX2_KT_IMAGE_H
#define CPUINFER_OPERATOR_AVX2_KT_IMAGE_H

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace kt_image {

constexpr int kBlock = 32;          // intermediate columns a block (the part granule)
constexpr int kGroup = 32;          // MXFP4 group
constexpr uint64_t kAlign = 4096;   // O_DIRECT offsets, lengths and buffers

// SHA-256 (FIPS 180-4), portable C++: the 4 Nov prepack loader's (prepack-loader-r1/expert_prepack.hpp), checked
// there against hashlib on edge lengths and real stripes; tests/test_kt_image.cpp checks it again on the FIPS vectors.
struct Sha256 {
  uint32_t h[8];
  uint64_t len = 0;
  uint8_t buf[64];
  size_t n = 0;

  Sha256() {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h, iv, sizeof(h));
  }
  static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  void update(const uint8_t* p, size_t m) {
    len += m;
    if (n) {
      const size_t t = std::min(m, (size_t)64 - n);
      std::memcpy(buf + n, p, t);
      n += t; p += t; m -= t;
      if (n < 64) return;
      block(buf);
      n = 0;
    }
    for (; m >= 64; p += 64, m -= 64) block(p);
    if (m) { std::memcpy(buf, p, m); n = m; }
  }
  void final(uint8_t out[32]) {
    const uint64_t bits = len * 8;
    const uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (n != 56) update(&zero, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(lb, 8);
    for (int i = 0; i < 8; i++) {
      out[4 * i] = (uint8_t)(h[i] >> 24); out[4 * i + 1] = (uint8_t)(h[i] >> 16);
      out[4 * i + 2] = (uint8_t)(h[i] >> 8); out[4 * i + 3] = (uint8_t)h[i];
    }
  }
};

// One layer of the image, as the Python side read it from the pinned manifest.
struct Spec {
  std::string path;
  uint64_t file_bytes = 0;
  int E = 0, H = 0, I = 0;
  std::string digests;   // 32 bytes each: weights (gate, up, down) x blocks, then scales in the same order
  bool ready = false;

  int nblk() const { return I / kBlock; }
  uint64_t w_per() const { return (uint64_t)kBlock * H / 2; }        // one expert's bytes in a weight chunk
  uint64_t s_per() const { return (uint64_t)kBlock * H / kGroup; }   // ... in a scale chunk
  uint64_t w_chunk() const { return (uint64_t)E * w_per(); }
  uint64_t s_chunk() const { return (uint64_t)E * s_per(); }
  uint64_t w_off(int p, int b) const { return ((uint64_t)p * nblk() + b) * w_chunk(); }
  uint64_t s_off(int p, int b) const { return 3ull * nblk() * w_chunk() + ((uint64_t)p * nblk() + b) * s_chunk(); }
  uint64_t layer_bytes() const { return 3ull * nblk() * (w_chunk() + s_chunk()); }
  const uint8_t* digest(bool scales, int p, int b) const {
    return (const uint8_t*)digests.data() + 32ull * ((uint64_t)((scales ? 3 : 0) + p) * nblk() + b);
  }
};

// geometry = [experts, hidden, intermediate, block, group]. "" or the reason.
inline std::string make_spec(Spec& s, const std::string& path, uint64_t file_bytes, const std::vector<int64_t>& g,
                             const std::string& digests) {
  s = Spec{};
  if (g.size() != 5) return "image: geometry must be [experts, hidden, intermediate, block, group]";
  if (g[3] != kBlock || g[4] != kGroup) return "image: block and group must be 32";
  if (g[0] <= 0 || g[0] > 65536 || g[1] <= 0 || g[1] > (1 << 20) || g[2] <= 0 || g[2] > (1 << 20))
    return "image: geometry out of range";
  s.E = (int)g[0];
  s.H = (int)g[1];
  s.I = (int)g[2];
  if (s.H % kGroup || s.I % kBlock || s.H % 8) return "image: geometry not on 32-blocks";
  if (s.w_chunk() % kAlign || s.s_chunk() % kAlign) return "image: chunks are not 4 KiB multiples";
  if (digests.size() != 32ull * 6 * (uint64_t)s.nblk()) return "image: the digest list is not 6 x blocks x 32 bytes";
  if (file_bytes != s.layer_bytes()) return "image: the manifest's file size is not the geometry's";
  if (path.empty()) return "image: no file";
  s.path = path;
  s.file_bytes = file_bytes;
  s.digests = digests;
  s.ready = true;
  return "";
}

// The layer's file, open for one load: O_DIRECT where the file system takes it.
struct File {
  int fd = -1;
  bool direct = false;
  std::string open(const Spec& s) {
    fd = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    direct = fd >= 0;
    if (fd < 0 && errno == EINVAL) fd = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return "image: cannot open " + s.path + ": " + std::strerror(errno);
    struct stat st;
    if (fstat(fd, &st) != 0) return "image: fstat of " + s.path + " failed";
    if ((uint64_t)st.st_size != s.file_bytes)
      return "image: " + s.path + " is " + std::to_string((long long)st.st_size) + " bytes, the manifest says " +
             std::to_string(s.file_bytes);
    return "";
  }
  ~File() {
    if (fd >= 0) ::close(fd);
  }
};

// [off, off + n) into buf (kAlign-aligned; n a multiple of kAlign), then its sha256 against want. "" or the reason.
inline std::string read_verify(int fd, uint64_t off, uint64_t n, uint8_t* buf, const uint8_t* want) {
  uint64_t got = 0;
  while (got < n) {
    const ssize_t r = ::pread(fd, buf + got, n - got, (off_t)(off + got));
    if (r < 0) {
      if (errno == EINTR) continue;
      return std::string("image: read error: ") + std::strerror(errno);
    }
    if (r == 0) return "image: short read at " + std::to_string(off + got);
    got += (uint64_t)r;
  }
  Sha256 h;
  h.update(buf, n);
  uint8_t d[32];
  h.final(d);
  if (std::memcmp(d, want, 32) != 0) return "image: a chunk's sha256 differs from the manifest (offset " + std::to_string(off) + ")";
  return "";
}

// One part's destinations. lid[e] < 0: physical expert e is not held by this part's CPU pool (a GPU expert);
// w[p][e] / s[p][e]: the base of e's bytes in the part's gate, up, down weight and scale-byte buffers.
struct PartDst {
  int Ip = 0, b0 = 0;
  std::vector<int64_t> lid;
  std::vector<uint8_t*> w[3], s[3];
};

// One verified chunk (projection p, weights or scales, block b) into the part's buffers, as his loader places them.
inline void scatter(const Spec& sp, const PartDst& d, int p, bool scales, int b, const uint8_t* chunk) {
  const int rel = b - d.b0;
  const int E = (int)d.lid.size();
  const int H = sp.H;
  if (p < 2) {   // gate, up: the block's 32 rows are contiguous in the part's [Ip rows] of the expert
    const uint64_t per = scales ? sp.s_per() : sp.w_per();
    for (int e = 0; e < E; e++) {
      if (d.lid[e] < 0) continue;
      std::memcpy((scales ? d.s[p][e] : d.w[p][e]) + (uint64_t)rel * per, chunk + (uint64_t)d.lid[e] * per, per);
    }
    return;
  }
  if (!scales) {   // down: 16 bytes into each of the H rows of Ip/2 bytes
    const uint64_t per = sp.w_per(), row = (uint64_t)d.Ip / 2;
    for (int e = 0; e < E; e++) {
      if (d.lid[e] < 0) continue;
      const uint8_t* src = chunk + (uint64_t)d.lid[e] * per;
      uint8_t* dst = d.w[2][e] + (uint64_t)rel * (kBlock / 2);
      for (int h = 0; h < H; h++) std::memcpy(dst + (uint64_t)h * row, src + (uint64_t)h * (kBlock / 2), kBlock / 2);
    }
    return;
  }
  const uint64_t per = sp.s_per(), row = (uint64_t)d.Ip / kGroup;   // down scales: one byte into each of the H rows
  for (int e = 0; e < E; e++) {
    if (d.lid[e] < 0) continue;
    const uint8_t* src = chunk + (uint64_t)d.lid[e] * per;
    uint8_t* dst = d.s[2][e] + (uint64_t)rel * (kBlock / kGroup);
    for (int h = 0; h < H; h++) dst[(uint64_t)h * row] = src[h];
  }
}

// Staging buffers of one part: as many as its tasks run at once, reused, freed with the object.
struct Staging {
  std::mutex mu;
  std::vector<uint8_t*> free_, all_;
  uint64_t bytes;
  explicit Staging(uint64_t n) : bytes((n + kAlign - 1) / kAlign * kAlign) {}
  uint8_t* get() {
    std::lock_guard<std::mutex> g(mu);
    if (!free_.empty()) {
      uint8_t* p = free_.back();
      free_.pop_back();
      return p;
    }
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, bytes) != 0) return nullptr;
    all_.push_back((uint8_t*)p);
    return (uint8_t*)p;
  }
  void put(uint8_t* p) {
    std::lock_guard<std::mutex> g(mu);
    free_.push_back(p);
  }
  ~Staging() {
    for (auto* p : all_) std::free(p);
  }
};

// Every chunk part d needs (6 x its blocks), read, verified and scattered as parallel tasks: parfor(n, fn) runs fn(t)
// for t in [0, n) on the part's threads. "" or the first reason; after a failure the remaining tasks do nothing.
template <class ParFor>
inline std::string load_part(const Spec& sp, int fd, const PartDst& d, ParFor&& parfor, uint64_t* bytes_read,
                             uint64_t* chunks_read) {
  if (!sp.ready) return "image: not prepared";
  if (d.Ip <= 0 || d.Ip % kBlock) return "image: the part's rows are not whole blocks";
  const int nb = d.Ip / kBlock;
  if (d.b0 < 0 || d.b0 + nb > sp.nblk()) return "image: the part's blocks lie outside the image";
  for (int p = 0; p < 3; p++)
    if (d.w[p].size() != d.lid.size() || d.s[p].size() != d.lid.size()) return "image: destination lists differ in length";
  for (size_t e = 0; e < d.lid.size(); e++) {
    if (d.lid[e] < 0) continue;
    if (d.lid[e] >= sp.E) return "image: the expert map points outside the image (" + std::to_string(d.lid[e]) + ")";
    for (int p = 0; p < 3; p++)
      if (!d.w[p][e] || !d.s[p][e]) return "image: a held expert has no destination";
  }
  Staging st(sp.w_chunk());
  std::mutex mu;
  std::string err;
  std::atomic<bool> bad{false};
  std::atomic<uint64_t> nbytes{0}, nchunks{0};
  parfor(6 * nb, [&](int t) {
    if (bad.load(std::memory_order_relaxed)) return;
    const bool scales = t >= 3 * nb;
    const int r = t % (3 * nb), p = r / nb, b = d.b0 + r % nb;
    const uint64_t n = scales ? sp.s_chunk() : sp.w_chunk();
    const uint64_t off = scales ? sp.s_off(p, b) : sp.w_off(p, b);
    uint8_t* buf = st.get();
    std::string why = buf ? read_verify(fd, off, n, buf, sp.digest(scales, p, b)) : std::string("image: no memory for staging");
    if (why.empty()) {
      scatter(sp, d, p, scales, b, buf);
      nbytes += n;
      nchunks++;
    }
    if (buf) st.put(buf);
    if (!why.empty()) {
      std::lock_guard<std::mutex> g(mu);
      if (err.empty()) err = why;
      bad = true;
    }
  });
  if (bytes_read) *bytes_read += nbytes.load();
  if (chunks_read) *chunks_read += nchunks.load();
  return err;
}

}  // namespace kt_image

#endif
