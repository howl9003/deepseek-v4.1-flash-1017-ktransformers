// pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3): host-only test of operators/avx2/kt_image.hpp, the image
// reader kt-kernel's load_weights_image uses. No kt-kernel header, no model.
//   test_kt_image <vector dir>   (vectors from concept-r1/trials/queue/staged-ns2/mk_image_vectors.py)
// Checks, printed as one JSON object; exit 0 only if every check holds:
//   sha256 on the FIPS 180-2 vectors (empty, "abc", the 448-bit message, one million 'a') and on the image's chunks
//   for each case (part table, expert map with duplicates and skipped experts): every part's buffers filled by
//     load_part (read with O_DIRECT where the file system takes it, verified, scattered on four threads) equal the bytes
//     the vector file says his loader puts in the arenas (computed in Python from the source, not from the image)
//   refusals: a flipped digest, a manifest file size that is not the geometry's, a short digest list, a part outside
//     the image, an expert map outside the image, a missing file, a file whose size differs
#include <atomic>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../operators/avx2/kt_image.hpp"

static std::string hex(const uint8_t* d) {
  static const char* x = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 32; i++) {
    s += x[d[i] >> 4];
    s += x[d[i] & 15];
  }
  return s;
}
static std::string sha(const std::string& m) {
  kt_image::Sha256 h;
  h.update((const uint8_t*)m.data(), m.size());
  uint8_t d[32];
  h.final(d);
  return hex(d);
}
static std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static void parfor4(int n, const std::function<void(int)>& fn) {
  std::atomic<int> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < 4; t++)
    th.emplace_back([&] {
      for (int i; (i = next.fetch_add(1)) < n;) fn(i);
    });
  for (auto& t : th) t.join();
}

struct Out {
  std::vector<std::pair<std::string, bool>> checks;
  std::vector<std::string> notes;
  void check(const std::string& name, bool ok) { checks.emplace_back(name, ok); }
};

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: test_kt_image <vector dir>\n");
    return 2;
  }
  const std::string V = argv[1];
  Out o;
  // FIPS 180-2 vectors
  o.check("sha256_empty", sha("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  o.check("sha256_abc", sha("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  o.check("sha256_448bit",
          sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  o.check("sha256_million_a", sha(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");

  // the vectors
  std::ifstream spec(V + "/spec.txt");
  int E, H, I, ncases;
  std::string path;
  uint64_t file_bytes;
  spec >> E >> H >> I >> path >> file_bytes >> ncases;
  const std::string digests = slurp(V + "/digests.bin");
  kt_image::Spec sp;
  std::string why = kt_image::make_spec(sp, path, file_bytes, {E, H, I, 32, 32}, digests);
  o.check("make_spec", why.empty());
  if (!why.empty()) o.notes.push_back(why);
  {   // the image's own chunk digests, recomputed here with this sha256
    kt_image::File f;
    why = f.open(sp);
    bool all = why.empty();
    std::vector<uint8_t> buf(sp.w_chunk() + kt_image::kAlign);
    uint8_t* b = (uint8_t*)(((uintptr_t)buf.data() + kt_image::kAlign - 1) & ~(uintptr_t)(kt_image::kAlign - 1));
    for (int s = 0; all && s < 2; s++)
      for (int p = 0; all && p < 3; p++)
        for (int k = 0; all && k < sp.nblk(); k++) {
          why = kt_image::read_verify(f.fd, s ? sp.s_off(p, k) : sp.w_off(p, k), s ? sp.s_chunk() : sp.w_chunk(), b,
                                      sp.digest(s, p, k));
          all = why.empty();
        }
    o.check("every_chunk_verifies", all);
    o.notes.push_back(std::string("direct I/O: ") + (f.direct ? "yes" : "no (buffered)"));
    if (!all) o.notes.push_back(why);
  }
  uint64_t total_bytes = 0;
  for (int c = 0; c < ncases; c++) {
    std::ifstream cf(V + "/case" + std::to_string(c) + ".txt");
    int P;
    cf >> P;
    std::vector<int> Ip(P), b0(P);
    for (int i = 0; i < P; i++) cf >> Ip[i] >> b0[i];
    std::vector<int64_t> lid(E);
    for (int e = 0; e < E; e++) cf >> lid[e];
    const std::string want = slurp(V + "/case" + std::to_string(c) + ".bin");
    std::string got;
    bool ok = true;
    kt_image::File f;
    why = f.open(sp);
    if (!why.empty()) {
      ok = false;
      o.notes.push_back(why);
    }
    for (int i = 0; ok && i < P; i++) {
      const uint64_t wper[3] = {(uint64_t)Ip[i] * H / 2, (uint64_t)Ip[i] * H / 2, (uint64_t)H * Ip[i] / 2};
      const uint64_t sper[3] = {(uint64_t)Ip[i] * H / 32, (uint64_t)Ip[i] * H / 32, (uint64_t)H * Ip[i] / 32};
      std::vector<std::vector<uint8_t>> wb(3), sb(3);
      kt_image::PartDst d;
      d.Ip = Ip[i];
      d.b0 = b0[i];
      d.lid = lid;
      for (int p = 0; p < 3; p++) {
        wb[p].assign(wper[p] * E, 0xA5);
        sb[p].assign(sper[p] * E, 0x5A);
        d.w[p].assign(E, nullptr);
        d.s[p].assign(E, nullptr);
        for (int e = 0; e < E; e++)
          if (lid[e] >= 0) {
            d.w[p][e] = wb[p].data() + wper[p] * e;
            d.s[p][e] = sb[p].data() + sper[p] * e;
          }
      }
      uint64_t nb = 0, nc = 0;
      why = kt_image::load_part(sp, f.fd, d, parfor4, &nb, &nc);
      if (!why.empty()) {
        ok = false;
        o.notes.push_back("case " + std::to_string(c) + " part " + std::to_string(i) + ": " + why);
        break;
      }
      total_bytes += nb;
      for (int e = 0; e < E; e++) {
        if (lid[e] < 0) continue;
        for (int p = 0; p < 3; p++) got.append((const char*)wb[p].data() + wper[p] * e, wper[p]);
        for (int p = 0; p < 3; p++) got.append((const char*)sb[p].data() + sper[p] * e, sper[p]);
      }
    }
    ok = ok && got == want;
    o.check("case" + std::to_string(c) + "_arena_bytes_equal", ok);
    if (got.size() != want.size()) o.notes.push_back("case " + std::to_string(c) + ": " + std::to_string(got.size()) +
                                                     " bytes against " + std::to_string(want.size()));
  }
  // refusals
  {
    std::string bad = digests;
    bad[32 * (3 * sp.nblk() + 5) + 7] ^= 1;   // a scale chunk of gate, block 5
    kt_image::Spec s2;
    kt_image::make_spec(s2, path, file_bytes, {E, H, I, 32, 32}, bad);
    kt_image::File f;
    f.open(s2);
    kt_image::PartDst d;
    d.Ip = 32 * sp.nblk();
    d.b0 = 0;
    d.lid.assign(E, -1);
    for (int p = 0; p < 3; p++) {
      d.w[p].assign(E, nullptr);
      d.s[p].assign(E, nullptr);
    }
    why = kt_image::load_part(s2, f.fd, d, parfor4, nullptr, nullptr);
    o.check("refuses_a_flipped_digest", why.find("sha256 differs") != std::string::npos);
    kt_image::Spec s3;
    o.check("refuses_a_file_size_not_the_geometrys", !kt_image::make_spec(s3, path, file_bytes + 4096, {E, H, I, 32, 32}, digests).empty());
    o.check("refuses_a_short_digest_list", !kt_image::make_spec(s3, path, file_bytes, {E, H, I, 32, 32}, digests.substr(32)).empty());
    o.check("refuses_block_size_not_32", !kt_image::make_spec(s3, path, file_bytes, {E, H, I, 64, 32}, digests).empty());
    kt_image::File f2;
    f2.open(sp);
    kt_image::PartDst d2 = d;
    d2.Ip = 64;
    d2.b0 = sp.nblk() - 1;
    o.check("refuses_a_part_outside_the_image", !kt_image::load_part(sp, f2.fd, d2, parfor4, nullptr, nullptr).empty());
    kt_image::PartDst d3 = d;
    std::vector<uint8_t> dummy(1 << 20);
    d3.lid[0] = E;
    for (int p = 0; p < 3; p++) d3.w[p][0] = d3.s[p][0] = dummy.data();
    why = kt_image::load_part(sp, f2.fd, d3, parfor4, nullptr, nullptr);
    o.check("refuses_an_expert_map_outside_the_image", why.find("outside the image") != std::string::npos);
    kt_image::Spec s4 = sp;
    s4.path = path + ".missing";
    kt_image::File f3;
    o.check("refuses_a_missing_file", !f3.open(s4).empty());
    kt_image::Spec s5 = sp;
    s5.path = V + "/digests.bin";
    kt_image::File f4;
    o.check("refuses_a_file_whose_size_differs", f4.open(s5).find("bytes") != std::string::npos);
  }
  bool all = true;
  printf("{\"checks\": {");
  for (size_t i = 0; i < o.checks.size(); i++) {
    printf("%s\"%s\": %s", i ? ", " : "", o.checks[i].first.c_str(), o.checks[i].second ? "true" : "false");
    all = all && o.checks[i].second;
  }
  printf("}, \"notes\": [");
  for (size_t i = 0; i < o.notes.size(); i++) {
    std::string n;
    for (char ch : o.notes[i]) n += (ch == '"' || ch == '\\') ? ' ' : ch;
    printf("%s\"%s\"", i ? ", " : "", n.c_str());
  }
  printf("], \"bytes_read_in_cases\": %llu, \"ok\": %s}\n", (unsigned long long)total_bytes, all ? "true" : "false");
  return all ? 0 : 1;
}
