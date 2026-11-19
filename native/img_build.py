"""Build, resume, finish and verify a block-ordered expert image from a pinned GGUF range inventory."""
import concurrent.futures as cf
import hashlib
import json
import mmap
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import img_format as F  # noqa: E402

GGUF = Path(os.environ["IMG_GGUF"])
RANGES = Path(os.environ["IMG_RANGES"])
RANGES_SHA = os.environ.get("IMG_RANGES_SHA256", "9246e760c554a746662f3537cb2aadb626a90d827d2edbdda2e31438c7f30d12")
NLAYERS = int(os.environ.get("IMG_NLAYERS", "40"))
RESERVE = 32 << 30
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES = {0: "gate", 1: "up", 2: "down"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def open_direct(path, flags):
    """O_DIRECT where the file system takes it, else buffered (recorded)."""
    flags |= F.O_BINARY
    if F.O_DIRECT and os.environ.get("IMG_DIRECT", "1") == "1":
        try:
            return os.open(path, flags | F.O_DIRECT, 0o644), True
        except OSError as ex:
            if ex.errno != 22:   # EINVAL: no O_DIRECT here (tmpfs, some test file systems)
                raise
    return os.open(path, flags, 0o644), False


def aligned(n):
    return mmap.mmap(-1, (n + F.ALIGN - 1) // F.ALIGN * F.ALIGN)


class Source:
    """The GGUF and its pinned range manifest; every range read is checked against its digest."""

    def __init__(self):
        if sha_file(RANGES) != RANGES_SHA:
            raise SystemExit(f"range manifest {RANGES} is not the pinned {RANGES_SHA}")
        m = json.load(open(RANGES))
        if m.get("schema") != "kt-native-mxfp4-ranges-v1":
            raise SystemExit("range manifest schema " + str(m.get("schema")))
        st = GGUF.stat()
        if st.st_size != m["file_size"]:
            raise SystemExit(f"GGUF size {st.st_size} is not the manifest's {m['file_size']}")
        with open(GGUF, "rb") as f:
            hdr = f.read(m["header_bytes"])
        if hashlib.sha256(hdr).hexdigest() != m["header_sha256"]:
            raise SystemExit("GGUF header sha256 differs from the range manifest")
        self.meta = {k: m[k] for k in ("file_size", "header_bytes", "header_sha256", "original_model_revision", "schema")}
        self.idx = {}
        for r in m["ranges"]:
            self.idx[(r["tensor"], int(r["expert"]))] = r
        self.fd = os.open(GGUF, os.O_RDONLY | F.O_BINARY)
        self.experts = 1 + max(int(r["expert"]) for r in m["ranges"])

    def expert(self, layer, p, e, n, k):
        r = self.idx[(f"blk.{layer}.ffn_{NAMES[p]}_exps.weight", e)]
        want = n * (k // 32) * 17
        if int(r["bytes"]) != want or list(r["shape"]) != [n, k]:
            raise SystemExit(f"range {r['tensor']} expert {e}: {r['bytes']} bytes {r['shape']}, want {want} [{n}, {k}]")
        raw = F.read_aligned(self.fd, int(r["offset"]), want)
        if hashlib.sha256(raw).hexdigest() != r["sha256"]:
            raise SystemExit(f"range {r['tensor']} expert {e}: sha256 differs from the range manifest")
        try:
            os.posix_fadvise(self.fd, int(r["offset"]), want, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        return raw


def layer_done(d, L):
    return (Path(d) / "layers" / f"L{L:02d}.json").is_file() and (Path(d) / f"L{L:02d}.img").is_file()


def build_layer(src, g, d, L, threads):
    t0 = time.time()
    d = Path(d)
    part = d / f"L{L:02d}.img.partial"
    if part.exists():
        part.unlink()   # an interrupted attempt: start the layer again
    fd, direct = open_direct(str(part), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        try:
            os.posix_fallocate(fd, 0, g.layer_bytes)
        except (AttributeError, OSError):
            pass
        digests = {"w": [[None] * g.nblk for _ in range(3)], "s": [[None] * g.nblk for _ in range(3)]}
        lock = threading.Lock()
        bufs = [aligned(g.w_chunk) for _ in range(threads)]
        free = list(range(threads))

        def put(sec, p, b, arr):
            with lock:
                i = free.pop()
            try:
                n = arr.nbytes
                mv = memoryview(bufs[i])
                mv[:n] = arr.tobytes() if not arr.flags.c_contiguous else memoryview(arr).cast("B")
                h = hashlib.sha256(mv[:n]).hexdigest()
                off = g.w_off(p, b) if sec == "w" else g.s_off(p, b)
                F.pwrite_all(fd, mv[:n], off)
                digests[sec][p][b] = h
            finally:
                with lock:
                    free.append(i)

        for p in range(3):
            n, k = F.proj_shape(g, p)
            W = np.empty((g.E, n, k // 2), np.uint8)
            S = np.empty((g.E, n, k // 32), np.uint8)

            def one(e):
                W[e], S[e] = F.gguf_to_arena(src.expert(L, p, e, n, k), n, k)

            with cf.ThreadPoolExecutor(threads) as ex:
                list(ex.map(one, range(g.E)))
            jobs = [("w", b) for b in range(g.nblk)] + [("s", b) for b in range(g.nblk)]
            with cf.ThreadPoolExecutor(threads) as ex:
                list(ex.map(lambda j: put(j[0], p, j[1], F.chunk_from_full(g, p, j[0] == "s", j[1], W if j[0] == "w" else S)),
                            jobs))
            W = S = None   # free the projection before the next one
        os.fsync(fd)
    finally:
        os.close(fd)
    if part.stat().st_size != g.layer_bytes:
        raise SystemExit(f"layer {L}: file is {part.stat().st_size} bytes, want {g.layer_bytes}")
    bad = verify_file(g, part, digests, threads)
    if bad:
        raise SystemExit(f"layer {L}: {bad} chunks differ when read back")
    os.rename(part, d / f"L{L:02d}.img")
    rec = {"layer": L, "file": f"L{L:02d}.img", "bytes": g.layer_bytes, "w": digests["w"], "s": digests["s"],
           "direct_io": direct, "seconds": round(time.time() - t0, 1), "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = d / "layers" / f".L{L:02d}.json.tmp"
    tmp.write_text(json.dumps(rec) + "\n")
    os.rename(tmp, d / "layers" / f"L{L:02d}.json")
    return rec


def verify_file(g, path, digests, threads):
    """Chunks of one layer file whose sha256 differs from digests (read with O_DIRECT where possible)."""
    fd, direct = open_direct(str(path), os.O_RDONLY)
    lock = threading.Lock()
    bufs = [aligned(g.w_chunk) for _ in range(threads)]
    free = list(range(threads))
    bad = [0]

    def chk(j):
        sec, p, b = j
        with lock:
            i = free.pop()
        try:
            n = g.w_chunk if sec == "w" else g.s_chunk
            off = g.w_off(p, b) if sec == "w" else g.s_off(p, b)
            mv = F.read_aligned(fd, off, n, bufs[i] if direct else None)
            if hashlib.sha256(mv).hexdigest() != digests[sec][p][b]:
                with lock:
                    bad[0] += 1
        finally:
            with lock:
                free.append(i)

    try:
        jobs = [(sec, p, b) for sec in ("w", "s") for p in range(3) for b in range(g.nblk)]
        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(chk, jobs))
    finally:
        os.close(fd)
    return bad[0]


def layers_arg(s):
    return list(range(NLAYERS)) if s == "all" else [int(v) for v in s.split(",")]


def cmd_build(out, d, layers, budget):
    t_start = time.time()
    out, d = Path(out), Path(d)
    (d / "layers").mkdir(parents=True, exist_ok=True)
    src = Source()
    g = F.Geometry(src.experts, *hidden_inter())
    threads = max(2, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4))
    rep = {"image_dir": str(d), "gguf": str(GGUF), "ranges": str(RANGES), "ranges_sha256": RANGES_SHA, "source": src.meta,
           "geometry": g.as_dict(), "threads": threads, "built": [], "skipped_done": [], "status": "RUNNING",
           "img_build_sha256": sha_file(__file__), "img_format_sha256": sha_file(HERE / "img_format.py")}
    save = lambda: (out / "img_build.json").write_text(json.dumps(rep, indent=1) + "\n")
    save()
    per = None
    for L in layers:
        if layer_done(d, L):
            rep["skipped_done"].append(L)
            continue
        if per is not None and time.time() - t_start + 1.5 * per > budget:
            rep["status"] = "PARTIAL"
            save()
            log(f"IMG PARTIAL: time budget; {sum(layer_done(d, x) for x in layers)} of {len(layers)} layers done")
            sys.exit(10)
        t0 = time.time()
        free = shutil.disk_usage(d).free
        if free < g.layer_bytes + RESERVE:
            rep["status"] = "FAIL: disk"
            save()
            raise SystemExit(f"refused: {free} bytes free on the image file system, need {g.layer_bytes + RESERVE}")
        rec = build_layer(src, g, d, L, threads)
        per = max(per or 0, time.time() - t0)
        rep["built"].append({"layer": L, "seconds": rec["seconds"], "direct_io": rec["direct_io"]})
        save()
        log(f"layer {L} built and read back in {rec['seconds']} s")
    rep["status"] = "DONE"
    save()
    log(f"IMG BUILD DONE: {sum(layer_done(d, x) for x in layers)} of {len(layers)} layers")


def hidden_inter():
    return int(os.environ.get("IMG_HIDDEN", "5120")), int(os.environ.get("IMG_INTER", "2304"))


def cmd_finish(d):
    d = Path(d)

    recs = {}
    for L in range(NLAYERS):
        if not layer_done(d, L):
            raise SystemExit(f"layer {L} is not built")
        r = json.load(open(d / "layers" / f"L{L:02d}.json"))
        if (d / r["file"]).stat().st_size != r["bytes"]:
            raise SystemExit(f"layer {L}: file size differs from its record")
        recs[str(L)] = {"file": r["file"], "bytes": r["bytes"], "w": r["w"], "s": r["s"]}
    src = Source()
    g = F.Geometry(src.experts, *hidden_inter())
    m = {"schema": F.SCHEMA, "geometry": dict(g.as_dict(), layers=list(range(NLAYERS))),
         "source": dict(src.meta, gguf=str(GGUF), range_manifest=str(RANGES), range_manifest_sha256=RANGES_SHA),
         "builder": {"img_build_sha256": sha_file(__file__), "img_format_sha256": sha_file(HERE / "img_format.py")},
         "layers": recs, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    raw = (json.dumps(m, indent=0, sort_keys=True) + "\n").encode()
    tmp = d / ".manifest.json.tmp"
    tmp.write_bytes(raw)
    if (d / "manifest.json").exists() and (d / "manifest.json").read_bytes() != raw:
        # the records changed only if a layer was rebuilt; keep the old one beside it, never silently replace a pin
        os.rename(d / "manifest.json", d / f"manifest.json.before-{int(time.time())}")
    os.rename(tmp, d / "manifest.json")
    print(hashlib.sha256(raw).hexdigest())


def cmd_verify(out, d, threads):
    out, d = Path(out), Path(d)
    m, h = F.load_manifest(d)
    g = F.Geometry(m["geometry"]["experts"], m["geometry"]["hidden"], m["geometry"]["intermediate"])
    res = {"manifest_sha256": h, "layers": {}, "bad_chunks": 0, "chunks": 0, "seconds": 0}
    t0 = time.time()
    for L, r in sorted(m["layers"].items(), key=lambda kv: int(kv[0])):
        p = d / r["file"]
        if p.stat().st_size != r["bytes"] or r["bytes"] != g.layer_bytes:
            res["layers"][L] = "size"
            res["bad_chunks"] += 1
            continue
        bad = verify_file(g, p, {"w": r["w"], "s": r["s"]}, threads)
        res["layers"][L] = bad
        res["bad_chunks"] += bad
        res["chunks"] += 6 * g.nblk
    res["seconds"] = round(time.time() - t0, 1)
    res["ok"] = res["bad_chunks"] == 0 and len(m["layers"]) == NLAYERS
    (out / "img_verify.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "layers"}))
    sys.exit(0 if res["ok"] else 3)


def cmd_status(d):
    n = sum(layer_done(d, L) for L in range(NLAYERS))
    print(f"{n} of {NLAYERS} layers done; manifest {'present' if (Path(d) / 'manifest.json').is_file() else 'absent'}")
    sys.exit(0 if n == NLAYERS else 1)


def cmd_space(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    g = F.Geometry(384, *hidden_inter())
    todo = sum(not layer_done(d, L) for L in range(NLAYERS))
    need = todo * g.layer_bytes + RESERVE
    free = shutil.disk_usage(d).free
    print(f"free {free / 2**30:.1f} GiB, need {need / 2**30:.1f} GiB for {todo} layers + 32 GiB reserve")
    sys.exit(0 if free >= need else 1)


def cmd_synth(d, E, H, I, nl, seed=1):
    """A small GGUF-like file (header + every layer's gate/up/down expert ranges) and its range manifest, for tests.
    Scale codes are drawn from 110..130 (a few 0 and 254 too); nibbles are random, including 0x8 (-0)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    hdr = rng.integers(0, 256, 4096 + 13, dtype=np.uint8).tobytes()
    ranges, off = [], len(hdr)
    with open(d / "synth.gguf", "wb") as f:
        f.write(hdr)
        for L in range(nl):
            for p in range(3):
                n, k = (I, H) if p < 2 else (H, I)
                for e in range(E):
                    b = rng.integers(0, 256, (n, k // 32, 17), dtype=np.uint8)
                    b[..., 0] = rng.integers(110, 131, (n, k // 32))
                    b[0, 0, 0], b[-1, -1, 0] = 0, 254
                    raw = b.tobytes()
                    f.write(raw)
                    ranges.append({"bytes": len(raw), "expert": e, "offset": off, "sha256": hashlib.sha256(raw).hexdigest(),
                                   "shape": [n, k], "tensor": f"blk.{L}.ffn_{NAMES[p]}_exps.weight"})
                    off += len(raw)
    man = {"file_size": off, "header_bytes": len(hdr), "header_sha256": hashlib.sha256(hdr).hexdigest(),
           "original_model_revision": "synthetic", "provenance": "synthetic test source", "ranges": ranges,
           "schema": "kt-native-mxfp4-ranges-v1"}
    (d / "ranges.json").write_text(json.dumps(man))
    print(sha_file(d / "ranges.json"))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        cmd_build(sys.argv[2], sys.argv[3], layers_arg(sys.argv[4]), float(sys.argv[5]))
    elif mode == "finish":
        cmd_finish(sys.argv[2])
    elif mode == "verify":
        cmd_verify(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6)
    elif mode == "status":
        cmd_status(sys.argv[2])
    elif mode == "space":
        cmd_space(sys.argv[2])
    elif mode == "synth":
        cmd_synth(sys.argv[2], *[int(v) for v in sys.argv[3:7]])
    else:
        sys.exit("unknown mode " + mode)
