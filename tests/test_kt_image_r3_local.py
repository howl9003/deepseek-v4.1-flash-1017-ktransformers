"""pathb-image-r3 local test (no kt module, no torch, no numpy): the preload made inside the CPUInfer task
(AVX2ContractV3_MOE.image_build), the drain before any main-thread module, r2's adoption rules and r1's refusals, in
python/utils/kt_image.py, and the two hooks of python/utils/patch_amx_image.py on his amx.py. Any Python 3.8+:

  python -B tests/test_kt_image_r3_local.py <kt tree (this branch)> <his amx.py> [<r2 kt_image.py>]

The hook functions run against stand-ins for MOEConfig, the contract v3 class and its ImageBuild record (the build's
task runs on a fake CPUInfer worker thread: it makes the module, checks its rows, prepares and loads), the wrapper and
sglang's expert-location metadata. The stand-in NumaJobDistributor counts every do_numa_job that starts while another
is running (the race r3 removes): the module's constructor and each load call it, as moe-tp.hpp and the image task do.
With the r2 kt_image.py (git show 7a83c4d:python/utils/kt_image.py) as the third argument, the same server order is run
on r2's preload as the negative control: it must count overlaps. One line per check, then PASS or FAIL (exit 1).
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

KT, HIS_AMX = Path(sys.argv[1]), Path(sys.argv[2])
R2_KTI = Path(sys.argv[3]) if len(sys.argv) > 3 else None
FMT = "pathb-arena-v1;blk=32;scale=e8m0;w=0123456789abcdef"
ROWS8 = [288] * 8
MAIN = threading.get_ident()
results = []


def check(name, ok, note=""):
    results.append(bool(ok))
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + note[:220]) if note else ''}")


class Dist:   # the stand-in NumaJobDistributor: do_numa_job calls that overlap are counted (r2's race)
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = 0
        self.overlaps = 0
        self.calls = 0
        self.threads = set()

    def job(self, secs):
        with self.lock:
            self.calls += 1
            self.threads.add(threading.get_ident())
            if self.busy:
                self.overlaps += 1
            self.busy += 1
        time.sleep(secs)
        with self.lock:
            self.busy -= 1


class T:   # a 1-D CPU tensor stand-in
    def __init__(self, vals, es=8):
        self.v, self.es = list(vals), es

    def element_size(self):
        return self.es

    def numel(self):
        return len(self.v)

    def contiguous(self):
        return self

    def clone(self):
        return T(self.v, self.es)

    def tolist(self):
        return list(self.v)

    def data_ptr(self):
        return id(self)


class Cfg:
    def __init__(self, *a):
        self.args = a
        self.quant_config = types.SimpleNamespace()


def make_backend(fmt=FMT, rows=ROWS8, status="ok", build=True, done=True, run_s=0.05, make_s=0.02, arenas=None):
    """The contract v3 class stand-in. build=False: an r2 module (image_done, no image_build); done=False: r1."""
    dist = Dist()

    class Moe:
        made = 0
        made_main = 0
        alive = 0
        ran = 0
        D = dist

        def __init__(self, cfg):
            Moe.made += 1
            Moe.made_main += threading.get_ident() == MAIN
            Moe.alive += 1
            self.cfg = cfg
            self._done = 0
            self._status = "not run"
            dist.job(make_s)   # TP_MOE_Common's constructor: do_numa_job (moe-tp.hpp)

        def __del__(self):
            Moe.alive -= 1

        def contract_v3_part_rows(self):
            return list(rows)

        def image_prepare(self, path, nbytes, geometry, digests):
            self._status = "prepared"
            return "ok"

        def load_weights_image_task(self, ptr):
            self.map_ptr = ptr
            return ("task", self)

        def run(self):   # load_weights_image on the worker: the read, the code tables, the collapse (do_numa_job each)
            dist.job(run_s)
            Moe.ran += 1
            self._status = status
            self._done = 1

        def image_status(self):
            return self._status

        def image_stats(self):
            return (432, 7219445760, 2.3, True)

        def expert_arena_infos(self):
            return arenas

    class Build:   # contract_v3h-moe.hpp ImageBuild, as written: run() on the worker, never raises
        def __init__(self, cfg, path, nbytes, geometry, digests, proven, any_rows, mp):
            self.cfg, self.proven, self.any, self.mp = cfg, proven, any_rows, mp
            self.moe, self.st, self.d, self.tid_, self.sub = None, "queued", 0, -1, 0
            Build.records.append(self)

        def task(self):
            if self.sub:
                raise RuntimeError("image build: its task was made already")
            self.sub = 1
            return ("task", self)

        def run(self):
            self.tid_ = threading.get_ident()
            m = Moe(self.cfg)
            r = m.contract_v3_part_rows()
            if not (self.any or r in self.proven):
                st = f"part rows {r} are not among the image's proven part rows {self.proven}"
            else:
                st = m.image_prepare("p", 0, [], b"")
                if st == "ok":
                    m.map_ptr = self.mp
                    m.run()
                    st = m.image_status()
            if st != "ok":
                m = None
            self.moe, self.st, self.d = m, st, 1

        def done(self):
            return self.d

        def status(self):
            return self.st if self.d else "queued"

        def seconds(self):
            return (0.01, 0.06) if self.d else (-1.0, -1.0)

        def tid(self):
            return self.tid_

        def take(self):
            if not self.d:
                raise RuntimeError("take() before the task has run")
            m, self.moe = self.moe, None
            return m

    Build.records = []
    if done:
        Moe.image_done = lambda self: self._done
    if build:
        Moe.image_build = staticmethod(Build)
    Moe.kt_arena_format = staticmethod(lambda: fmt)
    Moe.Build = Build
    return Moe


class CPUInfer:   # tasks run one at a time, in order, on one worker thread; sync() waits for all of them
    def __init__(self):
        self.q, self.submitted = [], 0
        self.backend_ = object()
        self.lock = threading.Lock()
        self.th = None

    def submit(self, task):
        self.submitted += 1
        with self.lock:
            self.q.append(task[1])
            if self.th is None or not self.th.is_alive():
                self.th = threading.Thread(target=self._work, daemon=True)
                self.th.start()

    def _work(self):
        while True:
            with self.lock:
                if not self.q:
                    return
                obj = self.q.pop(0)
            obj.run()

    def sync(self):
        while True:
            with self.lock:
                idle = not self.q
            if idle and (self.th is None or not self.th.is_alive()):
                return
            time.sleep(0.001)


def os_shim():
    ns = types.SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
    if not hasattr(os, "pread"):
        def pread(fd, n, off):
            os.lseek(fd, off, 0)
            return os.read(fd, n)
        ns.pread = pread
    return ns


def hook_ns(backend, src_path=None):
    src = (src_path or (KT / "python/utils/kt_image.py")).read_text(encoding="utf-8")
    ns = {"os": os_shim(), "MOEConfig": Cfg, "AVX2ContractV3_MOE": backend, "_select_mxfp4_backend": lambda: backend}
    exec(compile(src, "kt_image.py", "exec"), ns)
    return ns


def wrapper(gguf, layer=3, ci=None):
    return types.SimpleNamespace(layer_idx=layer, weight_path=str(gguf), num_experts=384, num_experts_per_tok=6,
                                 hidden_size=5120, moe_intermediate_size=2304, gpu_experts_mask=T([True] + [False] * 383, 1),
                                 cpu_infer=ci or CPUInfer(), chunked_prefill_size=8, swiglu_limit=10.0, _activation="silu",
                                 moe=None)


PMAP = [0] + list(range(383, 0, -1))


def sglang_metadata(present=True):
    for k in [k for k in sys.modules if k == "sglang" or k.startswith("sglang.")]:
        del sys.modules[k]
    if not present:
        return
    el = types.ModuleType("sglang.srt.eplb.expert_location")
    md = types.SimpleNamespace(physical_to_logical_map_cpu=[T(PMAP) for _ in range(40)])
    el.get_global_expert_location_metadata = lambda: md
    for name in ("sglang", "sglang.srt", "sglang.srt.eplb"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["sglang.srt.eplb.expert_location"] = el


def write_image(d, layers=(3,)):
    d.mkdir(parents=True, exist_ok=True)
    gguf = d / "model.gguf"
    hdr = b"GGUF" + bytes(range(60))
    gguf.write_bytes(hdr + b"\0" * 100)
    m = {"schema": "pathb-expert-image-v1",
         "source": {"file_size": gguf.stat().st_size, "header_bytes": len(hdr), "header_sha256": hashlib.sha256(hdr).hexdigest()},
         "geometry": {"experts": 384, "hidden": 5120, "intermediate": 2304, "block": 32, "group": 32},
         "layers": {str(L): {"file": f"L{L:02d}.img", "bytes": 7219445760, "w": [["00" * 32]] * 3, "s": [["11" * 32]] * 3}
                    for L in layers}}
    raw = json.dumps(m, indent=1).encode()
    (d / "manifest.json").write_bytes(raw)
    return gguf, hashlib.sha256(raw).hexdigest()


def sidecar(path, man_sha, fmt=FMT):
    c = {"schema": "pathb-image-arena-format-v1", "image_manifest_sha256": man_sha, "arena_format": fmt,
         "part_rows": [ROWS8]}
    raw = json.dumps(c, indent=1).encode()
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def setenv(env):
    for k in ("KT_IMAGE_DIR", "KT_IMAGE_MANIFEST_SHA256", "KT_IMAGE_STRICT", "KT_IMAGE_FORMAT_SHA256", "KT_IMAGE_FORMAT_FILE",
              "KT_ARENA_SCALE_BYTES", "KT_IMAGE_PRELOAD", "KT_IMAGE_DIGEST", "KT_MXFP4_EXPERT_CACHE_GIB"):
        os.environ.pop(k, None)
    os.environ.update({"KT_ARENA_SCALE_BYTES": "1"})
    os.environ.update(env)


def call(fn, *a):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            r = fn(*a)
        return r, None, buf.getvalue()
    except Exception as ex:
        return None, repr(ex), buf.getvalue()


def server_order(ns, be, gguf, n=40, run_s=None):
    """sglang's order: every wrapper made during the model build (each queues its layer), then load_weights in order."""
    ci = CPUInfer()
    ws = []
    for L in range(n):
        w = wrapper(gguf, layer=L, ci=ci)
        call(ns["_kt_image_init"], w)
        ws.append(w)
    outs = [call(ns["_kt_image_load_layer"], w, T(PMAP)) for w in ws]
    return ws, outs


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    img = td / "img"
    gguf, msha = write_image(img, layers=tuple(range(40)))
    fsha = sidecar(img / "arena_format.json", msha)
    base = {"KT_IMAGE_DIR": str(img), "KT_IMAGE_MANIFEST_SHA256": msha, "KT_IMAGE_FORMAT_SHA256": fsha}

    # 1. preload at init: the task makes the module; adopted at load_weights
    setenv(base)
    sglang_metadata(True)
    be = make_backend(run_s=0.2)
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_init"], w)
    check("init: image serves the layer -> True, one build task queued, no module made on the main thread",
          r is True and w.cpu_infer.submitted == 1 and w._kt_image_pre is not None and be.made_main == 0
          and len(ns["_KT_IMAGE_INFLIGHT"]) == 1, err or out)
    t0 = time.time()
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("load_weights: adopts the module the task made (worker thread), after its own task", r is True and w.moe is not None
          and be.made == 1 and be.made_main == 0 and be.ran == 1 and "preloaded" in out and "made in the task" in out
          and "format " + FMT in out and w.cpu_infer.submitted == 1, err or out.strip())
    check("load_weights: waited for the task (it was still running)", time.time() - t0 >= 0.1)
    check("adopted: record released, map kept on the wrapper, build emptied", len(ns["_KT_IMAGE_INFLIGHT"]) == 0
          and w._kt_image_map is not None and w._kt_image_pre is None and be.Build.records[0].moe is None)
    moe = w.moe
    w.moe = None
    del moe
    check("adopted module freed with the wrapper's reference (nothing else holds it)", be.alive == 0, f"alive {be.alive}")

    # 2. the map passed at load_weights differs: dropped after its task, the layer made again in the foreground
    setenv(base)
    be = make_backend()
    ns = hook_ns(be)
    w = wrapper(gguf)
    call(ns["_kt_image_init"], w)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(list(range(384))))
    check("map changed: preload dropped (its module freed), the layer loaded in the foreground", r is True
          and "preload dropped" in out and be.made == 2 and be.made_main == 1 and be.ran == 2 and be.alive == 1
          and len(ns["_KT_IMAGE_INFLIGHT"]) == 0, err or out.strip())

    # 3. the GPU mask object replaced
    setenv(base)
    be = make_backend()
    ns = hook_ns(be)
    w = wrapper(gguf)
    call(ns["_kt_image_init"], w)
    w.gpu_experts_mask = T([True] + [False] * 383, 1)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("mask replaced: preload dropped, foreground load", r is True and "preload dropped" in out and be.made == 2, err or out.strip())

    # 4. a preloaded image load that fails: GGUF path (non-strict), raise (strict); the task dropped its module
    for strict in ("0", "1"):
        setenv(dict(base, KT_IMAGE_STRICT=strict))
        be = make_backend(status="image: a chunk's sha256 differs from the manifest (offset 0)")
        ns = hook_ns(be)
        w = wrapper(gguf)
        call(ns["_kt_image_init"], w)
        r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
        if strict == "0":
            check("preload failed, non-strict: one GGUF-path line, False, no module kept", r is False and "GGUF path: image: a chunk"
                  in out and w.moe is None and be.alive == 0, err or out.strip())
        else:
            check("preload failed, strict: raises", r is None and "GGUF path: image: a chunk" in (err or ""), err)

    # 5. KT_IMAGE_PRELOAD=0, no sglang, an r2 module (no image_build), an r1 module: init True, nothing queued;
    #    load_weights loads in the foreground (r1)
    for name, env, present, kw in (("KT_IMAGE_PRELOAD=0", {"KT_IMAGE_PRELOAD": "0"}, True, {}),
                                   ("no sglang metadata", {}, False, {}),
                                   ("r2 module (image_done, no image_build)", {}, True, {"build": False}),
                                   ("r1 module (neither)", {}, True, {"build": False, "done": False})):
        setenv(dict(base, **env))
        sglang_metadata(present)
        be = make_backend(**kw)
        ns = hook_ns(be)
        w = wrapper(gguf)
        r, err, out = call(ns["_kt_image_init"], w)
        r2, err2, out2 = call(ns["_kt_image_load_layer"], w, T(PMAP))
        check(f"{name}: init True, nothing queued; load_weights loads in the foreground", r is True and getattr(w, "_kt_image_pre", None)
              is None and w.cpu_infer.submitted == 1 and r2 is True and be.made == 1 and "preloaded" not in out2
              and "s in all" in out2, (err or out) + (err2 or out2))
    sglang_metadata(True)

    # 6. image refused at init: False (the loader is made as before), no module; load_weights says why
    setenv(base)
    be = make_backend(fmt=FMT.replace("0123", "9999"))
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_init"], w)
    r2, err2, out2 = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("another arena format: init False, no module; load_weights GGUF path with the reason", r is False and be.made == 0
          and r2 is False and "the module writes" in out2, (err or out) + (err2 or out2))
    setenv(dict(base, KT_IMAGE_MANIFEST_SHA256="xyz"))
    r, err, out = call(ns["_kt_image_init"], wrapper(gguf))
    r2, err2, out2 = call(ns["_kt_image_load_layer"], wrapper(gguf), T(PMAP))
    check("malformed manifest pin: init False without raising; load_weights raises", r is False and err is None and r2 is None
          and "well-formed" in (err2 or ""), str(err) + str(err2))
    setenv(base)
    w = wrapper(gguf, layer=47)
    r, err, out = call(ns["_kt_image_init"], w)
    check("layer not in the image: init False", r is False)
    setenv(dict(base, KT_MXFP4_EXPERT_CACHE_GIB="8"))
    be = make_backend()
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_init"], w)
    r2, err2, out2 = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("expert cache on: init False, nothing queued (load_weights makes cache modules on the main thread)", r is False
          and w.cpu_infer.submitted == 0 and r2 is False and "expert cache off" in out2, (err or out) + (err2 or out2))

    # 7. part rows not proven: the foreground refuses (module made and dropped); with the preload on, the build task
    #    refuses with r1's reason and drops its module on the worker
    setenv(dict(base, KT_IMAGE_PRELOAD="0"))
    be = make_backend(rows=[320] * 7 + [64])
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("part rows not proven (foreground): GGUF path, the module made once and dropped", r is False
          and "not among the image's proven" in out and w.moe is None and be.made == 1 and be.alive == 0, out.strip())
    setenv(base)
    be = make_backend(rows=[320] * 7 + [64])
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_init"], w)
    r2, err2, out2 = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("part rows not proven (preload): the task refuses, no read, module dropped; load_weights GGUF path with the reason",
          r is True and w.cpu_infer.submitted == 1 and be.ran == 0 and r2 is False and "not among the image's proven" in out2
          and be.alive == 0 and be.made_main == 0, out2.strip())

    # 8. the server order (40 wrappers made back to back, then load_weights in order): no do_numa_job overlaps, every
    #    module made on the one worker thread, the first load_weights returns before the queue has drained
    setenv(base)
    be = make_backend(run_s=0.01, make_s=0.005)
    ns = hook_ns(be)
    ci = CPUInfer()
    ws = []
    for L in range(40):
        w = wrapper(gguf, layer=L, ci=ci)
        call(ns["_kt_image_init"], w)
        ws.append(w)
    r, err, out = call(ns["_kt_image_load_layer"], ws[0], T(PMAP))
    left = len(ns["_KT_IMAGE_INFLIGHT"])
    check("server order: the first load_weights returns while most builds are still queued", r is True and left >= 30,
          f"in flight after the first adoption: {left}")
    oks = [call(ns["_kt_image_load_layer"], x, T(PMAP))[0] for x in ws[1:]]
    tids = {b.tid() for b in be.Build.records}
    check("server order: 40 adopted, 40 makes, none on the main thread, all on one worker thread", all(oks) and be.made == 40
          and be.made_main == 0 and be.ran == 40 and len(tids) == 1 and MAIN not in tids and not ns["_KT_IMAGE_INFLIGHT"],
          f"made {be.made} main {be.made_main} tids {len(tids)}")
    check("server order: no do_numa_job started while another ran (r2's race absent)", be.D.overlaps == 0
          and be.D.calls == 80, f"overlaps {be.D.overlaps} calls {be.D.calls}")

    # 9. imgid2's pass b order (layer L+1's wrapper made while L's build is in flight; at most two layers held)
    setenv(base)
    be = make_backend(run_s=0.05, make_s=0.01)
    ns = hook_ns(be)
    ci = CPUInfer()
    cur = wrapper(gguf, layer=0, ci=ci)
    call(ns["_kt_image_init"], cur)
    inflight_at_make, ok = 0, True
    for L in range(40):
        nxt = None
        if L + 1 < 40:
            nxt = wrapper(gguf, layer=L + 1, ci=ci)
            inflight_at_make += cur._kt_image_pre is not None and not cur._kt_image_pre["build"].done()
            call(ns["_kt_image_init"], nxt)
        r, err, out = call(ns["_kt_image_load_layer"], cur, T(PMAP))
        ok = ok and r is True
        cur.moe = None
        cur = nxt
    check("pass-b order: layer L+1 made while L's build was in flight (39 of 39), all adopted, no overlap",
          ok and inflight_at_make >= 35 and be.D.overlaps == 0 and be.made_main == 0 and be.alive == 0,
          f"in flight at make {inflight_at_make}, overlaps {be.D.overlaps}, alive {be.alive}")

    # 10. a main-thread module while builds are queued: the GGUF path of a layer the image does not serve waits for
    #     every queued build first (the drain), and a failed preload does too
    setenv(base)
    be = make_backend(run_s=0.05)
    ns = hook_ns(be)
    ci = CPUInfer()
    ws = [wrapper(gguf, layer=L, ci=ci) for L in range(6)]
    for w in ws:
        call(ns["_kt_image_init"], w)
    stray = wrapper(gguf, layer=47, ci=ci)   # not in the image: init False, GGUF path in load_weights
    r, err, out = call(ns["_kt_image_load_layer"], stray, T(PMAP))
    check("GGUF path while 6 builds are queued: returns False only after all of them have run", r is False
          and all(b.done() for b in be.Build.records) and len(ns["_KT_IMAGE_INFLIGHT"]) == 6, out.strip())
    for w in ws:
        call(ns["_kt_image_load_layer"], w, T(PMAP))
    setenv(base)
    be = make_backend(run_s=0.05, status="image: short read")
    ns = hook_ns(be)
    ci = CPUInfer()
    ws = [wrapper(gguf, layer=L, ci=ci) for L in range(5)]
    for w in ws:
        call(ns["_kt_image_init"], w)
    r, err, out = call(ns["_kt_image_load_layer"], ws[0], T(PMAP))
    check("failed preload with 4 more queued: False only after all have run (the GGUF path makes on the main thread)",
          r is False and all(b.done() for b in be.Build.records), out.strip())

    # 11. the record: task made once; take() before done raises
    be = make_backend()
    b = be.image_build(Cfg(), "p", 1, [], b"", [], True, 0)
    b.task()
    try:
        b.task()
        check("build: a second task() raises", False)
    except RuntimeError:
        check("build: a second task() raises", True)
    try:
        b.take()
        check("build: take() before the task has run raises", False)
    except RuntimeError:
        check("build: take() before the task has run raises", True)

    # 12. KT_IMAGE_DIGEST=1: imgid_digest.py's formula over the six arenas of each part
    files = []
    infos = []
    for p in range(2):
        part = []
        for j in range(6):
            f = td / f"arena{p}{j}"
            f.write_bytes(bytes([p * 6 + j]) * (1000 + 7 * j))
            fd = os.open(str(f), os.O_RDONLY | getattr(os, "O_BINARY", 0))
            files.append(fd)
            part.append((fd, 1000 + 7 * j, 64 + j))
        part.append((-1, 1, 0))
        infos.append(part)
    want = []
    for p in range(2):
        canon = "".join(f"{j}:{1000 + 7 * j}:{64 + j}:{hashlib.sha256(bytes([p * 6 + j]) * (1000 + 7 * j)).hexdigest()}\n"
                        for j in range(6))
        want.append(hashlib.sha256(canon.encode()).hexdigest())
    setenv(dict(base, KT_IMAGE_DIGEST="1"))
    be = make_backend(arenas=infos)
    ns = hook_ns(be)
    w = wrapper(gguf)
    call(ns["_kt_image_init"], w)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    got = [ln.split()[-1] for ln in out.splitlines() if ln.startswith("[KT image digest] layer 3 part")]
    check("KT_IMAGE_DIGEST=1: per-part digests equal imgid_digest.py's formula", r is True and got == want, err or out.strip())
    for fd in files:
        os.close(fd)

    # 13. negative control: r2's kt_image.py, the same server order, an r2 module (made on the main thread): overlaps
    if R2_KTI is not None:
        setenv(base)
        be = make_backend(build=False, run_s=0.05, make_s=0.01)
        ns2 = hook_ns(be, R2_KTI)
        ws, outs = server_order(ns2, be, gguf, n=8)
        check("control (r2 order on an r2 module): modules made on the main thread while a task ran: overlaps counted",
              all(o[0] for o in outs) and be.made_main == 8 and be.D.overlaps >= 1,
              f"overlaps {be.D.overlaps} of {be.D.calls} calls, main-thread makes {be.made_main}")
        for w in ws:
            w.moe = None

# patch_amx_image (both hooks) on his amx.py with patch_amx_v3.py's two edits; the strings equal build_bs.py's
spec = importlib.util.spec_from_file_location("pai", KT / "python/utils/patch_amx_image.py")
pai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pai)
bs_ns = {"__file__": str(KT / "build_bs.py")}
exec((KT / "build_bs.py").read_text(encoding="utf-8").split("def main():")[0], bs_ns)
check("patch_amx_image's four strings are build_bs.py's", all(getattr(pai, k) == bs_ns[k] for k in
      ("IMAGE_ANCHOR", "IMAGE_HOOK", "IMAGE_INIT_ANCHOR", "IMAGE_INIT_HOOK")))
bs_txt = (KT / "build_bs.py").read_text(encoding="utf-8")
check("build_bs.py binds image_build and the AVX2ContractV3_ImageBuild record (task, done, status, rows, seconds, tid, take)",
      bs_txt.count('.def_static("image_build", &ContractV3PartialBindings::build)') == 1 and all(
          bs_txt.count(f'.def("{m}", &ImgBuildT::{m})') == 1 for m in ("done", "status", "rows", "seconds", "tid", "take"))
      and bs_txt.count('.def("task", &ContractV3PartialBindings::build_task)') == 1)
hdr = (KT / "contract_v3h-moe.hpp").read_text(encoding="utf-8")
check("contract_v3h-moe.hpp: ImageBuild makes the module inside run() (std::make_shared<TP_MOE>(cfg))",
      "struct ImageBuild {" in hdr and hdr.count("m = std::make_shared<TP_MOE>(cfg);") == 1)
check("ARENA_WRITERS untouched by r3 (contract_v3h-moe.hpp is not one)", "contract_v3h-moe.hpp" not in bs_ns["ARENA_WRITERS"])
his = HIS_AMX.read_bytes().decode("utf-8")
V3 = [('AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)\n',
       'AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)\n'
       'AVX2ContractV3_MOE = getattr(_moe_mod, "AVX2ContractV3_MOE", None)   # path B contract v3 (spec r3) builds only\n'),
      ('    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()\n\n    if forced == "amx":\n',
       '    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()\n\n'
       '    if forced == "contract_v3":\n'
       '        if AVX2ContractV3_MOE is None:\n'
       '            raise RuntimeError("KT_MXFP4_BACKEND=contract_v3 requested, but AVX2ContractV3_MOE is not compiled in.")\n'
       '        return AVX2ContractV3_MOE\n'
       '    if forced == "amx":\n')]
v3 = his
for old, new in V3:
    v3 = v3.replace(old, new)
kti = (KT / "python/utils/kt_image.py").read_text(encoding="utf-8")
p = pai.patched(v3, kti)
check("patched: both hooks once, kt_image.py appended", p.count(pai.IMAGE_HOOK) == 1 and p.count(pai.IMAGE_INIT_HOOK) == 1
      and p.endswith(kti) and len(p) == len(v3) + len(kti) + len(pai.IMAGE_HOOK) - len(pai.IMAGE_ANCHOR)
      + len(pai.IMAGE_INIT_HOOK) - len(pai.IMAGE_INIT_ANCHOR))
try:
    tree = ast.parse(p)
    check("patched amx.py parses", True)
except SyntaxError as ex:
    tree = None
    check("patched amx.py parses", False, repr(ex))
if tree is not None:
    init = [n for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == "NativeMoEWrapper"
            for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    src_init = ast.get_source_segment(p, init[0]) if init else ""
    check("the init hook sits in NativeMoEWrapper.__init__", "_kt_image_init(self)" in src_init)
for name, txt in (("no contract_v3 selector", his), ("already hooked", p), ("init anchor twice", v3 + "\n" + pai.IMAGE_INIT_ANCHOR)):
    try:
        pai.patched(txt, kti)
        check(f"refuses: {name}", False)
    except ValueError as ex:
        check(f"refuses: {name}", True, str(ex))

print("PASS" if all(results) else "FAIL", f"{sum(results)}/{len(results)}")
sys.exit(0 if all(results) else 1)
