"""pathb-image-r2 local test (no kt module, no torch, no numpy): the preload, the lazy GGUF loader and KT_IMAGE_DIGEST in
python/utils/kt_image.py, r1's refusals on the foreground path, and the two hooks of python/utils/patch_amx_image.py on
his amx.py (the same strings as build_bs.py). Any Python 3.8+:

  python -B tests/test_kt_image_r2_local.py <kt tree (this branch)> <his amx.py (path-b-r1/his-kt/python/utils/amx.py)>

The hook functions run against stand-ins for MOEConfig, the contract v3 class (its image task is run by a fake CPUInfer
worker thread, image_done raised when it returns), the wrapper and sglang's expert-location metadata, on a synthetic
manifest, GGUF header and sidecar. One line per check, then PASS or FAIL (exit 1).
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
FMT = "pathb-arena-v1;blk=32;scale=e8m0;w=0123456789abcdef"
ROWS8 = [288] * 8
results = []


def check(name, ok, note=""):
    results.append(bool(ok))
    print(f"{'ok  ' if ok else 'FAIL'} {name}{(': ' + note[:220]) if note else ''}")


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


def make_backend(fmt=FMT, rows=ROWS8, status="ok", has_done=True, run_s=0.05, arenas=None):
    class Moe:
        made = 0
        alive = 0
        ran = 0

        def __init__(self, cfg):
            Moe.made += 1
            Moe.alive += 1
            self.cfg = cfg
            self._done = 0

        def __del__(self):
            Moe.alive -= 1

        def contract_v3_part_rows(self):
            return list(rows)

        def image_prepare(self, path, nbytes, geometry, digests):
            return "ok"

        def load_weights_image_task(self, ptr):
            self.map_ptr = ptr
            return ("task", self)

        def run(self):   # what the CPUInfer worker does with the task
            time.sleep(run_s)
            Moe.ran += 1
            self._done = 1

        def image_status(self):
            return status

        def image_stats(self):
            return (432, 7219445760, 2.3, True)

        def expert_arena_infos(self):
            return arenas

    if has_done:
        Moe.image_done = lambda self: self._done
    Moe.kt_arena_format = staticmethod(lambda: fmt)
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
                moe = self.q.pop(0)
            moe.run()

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


def hook_ns(backend):
    src = (KT / "python/utils/kt_image.py").read_text(encoding="utf-8")
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


def write_image(d):
    d.mkdir(parents=True, exist_ok=True)
    gguf = d / "model.gguf"
    hdr = b"GGUF" + bytes(range(60))
    gguf.write_bytes(hdr + b"\0" * 100)
    m = {"schema": "pathb-expert-image-v1",
         "source": {"file_size": gguf.stat().st_size, "header_bytes": len(hdr), "header_sha256": hashlib.sha256(hdr).hexdigest()},
         "geometry": {"experts": 384, "hidden": 5120, "intermediate": 2304, "block": 32, "group": 32},
         "layers": {"3": {"file": "L03.img", "bytes": 7219445760, "w": [["00" * 32]] * 3, "s": [["11" * 32]] * 3}}}
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
              "KT_ARENA_SCALE_BYTES", "KT_IMAGE_PRELOAD", "KT_IMAGE_DIGEST"):
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


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    img = td / "img"
    gguf, msha = write_image(img)
    fsha = sidecar(img / "arena_format.json", msha)
    base = {"KT_IMAGE_DIR": str(img), "KT_IMAGE_MANIFEST_SHA256": msha, "KT_IMAGE_FORMAT_SHA256": fsha}

    # 1. preload at init, adopted at load_weights
    setenv(base)
    sglang_metadata(True)
    be = make_backend(run_s=0.2)
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_init"], w)
    check("init: image can serve the layer -> True (no GGUF loader), preload queued", r is True and w.cpu_infer.submitted == 1
          and w._kt_image_pre is not None and be.made == 1 and len(ns["_KT_IMAGE_INFLIGHT"]) == 1, err or out)
    t0 = time.time()
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("load_weights: adopts the preloaded module after its own task", r is True and w.moe is not None and be.made == 1
          and be.ran == 1 and "preloaded" in out and "format " + FMT in out and w.cpu_infer.submitted == 1, err or out.strip())
    check("load_weights: waited for the task (it was still running)", time.time() - t0 >= 0.1)
    check("adopted: the in-flight record released, the map kept on the wrapper", len(ns["_KT_IMAGE_INFLIGHT"]) == 0
          and w._kt_image_map is not None and w._kt_image_pre is None)
    moe = w.moe
    w.moe = None
    del moe
    check("adopted module freed with the wrapper's reference (nothing else holds it)", be.alive == 0, f"alive {be.alive}")

    # 2. the map passed at load_weights differs: dropped after its task, the layer loaded again (r1 path)
    setenv(base)
    be = make_backend()
    ns = hook_ns(be)
    w = wrapper(gguf)
    call(ns["_kt_image_init"], w)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(list(range(384))))
    check("map changed: preload dropped, layer loaded in the foreground", r is True and "preload dropped" in out and be.made == 2
          and be.ran == 2 and be.alive == 1 and len(ns["_KT_IMAGE_INFLIGHT"]) == 0, err or out.strip())

    # 3. the GPU mask object replaced
    setenv(base)
    be = make_backend()
    ns = hook_ns(be)
    w = wrapper(gguf)
    call(ns["_kt_image_init"], w)
    w.gpu_experts_mask = T([True] + [False] * 383, 1)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("mask replaced: preload dropped, foreground load", r is True and "preload dropped" in out and be.made == 2, err or out.strip())

    # 4. a preloaded image load that fails: GGUF path (non-strict), raise (strict)
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

    # 5. KT_IMAGE_PRELOAD=0, no sglang, an r1 module: True (lazy loader), nothing queued; load_weights in the foreground
    for name, env, present, done in (("KT_IMAGE_PRELOAD=0", {"KT_IMAGE_PRELOAD": "0"}, True, True),
                                     ("no sglang metadata", {}, False, True),
                                     ("module without image_done (r1)", {}, True, False)):
        setenv(dict(base, **env))
        sglang_metadata(present)
        be = make_backend(has_done=done)
        ns = hook_ns(be)
        w = wrapper(gguf)
        r, err, out = call(ns["_kt_image_init"], w)
        r2, err2, out2 = call(ns["_kt_image_load_layer"], w, T(PMAP))
        check(f"{name}: init True, nothing queued; load_weights loads in the foreground", r is True and getattr(w, "_kt_image_pre", None)
              is None and r2 is True and be.made == 1 and "preloaded" not in out2 and "s in all" in out2, (err or out) + (err2 or out2))
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
    w = wrapper(gguf, layer=7)
    r, err, out = call(ns["_kt_image_init"], w)
    check("layer not in the image: init False", r is False)

    # 7. r1's foreground refusals unchanged (part rows not proven: one module made and dropped)
    setenv(dict(base, KT_IMAGE_PRELOAD="0"))
    be = make_backend(rows=[320] * 7 + [64])
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("part rows not proven: GGUF path, the module made once and dropped", r is False and "not among the image's proven"
          in out and w.moe is None and be.made == 1 and be.alive == 0, out.strip())
    setenv(base)
    be = make_backend(rows=[320] * 7 + [64])
    ns = hook_ns(be)
    w = wrapper(gguf)
    r, err, out = call(ns["_kt_image_init"], w)
    r2, err2, out2 = call(ns["_kt_image_load_layer"], w, T(PMAP))
    check("part rows not proven with preload on: nothing queued, load_weights refuses with the reason", r is True
          and w.cpu_infer.submitted == 0 and r2 is False and "not among the image's proven" in out2 and be.alive == 0, out2.strip())

    # 8. forty layers queued at once run in order; each load_weights waits for its own layer only
    setenv(base)
    be = make_backend(run_s=0.01)
    ns = hook_ns(be)
    ci = CPUInfer()
    m = json.loads((img / "manifest.json").read_text())
    ws = []
    for L in range(40):
        w = wrapper(gguf, layer=3, ci=ci)   # the manifest lists layer 3 only: every stand-in layer uses it
        w.layer_idx = 3
        call(ns["_kt_image_init"], w)
        ws.append(w)
    first_done_at = None
    r, err, out = call(ns["_kt_image_load_layer"], ws[0], T(PMAP))
    left = sum(1 for x in ns["_KT_IMAGE_INFLIGHT"])
    check("40 queued: the first load_weights returns before the other 39 tasks have run", r is True and left >= 30,
          f"in flight after the first adoption: {left}")
    oks = [call(ns["_kt_image_load_layer"], x, T(PMAP))[0] for x in ws[1:]]
    check("40 queued: all adopted, one module each, nothing in flight", all(oks) and be.made == 40 and be.ran == 40
          and not ns["_KT_IMAGE_INFLIGHT"])

    # 9. KT_IMAGE_DIGEST=1: imgid_digest.py's formula over the six arenas of each part
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

# patch_amx_image (both hooks) on his amx.py with patch_amx_v3.py's two edits; the strings equal build_bs.py's
spec = importlib.util.spec_from_file_location("pai", KT / "python/utils/patch_amx_image.py")
pai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pai)
bs_ns = {"__file__": str(KT / "build_bs.py")}
exec((KT / "build_bs.py").read_text(encoding="utf-8").split("def main():")[0], bs_ns)
check("patch_amx_image's four strings are build_bs.py's", all(getattr(pai, k) == bs_ns[k] for k in
      ("IMAGE_ANCHOR", "IMAGE_HOOK", "IMAGE_INIT_ANCHOR", "IMAGE_INIT_HOOK")))
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


# the init hook's behaviour, run as written: the loader is made only when the image cannot serve the layer
class NativeMoEWrapper:
    _native_loader_instance = None
    made = 0

    @staticmethod
    def _create_loader(method, weight_path):
        NativeMoEWrapper.made += 1
        return object()


body = "def _init(self, method, weight_path):\n" + pai.IMAGE_INIT_HOOK
for serve in (True, False):
    NativeMoEWrapper._native_loader_instance, NativeMoEWrapper.made = None, 0
    g = {"os": os, "NativeMoEWrapper": NativeMoEWrapper, "_kt_image_init": lambda self, s=serve: s}
    exec(body, g)
    os.environ["KT_IMAGE_DIR"] = "x"
    obj = types.SimpleNamespace()
    g["_init"](obj, "MXFP4", "w.gguf")
    if serve:
        check("init hook, image serves: no GGUF loader made, self.loader None", NativeMoEWrapper.made == 0 and obj.loader is None)
    else:
        check("init hook, image refused: the loader made as before", NativeMoEWrapper.made == 1 and obj.loader is not None)
os.environ.pop("KT_IMAGE_DIR", None)
NativeMoEWrapper._native_loader_instance, NativeMoEWrapper.made = None, 0
g = {"os": os, "NativeMoEWrapper": NativeMoEWrapper, "_kt_image_init": lambda self: True}
exec(body, g)
obj = types.SimpleNamespace()
g["_init"](obj, "MXFP4", "w.gguf")
check("init hook, KT_IMAGE_DIR unset: his path (loader made)", NativeMoEWrapper.made == 1)

print("PASS" if all(results) else "FAIL", f"{sum(results)}/{len(results)}")
sys.exit(0 if all(results) else 1)
