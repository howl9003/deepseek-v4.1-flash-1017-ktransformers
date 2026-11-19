"""Prebuild, qualify and verify the torch extensions in a dedicated installation.

Commands: run ROOT LAUNCH OUTPUT, ready ROOT LAUNCH, verify ROOT [LAUNCH]."""
import hashlib
from functools import lru_cache
import importlib.util
import json
import os
import posixpath
import re
import shlex
import shutil
import tempfile
import subprocess
import sys
import time
import types

HELPER = os.path.abspath(__file__)
MARKER = "PREBUILT_EXT.json"
BENCH_LOCK = os.environ.get("C4_BUILD_LOCK", os.path.join(os.getcwd(), ".c4-build.lock"))
# Isolate runtime switches before applying the recorded launch configuration.
PAYLOAD_DROP = ("SGLANG_", "KT_", "DSV41_", "ACCEPT_", "FEATURE_", "PATHA_", "ROUTE3_", "SM89_")
SETUP_UNSET = ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")   # setup-env.sh
PAYLOAD_SET = (("PYTHONDONTWRITEBYTECODE", "1"), ("HF_HUB_OFFLINE", "1"))          
# what torch's JIT path, nvcc, the host compiler and ccache read
TORCH_KEYS = ("TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "CUDA_HOME", "CUDA_PATH", "CXX", "CC", "CUDAHOSTCXX",
              "PYTORCH_NVCC", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
              "LDFLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "PATH",
              "MAX_JOBS", "TORCH_NVCC_FLAGS", "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "TORCH_DONT_CHECK_COMPILER_ABI",
              "CUDA_VISIBLE_DEVICES", "HOME", "LC_ALL", "LANG", "PYTHONPATH", "SGLANG_V3_FZ_EXT_SHA256")
TORCH_PREFIXES = ("CCACHE_", "NVCC_", "TORCH_")
CPUS = os.environ.get("C4_PREBUILD_CPUS", "0-23")
# key, module file under the sglang package, how the server builds it
EXTS = (
    ("hash", "srt/mem_cache/cpp_utils/native_hash.py", "hash"),
    ("ll", "srt/distributed/device_communicators/ll_allreduce.py", "init:LLAllReduce"),
    ("v3h", "srt/layers/moe/v3_his.py", "init:_LLInt64"),
    ("xrp", "srt/layers/moe/v3_xrpush.py", "ext"),
    ("fz", "srt/layers/moe/v3_freeze.py", "ext"),
    ("l2pf", "srt/layers/moe/v3_l2pf.py", "ext"),
    ("ltl", "srt/v3_lane_tl.py", "ext"),
    ("lflag", "srt/speculative/v3_spec_lane_flag.py", "ext"),
)
LOAD_ORDER = ("hash", "fz", "xrp", "l2pf", "ltl", "lflag", "ll", "v3h")   # v3_his imports v3_freeze at its top
STOP_NAMES = ("barrier", "broadcast_object_list", "all_gather_object", "get_process_group_ranks", "all_reduce")


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(p):
    with open(p, "rb") as f:
        return _sha_bytes(f.read())


def _json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save(p, obj):
    tmp = f"{p}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def _n(p):
    return os.path.normcase(os.path.normpath(p))


def _under(p, root):
    return _n(p).startswith(_n(root) + os.sep)


def _canon(a, k):
    """load_inline's arguments as given (sources by sha256), for comparing two calls."""
    def v(x):
        if isinstance(x, str):
            return x if len(x) <= 160 else {"sha256": _sha_bytes(x.encode()), "chars": len(x)}
        if isinstance(x, (list, tuple)):
            return [v(y) for y in x]
        if isinstance(x, dict):
            return {str(n): v(y) for n, y in sorted(x.items())}
        if x is None or isinstance(x, (bool, int, float)):
            return x
        return repr(x)
    return {"args": [v(x) for x in a], "kwargs": {n: v(k[n]) for n in sorted(k)}}


def _digest(call):
    return _sha_bytes(json.dumps(call, sort_keys=True).encode())


def _torch_env(env):
    out = {k: env.get(k) for k in TORCH_KEYS}
    out.update({k: v for k, v in env.items() if k.startswith(TORCH_PREFIXES)})
    return dict(sorted(out.items()))


def _ninja_log_last(p):
    """{output: [mtime, command hash]} of the last entry per output (survives ninja's log recompaction)."""
    last = {}
    try:
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not line.startswith("#") and len(parts) == 5:
                    last[parts[3]] = [parts[2], parts[4]]
    except OSError:
        return None
    return last


def _dir_state(d, name):
    st = {}
    for f in ("build.ninja", name + ".so", "main.cpp", "cuda.cu"):
        p = os.path.join(d, f)
        st[f] = [_sha_file(p), os.stat(p).st_mtime_ns] if os.path.isfile(p) else None
    st["log_last"] = _ninja_log_last(os.path.join(d, ".ninja_log"))
    return st


# ------------------------------------------------------------------------------------------------ child (server's env)

class _Stop(BaseException):
    pass


def _load_tree(sglang_dir):
    """The extension modules, each loaded by path under its own dotted name. Their parent packages are empty stand-ins
    (no __path__ entries): no sglang __init__ runs, nothing else of the package imports, and v3_his's own
    `from sglang.srt.layers.moe import v3_freeze` finds the copy loaded here."""
    rels = {k: rel for k, rel, _ in EXTS}
    rels["loader"] = "srt/utils/cpp_extension_loader.py"
    mods, files = {}, {}
    for key in ("loader", *LOAD_ORDER):
        path = os.path.join(sglang_dir, *rels[key].split("/"))
        if not os.path.isfile(path):
            continue
        dotted = "sglang." + rels[key][:-3].replace("/", ".")
        parts = dotted.split(".")
        for i in range(1, len(parts)):
            pkg = ".".join(parts[:i])
            if pkg not in sys.modules:
                m = types.ModuleType(pkg)
                m.__path__ = []
                sys.modules[pkg] = m
                if i > 1:
                    setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], m)
        spec = importlib.util.spec_from_file_location(dotted, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = m
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], m)
        spec.loader.exec_module(m)
        if key != "loader":
            mods[key] = m
        files[key] = [path, _sha_file(path)]
    return mods, files


def _conditions(mods, tp, hicache):
    his = mods.get("v3h")
    his_on = bool(his is not None and getattr(his, "ENABLED", False))
    fz = mods.get("fz")
    on = lambda k: bool(mods.get(k) is not None and getattr(mods[k], "ENABLED", False))
    return {
        "hash": hicache,
        "ll": os.environ.get("SGLANG_LL_ALLREDUCE") == "1" and tp == 2,
        "v3h": his_on and tp == 2,
        "xrp": his_on and ((getattr(his, "XRPUSH", False) and tp == 2) or getattr(his, "POSTSYNC", 0) in (1, 2)),
        "fz": his_on and fz is not None and (getattr(fz, "A1", None) is not None or getattr(fz, "RP", None) is not None),
        "l2pf": on("l2pf"),
        "ltl": on("ltl"),
        "lflag": on("lflag"),
    }


def _drive(mod, how, build):
    """Run the module's own build code with load_inline recorded; plan: stop before the build."""
    import torch
    import torch.utils.cpp_extension as ce
    api = "load" if how == "hash" else "load_inline"
    real, calls = getattr(ce, api), []

    def rec(*a, **k):
        calls.append({"call": _canon(a, k), "name": a[0] if a else k.get("name")})
        if not build:
            raise _Stop("plan: not built")
        m = real(*a, **k)
        calls[-1]["module_file"] = os.path.abspath(m.__file__) if getattr(m, "__file__", None) else None
        return m

    def stop(*a, **k):
        raise _Stop("first collective")

    dist = torch.distributed
    saved = {n: getattr(dist, n) for n in STOP_NAMES if hasattr(dist, n)}
    setattr(ce, api, rec)
    for n in saved:
        setattr(dist, n, stop)
    try:
        if how == "hash":
            mod._load_native_hash_module()
        elif how == "ext":
            mod.ext()
        else:
            getattr(mod, how.split(":", 1)[1])(None, 0, 2, None)
    except _Stop:
        pass
    finally:
        setattr(ce, api, real)
        for n, f in saved.items():
            setattr(dist, n, f)
    if len(calls) != 1:
        raise RuntimeError(f"{how}: {len(calls)} load_inline calls before its first collective (want 1)")
    return calls[0]


def child(mode, ctx_path, result_path, key=None):
    t0 = time.time()
    with open(ctx_path) as f:
        ctx = json.load(f)
    spec = importlib.util.find_spec("sglang")   # locates the package without running it
    if spec is None or not spec.origin:
        raise RuntimeError("sglang is not on this environment's path")
    res = {"mode": mode, "key": key, "pid": os.getpid(), "executable": sys.executable, "prefix": sys.prefix,
           "sglang_dir": os.path.dirname(os.path.abspath(spec.origin))}
    import torch
    import torch.utils.cpp_extension  # noqa: F401
    res["torch"] = [torch.__version__, os.path.abspath(torch.__file__)]
    mods, files = _load_tree(res["sglang_dir"])
    res["files"] = files
    how = {k: h for k, _, h in EXTS}
    if mode == "plan":
        on = _conditions(mods, int(ctx["tp"]), ctx["hicache"])
        plan = []
        for k, _, h in EXTS:
            if k in mods and on[k]:
                c = _drive(mods[k], h, build=False)
                plan.append({"key": k, "how": h, "name": c["name"], "call": c["call"], "args_sha256": _digest(c["call"]),
                             "file": files[k]})
        res.update(plan=plan, inactive=[k for k in mods if not on[k]], absent=[k for k, _, _ in EXTS if k not in mods])
    else:
        c = _drive(mods[key], how[key], build=True)
        so = c.get("module_file") or ""
        d = os.path.dirname(so)
        res.update(name=c["name"], call=c["call"], args_sha256=_digest(c["call"]), so=so, dir=d,
                   state=_dir_state(d, c["name"]) if os.path.isfile(so) else None)
    res["cuda_initialized"] = bool(torch.cuda.is_initialized())
    res["seconds"] = round(time.time() - t0, 2)
    _save(result_path, res)


# ------------------------------------------------------------------------------------------------ controller

def _need(ok, code, msg):
    if not ok:
        print({2: "refused: ", 3: "bad input: "}.get(code, "FAILED: ") + msg, flush=True)
        raise SystemExit(code)


def server_env(base, launch_env):
    """Apply the serving environment used for precompilation."""
    env = {k: v for k, v in base.items() if not k.startswith(PAYLOAD_DROP)}
    for k in SETUP_UNSET:
        env.pop(k, None)
    env.update(launch_env)
    env.update(dict(PAYLOAD_SET))
    return env


def _tp(argv):
    v = None
    for i, a in enumerate(argv[:-1]):
        if a in ("--tensor-parallel-size", "--tp-size", "--tp"):
            v = argv[i + 1]
    return int(v) if v is not None and str(v).isdigit() else None


def _ancestors():
    pids, p = set(), os.getpid()
    while p > 1 and p not in pids:
        pids.add(p)
        try:
            with open(f"/proc/{p}/stat") as f:
                p = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def _argv_paths(argv, cwd):
    """Linux argv keeps path arguments intact. Also inspect joined compiler flags and shell/Python command strings."""
    program = posixpath.basename(argv[0])
    interpreter = program in ("sh", "bash", "dash", "zsh", "ksh") or program.startswith("python")
    forwarded = ("-Xcompiler", "--compiler-options", "-Xlinker", "--linker-options")
    for i, arg in enumerate(argv):
        words = [arg]
        code = interpreter and i > 0 and re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", argv[i - 1])
        option, sep, value = arg.partition("=")
        if code or (i > 0 and argv[i - 1] in forwarded) or (sep and option in forwarded):
            lexer = shlex.shlex(value if sep and option in forwarded else arg, posix=True)
            lexer.whitespace += "();[]{}<>|&,="
            lexer.whitespace_split = True
            try:
                words = list(lexer)
            except ValueError:  # process titles may have been truncated midway through a quote
                words = arg.split()
        for word in words:
            # Keep literal path operands intact. Unpack option values and linker/preprocessor lists.
            option = re.match(r"^(?:--?[\w-]+|[A-Za-z_]\w*)=", word)
            value = word[option.end():] if option else word
            if option and len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value.startswith(("-Wl,", "-Wp,", "-Wa,")):
                parts = re.split("[,:]", value)[1:]
            elif option and option.group().rstrip("=") in ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH",
                                                          "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "PYTHONPATH"):
                parts = value.split(":")
            else:
                parts = [value]
            for part in parts:
                part = re.sub(r"^(?:@|-(?:isystem|iquote|idirafter|include|imacros|MF|I|L|B|o|C))(?=.)", "", part)
                if i == 0 and "/" not in part:  # argv[0] can be a PATH lookup or an arbitrary process title
                    continue
                if part and not part.startswith("-"):
                    if posixpath.isabs(part):
                        yield part
                    elif cwd:
                        yield posixpath.join(cwd, part)


def _ns_processes(ns, te, verbose=False):
    _need(os.path.isdir("/proc"), 2, "no /proc: cannot rule out a live server in the namespace")
    ancestors, hits = _ancestors(), []
    roots = {ns, te, os.path.realpath(ns), os.path.realpath(te)}

    def in_scope(path):
        path = posixpath.normpath(path.removesuffix(" (deleted)"))
        return any(path == root or path.startswith(root + "/") for root in roots)

    # /proc links already name the opened object. argv can instead name a symlink into the namespace.
    @lru_cache(maxsize=4096)
    def arg_in_scope(path):
        return in_scope(path) or in_scope(os.path.realpath(path))

    for d in sorted((d for d in os.listdir("/proc") if d.isdigit()), key=int):
        if int(d) == os.getpid():
            if verbose:
                print(f"SKIP pid {d} (PREBUILD inspector)")
            continue
        proc = f"/proc/{d}"
        reasons, unavailable = [], []

        def link(field):
            try:
                target = os.readlink(f"{proc}/{field}")
            except OSError:
                unavailable.append(field)
                return None
            if in_scope(target):
                reasons.append(f"{field}={target}")
            return target

        cwd = link("cwd")
        link("exe")
        try:
            with open(f"{proc}/cmdline", "rb") as f:
                argv = os.fsdecode(f.read()).rstrip("\0").split("\0")
            # An invoking shell may contain this command literally in its own argv.
            launcher = posixpath.basename(argv[0]) in ("sh", "bash", "dash")
            invocation = int(d) in ancestors and launcher and any(
                re.search(r"(?:^|[/\s])(?:launch|prebuild)\.py(?:$|[\s'\"])", arg)
                for arg in argv)
            if not invocation:
                for path in dict.fromkeys(_argv_paths(argv, cwd)):
                    if arg_in_scope(path):
                        reasons.append(f"cmdline={path}")
        except OSError:
            unavailable.append("cmdline")
        try:
            for fd in sorted(os.listdir(f"{proc}/fd")):
                link(f"fd/{fd}")
        except OSError:
            unavailable.append("fd")
        # dlopen and mmap can close the fd while the process still holds the mapped file.
        try:
            with open(f"{proc}/maps", "rb") as maps:
                mapped = set()
                for line in maps:
                    fields = line.rstrip(b"\n").split(None, 5)
                    if len(fields) == 6 and fields[5].startswith(b"/"):
                        path = os.fsdecode(fields[5])
                        if in_scope(path):
                            mapped.add(path)
                reasons.extend(f"maps={path}" for path in sorted(mapped))
        except OSError:
            unavailable.append("maps")
        if reasons:
            hits.append(f"pid {d} ({', '.join(reasons)})")
        elif verbose:
            detail = ", unavailable " + ", ".join(unavailable) if unavailable else ""
            print(f"PASS pid {d} (no matching readable path{detail})")
    return hits


def _process_check(ns, te, verbose=False):
    hits = _ns_processes(ns, te, verbose=verbose)
    for hit in hits:
        print(f"REFUSE {hit}", flush=True)
    if verbose:
        print(f"{'REFUSE' if hits else 'PASS'} namespace {ns}, extension cache {te}: {len(hits)} matching processes")
    return not hits


def _locks(te):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(te) for f in fs if f == "lock")


def _fresh_cache(ns):
    slot = os.path.join(ns, "cache", "extension-build")
    if os.path.exists(slot):
        archive = tempfile.mkdtemp(prefix="extension-crashed-", dir=os.path.dirname(slot))
        os.rename(slot, os.path.join(archive, "extension-build"))
        print(f"PREBUILD kept interrupted build in {archive}", flush=True)
    os.mkdir(slot, mode=0o700)
    return tempfile.mkdtemp(prefix="torch_extensions-", dir=slot)


def _archive_pass_locks(te, tag):
    if not _locks(te):
        return
    # A torch build can leave a lock even on success. Rename the directory as a
    # unit, so a replacement published before the rename goes into the archive.
    # copytree creates a new destination exclusively. Ninja needs the same paths
    # and mtimes in the active copy to verify that the next load does no work.
    archive = os.path.join(os.path.dirname(te), tag)
    os.rename(te, archive)
    shutil.copytree(archive, te, symlinks=True, ignore=shutil.ignore_patterns("lock"))
    print(f"PREBUILD kept {tag} locks and artifacts in {archive}", flush=True)


def marked_cache(ns, rec):
    """Resolve the verified directory from the same receipt used by the real launch tools."""
    te = rec.get("torch_extensions")
    if (not te or rec.get("status") != "OK" or rec.get("namespace") != ns
            or os.path.dirname(te) != os.path.join(ns, "cache", "extension-build")
            or rec.get("env_launch_torch", {}).get("TORCH_EXTENSIONS_DIR") != te):
        return None
    try:
        info = os.stat(te)
    except OSError:
        return None
    if rec.get("cache_identity") == [info.st_dev, info.st_ino] and os.path.isdir(te):
        return te
    return None


def _ninja_dry(d, env):
    """`ninja -n` in an extension dir under the server's PATH: (no work?, its output)."""
    nin = shutil.which("ninja", path=env.get("PATH"))
    if not nin:
        return False, "ninja not on the launch's PATH"
    r = subprocess.run([nin, "-n", "-C", d], env=env, capture_output=True, text=True, timeout=120)
    text = (r.stdout + r.stderr).strip()
    return r.returncode == 0 and "no work to do" in text, text[-300:]


def _child_cmd(py, args):
    pre = ["taskset", "-c", CPUS, "nice", "-n", "10"] if CPUS else []
    return pre + [py, "-B", HELPER, "child"] + args


def _spawn(py, env, args, log, cwd):
    with open(log, "x") as lf:
        return subprocess.Popen(_child_cmd(py, args), env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)


def _wait(p, result, log, timeout):
    try:
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        rc = "timeout"
    res = _json(result) if rc == 0 else None
    tail = ""
    if res is None:
        try:
            with open(log, errors="replace") as f:
                tail = " | ".join(f.read().strip().splitlines()[-6:])[-600:]
        except OSError:
            pass
    return rc, res, tail


def _build_pass(tag, plan, py, env, ctxp, out):
    t = time.time()
    procs = {}
    for e in plan:
        r, lg = f"{out}/{tag}-{e['key']}.json", f"{out}/{tag}-{e['key']}.log"
        procs[e["key"]] = (_spawn(py, env, ["build", ctxp, r, e["key"]], lg, out + "/tmp"), r, lg)
    got = {k: _wait(p, r, lg, 1800) for k, (p, r, lg) in procs.items()}
    return got, round(time.time() - t, 1)


def run(ns, launch_path, out, base_env=None):
    import fcntl

    with open(BENCH_LOCK, "ab") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("transient: the bench lock is held", flush=True)
            return 75
        return _run_locked(ns, launch_path, out, base_env)


def _run_locked(ns, launch_path, out, base_env):
    t0 = time.time()
    base = dict(os.environ if base_env is None else base_env)
    ns = ns.rstrip("/")
    _need(os.path.isabs(ns) and os.path.isdir(ns), 3, f"{ns} is not a namespace dir")
    launch = _json(launch_path)
    _need(isinstance(launch, dict) and launch.get("argv") and isinstance(launch.get("environment"), dict), 3,
          f"{launch_path} is not a LAUNCH.json")
    argv, lenv = launch["argv"], launch["environment"]
    py, te = argv[0], lenv.get("TORCH_EXTENSIONS_DIR", "")
    _need(_n(py) == _n(ns + "/tree/.venv/bin/python") and os.path.isfile(py), 3,
          f"the launch interpreter {py} is not {ns}/tree/.venv/bin/python")
    _need(_n(te) == _n(ns + "/cache/torch_extensions"), 3, f"the launch's TORCH_EXTENSIONS_DIR {te!r} is not this namespace's")
    tp = _tp(argv)
    _need(tp is not None, 3, "no --tensor-parallel-size in the launch argv")
    arch = lenv.get("TORCH_CUDA_ARCH_LIST", "").strip()
    _need(bool(arch) and arch.lower() != "native", 2, "TORCH_CUDA_ARCH_LIST is unset or native in the launch: torch would "
          "derive the arch from the visible GPUs (use the serving GPU architecture)")
    _need(not os.path.exists(ns + "/jit-manifest.json"), 2, f"{ns} has served (jit-manifest.json)")
    _need(not os.path.exists(f"{ns}/cache/{MARKER}"), 2, f"prebuilt before (cache/{MARKER})")
    _need(os.path.isdir(te), 3, f"{te} missing (create the extension directory first)")
    _need(not os.path.exists(out) or not os.listdir(out), 3, f"{out} is not new")
    _need(_process_check(ns, te), 2, "a process is in the namespace or its extension cache (see all hits above)")
    te = _fresh_cache(ns)
    lenv = dict(lenv, TORCH_EXTENSIONS_DIR=te)
    os.makedirs(out + "/tmp", exist_ok=True)
    env = server_env(base, lenv)
    env["TMPDIR"] = out + "/tmp"
    ctxp = out + "/ctx.json"
    _save(ctxp, {"out": out, "tp": tp, "namespace": ns, "hicache": "--enable-hierarchical-cache" in launch["argv"]})

    # plan
    p = _spawn(py, env, ["plan", ctxp, out + "/plan.json"], out + "/plan.log", out + "/tmp")
    rc, pres, tail = _wait(p, out + "/plan.json", out + "/plan.log", 900)
    _need(rc == 0 and pres is not None, 4, f"plan failed (rc {rc}): {tail}")
    _need(_under(pres["sglang_dir"], ns + "/tree"), 4, f"sglang resolves to {pres['sglang_dir']}, outside {ns}/tree")
    _need(not pres["cuda_initialized"], 4, "the plan child initialised CUDA")
    plan = pres["plan"]
    names = [e["name"] for e in plan]
    print(f"plan: {len(plan)} extensions {names}; inactive {pres['inactive']}; absent {pres['absent']}; torch "
          f"{pres['torch'][0]} {pres['torch'][1]}", flush=True)

    # pass 1: the builds, in parallel
    got1, s1 = _build_pass("build1", plan, py, env, ctxp, out)
    for e in plan:
        rc, r, tail = got1[e["key"]]
        _need(rc == 0 and r is not None, 4, f"{e['name']}: build failed (rc {rc}): {tail}")
        _need(r["name"] == e["name"] and r["args_sha256"] == e["args_sha256"], 4,
              f"{e['key']}: built {r['name']} with other arguments than planned")
        _need(not r["cuda_initialized"], 4, f"{e['name']}: the build child initialised CUDA")
        _need(r["torch"] == pres["torch"] and r["state"] is not None and _n(r["dir"]) == _n(f"{te}/{e['name']}"), 4,
              f"{e['name']}: .so missing or outside {te}")
    dirs = {e["name"]: f"{te}/{e['name']}" for e in plan}
    _archive_pass_locks(te, "build1")

    # pass 2: the no-op proof (ninja -n, then the same builders in fresh processes)
    before = {nm: _dir_state(d, nm) for nm, d in dirs.items()}
    for e in plan:
        _need(before[e["name"]] == got1[e["key"]][1]["state"], 4,
              f"{e['name']}: cache differs from the completed build")
    dry = {nm: _ninja_dry(d, env) for nm, d in dirs.items()}
    got2, s2 = _build_pass("build2", plan, py, env, ctxp, out)
    _archive_pass_locks(te, "build2")
    after = {nm: _dir_state(d, nm) for nm, d in dirs.items()}
    for e in plan:
        nm = e["name"]
        rc, r, tail = got2[e["key"]]
        _need(rc == 0 and r is not None and not r["cuda_initialized"], 4, f"{nm}: second pass failed (rc {rc}): {tail}")
        _need(_n(r["dir"]) == _n(dirs[nm]) and r["state"] == after[nm], 4,
              f"{nm}: published cache differs from the verified module")
        _need(dry[nm][0], 4, f"{nm}: ninja -n would still work: {dry[nm][1]}")
        _need(before[nm] == after[nm], 4, f"{nm}: the second pass changed the build dir (not a no-op): "
              f"{[k for k in after[nm] if after[nm][k] != before[nm][k]]}")
    left = _locks(te)
    _need(not left, 4, f"locks left under {te}: {left}")

    others = {}
    for d in sorted(os.listdir(te)):
        if d not in dirs and os.path.isdir(f"{te}/{d}"):
            others[d] = _ninja_log_last(f"{te}/{d}/.ninja_log")
    exts = {}
    for e in plan:
        nm = e["name"]
        r1, r2 = got1[e["key"]][1], got2[e["key"]][1]
        exts[nm] = dict(key=e["key"], how=e["how"], file=e["file"], call=e["call"], args_sha256=e["args_sha256"],
                        state=after[nm], build_s=r1["seconds"], noop_s=r2["seconds"], ninja_n=dry[nm][1][-80:])
    info = os.stat(te)
    rec = {"cache_identity": [info.st_dev, info.st_ino], "kind": "extension prebuild", "helper_sha256": _sha_file(HELPER), "namespace": ns, "torch_extensions": te,
           "launch": os.path.abspath(launch_path), "launch_sha256": _sha_file(launch_path), "python": py, "tp": tp,
           "torch": pres["torch"], "sglang_dir": pres["sglang_dir"], "cpus": CPUS or None,
           "env_torch": _torch_env(env), "env_launch_torch": _torch_env(lenv), "env_overrides": {"TMPDIR": env["TMPDIR"]},
           "inherited_keys": sorted(k for k in env if k not in lenv and k not in dict(PAYLOAD_SET)),
           "plan_inactive": pres["inactive"], "plan_absent": pres["absent"],
           "exts": exts, "other_dirs": others, "pass1_s": s1, "pass2_s": s2,
           "seconds": round(time.time() - t0, 1), "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "status": "OK"}
    _save(f"{ns}/cache/{MARKER}", rec)
    print(f"PASS {len(plan)} extensions built in {s1} s; second pass took {s2} s and changed no builds.", flush=True)
    return 0


def verify(ns, launch_path=None):
    ns = ns.rstrip("/")
    rec = _json(f"{ns}/cache/{MARKER}")
    if not rec:
        print(f"PREBUILD_VERIFY NO_RECORD {ns}/cache/{MARKER}")
        return 3
    te, bad, ok = rec["torch_extensions"], [], 0
    if marked_cache(ns, rec) != te:
        bad.append("cache directory differs from the verified receipt")
    for nm, e in sorted(rec["exts"].items()):
        st = _dir_state(f"{te}/{nm}", nm)
        diffs = [k for k in ("build.ninja", nm + ".so") if (st[k] or [None])[0] != (e["state"][k] or [None])[0]]
        if st["log_last"] != e["state"]["log_last"]:
            diffs.append(".ninja_log")
        if diffs:
            bad.append(f"{nm} changed after the prebuild: {diffs}")
        else:
            ok += 1
    others = rec.get("other_dirs", {})
    for d in sorted(os.listdir(te)) if os.path.isdir(te) else []:
        if d in rec["exts"] or not os.path.isdir(f"{te}/{d}"):
            continue
        last = _ninja_log_last(f"{te}/{d}/.ninja_log")
        if (d not in others and last) or (d in others and last != others[d]):
            bad.append(f"{d} built after the prebuild (not in the plan)")
    if launch_path:
        real = _json(launch_path) or {}
        a = {k: v for k, v in _torch_env(real.get("environment", {})).items() if k != "TMPDIR"}
        b = {k: v for k, v in rec["env_launch_torch"].items() if k != "TMPDIR"}
        # The server can pin the binary only after the prebuild produced it.
        freezes = [(nm, e) for nm, e in rec["exts"].items() if e["key"] == "fz"]
        if len(freezes) == 1:
            nm, binary = freezes[0]
            b["SGLANG_V3_FZ_EXT_SHA256"] = binary["state"][nm + ".so"][0]
        dk = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if dk:
            bad.append(f"the real launch's torch keys differ from the prebuild's dry run: {dk}")
    n = len(rec["exts"])
    print(f"PREBUILD_VERIFY {'PASS' if not bad else 'FAIL'} {ns}: {ok}/{n} prebuilt extensions unchanged (build.ninja, .so and "
          f".ninja_log: the server's load did no build); {sum('not in the plan' in b for b in bad)} unplanned builds"
          + ("; " + "; ".join(bad) if bad else ""))
    return 0 if not bad else 1


def ready(ns, launch_path):
    rec = _json(os.path.join(ns, "cache", MARKER)) or {}
    launch = _json(launch_path) or {}
    exts = rec.get("exts", {})
    need_hash = "--enable-hierarchical-cache" in launch.get("argv", [])
    need_flag = launch.get("environment", {}).get("SGLANG_V3_SPEC") == "1"
    expected = {"ll", "v3h", "xrp", "fz", "l2pf"}
    if need_flag:
        expected.add("lflag")
    if need_hash:
        expected.add("hash")
    return bool(launch and marked_cache(ns, rec)
                and rec.get("helper_sha256") == _sha_file(HELPER)
                and rec.get("launch_sha256") == _sha_file(launch_path)
                and {v.get("key") for v in exts.values()} >= expected
                and ("hicache_hash_cpp" in exts) == need_hash)


if __name__ == "__main__":
    if sys.argv[1:2] == ["ready"]:
        sys.exit(0 if ready(*sys.argv[2:]) else 1)
    a = sys.argv[1:]
    if a[:1] == ["child"] and len(a) >= 4:
        child(a[1], a[2], a[3], a[4] if len(a) > 4 else None)
    elif a[:1] == ["run"] and len(a) == 4:
        sys.exit(run(a[1], a[2], a[3]))
    elif a[:1] == ["verify"] and len(a) in (2, 3):
        sys.exit(verify(a[1], a[2] if len(a) == 3 else None))
    else:
        print(__doc__)
        sys.exit(3)
