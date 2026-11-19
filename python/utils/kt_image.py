
# ---- pathb-nodescale (concept-r1 NODE_SCALING_r1.md section 3, 14 Nov): the block-ordered expert image ----
# Appended to amx.py by build_bs.py (it reads this module's globals: MOEConfig, AVX2ContractV3_MOE,
# _select_mxfp4_backend). NativeMoEWrapper.load_weights calls _kt_image_load_layer first when KT_IMAGE_DIR is set; it
# returns True when the layer was loaded from the image, False when the GGUF path should load it (one
# "[KT image] layer L: GGUF path: <reason>" line; KT_IMAGE_STRICT=1 raises instead). A layer loads from the image only
# if: KT_IMAGE_MANIFEST_SHA256 is the sha256 of <KT_IMAGE_DIR>/manifest.json (set without a well-formed pin: raise);
# the manifest's schema is pathb-expert-image-v1; the wrapper's checkpoint is the image's GGUF (size and header sha256);
# the layer and its geometry are listed; byte-scale arenas and the contract v3 backend are on; and kt-kernel's own
# checks pass (file size, every chunk's sha256 against the manifest, the expert map). Unset KT_IMAGE_DIR: nothing here
# runs.
# pathb-image-r1 (15 Nov, concept-r1 IMAGE_LOAD_DESIGN_r1.md section 2): and the image's proven arena format is the
# module's (AVX2ContractV3_MOE.kt_arena_format(), compiled in by build_bs.py from the arena writers' sources) and the
# layer's part rows are among the proven ones. The proof is the manifest's "arena_format" (+ "part_rows") when the
# manifest has it, else the sidecar KT_IMAGE_FORMAT_FILE (default <KT_IMAGE_DIR>/arena_format.json), schema
# pathb-image-arena-format-v1, whose sha256 must be KT_IMAGE_FORMAT_SHA256 (set but not 64 hex digits: raise) and whose
# image_manifest_sha256 must be KT_IMAGE_MANIFEST_SHA256. The sidecar is written only by a digest job that found the
# image's arenas equal to the GGUF path's (staged-imgid), so a module whose writers changed falls back to the GGUF path.
_KT_IMAGE_SCHEMA = "pathb-expert-image-v1"
_KT_IMAGE_FORMAT_SCHEMA = "pathb-image-arena-format-v1"
_KT_IMAGE_CACHE = {}
_KT_IMAGE_CLAIMS = {}


def _kt_image_write(text):
    import sys
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _kt_image_fail(layer, why):
    msg = f"[KT image] layer {layer}: GGUF path: {why}"
    _kt_image_write(msg)
    if os.environ.get("KT_IMAGE_STRICT", "") == "1":
        raise RuntimeError(msg)
    return False


def _kt_image_manifest(weight_path):
    """(manifest, "") or (None, reason); cached per process (directory, pin, checkpoint)."""
    import hashlib
    import json
    import re

    d = os.environ.get("KT_IMAGE_DIR", "")
    pin = os.environ.get("KT_IMAGE_MANIFEST_SHA256", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", pin):
        raise RuntimeError("KT_IMAGE_DIR is set without a well-formed KT_IMAGE_MANIFEST_SHA256")
    key = (d, pin, str(weight_path))
    if key in _KT_IMAGE_CACHE:
        return _KT_IMAGE_CACHE[key]
    res = (None, "unknown")
    try:
        with open(os.path.join(d, "manifest.json"), "rb") as f:
            raw = f.read()
        h = hashlib.sha256(raw).hexdigest()
        if h != pin:
            res = (None, f"manifest.json sha256 {h} is not KT_IMAGE_MANIFEST_SHA256 {pin}")
        else:
            m = json.loads(raw)
            src = m.get("source", {})
            if m.get("schema") != _KT_IMAGE_SCHEMA:
                res = (None, f"manifest schema {m.get('schema')!r}")
            elif not os.path.isfile(str(weight_path)):
                res = (None, f"the checkpoint {weight_path} is not a file (the image was built from a GGUF)")
            elif os.path.getsize(str(weight_path)) != int(src.get("file_size", -1)):
                res = (None, f"the checkpoint {weight_path} is not the image's GGUF (size)")
            else:
                with open(str(weight_path), "rb") as f:
                    hdr = f.read(int(src["header_bytes"]))
                if hashlib.sha256(hdr).hexdigest() != src.get("header_sha256"):
                    res = (None, f"the checkpoint {weight_path} is not the image's GGUF (header sha256)")
                else:
                    res = (m, "")
    except (OSError, ValueError, KeyError, TypeError) as ex:
        res = (None, f"manifest: {ex!r}")
    _KT_IMAGE_CACHE[key] = res
    return res


def _kt_image_claim(m):
    """pathb-image-r1: ((arena format, proven part rows or None), "") or (None, reason): the image's proven arena format,
    from the manifest's own field or the pinned sidecar; cached per process (manifest pin, sidecar path and pin)."""
    import hashlib
    import json
    import re

    pin = os.environ.get("KT_IMAGE_MANIFEST_SHA256", "").strip().lower()
    fpin = os.environ.get("KT_IMAGE_FORMAT_SHA256", "").strip().lower()
    if fpin and not re.fullmatch(r"[0-9a-f]{64}", fpin):
        raise RuntimeError("KT_IMAGE_FORMAT_SHA256 is set but is not 64 hex digits")
    path = os.environ.get("KT_IMAGE_FORMAT_FILE", "") or os.path.join(os.environ.get("KT_IMAGE_DIR", ""), "arena_format.json")
    key = (pin, path, fpin)
    if key in _KT_IMAGE_CLAIMS:
        return _KT_IMAGE_CLAIMS[key]
    res = (None, "unknown")
    try:
        if "arena_format" in m:
            c, src = m, "the manifest"
        elif not fpin:
            c, src = None, None
            res = (None, "the manifest has no arena_format and KT_IMAGE_FORMAT_SHA256 is unset (no proof for this image)")
        else:
            with open(path, "rb") as f:
                raw = f.read()
            h = hashlib.sha256(raw).hexdigest()
            c, src = None, path
            if h != fpin:
                res = (None, f"{path} sha256 {h} is not KT_IMAGE_FORMAT_SHA256 {fpin}")
            else:
                c = json.loads(raw)
                if c.get("schema") != _KT_IMAGE_FORMAT_SCHEMA:
                    res, c = (None, f"{path}: schema {c.get('schema')!r}"), None
                elif c.get("image_manifest_sha256") != pin:
                    res, c = (None, f"{path} is the proof for manifest {c.get('image_manifest_sha256')}, not {pin}"), None
        if c is not None:
            fmt, rows = c.get("arena_format"), c.get("part_rows")
            if not isinstance(fmt, str) or not fmt:
                res = (None, f"{src}: arena_format {fmt!r}")
            elif rows is not None and not (isinstance(rows, list) and all(
                    isinstance(r, list) and r and all(isinstance(v, int) and v > 0 for v in r) for r in rows)):
                res = (None, f"{src}: part_rows {rows!r}")
            else:
                res = ((fmt, rows), "")
    except (OSError, ValueError, TypeError, AttributeError) as ex:
        res = (None, f"arena format proof: {ex!r}")
    _KT_IMAGE_CLAIMS[key] = res
    return res


# pathb-image-r2 (16 Nov, concept-r1 STARTUP_CUTS_r1.md section 3): the image read queued when the wrapper is made, and no
# GGUF loader unless a layer falls back. NativeMoEWrapper.__init__ calls _kt_image_init(self) when KT_IMAGE_DIR is set
# (build_bs.py IMAGE_INIT_HOOK). If the checks below pass for the layer, the wrapper makes no GGUF loader (load_weights
# makes one only on a fallback). Unless KT_IMAGE_PRELOAD=0, and once sglang has published the expert map (it does so
# before it builds the model), the layer's module is made and its image task is queued on the CPUInfer right away. The
# task is r1's own load_weights_image task, so the writer, the verify-then-scatter order and the code tables do not
# change; only the time at which the task runs does. The CPUInfer runs tasks one at a time in order, so layer L's read
# runs while the dense weights load and while earlier layers are finished. load_weights adopts the module once that
# layer's task alone is done (image_done; a CPUInfer sync would wait for every queued layer). It adopts only if the
# expert map and the GPU mask are the ones the task ran with; otherwise it drops the preload, after the task is done,
# and loads the layer as r1 does.
# KT_IMAGE_DIGEST=1: after each image load, "[KT image digest] layer L part p <hex>", with staged-imgid's
# imgid_digest.py digest (sha256 of "j:size:stride:sha256(arena j)\n" for j = 0..5, each arena read whole from its memfd).
# pathb-image-r3 (16 Nov): r2 made each layer's module on the main thread while an earlier layer's image task ran on the
# CPUInfer worker. Both call NumaJobDistributor::do_numa_job (the module's constructor, moe-tp.hpp; the task's read,
# code tables and collapse), which stores one shared compute_func and spins on shared per-node flags with no lock, so
# the two could run each other's function or return early. Now the preload queues ONE task that makes the module, checks
# its part rows, prepares and reads the image (AVX2ContractV3_MOE.image_build, contract_v3h-moe.hpp ImageBuild), so every
# do_numa_job of the preloaded loads runs on the worker thread, in queue order, and the main thread goes on with the
# model build and the dense weights. The statements of the load are r2's (constructor, image_prepare, the map set,
# load_weights_image), so the arenas are byte for byte r2's. And the main thread makes or drops a module (the foreground
# path, a fallback to the GGUF path, a dropped preload) only after every queued build has run (_kt_image_drain). A module
# without image_build (r1, r2) gets no preload: its layers load in load_weights, in the foreground, as r1.
_KT_IMAGE_INFLIGHT = []   # queued builds, in queue order, until their own layer's load_weights has waited for them


def _kt_image_prep(self):
    """(context, "") or (None, reason): every check of the image path that comes before a module is made."""
    L = self.layer_idx
    m, why = _kt_image_manifest(self.weight_path)
    if m is None:
        return None, why
    rec = m.get("layers", {}).get(str(L))
    if rec is None:
        return None, "the layer is not in the image"
    g = m["geometry"]
    if (int(g["experts"]), int(g["hidden"]), int(g["intermediate"])) != (
            int(self.num_experts), int(self.hidden_size), int(self.moe_intermediate_size)):
        return None, "the image's geometry is not the layer's"
    if os.environ.get("KT_ARENA_SCALE_BYTES", "") != "1":
        return None, "the image needs byte-scale arenas (KT_ARENA_SCALE_BYTES=1)"
    try:   # r3: with the cache on, load_weights makes cache modules on the main thread: no preload may be queued
        cache_gib = float(os.environ.get("KT_MXFP4_EXPERT_CACHE_GIB", "0") or 0)
    except ValueError:
        cache_gib = 1.0
    if cache_gib > 0:
        return None, "the image needs the MXFP4 expert cache off (KT_MXFP4_EXPERT_CACHE_GIB)"
    backend_cls = _select_mxfp4_backend()
    if AVX2ContractV3_MOE is None or backend_cls is not AVX2ContractV3_MOE:
        return None, "the image needs the contract v3 backend (KT_MXFP4_BACKEND=contract_v3)"
    # pathb-image-r1: the image must be proven for the arena format this module writes
    mod_fmt = getattr(backend_cls, "kt_arena_format", None)
    if mod_fmt is None:
        return None, "the module has no kt_arena_format() (not a pathb-image-r1 build)"
    mod_fmt = str(mod_fmt())
    claim, why = _kt_image_claim(m)
    if claim is None:
        return None, why
    if claim[0] != mod_fmt:
        return None, f"the image is proven for arena format {claim[0]}, the module writes {mod_fmt}"
    try:
        digests = b"".join(bytes.fromhex(h) for sec in ("w", "s") for p in range(3) for h in rec[sec][p])
    except (KeyError, ValueError, TypeError) as ex:
        return None, f"the layer's digests: {ex!r}"
    return {"rec": rec, "g": g, "backend_cls": backend_cls, "mod_fmt": mod_fmt, "claim": claim, "digests": digests}, ""


def _kt_image_cfg(self):
    """The layer's MOEConfig for an image load (r1's, unchanged; r3: shared by the foreground path and the build task)."""
    E = int(self.num_experts)
    cfg = MOEConfig(E, self.num_experts_per_tok, self.hidden_size, self.moe_intermediate_size,
                    self.gpu_experts_mask.data_ptr())
    cfg.layer_idx = self.layer_idx
    cfg.pool = self.cpu_infer.backend_
    cfg.max_len = self.chunked_prefill_size
    cfg.swiglu_limit = self.swiglu_limit
    cfg.activation_type = 1 if getattr(self, "_activation", "silu") == "gelu_pytorch_tanh" else 0
    zero = [[0] * E]
    cfg.gate_projs, cfg.up_projs, cfg.down_projs = zero, zero, zero
    cfg.gate_scales, cfg.up_scales, cfg.down_scales = zero, zero, zero
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = 32
    cfg.quant_config.zero_point = False
    return cfg


def _kt_image_make(self, ctx):
    """(module, "") or (None, reason): the layer's contract v3 module, its part rows checked, the image prepared. Made on
    the calling (main) thread: r3 calls it only with nothing queued (_kt_image_drain first)."""
    cfg = _kt_image_cfg(self)
    moe = ctx["backend_cls"](cfg)
    rows = [int(r) for r in moe.contract_v3_part_rows()]   # pathb-image-r1: the layout must be a proven one
    if ctx["claim"][1] is not None and rows not in ctx["claim"][1]:
        del moe
        return None, f"part rows {rows} are not among the image's proven part rows {ctx['claim'][1]}"
    g, rec = ctx["g"], ctx["rec"]
    st = moe.image_prepare(os.path.join(os.environ["KT_IMAGE_DIR"], rec["file"]), int(rec["bytes"]),
                           [int(g["experts"]), int(g["hidden"]), int(g["intermediate"]), int(g["block"]), int(g["group"])],
                           ctx["digests"])
    if st != "ok":
        del moe
        return None, st
    return moe, ""


def _kt_image_preload_map(L):
    """sglang's physical-to-logical map of layer L (published before the model is built), or None outside sglang."""
    try:
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        md = get_global_expert_location_metadata()
        if md is None:
            return None
        return md.physical_to_logical_map_cpu[L].contiguous().clone()
    except Exception:
        return None


def _kt_image_preload(self, physical_to_logical_map_cpu, ctx=None):
    """pathb-image-r2/r3: queue layer L's image load now; True when queued. r3: the queued task makes the module itself
    (image_build; a module without it, r1 or r2, gets no preload). The record holds the build, the map and the GPU mask
    (the task reads both through raw pointers) until the layer's load_weights has waited for the task."""
    import time

    if getattr(self, "_kt_image_pre", None) is not None:
        return False
    if ctx is None:
        ctx, _ = _kt_image_prep(self)
        if ctx is None:
            return False
    if not hasattr(ctx["backend_cls"], "image_build"):
        return False
    pmap = physical_to_logical_map_cpu
    try:
        if int(pmap.element_size()) != 8 or int(pmap.numel()) != int(self.num_experts):
            return False
        pmap = pmap.contiguous()
        map_list = [int(v) for v in pmap.tolist()]
        mask = self.gpu_experts_mask
        mask_list = [bool(v) for v in mask.tolist()]
    except (AttributeError, TypeError, ValueError):
        return False
    g, lrec, proven = ctx["g"], ctx["rec"], ctx["claim"][1]
    build = ctx["backend_cls"].image_build(
        _kt_image_cfg(self), os.path.join(os.environ["KT_IMAGE_DIR"], lrec["file"]), int(lrec["bytes"]),
        [int(g["experts"]), int(g["hidden"]), int(g["intermediate"]), int(g["block"]), int(g["group"])],
        ctx["digests"], [] if proven is None else [[int(v) for v in r] for r in proven], proven is None,
        int(pmap.data_ptr()))
    rec = {"layer": int(self.layer_idx), "build": build, "map": pmap, "map_list": map_list, "mask": mask,
           "mask_list": mask_list, "fmt": ctx["mod_fmt"], "t0": time.time()}
    task = build.task()
    _KT_IMAGE_INFLIGHT.append(rec)
    self._kt_image_pre = rec
    self.cpu_infer.submit(task)
    return True


def _kt_image_init(self):
    """pathb-image-r2, from NativeMoEWrapper.__init__: True when the image can serve this layer (the wrapper then makes no
    GGUF loader), queueing the layer's image task when sglang's map is there and KT_IMAGE_PRELOAD is not 0. An image
    problem never raises here: load_weights repeats the checks and prints the reason (or raises, KT_IMAGE_STRICT=1)."""
    try:
        ctx, _ = _kt_image_prep(self)
    except Exception:
        return False
    if ctx is None:
        return False
    if os.environ.get("KT_IMAGE_PRELOAD", "1") != "0":
        pmap = _kt_image_preload_map(self.layer_idx)
        if pmap is not None:
            try:
                _kt_image_preload(self, pmap, ctx)
            except Exception as ex:   # the layer still loads in load_weights (r1's path)
                _kt_image_write(f"[KT image] layer {self.layer_idx}: preload not queued: {ex!r}")
    return True


def _kt_image_wait(rec):
    """Until the preload's task has run (status final), then release the record from the in-flight list."""
    import time

    build = rec["build"]
    while not build.done():
        time.sleep(0.002)
    if rec in _KT_IMAGE_INFLIGHT:
        _KT_IMAGE_INFLIGHT.remove(rec)


def _kt_image_drain():
    """pathb-image-r3: until every queued build has run (the records stay for their own layers), so that the main thread
    can make or drop a module with no do_numa_job running on the worker. Seconds waited."""
    import time

    t0 = time.time()
    for rec in list(_KT_IMAGE_INFLIGHT):
        while not rec["build"].done():
            time.sleep(0.002)
    return time.time() - t0


def _kt_image_arena_sha(fd, size):
    import hashlib

    h = hashlib.sha256()
    off = 0
    while off < size:
        b = os.pread(fd, min(8 << 20, size - off), off)
        if not b:
            raise OSError(f"short read of arena fd {fd} at {off} of {size}")
        h.update(b)
        off += len(b)
    return h.hexdigest()


def _kt_image_digest(moe, L):
    """KT_IMAGE_DIGEST=1: staged-imgid's per-part arena digests of this layer, printed; the list of part digests."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    if os.environ.get("KT_IMAGE_DIGEST", "") != "1":
        return None
    infos = [[tuple(int(v) for v in t) for t in p] for p in moe.expert_arena_infos()]
    if not infos or any(len(p) != 7 or p[6] != (-1, 1, 0) for p in infos):
        _kt_image_write(f"[KT image digest] layer {L}: no byte-arena infos, no digest")
        return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        jobs = {(i, j): ex.submit(_kt_image_arena_sha, infos[i][j][0], infos[i][j][1])
                for i in range(len(infos)) for j in range(6)}
        out = []
        for i in range(len(infos)):
            canon = "".join(f"{j}:{infos[i][j][1]}:{infos[i][j][2]}:{jobs[(i, j)].result()}\n" for j in range(6))
            out.append(hashlib.sha256(canon.encode()).hexdigest())
    for i, d in enumerate(out):
        _kt_image_write(f"[KT image digest] layer {L} part {i} {d}")
    return out


def _kt_image_load_layer(self, physical_to_logical_map_cpu):
    import time

    L = self.layer_idx
    pre = getattr(self, "_kt_image_pre", None)
    if pre is not None:   # pathb-image-r2: queued when the wrapper was made (r3: the task makes the module too)
        self._kt_image_pre = None
        t_w = time.time()
        _kt_image_wait(pre)
        waited = time.time() - t_w
        build = pre["build"]
        same = ([int(v) for v in physical_to_logical_map_cpu.tolist()] == pre["map_list"]
                and self.gpu_experts_mask is pre["mask"]
                and [bool(v) for v in self.gpu_experts_mask.tolist()] == pre["mask_list"])
        st = build.status()
        if same and st == "ok":
            moe = build.take()
            chunks, nbytes, secs, direct = moe.image_stats()
            mk, tot = build.seconds()
            self.moe = moe
            self._kt_image_map = pre["map"]   # the module's config points at this map
            _kt_image_write(f"[KT image] layer {L}: {chunks} chunks, {nbytes / 2**30:.2f} GiB, read and verified in {secs:.2f} s "
                            f"({'O_DIRECT' if direct else 'buffered'}); preloaded: queued {t_w - pre['t0']:.2f} s before "
                            f"load_weights, waited {waited:.2f} s; made in the task in {mk:.2f} s (thread {build.tid()}), "
                            f"task {tot:.2f} s; parts {','.join(str(r) for r in moe.contract_v3_part_rows())}; "
                            f"format {pre['fmt']}")
            _kt_image_digest(moe, L)
            return True
        # r3: the main thread makes the layer again (or the GGUF path does): nothing may be queued while it does
        _kt_image_drain()
        build.take()   # a module the task kept (a changed map or mask) is freed here, before the layer is made again
        pre = build = None
        if same:
            return _kt_image_fail(L, st)
        _kt_image_write(f"[KT image] layer {L}: preload dropped (the expert map or the GPU mask changed after it was queued); "
                        "loading the layer again")
    # r3: the foreground path below, and the GGUF path after a False, make the module on this thread: drain first
    _kt_image_drain()
    ctx, why = _kt_image_prep(self)
    if ctx is None:
        return _kt_image_fail(L, why)
    t0 = time.time()
    moe, why = _kt_image_make(self, ctx)
    if moe is None:
        return _kt_image_fail(L, why)
    self.cpu_infer.submit(moe.load_weights_image_task(physical_to_logical_map_cpu.data_ptr()))
    self.cpu_infer.sync()
    st = moe.image_status()
    if st != "ok":
        del moe
        return _kt_image_fail(L, st)
    chunks, nbytes, secs, direct = moe.image_stats()
    self.moe = moe
    _kt_image_write(f"[KT image] layer {L}: {chunks} chunks, {nbytes / 2**30:.2f} GiB, read and verified in {secs:.2f} s "
                    f"({'O_DIRECT' if direct else 'buffered'}); {time.time() - t0:.2f} s in all; parts "
                    f"{','.join(str(r) for r in moe.contract_v3_part_rows())}; format {ctx['mod_fmt']}")
    _kt_image_digest(moe, L)
    return True
