
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
_KT_IMAGE_SCHEMA = "pathb-expert-image-v1"
_KT_IMAGE_CACHE = {}


def _kt_image_fail(layer, why):
    msg = f"[KT image] layer {layer}: GGUF path: {why}"
    print(msg, flush=True)
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


def _kt_image_load_layer(self, physical_to_logical_map_cpu):
    import time

    L = self.layer_idx
    m, why = _kt_image_manifest(self.weight_path)
    if m is None:
        return _kt_image_fail(L, why)
    rec = m.get("layers", {}).get(str(L))
    if rec is None:
        return _kt_image_fail(L, "the layer is not in the image")
    g = m["geometry"]
    if (int(g["experts"]), int(g["hidden"]), int(g["intermediate"])) != (
            int(self.num_experts), int(self.hidden_size), int(self.moe_intermediate_size)):
        return _kt_image_fail(L, "the image's geometry is not the layer's")
    if os.environ.get("KT_ARENA_SCALE_BYTES", "") != "1":
        return _kt_image_fail(L, "the image needs byte-scale arenas (KT_ARENA_SCALE_BYTES=1)")
    backend_cls = _select_mxfp4_backend()
    if AVX2ContractV3_MOE is None or backend_cls is not AVX2ContractV3_MOE:
        return _kt_image_fail(L, "the image needs the contract v3 backend (KT_MXFP4_BACKEND=contract_v3)")
    try:
        digests = b"".join(bytes.fromhex(h) for sec in ("w", "s") for p in range(3) for h in rec[sec][p])
    except (KeyError, ValueError, TypeError) as ex:
        return _kt_image_fail(L, f"the layer's digests: {ex!r}")
    t0 = time.time()
    E = int(self.num_experts)
    cfg = MOEConfig(E, self.num_experts_per_tok, self.hidden_size, self.moe_intermediate_size,
                    self.gpu_experts_mask.data_ptr())
    cfg.layer_idx = L
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
    moe = backend_cls(cfg)
    st = moe.image_prepare(os.path.join(os.environ["KT_IMAGE_DIR"], rec["file"]), int(rec["bytes"]),
                           [int(g["experts"]), int(g["hidden"]), int(g["intermediate"]), int(g["block"]), int(g["group"])],
                           digests)
    if st != "ok":
        del moe
        return _kt_image_fail(L, st)
    self.cpu_infer.submit(moe.load_weights_image_task(physical_to_logical_map_cpu.data_ptr()))
    self.cpu_infer.sync()
    st = moe.image_status()
    if st != "ok":
        del moe
        return _kt_image_fail(L, st)
    chunks, nbytes, secs, direct = moe.image_stats()
    self.moe = moe
    print(f"[KT image] layer {L}: {chunks} chunks, {nbytes / 2**30:.2f} GiB, read and verified in {secs:.2f} s "
          f"({'O_DIRECT' if direct else 'buffered'}); {time.time() - t0:.2f} s in all; parts "
          f"{','.join(str(r) for r in moe.contract_v3_part_rows())}", flush=True)
    return True
