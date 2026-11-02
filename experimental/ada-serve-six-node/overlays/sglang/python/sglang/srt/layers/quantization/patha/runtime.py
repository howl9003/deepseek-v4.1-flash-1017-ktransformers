"""Default-off Path A TP1 glue. Native builds must be supplied by a hashed G0 manifest.

This adapter never builds CUDA code. The pinned upstream Python wrappers are used
unchanged, with their native module slots bound to the reviewed dual-target builds.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import weakref

from .policy import AMAX_FLOOR, DECODE_SPLITS, enabled, route_for, validate_policy

_BACKENDS = None
_WEIGHTS = {}
_WARMED = set()
_INDEXER_VALIDATED = False
_QUANTIZED = {}


def _hashed_native(item, expected_name):
    path = Path(item["path"]).resolve(strict=True)
    if item["module"] != expected_name:
        raise RuntimeError("unexpected Path A extension name")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise RuntimeError("Path A extension hash mismatch: " + str(path))
    spec = importlib.util.spec_from_file_location(expected_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[expected_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _backends(capturing):
    global _BACKENDS, _INDEXER_VALIDATED
    if _BACKENDS is None:
        if capturing:
            raise RuntimeError("Path A native modules must be loaded before graph capture")
        from .vendor.decode import patha_decode as decode
        from .vendor.prefill import patha_prefill as prefill
        manifest = json.loads(Path(os.environ["SGLANG_PATHA_NATIVE_MANIFEST"]).read_text())
        decode._EXT = _hashed_native(manifest["decode"], "patha_decode_g0_dual_s2")
        prefill._EXT = _hashed_native(manifest["prefill"], "patha_prefill_g0_dual")
        if [decode._EXT.build_config(i) for i in range(4)] != [1, 8, 1, 2]:
            raise RuntimeError("Path A requires dense LUT=1, repl=8, expert LUT=1, stages=2")
        # Opt-in is an authenticated identity receipt, not an environment bypass.
        receipt = manifest.get("indexer_prefill_identity")
        if receipt is not None:
            p = Path(receipt["path"])
            if hashlib.sha256(p.read_bytes()).hexdigest() != receipt["sha256"]:
                raise RuntimeError("indexer prefill identity receipt hash mismatch")
            evidence = json.loads(p.read_text())
            _INDEXER_VALIDATED = (
                evidence.get("status") == "PASS"
                and evidence.get("shape") == [4096, 1280]
                and evidence.get("native_sha256") == manifest["prefill"]["sha256"]
                and evidence.get("raw_i64_identity") is True
                and evidence.get("bf16_identity") is True
                and evidence.get("flags_zero") is True
            )
            if not _INDEXER_VALIDATED:
                raise RuntimeError("indexer prefill receipt has incomplete identity evidence")
        decode.error_flag(reset=True)
        prefill.error_flag(reset=True)
        _BACKENDS = decode, prefill
    return _BACKENDS


def _version(t):
    try:
        return t._version
    except RuntimeError:  # inference tensors do not track versions; serving weights are immutable
        return None


def _tensor_key(t):
    return (str(t.device), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, _version(t))


def _quantizer_context(A):
    import torch
    return (torch.cuda.is_current_stream_capturing(),
            str(A.device), torch.cuda.current_stream(A.device).cuda_stream)


def register_quantized(A, As):
    """One-use provenance; only the strict-floor v2 wrapper calls this helper.

    No value reads or device work. The producer and consumer must use the same
    tensor objects on the same stream in the same capture/eager context. A
    captured consumer therefore has the strict quantizer as its predecessor.
    Inference tensors need no version counter: their values are freshly written
    by that captured producer on every replay. Holding/reusing an old eager pair
    does not authenticate it for capture, and every normal dispatch consumes it.
    """
    key = (id(A), id(As))
    entry = {"context": _quantizer_context(A), "keys": (_tensor_key(A), _tensor_key(As))}
    def gone(_):
        if _QUANTIZED.get(key) is entry:
            del _QUANTIZED[key]
    entry["A"], entry["As"] = weakref.ref(A, gone), weakref.ref(As, gone)
    _QUANTIZED[key] = entry


def _consume_quantized(A, As):
    entry = _QUANTIZED.pop((id(A), id(As)), None)
    if (entry is None or entry["A"]() is not A or entry["As"]() is not As
            or entry["context"] != _quantizer_context(A)
            or entry["keys"] != (_tensor_key(A), _tensor_key(As))):
        raise RuntimeError("Path A requires a fresh strict-floor quantizer output on this stream/capture")


def _weights(B, Bs, decode, capturing, need_prefill):
    # Scale pointer/version is essential: reusing only B's pointer can return stale EW.
    key = (_tensor_key(B), _tensor_key(Bs))
    entry = _WEIGHTS.get(key)
    if entry is None or entry["B"]() is not B or entry["Bs"]() is not Bs:
        if capturing:
            raise RuntimeError("Path A weights must be prepared outside graph capture")
        wc, ws, EW = decode.prepare_dense_weights(B, Bs)
        entry = {"B": weakref.ref(B), "Bs": weakref.ref(Bs), "wc": wc, "ws": ws, "EW": EW}
        # Remove entries for mutated/replaced versions of this storage pair.
        for old in list(_WEIGHTS):
            if old[0][1] == key[0][1] and old[1][1] == key[1][1]:
                del _WEIGHTS[old]
        _WEIGHTS[key] = entry
    if need_prefill and "ew_full" not in entry:
        if capturing:
            raise RuntimeError("Path A prefill weight metadata must be prepared before capture")
        import torch
        entry["ew_full"] = (entry["ws"].to(torch.int16) - 127).to(torch.int8).repeat_interleave(32, 0).contiguous()
        entry["EW_full"] = entry["EW"].repeat_interleave(32).contiguous()
    return entry


def dispatch(A, B, As, Bs, block_size, output_dtype, C):
    """Return C when selected, else None. No host validation or conversion of static weights in capture."""
    if not enabled():
        return None
    validate_policy()
    import torch
    n, k = B.shape
    m = A.numel() // k
    if (n, k) not in DECODE_SPLITS or m == 0:
        return None
    if list(block_size) != [32, 32] or output_dtype != torch.bfloat16:
        raise RuntimeError("Path A dense chain requires 32x32 blocks and BF16 output")
    if A.dtype != torch.float8_e4m3fn or B.dtype != torch.float8_e4m3fn:
        raise TypeError("Path A dense chain requires E4M3 code tensors")
    if not A.is_cuda or B.device != A.device or As.device != A.device or Bs.device != A.device:
        raise ValueError("Path A tensors must share one CUDA device")
    if tuple(Bs.shape) != (n // 32, k // 32):
        raise ValueError("Path A requires unswizzled 32x32 block scales")
    # Shape warmup alone cannot authenticate externally prequantized inputs.
    # Check even in capture; no tensor value is read and no CUDA work is added.
    _consume_quantized(A, As)
    capturing = torch.cuda.is_current_stream_capturing()
    decode, prefill = _backends(capturing)
    route, splits = route_for(m, n, k, indexer_prefill_validated=_INDEXER_VALIDATED)
    entry = _weights(B, Bs, decode, capturing, route == "prefill")
    a = A.reshape(m, k)
    scales = As.reshape(m, k // 32)
    signature = (str(A.device), route, m, n, k)
    if capturing and signature not in _WARMED:
        raise RuntimeError("Warm every Path A batch/shape outside capture first")
    if not capturing:
        ea_checked = decode.exponents_from_pow2_scales(scales, check=True)
        if bool((ea_checked < -22).any()):
            raise ValueError("Path A activation scales violate the 1e-4 quantizer floor")
    if route == "decode":
        result = decode.dense_from_fp8_site(a, scales, entry["wc"], entry["ws"], entry["EW"],
                                          splits=splits, out=C.view(m, n))
    else:
        # The prefill wrapper's activation_exponents does host checks; use the
        # same graph-safe bit conversion as decode after the warmup checks.
        ea = decode.exponents_from_pow2_scales(scales, check=False)
        planes, ameta = prefill.prep_activations(a.view(torch.uint8), ea, **prefill.DENSE_PREP)
        result = prefill.dense(planes, ameta, ea, entry["wc"], entry["ew_full"], entry["EW_full"], out=C.view(m, n))
    if not capturing:
        active = decode if route == "decode" else prefill
        if active.error_flag(reset=True):
            raise RuntimeError("Path A warmup raised a numerical flag")
        _WARMED.add(signature)
    return C


def error_flags(reset=False):
    """Call between numerical/benchmark requests, never during graph capture."""
    if _BACKENDS is None:
        return {"decode": 0, "prefill": 0}
    return dict(zip(("decode", "prefill"), (b.error_flag(reset=reset) for b in _BACKENDS)))
