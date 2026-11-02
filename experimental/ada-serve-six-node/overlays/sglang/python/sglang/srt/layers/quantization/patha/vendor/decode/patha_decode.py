"""Path A kernel 1: loader and wrappers for the exact integer decode GEMV (M <= 8), default-off.

Builds patha_decode.cu with torch's extension loader using the active environment's nvcc (sm_89), the route 3 flags
(-O3 -gencode=arch=compute_89,code=sm_89 -fmad=false, no fast-math). Build knobs come from the environment and are
folded into the extension name so variants never share a cache entry:

    PATHA_DENSE_LUT   1 (default) 256-entry int32 smem table, replicated PATHA_LUT_REPL times; 0 six-op arithmetic form
    PATHA_LUT_REPL    8 (default); 1, 2, 4, 8, 16, 32
    PATHA_EXPERT_LUT  1 (default) 16-entry int32 smem table replicated 32 times; 0 prmt byte-table form
    PATHA_STAGES      3 (default); 2
    PATHA_EXT_NAME    overrides the extension name; PATHA_BUILD_VERBOSE=1 prints the build

Every variant computes the same bits; the knobs are for the microbenchmark (DESIGN_r1.md section 4, item 3).

Tensors (all CUDA, contiguous):
    activations  codes uint8 (or float8_e4m3fn viewed as uint8) [M, K], ea int8 [M, K/32]  ->  prep_activations
                 n int32 [M, K], EA int8 [M]
    dense        wc uint8 (or float8_e4m3fn) [N, K], ws uint8 E8M0 codes [N/32, K/32], EW int8 [N/32]
    expert       w uint8 nibbles [N, K/2], ws uint8 E8M0 codes [N, K/32], EW int8 [N]
    partials     P int64 [S, M, N];  output Y bfloat16 [M, N]
At the fea01 dense call site the scales are float32 powers of two: exponents_from_pow2_scales / dense_weight_codes
convert them (exact, validated) and dense_from_fp8_site wires the whole call.

Usage:
    from patha_decode import prep_activations, dense, expert, merge_finish, error_flag
    n, EA = prep_activations(codes, ea)
    Y = dense(n, ea, EA, wc, ws, EW, splits=4)         # bf16 [M, N]; splits 1 fuses the finish into the GEMV
    assert error_flag(reset=True) == 0                 # between runs, never inside a graph capture

Contract (CONTRACT_PINS_r1.md, CONTRACT_FIXES_r1.md): row base B = EA + EW - MARGIN with MARGIN_DENSE = 13 for the
dense projections and MARGIN_EXPERT = 20 for the experts (contract v2); the int64 accumulators cannot overflow on a
legal input. Finish rule: f = __ll2float_rn(A) (one direct rounding, never via double); y = f * 2^(B - SA - SW)
formed on the exponent field; A == 0 gives +0; a biased exponent above 254 gives a signed infinity and flag bit 1,
below 1 a signed zero and flag bit 1; then the integer RNE to BF16. Neither flagged case is reachable with the
contract quantiser (amax floor 1e-4, so ea >= -22) and weight scale codes >= 54 (dense) / 53 (expert), which
prepare_dense_weights / prepare_expert_weights enforce. Only the contract quantiser (1e-4 floor) may produce the
activation codes and exponents; the serving SGLang quantiser's 1e-10 floor is not it. error_flag bits: 1 int64
overflow (a defence, unreachable on legal inputs), 2 finish out of range, 4 NaN activation code, 8 exponent metadata
inconsistent (a staged ea_g above EA or ew_g above EW: a stale exponent table).
"""
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT = None

MARGIN_DENSE = 13
MARGIN_EXPERT = 20
MIN_SCALE_CODE_DENSE = 54      # finish-rule admission (M1): smallest weight scale code that keeps every finish normal
MIN_SCALE_CODE_EXPERT = 53
FLAG_OVERFLOW, FLAG_FINISH, FLAG_NAN_CODE, FLAG_META = 1, 2, 4, 8
SA = 9
SW_DENSE = 9
SW_EXPERT = 1
MAX_ROWS = 8
SPLITS = (1, 2, 4)
DENSE_SHAPES = {(1792, 5120), (32768, 1280), (5120, 8192), (4608, 5120), (5120, 2304), (4096, 1280)}
EXPERT_SHAPES = {(4608, 5120), (5120, 2304)}
KIND_DENSE, KIND_EXPERT = 0, 1


def build_knobs():
    return {
        "PATHA_DENSE_LUT": int(os.environ.get("PATHA_DENSE_LUT", "1")),
        "PATHA_LUT_REPL": int(os.environ.get("PATHA_LUT_REPL", "8")),
        "PATHA_EXPERT_LUT": int(os.environ.get("PATHA_EXPERT_LUT", "1")),
        "PATHA_STAGES": int(os.environ.get("PATHA_STAGES", "3")),
    }


DEFAULT_KNOBS = {"PATHA_DENSE_LUT": 1, "PATHA_LUT_REPL": 8, "PATHA_EXPERT_LUT": 1, "PATHA_STAGES": 3}


def extension_name():
    """patha_decode_sm89 for the default knobs; variants get a suffix so they never share a cache entry."""
    k = build_knobs()
    name = "patha_decode_sm89"
    if k != DEFAULT_KNOBS:
        name += "_d%d_r%d_e%d_s%d" % (k["PATHA_DENSE_LUT"], k["PATHA_LUT_REPL"], k["PATHA_EXPERT_LUT"], k["PATHA_STAGES"])
    return os.environ.get("PATHA_EXT_NAME", name)


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        k = build_knobs()
        cuda_cflags = ["-O3", "-gencode=arch=compute_89,code=sm_89", "-fmad=false", "--expt-relaxed-constexpr"]
        cuda_cflags += ["-D%s=%d" % (name, val) for name, val in k.items()]
        _EXT = load(
            name=extension_name(),
            sources=[str(_HERE / "patha_decode.cu")],
            extra_include_paths=[str(_HERE)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=cuda_cflags,
            verbose=os.environ.get("PATHA_BUILD_VERBOSE", "0") == "1",
        )
        built = {name: int(_EXT.build_config(i)) for i, name in enumerate(k)}
        if built != k:
            raise RuntimeError("patha_decode: built knobs %r differ from the requested %r (stale cache?)" % (built, k))
    return _EXT


# ----------------------------------------------------------------------------- conversions

def as_u8(t):
    """float8_e4m3fn (or uint8) codes viewed as uint8 without a copy."""
    if t.dtype == torch.uint8:
        return t
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    raise TypeError("expected uint8 or float8_e4m3fn codes, got %s" % t.dtype)


def exponents_from_pow2_scales(s, check=True):
    """float32 power-of-two scales -> int8 exponents (ea = log2(s)). Normal powers of two only; validated when check."""
    assert s.dtype == torch.float32
    bits = s.contiguous().view(torch.int32)
    if check:
        ok = ((bits & 0x7FFFFF) == 0) & (bits > 0) & ((bits >> 23) < 255)
        if not bool(ok.all()):
            raise ValueError("patha: scales must be finite positive normal powers of two")
    return ((bits >> 23) - 127).to(torch.int8)


def dense_weight_codes(Bs):
    """float32 power-of-two dense scales [N/32, K/32] -> uint8 E8M0 codes (exponent + 127); validated."""
    assert Bs.dtype == torch.float32
    bits = Bs.contiguous().view(torch.int32)
    ok = ((bits & 0x7FFFFF) == 0) & (bits > 0) & ((bits >> 23) < 255)
    if not bool(ok.all()):
        raise ValueError("patha: dense weight scales must be finite positive normal powers of two")
    return (bits >> 23).to(torch.uint8)


def prepare_dense_weights(wc, ws):
    """(codes [N, K], E8M0 codes uint8 or float32 pow2 scales [N/32, K/32]) -> (wc u8, ws u8, EW int8 [N/32]).
    Rejects NaN weight codes (the oracle raises on them)."""
    wc = as_u8(wc).contiguous()
    if ws.dtype == torch.float32:
        ws = dense_weight_codes(ws)
    assert ws.dtype == torch.uint8
    ws = ws.contiguous()
    if bool(((wc & 127) == 127).any()) or bool((ws == 255).any()):
        raise ValueError("patha: NaN code in dense weights or scales")
    if bool((ws < MIN_SCALE_CODE_DENSE).any()):
        raise ValueError("patha: dense scale code below %d would make an out-of-range finish reachable" % MIN_SCALE_CODE_DENSE)
    EW = (ws.to(torch.int32).max(dim=1).values - 127).to(torch.int8)
    return wc, ws, EW


def prepare_expert_weights(w, ws):
    """(nibbles [N, K/2], E8M0 codes [N, K/32]) -> (w u8, ws u8, EW int8 [N]); contract_v2.prepare_weights for EW."""
    w = w.contiguous()
    ws = ws.contiguous()
    assert w.dtype == torch.uint8 and ws.dtype == torch.uint8
    if bool((ws == 255).any()):
        raise ValueError("patha: E8M0 NaN scale")
    if bool((ws < MIN_SCALE_CODE_EXPERT).any()):
        raise ValueError("patha: expert scale code below %d would make an out-of-range finish reachable" % MIN_SCALE_CODE_EXPERT)
    EW = (ws.to(torch.int32).max(dim=1).values - 127).to(torch.int8)
    return w, ws, EW


# ----------------------------------------------------------------------------- kernels

def prep_activations(codes, ea, n=None, EA=None):
    """codes [M, K] (uint8 or float8_e4m3fn), ea int8 [M, K/32] -> (n int32 [M, K], EA int8 [M])."""
    ext = _load()
    codes = as_u8(codes).contiguous()
    ea = ea.contiguous()
    M, K = codes.shape
    if n is None:
        n = torch.empty((M, K), dtype=torch.int32, device=codes.device)
    if EA is None:
        EA = torch.empty((M,), dtype=torch.int8, device=codes.device)
    ext.prep_activations(codes, ea, n, EA)
    return n, EA


def dense_partials(n, ea, EA, wc, ws, EW, splits=4, out=None):
    """-> P int64 [splits, M, N]: the raw split partials (the oracle's accumulate_exact per split)."""
    ext = _load()
    M = n.shape[0]
    N = wc.shape[0]
    if out is None:
        out = torch.empty((splits, M, N), dtype=torch.int64, device=n.device)
    ext.dense_partials(n, ea, EA, as_u8(wc), ws, EW, out, int(splits))
    return out


def expert_partials(n, ea, EA, w, ws, EW, splits=4, out=None):
    ext = _load()
    M = n.shape[0]
    N = w.shape[0]
    if out is None:
        out = torch.empty((splits, M, N), dtype=torch.int64, device=n.device)
    ext.expert_partials(n, ea, EA, w, ws, EW, out, int(splits))
    return out


def merge_finish(P, EA, EW, kind, out=None):
    """P int64 [S, M, N] -> Y bf16 [M, N]: int64 sum over S in index order, then the finish."""
    ext = _load()
    S, M, N = P.shape
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=P.device)
    ext.merge_finish(P, EA, EW, int(kind), out)
    return out


def dense(n, ea, EA, wc, ws, EW, splits=4, out=None, fuse=True, partials=None):
    """Exact path A dense projection: bf16 [M, N]. splits=1 with fuse applies the finish inside the GEMV."""
    ext = _load()
    M = n.shape[0]
    N = wc.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=n.device)
    if splits == 1 and fuse:
        ext.dense_finish(n, ea, EA, as_u8(wc), ws, EW, out)
        return out
    P = dense_partials(n, ea, EA, wc, ws, EW, splits, out=partials)
    return merge_finish(P, EA, EW, KIND_DENSE, out=out)


def expert(n, ea, EA, w, ws, EW, splits=4, out=None, fuse=True, partials=None):
    """Exact path A expert projection (contract v2 arithmetic): bf16 [M, N]."""
    ext = _load()
    M = n.shape[0]
    N = w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=n.device)
    if splits == 1 and fuse:
        ext.expert_finish(n, ea, EA, w, ws, EW, out)
        return out
    P = expert_partials(n, ea, EA, w, ws, EW, splits, out=partials)
    return merge_finish(P, EA, EW, KIND_EXPERT, out=out)


def dense_from_fp8_site(A, As, wc, ws, EW, splits=4, out=None):
    """The fea01 dense call site's tensors: A float8_e4m3fn [M, K], As float32 pow2 [M, K/32]; wc/ws/EW from
    prepare_dense_weights (once per weight). Two small launches (exponent conversion, prep) then the GEMV."""
    ea = exponents_from_pow2_scales(As, check=False)       # validate with check=True outside captures
    n, EA = prep_activations(A, ea)
    return dense(n, ea, EA, wc, ws, EW, splits=splits, out=out)


def eligible(M, N, K, kind=KIND_DENSE, splits=4):
    """Admission: decode rows, the verified shapes, K % 128 == 0, N % 32 == 0, splits 1/2/4."""
    shapes = DENSE_SHAPES if kind == KIND_DENSE else EXPERT_SHAPES
    return 1 <= M <= MAX_ROWS and (N, K) in shapes and K % 128 == 0 and N % 32 == 0 and splits in SPLITS


def error_flag(reset=False):
    """Bitmask: 1 int64 overflow, 2 finish out of range, 4 NaN activation code, 8 exponent metadata inconsistent.
    Read between runs, never inside a capture; every bit is unreachable on legal inputs."""
    return int(_load().error_flag(bool(reset)))


def smem_bytes(M, kind=KIND_DENSE):
    return int(_load().smem_bytes(int(M), int(kind)))
