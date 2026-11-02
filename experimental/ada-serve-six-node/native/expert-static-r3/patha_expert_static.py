"""Path A r3-E step 1: loader and the static per-layer entry `expert_routes_patha` for the hot-slot GPU experts
(DESIGN_r3_dense.md section 2), default-off behind SGLANG_FP8_PATHA_EXPERT_STATIC (its own selector, separate from
the dense name SGLANG_FP8_PATHA_DECODE: open question 2 of integration-r1).

Builds patha_expert_static.cu with torch's extension loader using the active environment's nvcc (sm_89), the route 3
flags (-O3 -gencode=arch=compute_89,code=sm_89 -fmad=false, no fast-math). Build knobs come from the environment and
are folded into the extension name so variants never share a cache entry:

    PATHA_EXPERT_LUT   1 (default) 16-entry int32 smem table replicated 32 times; 0 prmt byte-table form
    PATHA_STAGES       3 (default); 2
    PATHA_EXPERT_BPC   8 (default) groups per chunk; 16 (design knob for the 9-chunk down projection)
    PATHA_EXPERT_TC    0 (default); 1 is the step 2 tensor-digit knob, designed but not implemented: refused here
    PATHA_EXT_NAME     overrides the extension name; PATHA_BUILD_VERBOSE=1 prints the build

Every built variant computes the same bits.

The per-layer entry (the signature of dp4a_routes_candidate.expert_routes_splitk, selected in moe.expert_routes ahead
of the DP4A branch; INTEGRATION.md section 1):

    out = expert_routes_patha(x, w13, s13, w2, s2, ids, weights, out=out)

    x bf16 [M, H]; w13 [E, 2I, H/2] and w2 [E, H, I/2] uint8 nibbles; s13 [E, 2I, H/32] and s2 [E, H, I/32] uint8
    E8M0; ids [M, topk] int32/int64; weights [M, topk] float32; out bf16 [M, topk, H] (inactive slots untouched).
    Five launches per layer: _quant_codes(x) -> codes_x, ea_x; gate/up static launch -> gu f32 [count, 2I]
    (BF16 values); the overlay's _activate_routes (unchanged) -> h f32 [count, I]; _quant_codes(h) -> codes_h, ea_h;
    down static launch -> out. Admission: 1 <= M <= 8, H = 5120, 2I = 4608, 1 <= topk <= 6, contiguous CUDA tensors.

The quantiser: `_quant_codes` is `_quant_v2` (contract_v2/kernels.py, the contract quantiser, amax floor 1e-4)
writing only the codes and the group exponents (no q/hi/lo/planes, no tables); same arithmetic line for line, so its
codes and ea equal `_quant_v2`'s and contract_v2.quant's bit for bit. One addition: a group with a non-finite element
gets NaN codes (127), which the kernel flags (bit 2); this replaces the DP4A path's two `_assert_async` nodes with a
launch-free fail-closed check. Only this quantiser may feed the kernel (CONTRACT_PINS_r1.md pin c).

Static workspaces (one `Workspace` per (M, topk), i.e. per captured batch graph) are allocated outside any capture:
codes_x [M, H] u8, ea_x [M, H/32] i8, gu [count, 2I] f32, h [count, I] f32, codes_h [count, I] u8, ea_h [count,
I/32] i8, and the flag word (int32 [1]). The kernel sets no attribute and allocates nothing at launch: `_load()`
calls `configure()` once (all 16 instantiations), and `warmup()` runs the Triton kernels once so their JIT happens
before a capture. `active` is `torch.where(valid, ids, -1)` computed per call, as the DP4A launcher does (captured
allocations come from the graph's pool).

Flags (one word per workspace, passed to every launch): bit 0 int64 overflow (defence), bit 1 finish out of range,
bit 2 NaN activation code, bit 3 exponent metadata inconsistent (defence), bit 4 weight scale code 255 or below
53 (the finish rule's admission, enforced in-kernel because arena-DMA'd slots never pass prepare_expert_weights),
bit 5 more routes share one expert than M slots (a token routed twice to one expert; surplus routes not served).
Eager mode: `expert_routes_patha` reads the word synchronously after the down launch and raises on a set bit (fail
closed at once). Graph mode: the read is skipped inside the capture; call `check_flags_after_step(ws)` once per step
after the replay's host sync (the graph-mode rule). `reset_flags(ws)` zeroes it outside captures.

Contract (CONTRACT_PINS_r1.md, CONTRACT_FIXES_r1.md): B = EA + EW - MARGIN_EXPERT (20); one direct int64 -> FP32
rounding (__ll2float_rn); the finish rule; FTZ/DAZ off (-fmad=false, no fast-math, no ftz); the contract quantiser.
"""
import os
from pathlib import Path

import torch

try:
    import triton as tr
    import triton.language as tl
except ImportError:  # the local self-test reads this file as text only; the kernels need Triton on the card
    tr = tl = None

_HERE = Path(__file__).resolve().parent
_EXT = None

SELECTOR = 'SGLANG_FP8_PATHA_EXPERT_STATIC'      # default off; read by moe.expert_routes (INTEGRATION.md section 1)
KERNEL_DIR = 'SGLANG_FP8_PATHA_EXPERT_STATIC_DIR'  # folder holding this file and patha_expert_static.cu (overlay import)
MARGIN_EXPERT = 20
SA = 9
SW_EXPERT = 1
MIN_SCALE_CODE_EXPERT = 53
FLAG_OVERFLOW, FLAG_FINISH, FLAG_NAN_CODE, FLAG_META, FLAG_BAD_SCALE, FLAG_SLOTS = 1, 2, 4, 8, 16, 32
FLAG_NAMES = {FLAG_OVERFLOW: 'int64 overflow', FLAG_FINISH: 'finish out of range', FLAG_NAN_CODE: 'NaN activation code',
              FLAG_META: 'exponent metadata inconsistent', FLAG_BAD_SCALE: 'weight scale code outside the admission',
              FLAG_SLOTS: 'more routes share an expert than M slots'}
MAX_ROWS = 8
MAX_TOPK = 6
MAX_COUNT = MAX_ROWS * MAX_TOPK
HIDDEN = 5120
TWO_I = 4608
INTER = TWO_I // 2
BLOCK_K = 32
EXPERT_SHAPES = {(TWO_I, HIDDEN), (HIDDEN, INTER)}      # (N, K): gate/up and down

DEFAULT_KNOBS = {'PATHA_EXPERT_LUT': 1, 'PATHA_STAGES': 3, 'PATHA_EXPERT_BPC': 8, 'PATHA_EXPERT_TC': 0}


def build_knobs():
    return {name: int(os.environ.get(name, str(default))) for name, default in DEFAULT_KNOBS.items()}


def extension_name():
    """patha_expert_static_sm89 for the default knobs; variants get a suffix so they never share a cache entry."""
    k = build_knobs()
    name = 'patha_expert_static_sm89'
    if k != DEFAULT_KNOBS:
        name += '_e%d_s%d_b%d_t%d' % (k['PATHA_EXPERT_LUT'], k['PATHA_STAGES'], k['PATHA_EXPERT_BPC'], k['PATHA_EXPERT_TC'])
    return os.environ.get('PATHA_EXT_NAME', name)


def selector_on(environ=None):
    environ = os.environ if environ is None else environ
    return environ.get(SELECTOR, '0') == '1'


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        k = build_knobs()
        if k['PATHA_EXPERT_TC'] != 0:
            raise RuntimeError('patha_expert_static: PATHA_EXPERT_TC=1 (step 2 tensor-digit planes) is designed but not implemented in r3-E step 1')
        if k['PATHA_EXPERT_BPC'] not in (8, 16) or k['PATHA_STAGES'] < 2 or k['PATHA_EXPERT_LUT'] not in (0, 1):
            raise RuntimeError('patha_expert_static: knobs out of range %r' % (k,))
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('patha_expert_static: load the extension outside any CUDA graph capture')
        cuda_cflags = ['-O3', '-gencode=arch=compute_89,code=sm_89', '-fmad=false', '--expt-relaxed-constexpr']
        cuda_cflags += ['-D%s=%d' % (name, val) for name, val in k.items()]
        ext = load(
            name=extension_name(),
            sources=[str(_HERE / 'patha_expert_static.cu')],
            extra_include_paths=[str(_HERE)],
            extra_cflags=['-O2'],
            extra_cuda_cflags=cuda_cflags,
            verbose=os.environ.get('PATHA_BUILD_VERBOSE', '0') == '1',
        )
        built = {name: int(ext.build_config(i)) for i, name in enumerate(k)}
        if built != k:
            raise RuntimeError('patha_expert_static: built knobs %r differ from the requested %r (stale cache?)' % (built, k))
        ext.configure()                                    # cudaFuncSetAttribute for every instantiation, outside any capture
        _EXT = ext
    return _EXT


# ----------------------------------------------------------------------------- the quantiser (codes and ea only)
if tr is not None:
    @tr.jit
    def _quant_codes(X, CODES, EA):
        # _quant_v2 (contract_v2/kernels.py) writing only the codes and the group exponents; the arithmetic lines are
        # its own, unchanged (the self-test compares them). X may be BF16 or FP32 (BF16-valued): exact widening.
        block = tl.program_id(0)
        j = tl.arange(0, 32)
        v = tl.load(X + block * 32 + j).to(tl.float32)
        nonfinite = tl.max(((v != v) | (tl.abs(v) == float('inf'))).to(tl.int32), 0)
        am = tl.maximum(tl.max(tl.abs(v), 0), 1.e-4)
        b = (am * (1. / 448.)).to(tl.uint32, bitcast=True)
        ex = ((b >> 23) & 255).to(tl.int32) + ((b & 0x7fffff) != 0).to(tl.int32) - 127
        scale = ((ex + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
        z = v / scale
        a = tl.abs(z); ab = a.to(tl.uint32, bitcast=True)
        normal = ((ab + 0x7ffff + ((ab >> 20) & 1)) >> 20).to(tl.int32) - 960
        sub = a * 512.; flo = tl.floor(sub); frac = sub - flo
        rounded = flo.to(tl.int32) + ((frac > .5) | ((frac == .5) & ((flo.to(tl.int32) & 1) != 0))).to(tl.int32)
        code = tl.where(a >= 448., 126, tl.where(a < .015625, rounded, normal))
        sign = ((z.to(tl.uint32, bitcast=True) >> 31) & 1).to(tl.int32)
        code = code | (sign << 7)
        code = tl.where(nonfinite != 0, 127, code)         # a non-finite group -> NaN codes -> kernel flag bit 2
        tl.store(CODES + block * 32 + j, code.to(tl.uint8))
        tl.store(EA + block, ex.to(tl.int8))


def quant_codes(x, codes, ea):
    """x bf16 or f32 [R, K] contiguous -> codes u8 [R, K], ea i8 [R, K/32] (preallocated). The contract quantiser."""
    if tr is None:
        raise ImportError('patha_expert_static: Triton is required for _quant_codes')
    R, K = x.shape
    assert x.is_contiguous() and K % BLOCK_K == 0 and codes.shape == (R, K) and ea.shape == (R, K // BLOCK_K)
    assert codes.dtype == torch.uint8 and ea.dtype == torch.int8 and x.dtype in (torch.bfloat16, torch.float32)
    _quant_codes[(R * K // BLOCK_K,)](x, codes, ea, enable_fp_fusion=False, num_warps=1)
    return codes, ea


# ----------------------------------------------------------------------------- workspaces and flags
class Workspace:
    """Static buffers of one (M, topk) launch shape, i.e. of one captured batch graph. Construct outside any capture."""

    def __init__(self, device, M, topk, hidden=HIDDEN, inter=INTER):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('patha_expert_static: allocate workspaces outside any CUDA graph capture')
        if not (1 <= M <= MAX_ROWS and 1 <= topk <= MAX_TOPK):
            raise ValueError('patha_expert_static: workspace needs 1 <= M <= 8 and 1 <= topk <= 6')
        self.device = torch.device(device)
        self.M, self.topk, self.hidden, self.inter = M, topk, hidden, inter
        count = M * topk
        self.count = count
        d = self.device
        self.codes_x = torch.empty((M, hidden), dtype=torch.uint8, device=d)
        self.ea_x = torch.empty((M, hidden // BLOCK_K), dtype=torch.int8, device=d)
        self.gu = torch.zeros((count, 2 * inter), dtype=torch.float32, device=d)
        self.h = torch.zeros((count, inter), dtype=torch.float32, device=d)
        self.codes_h = torch.empty((count, inter), dtype=torch.uint8, device=d)
        self.ea_h = torch.empty((count, inter // BLOCK_K), dtype=torch.int8, device=d)
        self.flag = torch.zeros((1,), dtype=torch.int32, device=d)


_WORKSPACES = {}


def workspace(device, M, topk):
    key = (str(torch.device(device)), int(M), int(topk))
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = Workspace(device, M, topk)
        _WORKSPACES[key] = ws
    return ws


def flags(ws):
    """The workspace's flag word (synchronous read; never inside a capture)."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('patha_expert_static: flags() inside a capture')
    return int(ws.flag.item())


def reset_flags(ws):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('patha_expert_static: reset_flags inside a capture')
    ws.flag.zero_()


def describe_flags(f):
    return ', '.join(name for bit, name in FLAG_NAMES.items() if f & bit) or 'clear'


def check_flags(ws, where=''):
    """Fail closed: a set bit raises; the caller discards the output."""
    f = flags(ws)
    if f:
        raise RuntimeError('patha_expert_static: flag word %#x (%s)%s; output discarded' % (f, describe_flags(f), where))


def check_flags_after_step(ws):
    """Graph mode: once per decode step after the replay has been synchronised (the sampler's host sync)."""
    check_flags(ws, ' at the end of the step')


# ----------------------------------------------------------------------------- the overlay pieces (called, not reimplemented)
def _overlay():
    """The serving overlay's SiLU table and its unchanged _activate_routes (dp4a_routes_candidate)."""
    from sglang.srt.layers.moe.contract_v2 import moe as _moe
    from sglang.srt.layers.moe.contract_v2.dp4a_routes_candidate import _activate_routes
    return _moe, _activate_routes


def _as_bytes(t, what):
    if t.dtype == torch.uint8:
        return t
    if t.dtype == torch.int8 or t.element_size() == 1:
        return t.view(torch.uint8)
    raise RuntimeError('expert_routes_patha: %s must be one byte per value (uint8, int8 or float8_e8m0fnu), got %s' % (what, t.dtype))


def eligible(M, hidden, two_i, topk):
    return 1 <= M <= MAX_ROWS and hidden == HIDDEN and two_i == TWO_I and 1 <= topk <= MAX_TOPK


def warmup(device, M_list=(1, 2, 4, 8), topk=MAX_TOPK):
    """Load and configure the extension, allocate the workspaces of the batch shapes, JIT the Triton kernels once.
    Outside any capture."""
    ext = _load()
    _moe, activate_routes = _overlay()
    silu = _moe._adapter(torch.device(device)).silu
    for M in M_list:
        ws = workspace(device, M, topk)
        quant_codes(torch.zeros((M, HIDDEN), dtype=torch.bfloat16, device=device), ws.codes_x, ws.ea_x)
        active = torch.full((M, topk), -1, dtype=torch.int64, device=device)
        weights = torch.zeros((M, topk), dtype=torch.float32, device=device)
        activate_routes[(ws.count, (INTER + 127) // 128)](ws.gu, active, weights, silu, ws.h, INTER, 128, enable_fp_fusion=False)
        quant_codes(ws.h, ws.codes_h, ws.ea_h)
        reset_flags(ws)
    torch.cuda.synchronize(device)
    return ext


def expert_routes_patha(x, w13, s13, w2, s2, ids, weights, *, out=None, silu=None, activate_routes=None, stages=False,
                        eager_check=True, debug_A=None):
    """The static per-layer entry (signature of dp4a_routes_candidate.expert_routes_splitk). Returns out bf16
    [M, topk, H]; with stages=True also the intermediates (active, codes_x, ea_x, gu, h, codes_h, ea_h). debug_A, if
    given as (A13 int64 [count, 2I], A2 int64 [count, H]), receives the raw row sums of both projections."""
    ext = _load()
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.ndim != 2 or not x.is_contiguous():
        raise ValueError('expert_routes_patha: contiguous CUDA BF16 activations [M, H] required')
    M, hidden = x.shape
    if ids.ndim != 2 or ids.shape[0] != M or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('expert_routes_patha: ids [M, topk] int32/int64 required')
    topk = ids.shape[1]
    if weights.shape != ids.shape or weights.dtype != torch.float32:
        raise ValueError('expert_routes_patha: weights [M, topk] float32 required')
    w13, s13, w2, s2 = [_as_bytes(t, name) for t, name in zip((w13, s13, w2, s2), ('w13', 's13', 'w2', 's2'))]
    E, two_i, hh = w13.shape
    inter = two_i // 2
    if not eligible(M, hidden, two_i, topk):
        raise ValueError('expert_routes_patha: admission is 1 <= M <= 8, H = 5120, 2I = 4608, 1 <= topk <= 6 (got M=%d, H=%d, 2I=%d, topk=%d)' % (M, hidden, two_i, topk))
    if hh * 2 != hidden or s13.shape != (E, two_i, hidden // BLOCK_K) or w2.shape != (E, hidden, inter // 2) or s2.shape != (E, hidden, inter // BLOCK_K):
        raise ValueError('expert_routes_patha: weight shapes do not match the contract layout')
    if any(not t.is_contiguous() or t.device != x.device for t in (w13, s13, w2, s2, ids, weights)):
        raise ValueError('expert_routes_patha: contiguous tensors on the device of x required')
    count = M * topk
    if out is None:
        out = torch.zeros((M, topk, hidden), dtype=torch.bfloat16, device=x.device)
    if out.shape != (M, topk, hidden) or out.dtype != torch.bfloat16 or not out.is_contiguous() or out.device != x.device:
        raise ValueError('expert_routes_patha: out must be a contiguous BF16 [M, topk, H] on the device of x')
    if silu is None or activate_routes is None:
        _moe, _act = _overlay()
        silu = _moe._adapter(x.device).silu if silu is None else silu
        activate_routes = _act if activate_routes is None else activate_routes
    ws = workspace(x.device, M, topk)
    active = torch.where((ids >= 0) & (ids < E) & (weights != 0), ids, -1).contiguous()   # [M, topk], -1 = inactive
    quant_codes(x, ws.codes_x, ws.ea_x)
    if debug_A is None:
        ext.expert_static(ws.codes_x, ws.ea_x, w13, s13, active, ws.gu, ws.flag, True, count, topk, M)
    else:
        ext.expert_static_debug(ws.codes_x, ws.ea_x, w13, s13, active, ws.gu, debug_A[0], ws.flag, True, count, topk, M)
    activate_routes[(count, (inter + 127) // 128)](ws.gu, active, weights, silu, ws.h, inter, 128, enable_fp_fusion=False)
    quant_codes(ws.h, ws.codes_h, ws.ea_h)
    out2 = out.view(count, hidden)
    if debug_A is None:
        ext.expert_static(ws.codes_h, ws.ea_h, w2, s2, active, out2, ws.flag, False, count, topk, M)
    else:
        ext.expert_static_debug(ws.codes_h, ws.ea_h, w2, s2, active, out2, debug_A[1], ws.flag, False, count, topk, M)
    if eager_check and not torch.cuda.is_current_stream_capturing():
        check_flags(ws, ' after expert_routes_patha(M=%d, topk=%d)' % (M, topk))
    if stages:
        return out, dict(active=active, codes_x=ws.codes_x, ea_x=ws.ea_x, gu=ws.gu, h=ws.h, codes_h=ws.codes_h, ea_h=ws.ea_h, workspace=ws)
    return out


def smem_bytes(M):
    return int(_load().smem_bytes(int(M)))
