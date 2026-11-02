"""Path A kernel 2 (chunked-fold INT8 IMMA prefill GEMM), revision r3-P3: loader and wrappers, default-off.

Builds patha_prefill.cu with torch's extension loader using the active environment's nvcc (fea01 trial venv, CUDA 13.x,
sm_89). No fast-math, no fmad contraction; the finish uses __ll2float_rn and integer exponent-field operations only.

Usage (dispatch described in INTEGRATION.md):

    import patha_prefill as pa
    ea = pa.activation_exponents(As)                      # float32 power-of-two scales [M, K/32] -> int8 exponents
    planes, ameta = pa.prep_activations(A.view(torch.uint8), ea)      # once per projection input (dense: 3 planes, 4 groups)
    ew = pa.weight_exponents_dense(Bs); EW = pa.row_max(ew)           # once per weight (cache with the weight)
    y = pa.dense(planes, ameta, ea, B.view(torch.uint8), ew, EW)      # bf16 [M, N]
    A64 = pa.dense(..., mode="i64")                                   # int64 row sums before the finish (identity tests)
    assert pa.error_flag(reset=True) == 0                             # between runs, never inside a capture

Experts (r3-P2): the kernel configuration follows the prepared activation operand:
    planes, ameta = pa.prep_activations(codes, ea, **pa.EXPERT_GATE_UP_PREP)   # 3 planes, fold chunk 4 groups (gate/up)
    planes, ameta = pa.prep_activations(codes, ea, **pa.EXPERT_DOWN_PREP)      # 3 planes, fold chunk 4 groups (down; same kernel)
    planes, ameta = pa.prep_activations(codes, ea, **pa.EXPERT_DOWN4_PREP)     # 4 planes, fold chunk 4 groups (the knob)
The 8-group fold chunk (fold_groups=8) is modelled and gated but not instantiated (Codex's r3 build: 8 bytes of stack).
r3-P3: `rows` (an int32 device scalar, or None) is the live row count of a static workspace: prep_activations skips the
rows at or beyond it (their planes and ameta untouched) and the ungrouped dense / expert launch computes only those rows
(its grid is still sized by max_rows / M; CTAs beyond the count exit before any barrier). Live rows produce the same
bytes as without the bound. A grouped launch is bounded by row_off and takes no rows.
    y = pa.expert(planes, ameta, ea, w_nibbles [E, N, K/2], ew [E, N, K/32], EW [E, N], row_off [E+1] int32, max_rows)
with the token rows sorted by expert (rows [row_off[e], row_off[e+1]) belong to expert e); E = 1 and row_off=None for a
single expert over all rows. The expert kernels take N % 64 == 0 (the dense kernel N % 32 == 0) and K % (32 fold_groups).

Contract (CONTRACT_PINS_r1.md, CONTRACT_FIXES_r1.md): row base B = EA + EW - MARGIN with MARGIN_DENSE = 13 (dense,
WP = 3) and MARGIN_EXPERT = 20 (experts, contract v2); the int64 row sums cannot overflow on a legal input. Finish
rule: f = __ll2float_rn(A) (one direct rounding, never via double); y = f * 2^(B - SA - SW) formed on the exponent
field; A == 0 gives +0; a biased exponent above 254 gives a signed infinity and flag bit 4, below 1 a signed zero and
flag bit 4; then the integer RNE to BF16 (bit-identical to __float2bfloat16_rn). Neither flagged case is reachable
with the contract quantiser (amax floor 1e-4, ea >= -22) and weight scale codes >= 54 (dense) / 53 (expert), which
weight_exponents_dense / weight_exponents_expert enforce. Only the contract quantiser (1e-4 floor) may produce the
activation codes and exponents fed to prep_activations; the serving SGLang quantiser's 1e-10 floor is not it.
"""
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT = None

# contract constants (patha_sequence.h)
MARGIN_DENSE = 13
MARGIN_EXPERT = 20
MIN_SCALE_CODE_DENSE = 54      # finish-rule admission (M1): weight exponent >= -73
MIN_SCALE_CODE_EXPERT = 53     # weight exponent >= -74
SA = 9
SW_DENSE = 9
SW_EXPERT = 1
STAGE_K = 128                  # one pipeline stage (4 groups)
CHUNK_K = STAGE_K              # kept for callers of r2 (the dense fold chunk)
BM = 64
BN_DENSE = 32                  # dense CTA tile 64 x 32 (r2 geometry)
BN_EXPERT = 64                 # r3-P expert CTA tile 64 x 64, eight warps
BN = BN_DENSE
# activation preparation per projection: planes (3 or 4) and the fold chunk in groups (4 or 8)
DENSE_PREP = dict(planes=3, fold_groups=4)
EXPERT_GATE_UP_PREP = dict(planes=3, fold_groups=4)       # r3-P2: F = 1, the configuration Codex built clean
EXPERT_DOWN_PREP = dict(planes=3, fold_groups=4)
EXPERT_DOWN4_PREP = dict(planes=4, fold_groups=4)       # the data-independent knob (budget 10 bits, +33 % MMAs)
ERR_BITS = {
    1: "int64 accumulator overflow",
    2: "negative chunk shift on the fold path",
    4: "finish outside the normal FP32 range",
    8: "NaN E4M3 activation code",
    16: "NaN E4M3 weight code",
    32: "top digit out of range",
    64: "shifted expert weight outside int8 (unreachable: 12 << 3 = 96; retained bit)",
    128: "exponent metadata inconsistent (a staged ea_g above EA or ew_g above EW, or a per-group term shift above the margin)",
}
DENSE_SHAPES = {(1792, 5120), (32768, 1280), (5120, 8192), (4608, 5120), (5120, 2304)}
EXPERT_SHAPES = {(4608, 5120), (5120, 2304)}


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        name = os.environ.get("PATHA_EXT_NAME", "patha_prefill_sm89_r3p")
        cuda_cflags = ["-O3", "-gencode=arch=compute_89,code=sm_89", "-fmad=false", "--expt-relaxed-constexpr",
                       "-Xptxas", "-v"]
        _EXT = load(
            name=name,
            sources=[str(_HERE / "patha_prefill.cu")],
            extra_include_paths=[str(_HERE)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=cuda_cflags,
            verbose=os.environ.get("PATHA_BUILD_VERBOSE", "0") == "1",
        )
    return _EXT


# ----------------------------------------------------------------------------- exponent helpers (host side, exact)

def _pow2_exponent(s):
    """float32 tensor of exact powers of two -> int8 exponents; raises on anything else (zero, sign, inf, nan, odd)."""
    assert s.dtype == torch.float32, "scales must be float32"
    mant, exp = torch.frexp(s)
    if not bool(torch.all(mant == 0.5)):
        raise ValueError("patha: scales must be exact powers of two")
    e = exp.to(torch.int32) - 1
    if int(e.min()) < -128 or int(e.max()) > 127:
        raise ValueError("patha: exponent outside int8")
    return e.to(torch.int8)


def activation_exponents(As):
    """As float32 [M, K/32] (UE8M0 values) -> ea int8 [M, K/32]."""
    return _pow2_exponent(As).contiguous()


def pow2_exponents(scales):
    """Any float32 tensor of exact powers of two (for example the streamed experts' per-row-per-group scales as the
    slot's float32 view holds them) -> int8 exponents of the same shape."""
    return _pow2_exponent(scales).contiguous()


def weight_exponents_dense(Bs):
    """Bs float32 [N/32, K/32] (32x32 block scales) -> ew int8 [N, K/32] (one row per output row). Rejects scale
    codes below MIN_SCALE_CODE_DENSE (exponents below -73): the finish rule's admission bound."""
    e = _pow2_exponent(Bs)
    if int(e.min()) < MIN_SCALE_CODE_DENSE - 127:
        raise ValueError("patha: dense scale code below %d would make an out-of-range finish reachable" % MIN_SCALE_CODE_DENSE)
    return e.repeat_interleave(32, dim=0).contiguous()


def weight_exponents_expert(ws):
    """E8M0 scale codes uint8 [..., N, K/32] -> ew int8 (code - 127); code 255 (NaN) and codes below
    MIN_SCALE_CODE_EXPERT (the finish rule's admission bound) are rejected."""
    assert ws.dtype == torch.uint8
    if bool((ws == 255).any()):
        raise ValueError("patha: E8M0 NaN scale")
    if bool((ws < MIN_SCALE_CODE_EXPERT).any()):
        raise ValueError("patha: expert scale code below %d would make an out-of-range finish reachable" % MIN_SCALE_CODE_EXPERT)
    return (ws.to(torch.int16) - 127).to(torch.int8).contiguous()


def row_max(ew):
    """EW: per-row maximum of the group exponents (int8), the oracle's EW / EA."""
    return ew.amax(dim=-1).contiguous()


# ----------------------------------------------------------------------------- kernels

def prep_activations(codes, ea, planes=3, fold_groups=4, rows=None, out=None):
    """codes uint8 [M, K] (E4M3 bit patterns; use A.view(torch.uint8) on a float8_e4m3fn tensor), ea int8 [M, K/32]
    -> planes int8 [planes, M, K], ameta int8 [K / (32 fold_groups), M, 4]. planes 3 (fold budget 3 bits) or 4
    (10 bits); fold_groups 4 (one stage; 8 is a model-only knob). The GEMM configuration follows these two numbers.
    rows: an int32 device scalar (or None) with the live row count; rows at or beyond it are skipped and their bytes
    left untouched (r3-P3). out: optional (planes, ameta) buffers of the right shapes to write into (a workspace)."""
    ext = _load()
    M, K = codes.shape
    assert planes in (3, 4) and fold_groups == 4, "planes 3 or 4; fold chunk 4 groups (8 is a model-only knob in r3-P2)"
    ck = 32 * fold_groups
    assert K % ck == 0, "K must be a multiple of the fold chunk (%d)" % ck
    if out is None:
        pl = torch.empty((planes, M, K), dtype=torch.int8, device=codes.device)
        ameta = torch.empty((K // ck, M, 4), dtype=torch.int8, device=codes.device)
    else:
        pl, ameta = out
    ext.prep_activations(codes.contiguous(), ea.contiguous(), pl, ameta, rows)
    return pl, ameta


def _outputs(M, N, device, mode, out):
    if mode == "bf16":
        o = out if out is not None else torch.empty((M, N), dtype=torch.bfloat16, device=device)
        return o, None, o
    if mode == "i64":
        o = out if out is not None else torch.empty((M, N), dtype=torch.int64, device=device)
        return None, o, o
    if mode == "both":
        ob = torch.empty((M, N), dtype=torch.bfloat16, device=device)
        oi = torch.empty((M, N), dtype=torch.int64, device=device)
        return ob, oi, (ob, oi)
    raise ValueError(mode)


def dense(planes, ameta, ea, wcodes, ew, EW, mode="bf16", out=None, rows=None):
    """Dense FP8 projection: wcodes uint8 [N, K] (B.view(torch.uint8)), ew int8 [N, K/32], EW int8 [N]. The operand
    must come from prep_activations(..., **DENSE_PREP). rows: optional int32 device scalar, the live row count
    (rows beyond it are neither read nor stored; r3-P3)."""
    ext = _load()
    M = planes.shape[1]
    N = wcodes.shape[0]
    ob, oi, ret = _outputs(M, N, planes.device, mode, out)
    ext.gemm(planes, ameta, ea, wcodes, ew, EW, None, M, ob, oi, False, rows)
    return ret


def expert(planes, ameta, ea, wnib, ew, EW, row_off=None, max_rows=None, mode="bf16", out=None, rows=None):
    """Routed experts: wnib uint8 [E, N, K/2] (adjacent nibbles, element 2b in the low nibble of byte b), ew int8
    [E, N, K/32], EW int8 [E, N]; row_off int32 [E+1] on the device (token rows sorted by expert), max_rows = the
    largest row count of any expert (host int, sizes the grid). The configuration (gate/up and down: 3 planes, 4
    groups, one kernel; the 4-plane knob) follows the shapes of planes and ameta. rows: optional int32 device scalar
    with the live row count of an ungrouped launch (row_off None); a grouped launch is bounded by row_off (r3-P3)."""
    ext = _load()
    M = planes.shape[1]
    N = wnib.shape[-2]
    if row_off is None:
        max_rows = M
    elif max_rows is None:
        raise ValueError("patha: max_rows is required with row_off (host-side, sizes the grid)")
    if row_off is not None and rows is not None:
        raise ValueError("patha: rows applies to the ungrouped launch; a grouped launch is bounded by row_off")
    ob, oi, ret = _outputs(M, N, planes.device, mode, out)
    ext.gemm(planes, ameta, ea, wnib, ew, EW, row_off, int(max_rows), ob, oi, True, rows)
    return ret


def error_flag(reset=False):
    return int(_load().error_flag(bool(reset)))


def describe_flag(v):
    return [name for bit, name in ERR_BITS.items() if v & bit]


def smem_bytes(expert_mode=False, planes=3, fold_groups=4):
    """Dynamic shared memory per CTA: dense 83,712; expert gate/up and down (3, 4) 76,288; the down knob (4, 4)
    92,672; the uninstantiated 8-group knob (3, 8) 76,800 (all under the 101,376 B opt-in limit of sm_89)."""
    return int(_load().smem_bytes(bool(expert_mode), int(planes), int(fold_groups)))


def eligible(M, N, K, expert_mode=False, fold_groups=4):
    """Dispatcher admission: the verified shapes, K % (32 fold_groups) == 0, N % 64 (expert) / 32 (dense) == 0,
    M above the decode threshold."""
    shapes = EXPERT_SHAPES if expert_mode else DENSE_SHAPES
    bn = BN_EXPERT if expert_mode else BN_DENSE
    return M >= 1 and (N, K) in shapes and K % (32 * fold_groups) == 0 and N % bn == 0
