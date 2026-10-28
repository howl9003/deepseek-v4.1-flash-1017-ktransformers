"""Route 3 phase 2: loader and wrapper for the CUDA decode kernel (M <= 16), default-off.

Builds route3_decode.cu with torch's extension loader using the active environment's nvcc (the trial venv on fea01,
CUDA 12.x, sm_89). No fast-math, no fmad contraction: the kernel's epilogue uses __fmul_rn / __fmaf_rn explicitly.

Usage (the dispatcher side is described in INTEGRATION.md):

    from route3_decode import route3_partials, error_flag
    P = route3_partials(A, As, B, Bs)            # [4, M, N] float32, the kernel's split partials
    _reduce_block_fp8_split_k[(cdiv(M*N, 256),)](P, C, M*N, 4, 256)   # the UNCHANGED Triton reduce
    assert error_flag(reset=True) == 0            # between runs, never inside a graph capture

Build variants: ROUTE3_L2_POLICY=1 in the environment adds -DROUTE3_L2_POLICY=1 (weight-stream L2 evict_first cache
hint, see ROUTE3_PHASE2_REVIEW_r2.md) and uses the extension name route3_decode_sm89_l2p1; unset, the build and its
name are the r1 ones.
"""
import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_EXT = None


def l2_policy_requested():
    """ROUTE3_L2_POLICY=1 builds the kernel with the weight-stream L2 evict_first cache hint (default off: the r1 binary)."""
    return os.environ.get("ROUTE3_L2_POLICY", "0") == "1"


def _load():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        l2p = l2_policy_requested()
        name = os.environ.get("ROUTE3_EXT_NAME", "route3_decode_sm89" + ("_l2p1" if l2p else ""))
        cuda_cflags = ["-O3", "-gencode=arch=compute_89,code=sm_89", "-fmad=false", "--expt-relaxed-constexpr"]
        if l2p:
            cuda_cflags.append("-DROUTE3_L2_POLICY=1")   # weight-stream L2 evict_first hint; same bytes, same arithmetic
        _EXT = load(
            name=name,
            sources=[str(_HERE / "route3_decode.cu")],
            extra_include_paths=[str(_HERE)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=cuda_cflags,
            verbose=os.environ.get("ROUTE3_BUILD_VERBOSE", "0") == "1",
        )
    return _EXT


SPLIT_K = 4
MAX_ROWS = 16
SHAPES = {(1792, 5120), (4608, 5120), (5120, 2304), (5120, 8192), (32768, 1280), (4096, 1280)}


def eligible(M, N, K, As, Bs):
    """The dispatcher's admission test: decode rows, the six verified shapes, float32 power-of-two scales."""
    return (1 <= M <= MAX_ROWS and (N, K) in SHAPES and K % 32 == 0 and N % 32 == 0
            and As.dtype == torch.float32 and Bs.dtype == torch.float32)


def route3_partials(A, As, B, Bs, out=None):
    """A [M,K] e4m3, As [M,K/32] fp32, B [N,K] e4m3, Bs [N/32,K/32] fp32 -> P [4,M,N] fp32 (split partials)."""
    ext = _load()
    M, K = A.shape
    N = B.shape[0]
    if out is None:
        out = torch.empty((SPLIT_K, M, N), dtype=torch.float32, device=A.device)
    ext.decode_fp8(A.contiguous(), As.contiguous(), B.contiguous(), Bs.contiguous(), out)
    return out


def error_flag(reset=False):
    return int(_load().error_flag(bool(reset)))


def smem_bytes():
    return int(_load().smem_bytes())
