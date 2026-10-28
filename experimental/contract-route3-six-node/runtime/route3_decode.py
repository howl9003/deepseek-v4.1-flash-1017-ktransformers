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
        import hashlib
        import importlib.util
        import json
        import sys
        assert not l2_policy_requested(), "This candidate selects the default binary only"
        binary = _HERE / 'route3_phase2_l2p0_r1.so'
        expected = '51c179a96c838f2b51d9a01707706534a9472fe50752bedbab7c659905dca649'
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest(binary) == expected
        spec = importlib.util.spec_from_file_location('route3_phase2_l2p0_r1', binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
        assert Path(module.__file__).resolve() == binary.resolve() and digest(binary) == expected
        receipt = _HERE.parent / 'route3-binary-origins' / (str(os.getpid()) + '.json')
        with receipt.open('x') as f:
            json.dump(dict(pid=os.getpid(), binary=str(binary.resolve()), binary_sha256=expected,
                           loader='direct_pinned_binary_no_compiler', policy=0), f, indent=2)
        _EXT = module
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
