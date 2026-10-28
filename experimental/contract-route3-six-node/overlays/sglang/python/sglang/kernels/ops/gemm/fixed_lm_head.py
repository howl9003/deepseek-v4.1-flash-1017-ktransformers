"""Uninstalled candidate: fixed head arithmetic with bounded split scratch.

Uses the already-tested fixed projection order. Chunking changes scheduling,
not the order within an output row. The normal runtime does not import this
module; a separate default-off integration would require causal/model checks.
"""
import torch
from sglang.kernels.ops.gemm.fixed_projection import linear

CHUNK_ROWS=16

def _fixed_fp32_head(hidden,weight,*,fp32_output=False):
    assert hidden.is_cuda and weight.is_cuda and hidden.device==weight.device
    assert hidden.dtype==weight.dtype==torch.bfloat16
    assert hidden.ndim==weight.ndim==2 and hidden.shape[1]==weight.shape[1]
    assert hidden.is_contiguous() and weight.is_contiguous()
    m=hidden.shape[0];n=weight.shape[0]
    dtype=torch.float32 if fp32_output else torch.bfloat16
    if m<=CHUNK_ROWS:
        return linear(hidden,weight).to(dtype)
    output=torch.empty((m,n),dtype=dtype,device=hidden.device)
    for start in range(0,m,CHUNK_ROWS):
        # copy_ performs the same final BF16 rounding as to(bfloat16), or
        # copies FP32 unchanged. At most16 rows use split-K scratch at once.
        output[start:start+CHUNK_ROWS].copy_(linear(hidden[start:start+CHUNK_ROWS],weight))
    return output


"""Private fixed-shape cuBLAS head experiments; not serving-integrated."""
import torch


def padded_bf16_head(hidden, weight, *, tile=16):
    """Use the same BF16 cuBLAS matrix shape for every row, including tails.

    Whether cuBLAS is invariant to row position/capture must be tested; this
    function deliberately makes no guarantee based on shape alone.
    """
    assert hidden.is_cuda and weight.is_cuda and hidden.device == weight.device
    assert hidden.dtype == weight.dtype == torch.bfloat16
    assert hidden.ndim == weight.ndim == 2 and hidden.shape[1] == weight.shape[1]
    assert hidden.is_contiguous() and weight.is_contiguous()
    assert tile in (1, 8, 16, 32, 64)
    m, k = hidden.shape
    result = torch.empty((m, weight.shape[0]), dtype=hidden.dtype, device=hidden.device)
    if not m:
        return result
    padded = torch.empty((tile, k), dtype=hidden.dtype, device=hidden.device)
    product = torch.empty((tile, weight.shape[0]), dtype=hidden.dtype, device=hidden.device)
    for start in range(0, m, tile):
        rows = min(tile, m - start)
        padded.zero_()
        padded[:rows].copy_(hidden[start:start + rows])
        torch.mm(padded, weight.t(), out=product)
        result[start:start + rows].copy_(product[:rows])
    return result


def fixed_lm_head(hidden, weight, *, fp32_output=False):
    """Default-off head: fixed 64-row BF16 cuBLAS calls, bounded tail padding.

    BF16 calls preserve the measured large-batch cuBLAS arithmetic. Explicit
    FP32 output keeps the previously checked fixed-projection implementation;
    it must not be silently produced by rounding BF16 logits back to FP32.
    """
    if fp32_output:
        return _fixed_fp32_head(hidden, weight, fp32_output=True)
    return padded_bf16_head(hidden, weight, tile=64)
