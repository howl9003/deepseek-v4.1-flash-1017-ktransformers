"""Contract v2 (exact E4M3 x E2M1) routed experts for the DeepSeek-V4.1-Flash-1017 hybrid stack.

Default off. KT_CONTRACT_V2=1 routes the KT EP wrapper's MoE through `adapter.apply`: CPU experts by the
kt-kernel contract backend (KT_MXFP4_BACKEND=contract_v2, per-route output), GPU-resident, hot-lend and
streamed experts by the contract Triton kernels, one FP32 combine in ascending expert id.
"""
import os

ENABLED = os.environ.get('KT_CONTRACT_V2') == '1'


def finalize_output(value):
    """Round a completed contract sum once, at the BF16 model boundary.

    Routed partials, shared additions and collectives remain FP32. Call only
    after the last required addition/reduction, never on an unfinished partial.
    The disabled path is an identity, including object identity and dtype.
    """
    if not ENABLED:
        return value
    import torch

    if value.dtype == torch.bfloat16:
        return value
    if value.dtype != torch.float32:
        raise TypeError("contract v2 expects a completed FP32 or BF16 MoE output")
    return value.to(torch.bfloat16)
