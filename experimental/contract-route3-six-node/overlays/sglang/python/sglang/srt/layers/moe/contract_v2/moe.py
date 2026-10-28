"""Contract v2 routed experts on the GPU, per route, and the contract combine.

expert_routes(x, w13, s13, w2, s2, ids, weights, *, out=None) -> bf16 [T, K, H]
    For every (token, slot) whose expert id is in [0, E) and whose weight is not zero, the expert's output
    under the contract (E4M3 digits, exact int64 projections, BF16 gate/up, table SiLU x clamp(up) x routing
    weight, BF16, E4M3, exact down, one RNE). The routing weight is already applied. Weights as stored:
    w13 [E, 2I, H/2] and w2 [E, H, I/2] uint8 (two E2M1 values a byte, low nibble first along K), scales
    [E, 2I, H/32] and [E, H, I/32] uint8 E8M0. Slots with an id outside [0, E) or a zero weight are untouched,
    so several placements (resident GPU experts, hot-lend shadow, streamed prefill, CPU) can fill one buffer.
    Bit-identical to the CPU contract backend's per-route output (contract-v2-r1 GPU validation on SM89).

combine_routes(routes, ids) -> fp32 [T, H]
    The contract's combine: the routes of a token summed in ascending expert id in FP32, exact zero for
    skipped slots. The caller adds the shared expert in FP32 and casts once (the model's `shared.add_(routed)`
    does exactly that when `routed` is FP32).

contract_v2_moe(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, ...) -> bf16 [M, H]
    Drop-in for callers that expect the BF16 [M, H] result of a fused MoE (one extra BF16 rounding).

Kernel mode: KT_CONTRACT_V2_GPU_MODE = int8 (default; three signed 6-bit planes through int8 tensor cores,
exact on SM75+) or gemv (int32 CUDA cores, any SM). fp8hl is exact only on SM100/SM120 and is refused here.
"""
import os
import torch
from .kernels import GPU_V2

_adapters = {}
_FAST = os.environ.get('KT_CONTRACT_V2_FAST_ADAPTER', '0') == '1'
_STATIC_GPU = os.environ.get('KT_CONTRACT_V2_STATIC_GPU', '0') == '1'
_DP4A = os.environ.get('KT_CONTRACT_V2_GPU_DP4A', '0') == '1'
_DECODE_SPLITS = int(os.environ.get('KT_CONTRACT_V2_DECODE_SPLITK', '0'))
if _DECODE_SPLITS not in (0, 1, 2, 4, 8):
    raise ValueError('KT_CONTRACT_V2_DECODE_SPLITK must be 0, 1, 2, 4 or 8')

def _adapter(device):
    key = (device.type, device.index)
    if key not in _adapters:
        mode = os.environ.get('KT_CONTRACT_V2_GPU_MODE', 'int8').strip().lower()
        if mode not in ('int8', 'gemv'):
            raise RuntimeError(f'KT_CONTRACT_V2_GPU_MODE={mode!r}: only int8 and gemv are exact on this hardware class')
        _adapters[key] = GPU_V2(enabled=True, device=device, mode=mode, fast_checks=_FAST)
    return _adapters[key]


def _as_bytes(t, what):
    """The live layer keeps the packed weights as signed int8 and the scales as float8_e8m0fnu; the saved
    component payloads are uint8. Both are the same bytes: view, never convert."""
    if t.dtype == torch.uint8: return t
    if t.dtype == torch.int8 or t.element_size() == 1: return t.view(torch.uint8)
    raise RuntimeError(f'expert_routes: {what} must be one byte per value (uint8, int8 or float8_e8m0fnu), got {t.dtype}')


def expert_routes(x, w13, s13, w2, s2, ids, weights, *, out=None, **_ignored):
    if _STATIC_GPU:
        # Opt-in exact packed-integer decode. Rows beyond four retain the
        # tensor-core route: the component measured DP4A slower at eight rows.
        if (_DP4A and 1 <= x.shape[0] <= 4 and x.shape[1] == 5120
                and w13.shape[1] == 4608 and 1 <= ids.shape[1] <= 6):
            from .dp4a_routes_candidate import expert_routes_splitk
            return expert_routes_splitk(x, w13, s13, w2, s2, ids, weights,
                                        out=out, tile=16, splits=2)
        # Restrict the opt-in to the model geometry and decode sizes whose raw
        # int64 sums and graph/DMA behavior were checked. No rounding changes.
        if (_DECODE_SPLITS and 1 <= x.shape[0] <= 8 and x.shape[1] == 5120
                and w13.shape[1] == 4608 and 1 <= ids.shape[1] <= 6):
            from .decode_splitk_candidate import expert_routes_splitk
            return expert_routes_splitk(x, w13, s13, w2, s2, ids, weights,
                                        out=out, tile=16, splits=_DECODE_SPLITS)
        from .static_routes_candidate import expert_routes_static
        return expert_routes_static(x, w13, s13, w2, s2, ids, weights, out=out, tile=16 if x.shape[0] < 128 else 64)
    if x.dtype != torch.bfloat16 or x.ndim != 2: raise RuntimeError('expert_routes: x must be BF16 [T, H]')
    if ids.ndim != 2 or weights.shape != ids.shape or weights.dtype != torch.float32: raise RuntimeError('expert_routes: ids [T, K] int64 and weights [T, K] fp32')
    w13, s13, w2, s2 = _as_bytes(w13, 'w13'), _as_bytes(s13, 'w13 scales'), _as_bytes(w2, 'w2'), _as_bytes(s2, 'w2 scales')
    T, K = ids.shape; H = x.shape[1]; E, twoI, Hh = w13.shape
    if Hh * 2 != H or s13.shape != (E, twoI, H // 32) or w2.shape[:2] != (E, H) or s2.shape != (E, H, w2.shape[2] * 2 // 32):
        raise RuntimeError('expert_routes: weight shapes do not match the contract layout')
    if out is None: out = torch.zeros((T, K, H), dtype=torch.bfloat16, device=x.device)
    elif out.shape != (T, K, H) or out.dtype != torch.bfloat16 or out.device != x.device or not out.is_contiguous():
        raise RuntimeError('expert_routes: out must be a contiguous BF16 [T, K, H] on the device of x')
    gpu = _adapter(x.device)
    ids = ids.to(torch.int64)
    valid = (ids >= 0) & (ids < E) & (weights != 0)
    if not bool(valid.any()): return out
    xf = x.float()   # exact: BF16 values
    for e in torch.unique(ids[valid]).tolist():
        tok, slot = (valid & (ids == e)).nonzero(as_tuple=True)
        y = gpu.expert(xf[tok].contiguous(), w13[e], s13[e], w2[e], s2[e], weights[tok, slot].contiguous().float())
        out[tok, slot] = y.to(torch.bfloat16)   # exact: y holds BF16 values in FP32
    return out


def combine_routes(routes, ids):
    if routes.ndim != 3 or routes.dtype != torch.bfloat16 or ids.shape != routes.shape[:2]: raise RuntimeError('combine_routes: routes BF16 [T, K, H] and ids [T, K]')
    if _FAST:
        from .fast_combine import combine
        return combine(routes, ids)
    T, K, H = routes.shape
    ids = ids.to(torch.int64)
    valid = ids >= 0
    order = torch.argsort(torch.where(valid, ids, torch.full_like(ids, torch.iinfo(torch.int64).max)), dim=1, stable=True)
    rows = torch.arange(T, device=routes.device)
    acc = torch.zeros((T, H), dtype=torch.float32, device=routes.device)
    for j in range(K):   # ascending expert id, one FP32 add per slot, skipped slots leave acc untouched (signed zero kept)
        slot = order[:, j]
        term = routes[rows, slot].float()
        acc = torch.where(valid[rows, slot][:, None], acc + term, acc)
    return acc


def contract_v2_moe(hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, swiglu_limit=None, out=None, **_ignored):
    routes = expert_routes(hidden_states, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights.float())
    y = combine_routes(routes, topk_ids).to(torch.bfloat16)
    if out is not None: out.copy_(y); return out
    return y
