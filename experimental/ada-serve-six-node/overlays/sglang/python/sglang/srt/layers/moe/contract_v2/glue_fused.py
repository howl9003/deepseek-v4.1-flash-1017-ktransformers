"""Fused route glue for the contract v2 adapter, default off (KT_CONTRACT_V2_GLUE_FUSED=1, Claude, 31 Oct 2026).

Captured decode only. adapter.apply launches about 50 small torch kernels a layer around the real work: the ownership
masks before the CPU handoff, the active ids of the two GPU passes, the CPU/GPU select and two asserts after it. Here
one route kernel computes all of that, and the combine reads each route from the half of `both` that owns it. Same
integers, same FP32 adds in the same order.

Checks: a value check a layer used to assert goes to the static expert path's sticky flag word, which the worker clones
after every forward and the result processor reads before any output leaves (the graph-mode rule accepted 20 Oct).
Eager calls take the adapter's own path and its immediate asserts; they only warm these kernels.
"""
import logging
import torch
import triton as tr
import triton.language as tl

logger = logging.getLogger(__name__)

FLAG_OWNERS, FLAG_WEIGHT = 64, 128
OWN_NONE, OWN_GPU, OWN_CPU = 0, 1, 2
_both = {}
_ready = set()
_logged = set()


@tr.jit
def _route_glue(ids_ptr, w_ptr, tbl_ptr, rtbl_ptr, cnt_ptr, cid_ptr, cw_ptr, rid_ptr, sid_ptr, own_ptr, flag_ptr, n, nres, nslot,
                BLOCK: tl.constexpr):
    """kt_kv_hot_lend._route_kernel with the resident rows' table, plus what adapter.apply derives from its outputs:
    rid and sid are the two passes' active ids, own says who serves each route."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    e = tl.load(ids_ptr + offs, mask=m, other=-1).to(tl.int64)
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    valid = m & (e >= 0)
    ec = tl.where(valid, e, 0)
    tl.atomic_add(cnt_ptr + ec, tl.where(valid, 1.0, 0.0), mask=valid)
    s = tl.load(tbl_ptr + ec, mask=valid, other=-1)
    r = tl.load(rtbl_ptr + ec, mask=valid, other=-1)
    res = valid & (r >= 0)
    lent = valid & (s >= 0) & (r < 0)
    gpu = lent | res
    live = w != 0
    rok = res & (r < nres) & live
    sok = lent & (s < nslot) & live
    tl.store(cid_ptr + offs, tl.where(gpu, -1, e).to(cid_ptr.dtype.element_ty), mask=m)
    tl.store(cw_ptr + offs, tl.where(gpu, 0.0, w).to(cw_ptr.dtype.element_ty), mask=m)
    tl.store(rid_ptr + offs, tl.where(rok, r, -1).to(rid_ptr.dtype.element_ty), mask=m)
    tl.store(sid_ptr + offs, tl.where(sok, s, -1).to(sid_ptr.dtype.element_ty), mask=m)
    cpu = valid & live & tl.where(gpu, False, True)
    tl.store(own_ptr + offs, tl.where(cpu, 2, tl.where(rok | sok, 1, 0)).to(tl.uint8), mask=m)   # OWN_CPU, OWN_GPU, OWN_NONE
    # owners == expected fails only here: cpu, rok and sok exclude each other and each needs valid & live
    lost = tl.sum((valid & live & gpu & tl.where(rok | sok, False, True)).to(tl.int32), 0)
    if lost != 0:
        tl.atomic_or(flag_ptr, 64)   # FLAG_OWNERS
    wild = tl.sum((m & ((w != w) | (tl.abs(w) == float('inf')))).to(tl.int32), 0)
    if wild != 0:
        tl.atomic_or(flag_ptr, 128)   # FLAG_WEIGHT


@tr.jit
def _combine_owned(R, I, OWN, O, Y, COUNT, K: tl.constexpr, H: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                   BK: tl.constexpr, BH: tl.constexpr, NE: tl.constexpr):
    # fast_combine._combine_ordered, except for where a route's term comes from
    row = tl.program_id(0); col = tl.program_id(1) * BH + tl.arange(0, BH)
    slot = tl.arange(0, BK)
    ids = tl.load(I + row * S0 + slot * S1, slot < K, -1).to(tl.int64)
    sentinel = tl.full((), 0x7fffffffffffffff, tl.int64)
    order_ids = tl.load(O + ids, (slot < K) & (ids >= 0) & (ids < NE), -1).to(tl.int64)
    keys = tl.where((slot < K) & (ids >= 0), order_ids * BK + slot, sentinel)
    acc = tl.full((BH,), 0, tl.float32); previous = tl.full((), -1, tl.int64)
    for _ in range(K):
        key = tl.min(tl.where(keys > previous, keys, sentinel), 0)
        valid = key != sentinel
        chosen = tl.where(valid, key % BK, 0)
        route = row * K + chosen
        own = tl.load(OWN + route, valid, 0).to(tl.int32)
        term = tl.load(R + (route + tl.where(own == 2, COUNT, 0)) * H + col, valid & (own != 0) & (col < H), 0).to(tl.float32)
        acc = tl.where(valid, acc + term, acc)
        previous = key
    tl.store(Y + row * H + col, acc, col < H)


def route(hot, L, ids, w, nres, nslot, flag):
    cid, cw, rid, own = torch.empty_like(ids), torch.empty_like(w), torch.empty_like(ids), torch.empty(ids.shape, dtype=torch.uint8, device=ids.device)
    sid = torch.empty(ids.shape, dtype=torch.int64, device=ids.device) if hot._MASK_ZERO else torch.empty_like(ids)
    _route_glue[(tr.cdiv(ids.numel(), 128),)](ids, w, hot.STATE['tbl'][L.idx], hot.STATE['rtbl'][L.idx], hot.STATE['cnt'][L.idx],
                                              cid, cw, rid, sid, own, flag, ids.numel(), nres, nslot, BLOCK=128)
    return cid, cw, rid, sid, own


def combine(both, ids, own, order):
    """both bf16 [2 count, H]: the GPU's routes, then the CPU's."""
    t, k = ids.shape; h = both.shape[1]
    out = torch.empty((t, h), dtype=torch.float32, device=both.device)
    _combine_owned[(t, tr.cdiv(h, 256))](both, ids, own, order, out, t * k, k, h, *ids.stride(), tr.next_power_of_2(k), 256, 384, enable_fp_fusion=False)
    return out


def expert_pair(pa, x, resident, shadow, rid, sid, w, routes):
    """expert_routes_patha for the resident rows, then for the slots: the same launches on one workspace, with the active
    ids from the route kernel and x quantised once (the second quantisation wrote the same codes over themselves)."""
    t, hidden = x.shape; topk = rid.shape[1]; count = t * topk
    moe, activate = pa._overlay()
    silu = moe._adapter(x.device).silu
    ws = pa.workspace(x.device, t, topk)
    out = routes.view(count, hidden)
    pa.quant_codes(x, ws.codes_x, ws.ea_x)
    for payload, active in ((resident, rid), (shadow, sid)):
        w13, s13, w2, s2 = [pa._as_bytes(p, name) for p, name in zip(payload, ('w13', 's13', 'w2', 's2'))]
        pa._EXT.expert_static(ws.codes_x, ws.ea_x, w13, s13, active, ws.gu, ws.flag, True, count, topk, t)
        activate[(count, tr.cdiv(pa.INTER, 128))](ws.gu, active, w, silu, ws.h, pa.INTER, 128, enable_fp_fusion=False)
        pa.quant_codes(ws.h, ws.codes_h, ws.ea_h)
        pa._EXT.expert_static(ws.codes_h, ws.ea_h, w2, s2, active, out, ws.flag, False, count, topk, t)


def _buffer(native, x, k):
    # one per slot, as adapter._route_buffers: graphs retain the address
    from kt_kernel.experts_base import KExpertsCPUBuffer
    key = (x.device, x.shape[0], x.shape[1], k, native.layer_idx % KExpertsCPUBuffer.buffer_depth)
    if key not in _both:
        _both[key] = torch.empty((2 * x.shape[0] * k, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    return _both[key]


def _fits(wrapper, layer, hot, moe, pa, L, x, ids, order):
    return (L is not None and L.shadow is not None and order is not None and moe._FAST
            and hot.DYNAMIC and hot.DYN_RES and hot._route_kernel is not None and not hot.ROUTE_TORCH and not hot.CHECK
            and not hot._in_window() and wrapper.num_gpu_experts > 0 and layer.w13_weight.shape[0] == wrapper.num_gpu_experts
            and pa.eligible(x.shape[0], x.shape[1], layer.w13_weight.shape[1], ids.shape[1]))


def _warm(wrapper, hot, static, pa, L, x, ids, w, order, nres, nslot):
    """JIT both kernels outside the capture, on no routes: nothing is counted and no flag can rise."""
    key = (ids.shape[0], ids.shape[1], ids.dtype)
    if key in _ready:
        return
    static.warmup(x.device)
    pa.FLAG_NAMES.update({FLAG_OWNERS: 'route ownership not exactly once', FLAG_WEIGHT: 'non-finite MoE weights'})
    blank = torch.full_like(ids, -1)
    own = route(hot, L, blank, torch.zeros_like(w), nres, nslot, static._flag)[4]
    combine(_buffer(wrapper.wrapper, x, ids.shape[1]), blank, own, order)
    _ready.add(key)


def apply(wrapper, layer, dispatch, order):
    """The adapter's body for a captured decode layer, or None: the adapter's own path serves it."""
    from sglang.srt.layers.moe import kt_kv_hot_lend as hot, kt_ep_wrapper
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
    from . import adapter, moe, patha_static_runtime as static
    if not (moe._STATIC_GPU and moe._PATHA_STATIC):
        return None
    pa = static._module()
    x = dispatch.hidden_states
    ids, w = dispatch.topk_output.topk_ids.contiguous(), dispatch.topk_output.topk_weights.contiguous()
    L = wrapper.__dict__.get('_kt_kv_hot')
    if not _fits(wrapper, layer, hot, moe, pa, L, x, ids, order):
        return None
    nres, nslot = int(wrapper.num_gpu_experts), int(L.shadow.w13_weight.shape[0])
    if not torch.cuda.is_current_stream_capturing():
        _warm(wrapper, hot, static, pa, L, x, ids, w, order, nres, nslot)
        return None
    if ids.dtype not in (torch.int32, torch.int64) or w.dtype != torch.float32:
        raise ValueError('fused route glue requires int32 or int64 IDs and float32 weights')
    if static._flag is None:
        raise RuntimeError('fused route glue needs the flag word from eager warm-up')
    native = wrapper.wrapper
    t, k = ids.shape; h = x.shape[1]
    both = _buffer(native, x, k)
    routes, cpu = both[:t * k].view(t, k, h), both[t * k:].view(t, k, h)
    cid, cw, rid, sid, own = route(hot, L, ids, w, nres, nslot, static._flag)
    adapter.submit_routes(native, x, cid, cw, kt_ep_wrapper.pack_to_host if getattr(kt_ep_wrapper, '_KT_HANDOFF_PACK', False) else None)
    if getattr(kt_ep_wrapper, '_KT_SHARED_AFTER_SUBMIT', False):
        wrapper._kt_post_submit_event = torch.cuda.current_stream(x.device).record_event()
    expert_pair(pa, x, adapter._payload(layer), adapter._payload(L.shadow), rid, sid, w, routes)
    host, _ = adapter._route_buffers(native, x, k)
    native.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream, 0)
    cpu.copy_(host, non_blocking=True)
    if t not in _logged:
        _logged.add(t)
        logger.warning('KT_CONTRACT_V2_GLUE_FUSED: captured %d rows on the fused route glue, layer %d onward', t, wrapper.kt_config.layer_idx)
    return StandardCombineInput(hidden_states=combine(both, ids, own, order))
