"""KT EP wrapper adapter for the contract v2 build (KT_CONTRACT_V2=1).

Mirrors the canonical-routes adapter of the moe_math_r6 draft: every route of a token is produced exactly once,
as a BF16 row in `routes [T, K, H]`, by whichever placement owns it:
  - CPU experts: kt-kernel's contract backend through its per-route API (`forward_routes_task`), routing weight
    already applied, exact K-partition merge inside kt-kernel;
  - GPU-resident experts, the hot-lend shadow layer and streamed prefill groups: the contract Triton kernels
    (`moe.expert_routes`), the same arithmetic.
Then one FP32 combine in ascending expert id (`moe.combine_routes`). The result is returned as FP32 so the
model's `shared.add_(routed, alpha)` sums the shared expert in FP32 and casts once, as the reference does.
"""
import torch
from .moe import expert_routes, combine_routes

_buffers = {}


def _route_buffers(native, x, k):
    # CUDA graphs retain these addresses: share by slot, never by layer
    from kt_kernel.experts_base import KExpertsCPUBuffer
    key = (x.device, x.shape[0], x.shape[1], k, native.layer_idx % KExpertsCPUBuffer.buffer_depth)
    if key not in _buffers:
        shape = (x.shape[0], k, x.shape[1])
        _buffers[key] = (torch.empty(shape, dtype=torch.bfloat16, pin_memory=True),
                         torch.empty(shape, dtype=torch.bfloat16, device=x.device))
    return _buffers[key]


def submit_routes(native, x, ids, weights, packer=None):
    from kt_kernel.experts_base import KExpertsCPUBuffer, BaseMoEWrapper
    if native.max_deferred_experts_per_token or not hasattr(native.moe, 'forward_routes_task'):
        raise RuntimeError('contract v2 routes require the native route API (AVX2ContractV2_MOE) and zero deferred experts')
    if BaseMoEWrapper._layer_has_pending_deferred.get(native.layer_idx - 1, False):
        raise RuntimeError('contract v2 routes cannot consume pending deferred output')
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError('contract v2 expects CUDA BF16 hidden states')
    if not 1 <= x.shape[0] <= native.chunked_prefill_size:
        raise ValueError(f'contract v2 rows {x.shape[0]} exceed native capacity {native.chunked_prefill_size}')
    if ids.shape != (x.shape[0], native.num_experts_per_tok) or weights.shape != ids.shape:
        raise ValueError('contract v2 route shapes mismatch')
    if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
        raise ValueError('contract v2 route IDs/weights require int32 or int64 IDs and float32 weights')
    if ids.device != x.device or weights.device != x.device:
        raise ValueError('contract v2 inputs must share a CUDA device')
    bufs = KExpertsCPUBuffer.get_buffer(x, native.num_experts_per_tok)
    slot = native.layer_idx % KExpertsCPUBuffer.buffer_depth
    host_x, host_ids, _, host_weights, _, host_bsz, _ = bufs
    host_out, _ = _route_buffers(native, x, ids.shape[1])
    packed = bool(packer is not None and packer(x, weights, ids, host_x[slot], host_weights[slot], host_ids[slot]))
    if not packed:
        host_x[slot].copy_(x, non_blocking=True)
        host_ids[slot].copy_(ids, non_blocking=True)
        host_weights[slot].copy_(weights, non_blocking=True)
    stream = torch.cuda.current_stream(x.device)
    task = native.moe.forward_routes_task(host_bsz[slot].data_ptr(), ids.shape[1], host_ids[slot].data_ptr(),
                                          host_weights[slot].data_ptr(), host_x[slot].data_ptr(), host_out.data_ptr())
    native.cpu_infer.submit_with_cuda_stream(stream.cuda_stream, task)
    BaseMoEWrapper._layer_has_pending_deferred[native.layer_idx] = False


def sync_routes(native, x, k):
    host, out = _route_buffers(native, x, k)
    native.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream, 0)
    out.copy_(host, non_blocking=True)
    return out


def _payload(layer):
    return tuple(getattr(layer, n) for n in ('w13_weight', 'w13_weight_scale_inv', 'w2_weight', 'w2_weight_scale_inv'))


def apply(wrapper, layer, dispatch):
    from sglang.srt.layers.moe import kt_kv_hot_lend, kt_stream_prefill, kt_ep_wrapper
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
    x = dispatch.hidden_states
    ids = dispatch.topk_output.topk_ids
    weights = dispatch.topk_output.topk_weights
    if wrapper.tp_rank != 0 or layer.moe_tp_size != 1 or layer.moe_ep_size != 1:
        raise RuntimeError('contract v2 currently requires one GPU rank')
    streamer = kt_stream_prefill.stream_prefill_for(wrapper, layer, x.shape[0])
    hot = kt_kv_hot_lend.route(wrapper, dispatch) if streamer is None else None
    resident = ids < wrapper.num_gpu_experts
    rid = torch.where(resident, ids, -1)
    rw = weights
    if streamer is not None:
        cutoff = kt_stream_prefill.stream_cpu_cutoff(wrapper, streamer)
        cid = torch.where(ids >= cutoff, ids, -1)
        cw = weights
        use_cpu = cutoff < wrapper.global_num_experts
    elif hot is not None:
        cid = hot[0].topk_output.topk_ids
        cw = hot[0].topk_output.topk_weights
        if len(hot) > 4 and hot[4] is not None: rid, rw = hot[4]
        elif hot[3] is not None: rid = hot[3]
        use_cpu = True
    else:
        cid = torch.where(ids >= wrapper.num_gpu_experts, ids, -1)
        cw = weights
        use_cpu = True
    cpu_owned = (cid >= 0) & (cw != 0)
    resident_owned = (rid >= 0) & (rid < wrapper.num_gpu_experts) & (rw != 0)
    owners = cpu_owned.to(torch.int32) + resident_owned.to(torch.int32)
    routes = torch.zeros((*ids.shape, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    if use_cpu:
        submit_routes(wrapper.wrapper, x, cid, cw, kt_ep_wrapper.pack_to_host if getattr(kt_ep_wrapper, '_KT_HANDOFF_PACK', False) else None)
        if getattr(kt_ep_wrapper, '_KT_SHARED_AFTER_SUBMIT', False):
            wrapper._kt_post_submit_event = torch.cuda.current_stream(x.device).record_event()
    if wrapper.num_gpu_experts:
        expert_routes(x, *_payload(layer), rid.contiguous(), rw.contiguous().float(), out=routes)
    if hot is not None and hot[1] is not None:
        shadow = wrapper.__dict__['_kt_kv_hot'].shadow
        sid = hot[1].topk_output.topk_ids; sw = hot[1].topk_output.topk_weights
        owners = owners + ((sid >= 0) & (sid < shadow.w13_weight.shape[0]) & (sw != 0)).to(torch.int32)
        expert_routes(x, *_payload(shadow), sid.contiguous(), sw.contiguous().float(), out=routes)
    if streamer is not None:
        owners = owners + ((ids >= wrapper.num_gpu_experts) & (ids < cutoff) & (weights != 0)).to(torch.int32)
        stream_routes(streamer, wrapper, layer, x, ids, weights, routes)
    if use_cpu:
        cpu = sync_routes(wrapper.wrapper, x, ids.shape[1])
        torch.where(cpu_owned[:, :, None], cpu, routes, out=routes)
    expected = ((ids >= 0) & (weights != 0)).to(torch.int32)
    torch._assert_async((owners == expected).all(), 'contract v2 route ownership is not exactly once')
    torch._assert_async(torch.isfinite(weights).all(), 'non-finite MoE weights')
    return StandardCombineInput(hidden_states=combine_routes(routes, ids))


def _scale_codes(scales):
    # E8M0 powers are exact in the CPU's FP32 scale arenas; code 0 (2^-127) has zero exponent bits as well
    bits = scales.contiguous().view(torch.int32)
    code = ((bits >> 23) & 255).to(torch.uint8)
    valid = ((bits & 0x7fffff) == 0) & (bits > 0) & (code < 255)
    valid = valid | (bits == 0x00400000)
    torch._assert_async(valid.all(), 'streamed payload contains a non-E8M0 scale')
    return code


def pack_stream_payload(streamer, slot, n):
    """Reassemble the streamed experts' original bytes from the NUMA slices (no dequant/requant)."""
    if hasattr(streamer, '_contract_arena_layout'):
        from .arena_stream_candidate import pack_payload
        return pack_payload(streamer, slot, n)
    N, K = streamer.N, streamer.K
    if streamer.world != 1: raise RuntimeError('unsupported streamed payload layout')
    device = slot.raw.device
    w13 = torch.empty((n, 2 * N, K // 2), dtype=torch.uint8, device=device)
    s13 = torch.empty((n, 2 * N, K // 32), dtype=torch.uint8, device=device)
    w2 = torch.empty((n, K, N // 2), dtype=torch.uint8, device=device)
    s2 = torch.empty((n, K, N // 32), dtype=torch.uint8, device=device)
    segments = [(streamer._segs[0], slot.raw, slot.raw_s)] if streamer._seg_whole else list(zip(streamer._segs, slot.raws, slot.raw_ss))
    next_row = 0
    for (_, lo, count, d, m), raw, raw_s in segments:
        if d != next_row or lo % 32 or count % 32 or m % 32 or lo + count > m:
            raise RuntimeError('streamed segments must cover aligned rows exactly once')
        next_row += count
        scales = raw_s.view(torch.float32)
        for pi, offset in ((0, 0), (1, N)):
            w13[:, offset + d:offset + d + count].copy_(raw[pi, :n, lo * K // 2:(lo + count) * K // 2].reshape(n, count, K // 2))
            sf = scales[pi, :n, lo * (K // 32):(lo + count) * (K // 32)].reshape(n, count, K // 32)
            s13[:, offset + d:offset + d + count].copy_(_scale_codes(sf))
        w2[:, :, d // 2:(d + count) // 2].copy_(raw[2, :n, :K * m // 2].reshape(n, K, m // 2)[:, :, lo // 2:(lo + count) // 2])
        sf = scales[2, :n, :K * (m // 32)].reshape(n, K, m // 32)[:, :, lo // 32:(lo + count) // 32]
        s2[:, :, d // 32:(d + count) // 32].copy_(_scale_codes(sf))
    if next_row != N: raise RuntimeError('incomplete streamed payload')
    return w13, s13, w2, s2


def stream_routes(streamer, wrapper, layer, x, ids, weights, routes):
    from sglang.srt.layers.moe import kt_stream_prefill as source
    if source.CONTRACT_STREAM:
        from .stream_graph_candidate import run
        return run(streamer, wrapper, layer, x, ids, weights, routes)
    if not source.ZEROCOPY or not source.W16 or torch.cuda.is_current_stream_capturing():
        raise RuntimeError('contract v2 prefill requires eager zero-copy W16 staging')
    first = wrapper.num_gpu_experts
    last = source.stream_cpu_cutoff(wrapper, streamer)
    groups = [list(range(b, min(b + streamer.G, last))) for b in range(first, last, streamer.G)]
    if not groups: return
    stats = {'write': 0., 'hostwait': 0., 'dma_enq': 0., 'prep_enq': 0., 'moe_enq': 0.}
    lid = wrapper.kt_config.layer_idx
    streamer._stage_zc(lid, 0, groups[0], stats)
    cur = torch.cuda.current_stream(x.device)
    for gi, experts in enumerate(groups):
        slot = streamer.slots[gi % 2]
        cur.wait_event(slot.dma_done)
        payload = pack_stream_payload(streamer, slot, len(experts))
        local = torch.where((ids >= experts[0]) & (ids <= experts[-1]), ids - experts[0], -1).contiguous()
        expert_routes(x, *payload, local, weights.float(), out=routes)
        slot.compute_done.record(cur)
        slot.compute_used = True
        if gi + 1 < len(groups): streamer._stage_zc(lid, gi + 1, groups[gi + 1], stats)
    streamer.calls += 1
