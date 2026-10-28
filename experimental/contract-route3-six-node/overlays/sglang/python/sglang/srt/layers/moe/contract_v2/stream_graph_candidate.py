"""Prepared contract-v2 streamed group graphs; preserves per-route BF16 values.

Each physical expert owns its route once. Group graphs fill those slots directly;
the adapter performs the existing ascending-expert FP32 combine afterwards.
No intermediate group sum or additional rounding is introduced.
"""
import torch


def graph_inputs(st, x, ids, weights):
    from sglang.srt.layers.moe import kt_stream_prefill as source
    rows, hidden = x.shape
    topk = ids.shape[1]
    g = st._gin.get(rows)
    if g is None:
        if source.SHARED_INPUTS:
            alloc = lambda name, shape, dtype: st._gin_view(
                'contract_' + name, rows, shape, dtype, x.device)
        else:
            alloc = lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device=x.device)
        # _gin_view may invalidate old graphs when growing a shared base. Only
        # publish this complete set after every allocation has finished.
        g = dict(xq=alloc('x', x.shape, x.dtype), sf=None,
                 ids=alloc('ids', ids.shape, torch.int32),
                 w=alloc('w', weights.shape, torch.float32),
                 acc=alloc('routes', (rows, topk, hidden), torch.bfloat16))
        g['out'] = g['acc']  # Common graph helper signature; no separate subtotal.
        st._gin[rows] = g
    if g['acc'].shape != (rows, topk, hidden) or g['acc'].dtype != torch.bfloat16:
        raise RuntimeError('contract stream buffers have an incompatible layout')
    g['xq'].copy_(x)
    g['ids'].copy_(ids)
    g['w'].copy_(weights)
    g['acc'].zero_()
    return g


def group_body(st, gi, base, n, x, weights, ids, routes):
    from .adapter import pack_stream_payload
    from .static_routes_candidate import expert_routes_static
    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows
    payload = pack_stream_payload(st, st.slots[gi % 2], n)
    local = torch.where((ids >= base) & (ids < base + n), ids - base, -1).contiguous()
    if routes.dtype != torch.bfloat16 or routes.shape != (*ids.shape, x.shape[1]):
        raise RuntimeError('contract stream requires BF16 per-route output')
    step = _block_rows(x.shape[0])
    for first in range(0, x.shape[0], step):
        last = min(first + step, x.shape[0])
        tile = 64 if last - first >= 128 else 16
        expert_routes_static(x[first:last], *payload, local[first:last], weights[first:last],
                             out=routes[first:last], tile=tile)


def run(st, wrapper, layer, x, ids, weights, routes):
    from sglang.srt.layers.moe import kt_stream_prefill as source
    if torch.cuda.is_current_stream_capturing() or not (source.ZEROCOPY and source.W16 and source.USE_GRAPH):
        raise RuntimeError('contract group graphs require eager zero-copy W16 dispatch')
    if x.dtype != torch.bfloat16 or st.world != 1:
        raise ValueError('contract stream requires BF16 activations and one GPU rank')
    first = wrapper.num_gpu_experts
    last = source.stream_cpu_cutoff(wrapper, st)
    groups = [list(range(b, min(b + st.G, last))) for b in range(first, last, st.G)]
    if not groups:
        return
    g = graph_inputs(st, x, ids, weights)
    stats = dict(write=0., hostwait=0., dma_enq=0., prep_enq=0., moe_enq=0.)
    lid = wrapper.kt_config.layer_idx
    cur = torch.cuda.current_stream(x.device)
    st._stage_zc(lid, 0, groups[0], stats)
    for gi, experts in enumerate(groups):
        slot = st.slots[gi % 2]
        # Capture warm-up reads scale bytes too: wait before _graph_for, not
        # merely before replay. This also orders raw DMA metadata reductions.
        cur.wait_event(slot.dma_done)
        graph = st._graph_for(layer, x.shape[0], gi, experts[0], len(experts), g)
        graph.replay()
        slot.compute_done.record(cur)
        slot.compute_used = True
        if gi + 1 < len(groups):
            st._stage_zc(lid, gi + 1, groups[gi + 1], stats)
    owned = ((ids >= first) & (ids < last) & (weights != 0))[:, :, None]
    torch.where(owned, g['acc'], routes, out=routes)
    st.calls += 1
