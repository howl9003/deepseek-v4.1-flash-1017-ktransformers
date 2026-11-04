"""05yuki's KT_STREAM_CPU_SHORT (upstream 46da340, tier 1 patch 14) ported to our contract v2 stream, as an overlay on
the served tree ada-serve-c2-20261103-r3 (Claude, 3 Nov 2026; upstream-picks-r1).

KT_STREAM_CPU_SHORT=<tokens>:<k>,... lets a streamed forward of at most <tokens> rows leave the last k of a layer's
streamed expert groups to kt-kernel on the CPU, computed while the rest crosses PCIe. Unset (the default) nothing
changes: every helper below falls back to KT_STREAM_CPU_GROUPS exactly as before.

Our streamed prefill is not his CUTLASS group path. KT_CONTRACT_V2=1 sends every MoE call to contract_v2/adapter.apply,
which already splits on kt_stream_prefill.stream_cpu_cutoff (the static KT_STREAM_CPU_GROUPS share): routes at or past
the cutoff go to kt-kernel's per-route API before the stream, the groups below it to the stream (graph path:
stream_graph_candidate.run; eager fallback: adapter.stream_routes), and each route lands once as a BF16 row in the
same `routes` buffer before the logical-order FP32 combine. The port gives stream_cpu_cutoff the forward's row count
and passes it from all three places, so the split, the stream's groups and the ownership check agree.

One addition of ours: a share pads the CPU's rows to a bucket (the next power of two, at most the entry's bound) with
routes of id -1 and weight 0. kt-kernel's KExpertsCPUBuffer and the adapter's _route_buffers keep one set of buffers
per row count forever; without the bucket every short prompt length would add about 150 MB of VRAM and as much pinned
memory at 1,024 rows. The bucket's per-route output is pinned host memory; its device copy is a transient of the
prefill window.

Files (argv: the sglang package directory, patched in place):
  srt/layers/moe/kt_stream_prefill.py              parse, cpu_groups_for, stream_cpu_cutoff(..., tokens)
  srt/layers/moe/contract_v2/adapter.py            row count to the cutoff, the padded CPU share
  srt/layers/moe/contract_v2/stream_graph_candidate.py   row count to the cutoff
kt_ep_wrapper.py is not touched: its own split serves KT_CONTRACT_V2=0 only, and kt_stream_prefill refuses
KT_STREAM_CPU_SHORT without KT_CONTRACT_V2=1 at import."""
import hashlib
import sys
from pathlib import Path

PINS = {
    'srt/layers/moe/kt_stream_prefill.py': '7827b4961219076e8cce930e65ec673f7a215517065369f490626ab76f179f95',
    'srt/layers/moe/contract_v2/adapter.py': '01beed62fdf94920144c42b939546b4b4612c6c44ba70ec1f13f9045e23687c8',
    'srt/layers/moe/contract_v2/stream_graph_candidate.py': '6c757f1fea59dc1c1c241624fe33724f9dfc9c535b17a5e911f7f82bdd1d8627',
}


def sub(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:80])
    return s.replace(old, new)


def patch_stream_prefill(s):
    old = ('CPU_GROUPS = int(os.environ.get("KT_STREAM_CPU_GROUPS", "0"))\n')
    new = (old +
           '# KT_STREAM_CPU_SHORT=<tokens>:<k>,... (05yuki 46da340; overlay upstream-picks-r1, default off): a forward of\n'
           '# at most <tokens> rows leaves k of a layer\'s streamed groups to the CPU instead (the first entry that fits,\n'
           '# smallest bound first); a longer forward keeps KT_STREAM_CPU_GROUPS. A short prompt streams nearly every\n'
           '# expert, so its stream costs what a long one\'s does while the CPU\'s share costs it little. Ported to the\n'
           '# contract v2 adapter, which submits the share before the stream and takes its per-route rows after.\n'
           'def _parse_cpu_short(spec: str):\n'
           '    entries = []\n'
           '    for item in spec.split(","):\n'
           '        item = item.strip()\n'
           '        if not item:\n'
           '            continue\n'
           '        tokens, sep, groups = item.partition(":")\n'
           '        if not sep:\n'
           '            raise ValueError(f"KT_STREAM_CPU_SHORT entry {item!r} is not <tokens>:<k>")\n'
           '        tokens, groups = int(tokens), int(groups)\n'
           '        if tokens <= 0 or groups < 0:\n'
           '            raise ValueError(f"KT_STREAM_CPU_SHORT entry {item!r}: tokens must be > 0 and k >= 0")\n'
           '        entries.append((tokens, groups))\n'
           '    entries.sort()\n'
           '    if len({t for t, _ in entries}) != len(entries):\n'
           '        raise ValueError(f"KT_STREAM_CPU_SHORT repeats a token bound: {spec!r}")\n'
           '    return entries\n'
           '\n'
           '\n'
           'CPU_SHORT = _parse_cpu_short(os.environ.get("KT_STREAM_CPU_SHORT", ""))\n'
           'if CPU_SHORT and os.environ.get("KT_CONTRACT_V2", "0") != "1":\n'
           '    raise RuntimeError("KT_STREAM_CPU_SHORT is ported to the contract v2 adapter only (KT_CONTRACT_V2=1)")\n'
           '\n'
           '\n'
           'def cpu_short_entry(tokens):\n'
           '    """The (bound, k) KT_STREAM_CPU_SHORT entry serving a forward of `tokens` rows, None when none does."""\n'
           '    if tokens is None:\n'
           '        return None\n'
           '    for bound, groups in CPU_SHORT:\n'
           '        if tokens <= bound:\n'
           '            return bound, groups\n'
           '    return None\n'
           '\n'
           '\n'
           'def cpu_groups_for(tokens) -> int:\n'
           '    """How many of a layer\'s streamed groups the CPU takes for a forward of `tokens` rows (None: KT_STREAM_CPU_GROUPS)."""\n'
           '    entry = cpu_short_entry(tokens)\n'
           '    return CPU_GROUPS if entry is None else entry[1]\n'
           '\n'
           '\n')
    s = sub(s, old, new)
    old = ('def stream_cpu_cutoff(method, streamer) -> int:\n'
           '    """First physical expert id the CPU takes under KT_STREAM_CPU_GROUPS\n'
           '    (== global_num_experts when the share is off)."""\n'
           '    first = method.num_gpu_experts\n'
           '    last = method.global_num_experts\n'
           '    if not CPU_GROUPS:\n'
           '        return last\n'
           '    n_groups = -(-(last - first) // streamer.G)\n'
           '    kept = max(1, n_groups - CPU_GROUPS)\n'
           '    return min(last, first + kept * streamer.G)\n')
    new = ('def stream_cpu_cutoff(method, streamer, tokens=None) -> int:\n'
           '    """First physical expert id the CPU takes under KT_STREAM_CPU_GROUPS, or under KT_STREAM_CPU_SHORT for a\n'
           '    forward of `tokens` rows (== global_num_experts when the share is off; tokens None: KT_STREAM_CPU_GROUPS)."""\n'
           '    first = method.num_gpu_experts\n'
           '    last = method.global_num_experts\n'
           '    cpu_groups = cpu_groups_for(tokens)\n'
           '    if not cpu_groups:\n'
           '        return last\n'
           '    n_groups = -(-(last - first) // streamer.G)\n'
           '    kept = max(1, n_groups - cpu_groups)\n'
           '    return min(last, first + kept * streamer.G)\n')
    assert s.endswith(old), 'stream_cpu_cutoff is no longer the last function'
    return sub(s, old, new)


def patch_adapter(s):
    s = sub(s, '_buffers = {}\n',
            '_buffers = {}\n'
            '_short_hosts = {}  # KT_STREAM_CPU_SHORT: pinned per-route rows, one per (bucket, slot)\n')
    old = ('def submit_routes(native, x, ids, weights, packer=None):\n')
    s = sub(s, old, 'def submit_routes(native, x, ids, weights, packer=None, host_out=None):\n')
    s = sub(s, '    host_out, _ = _route_buffers(native, x, ids.shape[1])\n',
            '    if host_out is None:\n'
            '        host_out, _ = _route_buffers(native, x, ids.shape[1])\n')
    old = ('def sync_routes(native, x, k):\n'
           '    host, out = _route_buffers(native, x, k)\n'
           '    native.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream, 0)\n'
           '    out.copy_(host, non_blocking=True)\n'
           '    return out\n')
    new = ('def sync_routes(native, x, k, host=None):\n'
           '    if host is None:\n'
           '        host, out = _route_buffers(native, x, k)\n'
           '    else:\n'
           '        # KT_STREAM_CPU_SHORT: the share\'s pinned rows are padded to its bucket; the device copy is a\n'
           '        # transient of the prefill window, not a buffer kept per row count\n'
           '        host = host[:x.shape[0]]\n'
           '        out = torch.empty((x.shape[0], k, x.shape[1]), dtype=torch.bfloat16, device=x.device)\n'
           '    native.cpu_infer.sync_with_cuda_stream(torch.cuda.current_stream(x.device).cuda_stream, 0)\n'
           '    out.copy_(host, non_blocking=True)\n'
           '    return out\n'
           '\n'
           '\n'
           'def _short_rows(rows):\n'
           '    """KT_STREAM_CPU_SHORT: the row count the CPU share of a `rows`-row streamed forward is padded to (the next\n'
           '    power of two, at most its entry\'s bound), so kt-kernel\'s staging and the per-route rows are made once a\n'
           '    bucket, not once a prompt length; 0 when no short entry gives this forward a share."""\n'
           '    from sglang.srt.layers.moe import kt_stream_prefill as source\n'
           '    entry = source.cpu_short_entry(rows)\n'
           '    if entry is None or entry[1] <= 0:\n'
           '        return 0\n'
           '    return max(rows, min(entry[0], 1 << (rows - 1).bit_length()))\n'
           '\n'
           '\n'
           'def _short_host(native, rows, hidden, k):\n'
           '    from kt_kernel.experts_base import KExpertsCPUBuffer\n'
           '    key = (rows, hidden, k, native.layer_idx % KExpertsCPUBuffer.buffer_depth)\n'
           '    if key not in _short_hosts:\n'
           '        _short_hosts[key] = torch.empty((rows, k, hidden), dtype=torch.bfloat16, pin_memory=True)\n'
           '    return _short_hosts[key]\n')
    s = sub(s, old, new)
    s = sub(s, '        cutoff = kt_stream_prefill.stream_cpu_cutoff(wrapper, streamer)\n',
            '        cutoff = kt_stream_prefill.stream_cpu_cutoff(wrapper, streamer, x.shape[0])\n')
    old = ('    if use_cpu:\n'
           '        submit_routes(wrapper.wrapper, x, cid, cw, kt_ep_wrapper.pack_to_host if getattr(kt_ep_wrapper, \'_KT_HANDOFF_PACK\', False) else None)\n')
    new = ('    short = _short_rows(x.shape[0]) if streamer is not None and use_cpu else 0\n'
           '    host_short = None\n'
           '    if use_cpu and short:\n'
           '        # KT_STREAM_CPU_SHORT: the CPU\'s rows padded to the bucket with copies of the last row whose routes\n'
           '        # are id -1 at weight 0 (no route: kt-kernel skips them, and no padded row is ever read back)\n'
           '        pad = short - x.shape[0]\n'
           '        xs, cs, ws = x, cid, cw\n'
           '        if pad:\n'
           '            xs = torch.cat([x, x[-1:].expand(pad, x.shape[1])])\n'
           '            cs = torch.cat([cid, cid.new_full((pad, cid.shape[1]), -1)])\n'
           '            ws = torch.cat([cw, cw.new_zeros((pad, cw.shape[1]))])\n'
           '        host_short = _short_host(wrapper.wrapper, short, x.shape[1], ids.shape[1])\n'
           '        submit_routes(wrapper.wrapper, xs, cs, ws, kt_ep_wrapper.pack_to_host if getattr(kt_ep_wrapper, \'_KT_HANDOFF_PACK\', False) else None,\n'
           '                      host_out=host_short)\n'
           '        if getattr(kt_ep_wrapper, \'_KT_SHARED_AFTER_SUBMIT\', False):\n'
           '            wrapper._kt_post_submit_event = torch.cuda.current_stream(x.device).record_event()\n'
           '    elif use_cpu:\n'
           '        submit_routes(wrapper.wrapper, x, cid, cw, kt_ep_wrapper.pack_to_host if getattr(kt_ep_wrapper, \'_KT_HANDOFF_PACK\', False) else None)\n')
    s = sub(s, old, new)
    old = ('    if use_cpu:\n'
           '        cpu = sync_routes(wrapper.wrapper, x, ids.shape[1])\n')
    new = ('    if use_cpu:\n'
           '        cpu = sync_routes(wrapper.wrapper, x, ids.shape[1], host=host_short) if short else sync_routes(wrapper.wrapper, x, ids.shape[1])\n')
    s = sub(s, old, new)
    s = sub(s, '    last = source.stream_cpu_cutoff(wrapper, streamer)\n',
            '    last = source.stream_cpu_cutoff(wrapper, streamer, x.shape[0])\n')
    return s


def patch_graph(s):
    return sub(s, '    last = source.stream_cpu_cutoff(wrapper, st)\n',
               '    last = source.stream_cpu_cutoff(wrapper, st, x.shape[0])\n')


def main(root):
    root = Path(root)
    steps = {'srt/layers/moe/kt_stream_prefill.py': patch_stream_prefill,
             'srt/layers/moe/contract_v2/adapter.py': patch_adapter,
             'srt/layers/moe/contract_v2/stream_graph_candidate.py': patch_graph}
    out = {}
    for rel, fn in steps.items():
        p = root / rel
        s = p.read_bytes().decode('utf-8')
        assert hashlib.sha256(s.encode('utf-8')).hexdigest() == PINS[rel], rel + ' differs from the served pin'
        s = fn(s)
        compile(s, str(p), 'exec')
        out[p] = s
    for p, s in out.items():   # nothing is written unless every file patched and compiled
        p.write_bytes(s.encode('utf-8'))
        print(hashlib.sha256(s.encode('utf-8')).hexdigest(), p.relative_to(root).as_posix())


if __name__ == '__main__':
    main(sys.argv[1])
