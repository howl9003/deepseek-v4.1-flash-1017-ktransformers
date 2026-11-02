"""Path A prefill expert kernel (r3-P2) behind the default-off selector KT_GPU_STREAM_PATHA=1, for the contract
build's streamed-prefill expert path (contract_v2/stream_graph_candidate.group_body, reached only under
KT_CONTRACT_V2_STREAM_GRAPH=1).

Glue only: no arithmetic lives here. With the selector off, stream_graph_candidate never imports this module. With
it on, group_body's per-row-block streamed-expert projections of a group of n experts run through the r3-P2
grouped launch (numeric-path-proposals-20261028/kernels-r1/prefill-imma/patha_prefill.cu): the group's routes
sorted by expert with 64-row padding (moe_align_block_size, the contract path's own alignment), the gate/up GEMM
over all experts in one launch (gridDim.z, row offsets), the contract's SwiGLU step and routing weight (unchanged
arithmetic, _activate_rows), the contract quantiser of the activation, the kernel's prep, the down GEMM, the
kernel's own finish. The per-route BF16 outputs land in the same `routes` buffer the existing path writes; the
combine across experts (run()'s owned-route select, the adapter's ascending-expert FP32 combine) is untouched.

The activation operand of gate/up is prepared once per layer (the contract quantiser of x into E4M3 codes and
group exponents, then the pinned prep kernel over all rows: the per-layer inputs hold exactly the chunk's rows, so
every row is live and no bound is needed), inside the first group's graph, and gathered per group; the down input
is quantised and prepared per group over the live rows only: the glue's row kernels and the r3-P3 prep kernel
(`prep_activations(..., rows=live, out=...)`) take the device-side live row count of the (block, group), so rows of
the static workspace at or beyond it are neither read nor written.

CUDA-graph safety: the stream path captures one graph per (rows, group) and replays it for every layer, so every
buffer of this path is static: the per-layer operands are graph inputs of the row count (st._gin_view, grown with
the chunk like the existing inputs), the per-group workspaces are allocated once at the plan's maximum block
(KT_GPU_STREAM_ROW_BLOCK rows x min(topk, group) routes + 64-row padding) in the lend's scratch region; nothing is
allocated or synchronised inside a capture except the routing masks and moe_align's tables, which the existing
path allocates in the same way (under 1 MB per block, from the graph pool). The kernel's device flag word is
zeroed at construction and read once per layer at the end of run() (outside any capture, after a stream
synchronisation); a set bit raises and the layer's output is discarded (fail closed).

Environment (read once):
    KT_GPU_STREAM_PATHA             1 enables the path (stream_graph_candidate reads it; this module checks it)
    KT_GPU_STREAM_PATHA_DOWN_PLANES 3 (default: the r4 grouped rows) or 4 (the four-plane down knob, CfgDown4)
    KT_GPU_STREAM_PATHA_MAX_ROWS    per-expert padded row bound of the GEMM grid (default 0: the block's rows, the
                                    only value that cannot fail); a lower bound is checked on the device per block
    KT_GPU_STREAM_PATHA_DIR         folder holding patha_prefill.py / .cu / patha_sequence.h (else plain import)
    KT_GPU_STREAM_PATHA_SOURCE_SHA256  'skip' disables the source pins (never in a receipt run)
    PATHA_EXT_NAME                  the loader's extension name (the r3-P3 build: patha_prefill_sm89_r3p3)
"""
import hashlib
import os
import sys

SELECTOR = 'KT_GPU_STREAM_PATHA'
DOWN_PLANES_ENV = 'KT_GPU_STREAM_PATHA_DOWN_PLANES'
MAX_ROWS_ENV = 'KT_GPU_STREAM_PATHA_MAX_ROWS'
KERNEL_DIR_ENV = 'KT_GPU_STREAM_PATHA_DIR'
SOURCE_SHA_ENV = 'KT_GPU_STREAM_PATHA_SOURCE_SHA256'
PINNED_SOURCES = {                      # kernels-r1/prefill-imma, r3-P3 (PREPARATION.json there, 1a73cfff...; device row count)
    'patha_prefill.cu': 'c55f90e8e3a5c33c154952714c60bc55b535900552f3246147150f690532c8fa',
    'patha_sequence.h': 'f654d81652cfdcd4437d25c396c07763ef82b9936ad55a4a5f46190cb765d7ba',
    'patha_prefill.py': 'ddad7e400b68c7a62596fda66c229ca9c956aa44962a55908d992ceaf8473e1f',
}
BM = 64                                 # the kernel's CTA row tile = the alignment tile
MIN_SCALE_CODE = 53                     # finish-rule admission of the expert kernel (CONTRACT_PINS_r1.md section 1)
GATE_UP_PLANES = 3                      # EXPERT_GATE_UP_PREP
DEFAULT_DOWN_PLANES = 3                 # EXPERT_DOWN_PREP: r4 group32 rows, down3 beats down4 by 9 to 11 percent
DOWN_PLANES_ALLOWED = (3, 4)
EXPERT_SHAPES = {(4608, 5120), (5120, 2304)}       # (N, K) verified by Codex (r4): gate/up and down
ERR_BITS = {1: 'int64 accumulator overflow', 2: 'negative chunk shift on the fold path', 4: 'finish outside the normal FP32 range',
            8: 'NaN E4M3 activation code', 16: 'NaN E4M3 weight code', 32: 'top digit out of range',
            64: 'shifted expert weight outside int8', 128: 'exponent metadata inconsistent'}


# ----------------------------------------------------------------------------- selector and admission (no torch)

def selector_on(environ=None):
    environ = os.environ if environ is None else environ
    return environ.get(SELECTOR, '0') == '1'


def down_planes(environ=None):
    environ = os.environ if environ is None else environ
    v = int(environ.get(DOWN_PLANES_ENV, str(DEFAULT_DOWN_PLANES)) or DEFAULT_DOWN_PLANES)
    if v not in DOWN_PLANES_ALLOWED:
        raise ValueError('%s must be 3 or 4' % DOWN_PLANES_ENV)
    return v


def check_environment(environ=None):
    """The selector is honoured only on the contract stream-graph path with the row block set."""
    environ = os.environ if environ is None else environ
    if environ.get('KT_CONTRACT_V2_STREAM_GRAPH', '0') != '1' or environ.get('KT_CONTRACT_V2', '0') != '1' \
            or environ.get('KT_CONTRACT_V2_STATIC_GPU', '0') != '1':
        raise RuntimeError('%s=1 requires KT_CONTRACT_V2=1 KT_CONTRACT_V2_STATIC_GPU=1 KT_CONTRACT_V2_STREAM_GRAPH=1' % SELECTOR)
    if environ.get('KT_MOE_FP32_STREAM', '0') == '1':
        raise RuntimeError('%s=1 excludes KT_MOE_FP32_STREAM' % SELECTOR)
    block = int(environ.get('KT_GPU_STREAM_ROW_BLOCK', '0') or 0)
    if block <= 0 or block % BM:
        raise RuntimeError('%s=1 requires KT_GPU_STREAM_ROW_BLOCK set to a multiple of 64 (the plan\'s maximum block sizes the static workspaces)' % SELECTOR)
    return block


def live_cap(step, topk, n_max):
    """Rows the sorted route table of one (block, group) can hold: a token routes to an expert at most once, so a
    group of n experts receives at most step x min(topk, n) routes; 64-row padding per expert on top."""
    return step * min(topk, n_max) + n_max * (BM - 1)


def max_rows_bound(step, environ=None):
    """The per-expert padded row bound the GEMM grid is sized for (host int): the block's rows unless lowered."""
    environ = os.environ if environ is None else environ
    v = int(environ.get(MAX_ROWS_ENV, '0') or 0)
    rows = step if v <= 0 else min(v, step)
    return -(-rows // BM) * BM


def workspace_bytes(step, topk, n_max, hidden, inter, dplanes=DEFAULT_DOWN_PLANES):
    """Static per-group workspace of the path (bytes per buffer); the memory statement of the README."""
    cap = live_cap(step, topk, n_max)
    K, I, H = hidden, inter, hidden
    return dict(cap_rows=cap,
                P=3 * cap * K, AM=(K // 128) * cap * 4, EAg=cap * (K // 32),
                gu=cap * 2 * I * 2, h=cap * I * 2, hcodes=cap * I, hea=cap * (I // 32),
                hplanes=dplanes * cap * I, hameta=(I // 128) * cap * 4, y=cap * H * 2,
                ew13=n_max * 2 * I * (K // 32), EW13=n_max * 2 * I, ew2=n_max * H * (I // 32), EW2=n_max * H,
                active=step * topk * 4, tables=4 * (n_max + 8))


def layer_bytes_per_row(hidden):
    """Per-layer graph inputs of the path per token row: codes, three planes, group exponents, metadata."""
    return hidden + 3 * hidden + hidden // 32 + (hidden // 128) * 4


def _import_patha(environ=None):
    environ = os.environ if environ is None else environ
    folder = environ.get(KERNEL_DIR_ENV)
    if folder and folder not in sys.path:
        sys.path.insert(0, folder)
    try:
        import patha_prefill
    except ImportError as e:
        raise ImportError('path A prefill: patha_prefill.py is not importable; set %s to its folder (%s)' % (KERNEL_DIR_ENV, e))
    if environ.get(SOURCE_SHA_ENV, '') != 'skip':
        here = os.path.dirname(os.path.abspath(patha_prefill.__file__))
        for name, expected in PINNED_SOURCES.items():
            with open(os.path.join(here, name), 'rb') as f:
                got = hashlib.sha256(f.read()).hexdigest()
            if got != expected:
                raise RuntimeError('path A prefill: %s sha256 %s is not the pinned r3-P3 %s (set %s=skip only outside receipt runs)'
                                   % (name, got, expected, SOURCE_SHA_ENV))
    if environ.get('KT_GPU_STREAM_PATHA_NATIVE_PIN'):
        from .patha_prefill_runtime import bind_native
        bind_native(patha_prefill, environ)
    return patha_prefill


# ----------------------------------------------------------------------------- the device side

class PathAStream:
    """One per KTStreamPrefill (st._patha). Construct outside any CUDA graph capture (graph_inputs is eager): loads
    the extension, launches every instantiation this path uses once (the kernel sets its dynamic shared-memory
    attribute on first launch), zeroes the flag word."""

    def __init__(self, device, environ=None):
        import torch
        self.torch = torch
        self.environ = os.environ if environ is None else environ
        self.device = torch.device(device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('path A prefill: construct outside CUDA graph capture')
        self.step = check_environment(self.environ)
        self.dplanes = down_planes(self.environ)
        self.pa = _import_patha(self.environ)
        self.ext = self.pa._load()
        try:
            from . import stream_patha_kernels as kk        # the overlay package (contract_v2/)
        except ImportError:
            import stream_patha_kernels as kk              # this folder on sys.path (component tests)
        self.kk = kk
        self.ws = None
        with torch.cuda.device(self.device):
            self.warmup()

    # ------------------------------------------------------------------ buffers
    def layer_buffers(self, alloc, rows, hidden):
        """The per-layer operands of gate/up as graph inputs of this row count (allocated by graph_inputs' alloc)."""
        t = self.torch
        return dict(patha_codes=alloc('patha_codes', (rows, hidden), t.uint8),
                    patha_planes=alloc('patha_planes', (GATE_UP_PLANES, rows, hidden), t.int8),
                    patha_ea=alloc('patha_ea', (rows, hidden // 32), t.int8),
                    patha_ameta=alloc('patha_ameta', (hidden // 128, rows, 4), t.int8))

    def workspace(self, topk, n_max, hidden, inter):
        """The static per-group workspace at the plan's maximum block, allocated once in the lend's scratch region."""
        if self.ws is not None:
            if self.ws['key'] != (topk, n_max, hidden, inter):
                raise RuntimeError('path A prefill: the group geometry changed after the workspace was allocated')
            return self.ws
        t = self.torch
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('path A prefill: workspace requested inside a capture')
        cap = live_cap(self.step, topk, n_max)
        K, I, H = hidden, inter, hidden
        if (2 * I, K) not in EXPERT_SHAPES or (H, I) not in EXPERT_SHAPES:
            raise RuntimeError('path A prefill: (N, K) = (%d, %d) / (%d, %d) is not a verified expert shape' % (2 * I, K, H, I))
        dev = self.device

        def z(shape, dtype):
            return t.zeros(shape, dtype=dtype, device=dev)
        try:
            from sglang.srt.layers.moe import kt_lend
            region = kt_lend.scratch_region()
        except ImportError:      # outside the serving tree (component tests)
            import contextlib
            region = contextlib.nullcontext()
        with region:
            ws = dict(key=(topk, n_max, hidden, inter), cap=cap, step=self.step,
                      P=z((GATE_UP_PLANES, cap, K), t.int8), AM=z((K // 128, cap, 4), t.int8), EAg=z((cap, K // 32), t.int8),
                      gu=z((cap, 2 * I), t.bfloat16), h=z((cap, I), t.bfloat16), hcodes=z((cap, I), t.uint8),
                      hea=z((cap, I // 32), t.int8), hplanes=z((self.dplanes, cap, I), t.int8), hameta=z((I // 128, cap, 4), t.int8),
                      y=z((cap, H), t.bfloat16),
                      ew13=z((n_max, 2 * I, K // 32), t.int8), EW13=z((n_max, 2 * I), t.int8),
                      ew2=z((n_max, H, I // 32), t.int8), EW2=z((n_max, H), t.int8),
                      active=z((self.step * topk,), t.int32), row_off=z((n_max + 1,), t.int32), bad=z((1,), t.int32))
        self.ws = ws
        return ws

    def warmup(self):
        """Every GEMM / prep instantiation once on zero inputs (attribute set, Triton compiled), then the flag reset."""
        t = self.torch
        K, N = 128, 64
        codes = t.zeros((BM, K), dtype=t.uint8, device=self.device)
        ea = t.zeros((BM, K // 32), dtype=t.int8, device=self.device)
        w = t.zeros((2, N, K // 2), dtype=t.uint8, device=self.device)
        ew = t.zeros((2, N, K // 32), dtype=t.int8, device=self.device)
        EW = t.zeros((2, N), dtype=t.int8, device=self.device)
        row_off = t.tensor([0, BM, BM], dtype=t.int32, device=self.device)
        live = row_off[1:2]                                   # a device int32 scalar: the r3-P3 row-count path too
        for planes in sorted({GATE_UP_PLANES, self.dplanes}):
            pl, am = self.pa.prep_activations(codes, ea, planes=planes, fold_groups=4)
            self.pa.prep_activations(codes, ea, planes=planes, fold_groups=4, rows=live, out=(pl, am))
            self.pa.expert(pl, am, ea, w, ew, EW, row_off, BM)
        t.cuda.synchronize(self.device)
        if self.pa.error_flag(reset=True) != 0:
            raise RuntimeError('path A prefill: flag word set during warm-up on zero inputs')

    # ------------------------------------------------------------------ the layer and the group
    def prep_layer(self, x, g):
        """Once per layer (inside the first group's graph): the contract quantiser of x (codes, group exponents) and the
        pinned prep kernel over all rows into the graph inputs of this row count."""
        self.kk.quant_codes_rows(x, g['patha_codes'], g['patha_ea'])
        self.ext.prep_activations(g['patha_codes'], g['patha_ea'], g['patha_planes'], g['patha_ameta'])

    def group(self, st, gi, n, x, weights, local, routes, payload, step):
        """group_body with the selector on: the same payload, group-local ids, weights and per-route output buffer."""
        t = self.torch
        from .moe import _adapter
        T, hidden = x.shape
        topk = local.shape[1]
        if step != min(T, self.step):
            raise RuntimeError('path A prefill: the row block changed (%d, workspace %d)' % (step, self.step))
        g = st._gin[T]
        ws = self.workspace(topk, st.G, hidden, st.N)
        if n > st.G or routes.shape != (T, topk, hidden) or routes.dtype != t.bfloat16:
            raise RuntimeError('path A prefill: group or route buffer outside the workspace geometry')
        if gi == 0:
            self.prep_layer(x, g)
        w13, s13, w2, s2 = payload
        gpu = _adapter(x.device)
        kk = self.kk
        ws['bad'].zero_()
        kk.weight_exponents(s13, ws['ew13'][:n], ws['EW13'][:n], ws['bad'])
        kk.weight_exponents(s2, ws['ew2'][:n], ws['EW2'][:n], ws['bad'])
        t._assert_async(ws['bad'] == 0, 'path A prefill: streamed expert scale code outside the finish-rule admission')
        max_rows = max_rows_bound(step, self.environ)
        for first in range(0, T, step):
            last = min(first + step, T)
            if last - first < 128:
                from .static_routes_candidate import expert_routes_static
                expert_routes_static(x[first:last], *payload, local[first:last], weights[first:last],
                                     out=routes[first:last], tile=16)
                continue
            self._block(st, ws, g, gpu, n, first, last, topk, local, weights, routes, w13, w2, max_rows)

    def _block(self, st, ws, g, gpu, n, first, last, topk, local, weights, routes, w13, w2, max_rows):
        t, kk = self.torch, self.kk
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        m = last - first
        count = m * topk
        hidden = routes.shape[2]
        active = ws['active'][:count].view(m, topk)
        lb, wb = local[first:last], weights[first:last]
        active.fill_(n)                                                         # inactive routes carry id n: sorted last
        t.where((lb >= 0) & (wb != 0), lb, active, out=active)
        sorted_ids, expert_ids, ntp = moe_align_block_size(active, BM, n + 1)   # the contract path's alignment, tile 64
        bound = min(max_rows, -(-m // BM) * BM, ws['cap'])                       # per-expert padded rows the grid covers
        ws['bad'].zero_()
        kk.row_offsets(expert_ids, ntp, ws['row_off'], ws['bad'], n, ws['cap'], bound)
        t._assert_async(ws['bad'] == 0, 'path A prefill: live rows beyond the workspace or an expert beyond the grid bound')
        live = ws['row_off'][n:n + 1]
        row_off = ws['row_off'][:n + 1]
        # gate/up: gather the prepared rows in sorted order, one grouped launch over the n experts
        kk.gather_rows(g['patha_planes'], g['patha_ameta'], g['patha_ea'], sorted_ids, live,
                       ws['P'], ws['AM'], ws['EAg'], first, count, topk)
        self.pa.expert(ws['P'], ws['AM'], ws['EAg'], w13, ws['ew13'][:n], ws['EW13'][:n], row_off, bound, out=ws['gu'])
        # the contract's SiLU / clamp / routing weight / BF16 (unchanged), then its quantiser and the kernel's prep,
        # all over the live rows only (the r3-P3 prep takes the device row count; the workspace beyond it is untouched)
        kk.activate_rows(ws['gu'], sorted_ids, live, wb.reshape(-1), gpu.silu, ws['h'], count)
        kk.quant_codes_rows(ws['h'], ws['hcodes'], ws['hea'], live)
        self.pa.prep_activations(ws['hcodes'], ws['hea'], planes=self.dplanes, fold_groups=4, rows=live,
                                 out=(ws['hplanes'], ws['hameta']))
        # down: the same grouped launch, then the real routes into the per-route buffer
        self.pa.expert(ws['hplanes'], ws['hameta'], ws['hea'], w2, ws['ew2'][:n], ws['EW2'][:n], row_off, bound, out=ws['y'])
        kk.scatter_rows(ws['y'], sorted_ids, live, routes[first:last].reshape(count, hidden), count)

    # ------------------------------------------------------------------ the flag protocol
    def flags(self):
        return int(self.pa.error_flag(reset=False))

    def check_flags(self, where=''):
        """Once per layer at the end of run(), outside any capture: synchronise, read the flag word, fail closed."""
        t = self.torch
        if t.cuda.is_current_stream_capturing():
            raise RuntimeError('path A prefill: check_flags inside a capture')
        t.cuda.current_stream(self.device).synchronize()
        f = self.flags()
        if f:
            names = [name for bit, name in ERR_BITS.items() if f & bit]
            raise RuntimeError('path A prefill: flag word %#x (%s)%s; layer output discarded' % (f, ', '.join(names), where))


# ----------------------------------------------------------------------------- entry points used by the patch

def patha_layer_buffers(st, alloc, rows, hidden, device):
    """graph_inputs (eager): construct the path once per streamer, return the per-layer buffers for this row count."""
    pa_st = getattr(st, '_patha', None)
    if pa_st is None:
        pa_st = PathAStream(device)
        st._patha = pa_st
    return pa_st.layer_buffers(alloc, rows, hidden)


def group_body_patha(st, gi, n, x, weights, local, routes, payload, step):
    pa_st = getattr(st, '_patha', None)
    if pa_st is None:
        raise RuntimeError('path A prefill: graph_inputs must run before group_body')
    return pa_st.group(st, gi, n, x, weights, local, routes, payload, step)


def check_flags(st):
    pa_st = getattr(st, '_patha', None)
    if pa_st is not None:
        pa_st.check_flags()
        if os.environ.get('KT_GPU_STREAM_PATHA_FLAGS_LOG', '0') == '1':
            checked = getattr(st, '_patha_checked_layers', 0) + 1
            st._patha_checked_layers = checked
            print('[patha-prefill] layer-check=%d flags=0' % checked, flush=True)
