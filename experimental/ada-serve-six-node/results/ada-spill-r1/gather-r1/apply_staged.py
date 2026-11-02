"""Overlay sm89_dual_scope.py: SGLANG_SM89_ATTN_STAGED=1 sends small unsplit calls to the staged gather, default off. argv: source, destination."""
import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text(encoding='utf-8')
PAIRS = [
    ('_MAX_ROWS = int(os.environ.get("SGLANG_SM89_ATTN_ROWS", "1024"))\n',
     '_MAX_ROWS = int(os.environ.get("SGLANG_SM89_ATTN_ROWS", "1024"))\n'
     '# SGLANG_SM89_ATTN_STAGED=1: up to this many rows, the unsplit gather runs as three staged kernels on the whole card, the same bits.\n'
     '_STAGED_ROWS = 8 if os.environ.get("SGLANG_SM89_ATTN_STAGED", "0") == "1" else 0\n'),
    ('    split_k = _split_k(rows, h_q, topk_main + topk_extra)\n    if split_k == 0:\n',
     '    split_k = _split_k(rows, h_q, topk_main + topk_extra)\n'
     '    if split_k == 0 and rows <= _STAGED_ROWS and h_q <= 64:\n'
     '        from sglang.kernels.ops.attention import sm89_gather_staged\n'
     '\n'
     '        return sm89_gather_staged.decode(\n'
     '            q3,\n'
     '            k_cache,\n'
     '            idx_main.contiguous(),\n'
     '            topk_length,\n'
     '            attn_sink,\n'
     '            softmax_scale,\n'
     '            extra_k_cache if has_extra else None,\n'
     '            idx_extra.contiguous() if has_extra else None,\n'
     '            extra_topk_length if has_extra else None,\n'
     '        ).unsqueeze(1)\n'
     '    if split_k == 0:\n'),
]
for old, new in PAIRS:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
compile(s, 'sm89_dual_scope.py', 'exec')
Path(sys.argv[2]).write_bytes(s.encode('utf-8'))
