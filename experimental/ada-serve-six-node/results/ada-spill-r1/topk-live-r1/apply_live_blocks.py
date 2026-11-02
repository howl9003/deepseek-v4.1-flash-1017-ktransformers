"""Makes the live among-blocks kernel drop live -inf logits, as topk_among_blocks does, in tree r6's live_topk.py (Claude, 1 Nov 2026). argv: source, destination, base sha."""
import hashlib
import sys
from pathlib import Path

src, dst, base = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = src.read_bytes()
assert hashlib.sha256(data).hexdigest() == base, 'live_topk differs from tree r6'
text = data.decode('utf-8')
HELPER = '''@triton.jit
def _emit_finite(scores_ptr, slots_ptr, out_ptr, b, n, K, stride_sc, stride_sl, stride_o, BLOCK: tl.constexpr):
    # n <= K among blocks: every live position but the -inf ones topk_among_blocks drops.
    offs0 = tl.arange(0, BLOCK)
    sel_carry = 0 * n
    for start in range(0, n, BLOCK):
        j = start + offs0
        m = j < n
        v = tl.load(scores_ptr + b * stride_sc + j, mask=m, other=0.0)
        sel = m & (v != float("-inf"))
        seli = sel.to(tl.int32)
        dst = sel_carry + tl.cumsum(seli, 0) - seli
        s = tl.load(slots_ptr + b * stride_sl + j, mask=sel, other=0)
        tl.store(out_ptr + b * stride_o + dst, s.to(tl.int32), mask=sel)
        sel_carry += tl.sum(seli, 0)
    for start in range(0, K, BLOCK):
        j = start + offs0
        tl.store(out_ptr + b * stride_o + j, tl.full([BLOCK], -1, tl.int32),
                 mask=(j >= sel_carry) & (j < K))


'''
PAIRS = [
    ('@triton.jit\ndef _live_topk_full_kernel(\n', HELPER + '@triton.jit\ndef _live_topk_full_kernel(\n'),
    ('    if all_live_published:\n        _emit_sequential(slots_ptr, out_ptr, b, n, K, stride_sl, stride_o, BLOCK)\n',
     '    if all_live_published:\n        _emit_finite(scores_ptr, slots_ptr, out_ptr, b, n, K, stride_sc, stride_sl, stride_o, BLOCK)\n'),
    ('                v = tl.load(scores_ptr + b * stride_sc + pos, mask=m, other=0.0)\n                key = _ord_key(v)\n',
     '                v = tl.load(scores_ptr + b * stride_sc + pos, mask=m, other=0.0)\n                m = m & (v != float("-inf"))\n                key = _ord_key(v)\n'),
    ('            v = tl.load(scores_ptr + b * stride_sc + pos, mask=m, other=0.0)\n            kp = _ord_key(v) >> TOPSHIFT\n',
     '            v = tl.load(scores_ptr + b * stride_sc + pos, mask=m, other=0.0)\n            m = m & (v != float("-inf"))\n            kp = _ord_key(v) >> TOPSHIFT\n'),
    ('  ties in flat (block-rank) order, marks the picks, and then emits in position\n  order like the full path.',
     '  ties in flat (block-rank) order, marks the picks, and then emits in position\n  order like the full path. Live -inf logits are dropped, as topk_among_blocks does.'),
]
for old, new in PAIRS:
    assert text.count(old) == 1, old
    text = text.replace(old, new)
compile(text, str(dst), 'exec')
dst.write_bytes(text.encode('utf-8'))
print(hashlib.sha256(text.encode('utf-8')).hexdigest())
