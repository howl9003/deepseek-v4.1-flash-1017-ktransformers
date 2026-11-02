"""Overlay kt_kv_prefill_lend.py with 05yuki's 7fef854e behind a selector: the lend reaches every pool tensor laid out by the allocator's pages. argv: source, destination."""
import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text(encoding='utf-8')
PAIRS = [
    ('CHECK = os.environ.get("KT_PREFILL_KV_LEND_CHECK", "0") == "1"\n',
     '# KT_PREFILL_KV_LEND_ALL_TENSORS=1: every pool tensor laid out by the allocator\'s pages lends, the compress\n'
     '# ratios\' and the indexers\' smaller pages included, not only a row or a token\'s rows a page.\n'
     'ALL_TENSORS = os.environ.get("KT_PREFILL_KV_LEND_ALL_TENSORS", "0") == "1"\n'
     'CHECK = os.environ.get("KT_PREFILL_KV_LEND_CHECK", "0") == "1"\n'),
    ('    (n + 1) * page_size rows of a token, on kv_vmm memory; [] when there are none.\n'
     '    A buffer of any other length (another page space) is left alone. extra:\n',
     '    (n + 1) * page_size rows of a token, on kv_vmm memory; [] when there are none.\n'
     '    With ALL_TENSORS: k rows a page from row 0 for the n + 1 pages and up to k + 1 rows of\n'
     '    padding after them. A buffer of any other length (another page space) is left alone. extra:\n'),
    ('        if t.is_cuda and t.is_contiguous() and t.dim() >= 1 and t.shape[0] in (n + 1, (n + 1) * ps):\n'
     '            tensors.append((t.view(-1).view(torch.uint8), t.nbytes // (n + 1)))\n',
     '        rows = t.shape[0] if t.dim() >= 1 else 0\n'
     '        k = rows // (n + 1)\n'
     '        paged = k >= 1 and rows - k * (n + 1) <= k + 1 if ALL_TENSORS else rows in (n + 1, (n + 1) * ps)\n'
     '        if t.is_cuda and t.is_contiguous() and paged:\n'
     '            tensors.append((t.view(-1).view(torch.uint8), t.nbytes // rows * k))\n'),
]
for old, new in PAIRS:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
compile(s, 'kt_kv_prefill_lend.py', 'exec')
Path(sys.argv[2]).write_bytes(s.encode('utf-8'))
