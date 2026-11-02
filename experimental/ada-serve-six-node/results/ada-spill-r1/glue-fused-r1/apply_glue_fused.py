"""Adds the KT_CONTRACT_V2_GLUE_FUSED selector to tree r6's contract_v2/adapter.py (Claude, 31 Oct 2026). argv: source, destination."""
import hashlib
import sys
from pathlib import Path

BASE = '8037ce074a7b3531ee04d641762b29113d051c83eea1853a98bab6795d82f3af'
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
data = src.read_bytes()
assert hashlib.sha256(data).hexdigest() == BASE, 'adapter differs from tree r6'
text = data.decode('utf-8')
PAIRS = [
    ('_TP2_EXACT = os.environ.get("SGLANG_PATHA_TP2_EXACT", "0") == "1"\n',
     '_TP2_EXACT = os.environ.get("SGLANG_PATHA_TP2_EXACT", "0") == "1"\n'
     '_GLUE_FUSED = os.environ.get("KT_CONTRACT_V2_GLUE_FUSED", "0") == "1"\n'),
    ('        tp2.check_stream(streamer)\n    hot = kt_kv_hot_lend.route(wrapper, dispatch) if streamer is None else None\n',
     '        tp2.check_stream(streamer)\n'
     '    if _GLUE_FUSED and streamer is None and not _TP2_EXACT:\n'
     '        from .glue_fused import apply as fused\n'
     '        output = fused(wrapper, layer, dispatch, combine_order)\n'
     '        if output is not None:\n'
     '            return output\n'
     '    hot = kt_kv_hot_lend.route(wrapper, dispatch) if streamer is None else None\n'),
]
for old, new in PAIRS:
    assert text.count(old) == 1, old
    text = text.replace(old, new)
compile(text, str(dst), 'exec')
dst.write_bytes(text.encode('utf-8'))
print(hashlib.sha256(text.encode('utf-8')).hexdigest())
