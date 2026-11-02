"""Overlay kt_kv_hot_lend.py: KT_KV_HOT_LAYER_QUOTA shares the slots between layers in proportion, default off. argv: source, destination."""
import sys
from pathlib import Path

s = Path(sys.argv[1]).read_text(encoding='utf-8')
PAIRS = [
    ('def _plan(budget: int) -> List[int]:\n'
     '    """Slots per layer within budget bytes: one slot at a time to the layer with\n'
     '    the fewest, while the next one\'s chunks fit."""\n'
     '    ks = [0] * len(LAYERS)\n',
     'def _quota() -> List[int]:\n'
     '    """KT_KV_HOT_LAYER_QUOTA=<json file>: one positive number a decoder layer, the share of the slots it gets.\n'
     '    Which experts hold them stays the dynamic rule\'s. Unset: one each, the even split."""\n'
     '    path = os.environ.get("KT_KV_HOT_LAYER_QUOTA", "")\n'
     '    if not path:\n'
     '        return [1] * len(LAYERS)\n'
     '    with open(path) as f:\n'
     '        q = json.load(f)\n'
     '    assert len(q) == len(LAYERS) and all(isinstance(x, int) and x > 0 for x in q), "layer quota"\n'
     '    return [q[L.idx] for L in LAYERS]\n'
     '\n\n'
     'def _plan(budget: int) -> List[int]:\n'
     '    """Slots per layer within budget bytes: one slot at a time to the layer with\n'
     '    the fewest for its quota, while the next one\'s chunks fit."""\n'
     '    ks = [0] * len(LAYERS)\n'
     '    q = _quota()\n'),
    ('            if best is None or ks[i] < ks[best]:\n',
     '            if best is None or ks[i] * q[best] < ks[best] * q[i]:\n'),
    ('            cheaper = [i for i, M in enumerate(LAYERS) if ks[i] < M.x and ks[i] == ks[best]\n',
     '            cheaper = [i for i, M in enumerate(LAYERS) if ks[i] < M.x and ks[i] * q[best] == ks[best] * q[i]\n'),
]
for old, new in PAIRS:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
if '\nimport json\n' not in s:
    anchor = '\nimport os\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, '\nimport json\nimport os\n')
compile(s, 'kt_kv_hot_lend.py', 'exec')
Path(sys.argv[2]).write_bytes(s.encode('utf-8'))
