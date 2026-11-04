"""Overlay: DSpark decode results carry the path A error flags (Claude, 3 Nov 2026). Applied after apply_dspark_cpu.py.

With SGLANG_FP8_PATHA_EXPERT_STATIC=1 the scheduler refuses any decode result without `patha_static_flags`
(`patha_static_runtime.check_result`). The target worker takes that snapshot after its verify forward, but DSpark's
`_forward_decode` builds a new result for the scheduler and drops it, so the first decode under DSpark was refused
(adadsi, 3 Nov: "C1 result is missing its error snapshot; output discarded"). This takes the snapshot after the whole
speculative step, verify and draft, with the same call the target worker makes. The check itself is unchanged; prefill
already returns the target's own result. argv: sglang package, patched in place."""
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1]) / 'srt/speculative/dspark_components/dspark_worker_v2.py'
s = p.read_bytes().decode('utf-8')
assert hashlib.sha256(s.encode('utf-8')).hexdigest() == 'a78fb549cce41534f4e2b2a8e93c92995de217e4ff8e48dc707f68bf8bd28627', 'dspark_worker_v2.py differs'


def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


sub('import logging\n', 'import logging\nimport os\n')
sub('        return GenerationBatchResult(\n            logits_output=logits_output,\n            next_token_ids=accept.out_tokens.reshape(-1),\n',
    '        patha_static_flags = None\n'
    '        if os.environ.get("SGLANG_FP8_PATHA_EXPERT_STATIC", "0") == "1":\n'
    '            # The scheduler refuses a decode result without the path A error flags; snapshot after the whole step.\n'
    '            from sglang.srt.layers.moe.contract_v2.patha_static_runtime import snapshot_after_forward\n'
    '\n'
    '            patha_static_flags = snapshot_after_forward()\n'
    '        return GenerationBatchResult(\n'
    '            patha_static_flags=patha_static_flags,\n'
    '            logits_output=logits_output,\n'
    '            next_token_ids=accept.out_tokens.reshape(-1),\n')
compile(s, str(p), 'exec')
p.write_bytes(s.encode('utf-8'))
print(hashlib.sha256(s.encode('utf-8')).hexdigest())
