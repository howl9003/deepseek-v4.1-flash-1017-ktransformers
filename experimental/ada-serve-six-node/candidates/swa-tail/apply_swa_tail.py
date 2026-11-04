"""05yuki's tier 1 patch 23, swa-checkpoint-tail (upstream 3e1fea9, fork commit ef268a9337), as an overlay on our tree
(Claude, 3 Nov 2026; owner: "Yes, both").

With decoder SWA bounded replay a prefix match stops swa_reprefill_tail_tokens (128) short of the key's end and needs
live SWA in the window before that point. Checkpointing a running request into the tree (end of its prefill) freed the
out-of-window SWA measured from the key's end, so a prompt sent again after an answer of 2 or more tokens fell back to
the previous chunk boundary (32,625 tokens: 16,384 read from the cache instead of 32,256). The free now starts one
window earlier; a checkpoint keeps one more page of SWA. Cache bookkeeping only: no arithmetic changes.

Both call sites of our unified_radix_cache.py match the patch's context; the file is pinned. argv: sglang package,
patched in place."""
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1]) / 'srt/mem_cache/unified_radix_cache.py'
s = p.read_bytes().decode('utf-8')
assert hashlib.sha256(s.encode('utf-8')).hexdigest() == '8280591e1c825d7c29c2e227fa07e361a210715bbcfa7f089049c9dd98078540', 'unified_radix_cache.py differs'
old = '                comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)\n'
new = ('                # A later match of this key stops swa_reprefill_tail_tokens short of its\n'
       '                # end; keep the window before that point live too, or the match falls\n'
       '                # back to the previous checkpoint.\n'
       '                comp.free_out_of_window_slots(\n'
       '                    req,\n'
       '                    len(radix_key) - 1 - self.swa_reprefill_tail_tokens(req),\n'
       '                    insert_params,\n'
       '                )\n')
assert s.count(old) == 2, s.count(old)
s = s.replace(old, new)
compile(s, str(p), 'exec')
p.write_bytes(s.encode('utf-8'))
print(hashlib.sha256(s.encode('utf-8')).hexdigest())
