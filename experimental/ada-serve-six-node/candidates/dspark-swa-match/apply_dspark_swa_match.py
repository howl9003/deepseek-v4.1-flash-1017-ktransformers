"""Overlay: a DSpark draft pool holds every SWA slot the shared allocator hands out (Claude, 4 Nov 2026). Selector
KT_DSPARK_SWA_MATCH, default off.

With the SWA lend (KT_PREFILL_SWA_LEND=1) the KV cache configurator widens the target's SWA pool (7,680 -> 77,312 tokens on the
C2 profile) but skips the draft worker (`... and not self.is_draft_worker`). The DSpark draft shares the target's allocator and
writes its SWA KV at `full_to_swa_index_mapping[cache_loc]`, so once a prompt's SWA slots pass 7,680 the draft writes past its own
buffers: the illegal address after any prompt over about 7K tokens (adadsi, adadsj, adadsw, adadsy, adadsx; the log shows
"swa_size=77312" for the target pool and "swa_size=7680" for the draft's). With the selector on, the draft's SWA pool takes the
target's widened size; its buffers are made after the lend is set up, so they are fully mapped (about 1.5 KiB a token for the
draft's three layers, about 110 MB at 77,312 tokens). Sizing only: no arithmetic changes.

argv: sglang package, patched in place."""
import hashlib
import sys
from pathlib import Path

PINS = {'srt/layers/moe/kt_kv_prefill_lend.py': '281ce3141d6075aab77a2ff6d2a98520902418f32ae1bf3aacd9e634c4f33d64',
        'srt/mem_cache/kv_cache_configurator.py': '6d5db5f0c2fa6c7e4882b108bce450b60caedc7d755b0ccb149611cacdb742d4'}
root = Path(sys.argv[1])
text = {}
for rel, pin in PINS.items():
    s = (root / rel).read_bytes().decode('utf-8')
    assert hashlib.sha256(s.encode('utf-8')).hexdigest() == pin, rel + ' differs'
    text[rel] = s


def sub(rel, old, new):
    assert text[rel].count(old) == 1, (rel, text[rel].count(old), old[:80])
    text[rel] = text[rel].replace(old, new)


lend = 'srt/layers/moe/kt_kv_prefill_lend.py'
sub(lend, "    _lazy_claim(want, page, follow)\n    return want, follow\n",
    "    _lazy_claim(want, page, follow)\n"
    "    SWA[\"widened\"] = want   # KT_DSPARK_SWA_MATCH: a draft pool on the same allocator takes this size\n"
    "    return want, follow\n")
sub(lend, "def swa_tokens(size: int) -> int:\n",
    "def swa_size_for_draft(size: int) -> int:\n"
    "    \"\"\"KT_DSPARK_SWA_MATCH=1: a draft pool that shares the target's allocator must hold every SWA slot id the\n"
    "    widened pool hands out; its buffers are made after swa_init, so they are fully mapped.\"\"\"\n"
    "    return max(int(size), int(SWA.get(\"widened\") or 0))\n"
    "\n"
    "\n"
    "def swa_tokens(size: int) -> int:\n")
conf = 'srt/mem_cache/kv_cache_configurator.py'
sub(conf, "        if swa_max_total_num_tokens is not None and not self.is_draft_worker:\n",
    "        if (\n"
    "            swa_max_total_num_tokens is not None\n"
    "            and self.is_draft_worker\n"
    "            and os.environ.get(\"KT_DSPARK_SWA_MATCH\", \"0\") == \"1\"\n"
    "        ):\n"
    "            # KT_DSPARK_SWA_MATCH (dspark-cpu-r1): the draft shares the target's SWA allocator, so its SWA pool takes\n"
    "            # the size the SWA lend widened the target's to, or the draft writes past its buffers.\n"
    "            from sglang.srt.layers.moe import kt_kv_prefill_lend\n"
    "\n"
    "            swa_max_total_num_tokens = kt_kv_prefill_lend.swa_size_for_draft(swa_max_total_num_tokens)\n"
    "        if swa_max_total_num_tokens is not None and not self.is_draft_worker:\n")
if '\nimport os\n' not in text[conf]:
    first = text[conf].index('\nimport ')
    text[conf] = text[conf][:first] + '\nimport os' + text[conf][first:]
for rel, s in text.items():
    compile(s, rel, 'exec')
    (root / rel).write_bytes(s.encode('utf-8'))
    print(rel, hashlib.sha256(s.encode('utf-8')).hexdigest())
