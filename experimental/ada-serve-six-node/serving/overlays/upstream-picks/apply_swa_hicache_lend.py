"""The two SWA-lend parts of 05yuki's tier 1 patch 21 (upstream cef35da, hicache) as an overlay on the served tree
ada-serve-c2-20261103-r3 (Claude, 4 Nov 2026; upstream-picks-r2). Both only act with KT_PREFILL_SWA_LEND=1 and
HiCache attached; each has its own default-off switch.

KT_PREFILL_SWA_LOAD_MAP=1: a load-back maps its SWA pages before the copy. The SWA lend unmaps the free SWA pages
beyond KT_PREFILL_SWA_LEND_KEEP outside prefill; an allocation that needs lent pages gets them at once in the free
list, but their chunks are mapped at the next forward's start (swa_map(defer=True), flushed by
kt_lend.before_forward). A HiCache load-back allocates its SWA slots in the scheduler (init_load_back, during
get_new_batch_prefill) and starts the host-to-device copy there too (ready_to_load_host_cache -> start_loading),
before that forward, into memory not mapped yet. His third load-back faulted both ranks. bind_tree now wraps the
cache controller's start_loading to flush the pending SWA maps first. Ours has the same order: scheduler.py calls
ready_to_load_host_cache before prepare_for_extend and the forward.

KT_PREFILL_SWA_HOST_TRIM=1: with HiCache attached the SWA trim keeps KT_PREFILL_SWA_LEND_KEEP tokens outside prefill
instead of the pool's budget (ours 11,520 tokens: the 7,424-token pool at the 2,048 sizing chunk plus SAFE - 2,048).
What the cache gives back is in host memory where HiCache backed it up. His default is on; ours is off.

Nothing else of patch 21 is taken here: the staging cap is apply_hicache_staging_cap.py (memory_pool_host.py, a
different file, so these scripts do not stack), and the first-launch calibration change is his serve.sh, a
launcher step our stage does not have.

argv: the sglang package directory; srt/layers/moe/kt_kv_prefill_lend.py is patched in place."""
import hashlib
import sys
from pathlib import Path

REL = 'srt/layers/moe/kt_kv_prefill_lend.py'
PIN = '281ce3141d6075aab77a2ff6d2a98520902418f32ae1bf3aacd9e634c4f33d64'


def sub(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:80])
    return s.replace(old, new)


def patch(s):
    old = 'SWA = {"alloc": None, "tensors": None, "lent": None, "bytes": 0, "lends": 0, "maps": 0, "ticks": 0}\n'
    s = sub(s, old, old +
            '# KT_PREFILL_SWA_LOAD_MAP=1 (05yuki tier 1 patch 21, upstream cef35da; overlay upstream-picks-r2, default\n'
            '# off): a HiCache load-back allocates its SWA pages in the scheduler and starts the host-to-device copy\n'
            '# before the next forward, where swa_map(defer=True) maps the lent pages it handed out: the cache\n'
            '# controller\'s start_loading maps them first (_wrap_hicache_load). Needed with HiCache and the SWA lend.\n'
            'SWA_LOAD_MAP = os.environ.get("KT_PREFILL_SWA_LOAD_MAP", "0") == "1"\n'
            '# KT_PREFILL_SWA_HOST_TRIM=1 (the same patch; his default on, ours off): with HiCache attached the SWA trim\n'
            '# keeps SWA_KEEP tokens outside prefill instead of the pool\'s budget; what the cache gives back is backed\n'
            '# up in host memory and loaded on the next match\n'
            'SWA_HOST_TRIM = os.environ.get("KT_PREFILL_SWA_HOST_TRIM", "0") == "1"\n')
    old = ('def bind_tree(scheduler) -> None:\n'
           '    """kt_prefill_plan, from the scheduler: its prefix cache, for swa_trim."""\n'
           '    if SWA_ENABLED and "tree" not in SWA:\n'
           '        SWA["tree"] = getattr(scheduler, "tree_cache", None)\n')
    new = ('def bind_tree(scheduler) -> None:\n'
           '    """kt_prefill_plan, from the scheduler: its prefix cache, for swa_trim (and, with KT_PREFILL_SWA_LOAD_MAP,\n'
           '    its HiCache controller, for _wrap_hicache_load)."""\n'
           '    if SWA_ENABLED and "tree" not in SWA:\n'
           '        SWA["tree"] = getattr(scheduler, "tree_cache", None)\n'
           '        if SWA_LOAD_MAP:\n'
           '            _wrap_hicache_load(SWA["tree"])\n'
           '\n'
           '\n'
           'def _wrap_hicache_load(tree) -> None:\n'
           '    """A HiCache load-back allocates its SWA slots in the scheduler and starts the host-to-device copy before\n'
           '    the next forward, which is where swa_map(defer=True) maps the pages it handed back: map them before the\n'
           '    copy starts."""\n'
           '    cc = getattr(tree, "cache_controller", None)\n'
           '    if cc is None or getattr(cc, "_kt_swa_load", False) or not hasattr(cc, "start_loading"):\n'
           '        return\n'
           '    orig = cc.start_loading\n'
           '\n'
           '    def start_loading(*args, **kw):\n'
           '        swa_flush("HiCache load", quiet=SWA["maps"] > 3 and not CHECK)\n'
           '        return orig(*args, **kw)\n'
           '\n'
           '    cc.start_loading = start_loading\n'
           '    cc._kt_swa_load = True\n'
           '    _log(f"SWA lend: {type(cc).__name__}.start_loading maps the SWA pages it loads into first")\n')
    s = sub(s, old, new)
    old = ('    a, tree, budget = SWA["alloc"], SWA.get("tree"), SWA.get("budget")\n'
           '    if a is None or tree is None or not budget:\n'
           '        return\n'
           '    over = _swa_in_use(a) - budget\n')
    new = ('    a, tree, budget = SWA["alloc"], SWA.get("tree"), SWA.get("budget")\n'
           '    if a is None or tree is None or not budget:\n'
           '        return\n'
           '    if SWA_HOST_TRIM and getattr(tree, "cache_controller", None) is not None:\n'
           '        # HiCache backs up what the cache gives back and loads it on the next match:\n'
           '        # keep only what the SWA lend keeps, the rest goes to the lends\n'
           '        if not SWA.get("host_trim_logged"):\n'
           '            SWA["host_trim_logged"] = True\n'
           '            _log(f"SWA trim: HiCache backs the cache in host memory, {SWA_KEEP} tokens kept outside prefill "\n'
           '                 f"(not the {budget}-token budget)")\n'
           '        budget = SWA_KEEP\n'
           '    over = _swa_in_use(a) - budget\n')
    return sub(s, old, new)


def main(root):
    p = Path(root) / REL
    s = p.read_bytes().decode('utf-8')
    assert hashlib.sha256(s.encode('utf-8')).hexdigest() == PIN, 'kt_kv_prefill_lend.py differs from the served pin'
    s = patch(s)
    compile(s, str(p), 'exec')
    p.write_bytes(s.encode('utf-8'))
    print(hashlib.sha256(s.encode('utf-8')).hexdigest(), REL)


if __name__ == '__main__':
    main(sys.argv[1])
