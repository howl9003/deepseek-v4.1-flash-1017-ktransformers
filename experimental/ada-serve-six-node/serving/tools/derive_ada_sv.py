"""C2 serving with vision (4 Nov 2026; owner: "can we serve with vision please", "download original"): adasv = adasc3 (the
served C2 serving trial: radix cache, HiCache with the disk tier, four sessions) on the vision checkpoint view without
--language-model-only, vision tower on demand. Client: adasc3's serving timing unchanged, then the adac2v image test, so
the trial that would be served is validated with an image. To serve it: SERVED_TAG/SERVED_TRIAL in serve_ada_r1.py and
RUNS in swap_entry_ada_r1.py, with C2's values kept as the rollback."""
from pathlib import Path
import os, re
os.chdir(r'C:\Users\howl\Downloads\Claude\strata-deepseek1017\six-node-performance-r1\tools')
exec(Path(r'C:\Users\howl\AppData\Local\Temp\claude\C--Users-howl-Downloads-Claude-strata-deepseek1017\b00a77c4-432d-4001-9414-9702e1d66640\scratchpad\derive_ada_c2v.py').read_text(encoding='utf-8').split("def subs(")[0])  # VIEW, ARGV_*, VISION_FN


def subs(s, pairs):
    for a, b in pairs:
        assert s.count(a) == 1, (a[:90], s.count(a))
        s = s.replace(a, b)
    return s


def write(n, s):
    assert not Path(n).exists(), n
    compile(s, n, 'exec')
    Path(n).write_bytes(s.encode('utf-8').replace(b'\r\n', b'\n'))
    print('wrote', n, len(s.splitlines()))


table = [('ada-serve-c2-20261103-r3', 'ada-serve-c2-vision-20261104-r1'), ('ADASC3', 'ADASV'), ('adasc3', 'adasv')]


def load(n):
    s = Path(n).read_text(encoding='utf-8')
    for a, b in table:
        s = s.replace(a, b)
    return s


write('stage_phase_g_g1v_adasv_r1.py', subs(load('stage_phase_g_g1v_adasc3_r1.py'), [(ARGV_OLD, ARGV_NEW)]))
for kind in ('profile', 'launch', 'dispatch', 'controller'):
    write(f'{kind}_phase_g_g1v_adasv_r1.py', load(f'{kind}_phase_g_g1v_adasc3_r1.py'))
client = load('client_ada_adasc3_r1.py')
vision = VISION_FN.replace('    save(out / ', '    write(out / ')
client = subs(client, [
    ("        result['tool_call'] = tool_call()\n        result.update(status='ADA_SERVE_TIMED_REVIEW_REQUIRED')\n",
     "        result['tool_call'] = tool_call()\n"
     "        result['vision'] = vision_test(out, args.trial / ('model-' + args.namespace) / 'server/launcher.log')\n"
     "        result.update(status='ADA_SERVE_TIMED_REVIEW_REQUIRED')\n"),
    ("\n\ndef main():\n", vision + "\n\ndef main():\n"),
    ('"""Serving profile timing (Claude, 2 Nov 2026)', '"""C2 serving with vision (adasv, 4 Nov 2026): the adasc3 serving client unchanged, then the adac2v image test. Earlier text: Serving profile timing (Claude, 2 Nov 2026)')])
write('client_ada_adasv_r1.py', client)
