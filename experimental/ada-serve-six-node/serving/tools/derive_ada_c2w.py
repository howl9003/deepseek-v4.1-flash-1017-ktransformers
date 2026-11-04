"""C2 vision identity r2 (4 Nov 2026): adac2w = adac2v with one client change. adac2v was exact vs C2 on the 11,496
scores, both replays, the 4K case and rotation, then differed on 35k-first, whose random prompt (ids 1000..129279)
holds the image placeholder token 129264 once at position 8519 (the 4K and 35k-second prompts hold none). With the
vision tower on, a bare image token is read as an image, so that case cannot match the text-only C2 archive by design.
A case whose prompt holds the model's image_token_id is now recorded as skipped, with the position; at least one 35K
case must still be compared and exact."""
from pathlib import Path
import os
os.chdir(r'C:\Users\howl\Downloads\Claude\strata-deepseek1017\six-node-performance-r1\tools')


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


table = [('ada-identity-c2-vision-20261104-r1', 'ada-identity-c2-vision-20261104-r2'), ('ADAC2V', 'ADAC2W'), ('adac2v', 'adac2w')]


def load(n):
    s = Path(n).read_text(encoding='utf-8')
    for a, b in table:
        s = s.replace(a, b)
    return s


for kind in ('stage', 'profile', 'launch', 'dispatch', 'controller'):
    write(f'{kind}_phase_g_g1v_adac2w_r1.py', load(f'{kind}_phase_g_g1v_adac2v_r1.py'))
write('client_ada_adac2w_r1.py', subs(load('client_ada_adac2v_r1.py'), [
    ("        for label, length, seed in CASES:\n            body = body_for(length, seed)\n",
     "        image_token = json.loads((Path(profile['server_args']['model_path']) / 'config.json').read_text())['image_token_id']\n"
     "        for label, length, seed in CASES:\n            body = body_for(length, seed)\n"
     "            if image_token in body['input_ids']:\n"
     "                # adac2w: with the vision tower on, a bare image placeholder in a random prompt is read as an image.\n"
     "                result['cases'].append(dict(case=label, prompt_tokens=length, skipped='holds the image token %d at position %d'\n"
     "                                            % (image_token, body['input_ids'].index(image_token))))\n"
     "                save(out / 'RESULT.json', result)\n"
     "                continue\n"),
    ("        assert all(v is True for k, v in result['rotation'].items() if k not in ('numeric', 'swaps_logged')), 'rotation changed the result'\n",
     "        assert any(c['case'].startswith('35k') and 'gate' in c for c in result['cases']), 'no 35K case compared'\n"
     "        assert all(v is True for k, v in result['rotation'].items() if k not in ('numeric', 'swaps_logged')), 'rotation changed the result'\n"),
    ('"""C2 vision identity client (adac2w, Claude, 4 Nov 2026):', '"""C2 vision identity client r2 (adac2w, 4 Nov 2026): adac2v\'s client, skipping a case whose prompt holds the image token. adac2v text: C2 vision identity client (adac2v, Claude, 4 Nov 2026):')]))
