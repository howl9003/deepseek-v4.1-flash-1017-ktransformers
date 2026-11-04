"""Checks over the six test_registration.py snapshots. argv: their folder. Writes REGISTRATION_RESULT.json there."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
S = {p.stem.split('_')[0]: json.loads(p.read_text()) for p in D.glob('*.json') if p.stem[:2] in ('A_', 'B_', 'C_', 'D_', 'E_', 'F_')}
VOLATILE = ('tree', 'selector')


def core(s, keys=None):
    return {k: v for k, v in s.items() if k not in VOLATILE and (keys is None or k in keys)}


TARGET_KEYS = ('KT_LAYERS', 'hot_LAYERS', 'hot_STATE_keys', 'targets', 'class_loader')
checks = []


def check(name, ok, detail=''):
    checks.append(dict(check=name, ok=bool(ok), detail=detail))


A, B, C, Dd, E, F = (S[k] for k in 'ABCDEF')
check('selector off, target alone: overlay == served', core(C) == core(A))
check('selector off, target + draft: overlay == served (the served behaviour, bugs included)', core(Dd) == core(B))
check('selector on, target alone: overlay == served', core(E) == core(A))
check('selector on, target + draft: target registry, layer ids, hot-lend list, target configs == target alone',
      core(F, TARGET_KEYS) == core(A, TARGET_KEYS))
tc = lambda s: [c for c in s['kt_constructions'] if c[0].startswith('target')]
check('selector on: target kt-kernel constructions and load maps == target alone', tc(F) == tc(A))
check('selector on: target split calls == target alone', [c for c in F['split_calls'] if c.startswith('target')] == A['split_calls'])
check('selector on: draft registry is KT layers 40-42 only', F['KT_DRAFT_LAYERS'] == {'40': 'draft0', '41': 'draft1', '42': 'draft2'}
      and not any(v.startswith('draft') for v in F['KT_LAYERS'].values()))
dr = [F['drafts'][f'draft{s}'] for s in range(3)]
check('selector on: draft wrappers 128 experts, top-3, all on CPU, draft GGUF, own loader, base key blk.s, identity map',
      all(d['wrapper']['kw']['layer_idx'] == 40 + s and d['wrapper']['kw']['num_experts'] == 128 and d['wrapper']['kw']['num_experts_per_tok'] == 3
          and d['wrapper']['kw']['gpu_mask_true'] == 0 and d['num_gpu_experts'] == 0 and d['gpu_rows'] == [0]
          and d['wrapper']['kw']['weight_path'].endswith('DSpark-MXFP4.gguf') and d['wrapper']['own_loader_at_init']
          and d['wrapper']['weight_base_key'] == f'model.layers.{s}' and d['wrapper']['load_map_identity']
          and d['wrapper']['load_map_len'] == 128 and not d['shadow'] and d['hot_layer'] is None and not d['cpu_holds_all']
          and d['cfg']['num_layers'] is None for s, d in enumerate(dr)))
check('selector on: no hot-lend entry, no class-level loader touched', len(F['hot_LAYERS']) == 40 and not F['class_loader']
      and all(not c[1]['own_loader_at_init'] for c in tc(F)))
check('selector on: draft weight loader keeps no routed expert on the GPU',
      F['draft_weight_loader'] == {'gpu_rows': 0, 'result': {'0': 'skipped', '1': 'skipped', '64': 'skipped', '127': 'skipped'}})
check('selector on: adapter sends draft layers to kt_dspark_draft.apply, target layers down the served path',
      F['adapter_paths']['draft0'].startswith('kt_dspark_draft.apply(layer 40)') and F['adapter_paths']['target0'].startswith('target path'))
check('selector on: draft handoff buffers [T, 3], kt-kernel buffers untouched',
      F['draft_host_buffers']['ids_shape'] == [4, 3] and F['draft_host_buffers']['kt_buffers_untouched'])
# what the served tree does with a draft (B): the bugs this overlay fixes
bugs = dict(
    KT_LAYERS_0_2=[B['KT_LAYERS'][str(i)] for i in range(3)],
    hot_LAYERS=len(B['hot_LAYERS']),
    hot_draft_entries=[(L['idx'], L['layer_idx'], L['wrapper'], L['e']) for L in B['hot_LAYERS'] if L['wrapper'].startswith('draft')],
    draft_kt=[dict(layer_idx=c[1]['kw']['layer_idx'], num_experts=c[1]['kw']['num_experts'], weight_path=c[1]['kw']['weight_path'],
                   load_map_identity=c[1].get('load_map_identity'), load_map_len=c[1].get('load_map_len'),
                   same_map_as_target=c[1].get('load_map_sha') == next(t[1].get('load_map_sha') for t in B['kt_constructions'] if t[1]['kw']['layer_idx'] == c[1]['kw']['layer_idx'] and t[0].startswith('target')))
              for c in B['kt_constructions'] if c[0].startswith('draft')],
    draft_weight_loader=B['draft_weight_loader'],
    adapter_paths=B['adapter_paths'])
result = dict(status='PASS' if all(c['ok'] for c in checks) else 'FAIL', checks=checks, served_with_draft=bugs)
(D / 'REGISTRATION_RESULT.json').write_text(json.dumps(result, indent=1) + '\n')
for c in checks:
    print('PASS' if c['ok'] else 'FAIL', c['check'])
print(json.dumps(bugs, indent=1))
print(result['status'])
