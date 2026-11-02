"""_plan with the quota against the accepted _plan: no quota equals it on every budget and cost shape; a quota shares in proportion. Runs the two functions out of the two files, no torch."""
import ast, os, random, sys, tempfile, json
from typing import List

OLD, NEW = sys.argv[1:3]


def pick(path, names):
    tree = ast.parse(open(path, encoding='utf-8').read())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    return compile(ast.Module(nodes, []), path, 'exec')


class L:
    def __init__(self, idx, x, n):
        self.idx, self.x, self.n = idx, x, n


def run(path, names, layers, budget, chunk, q=None):
    env = dict(List=List, os=os, json=json, LAYERS=layers, SLOTS_MAX=0)
    env['_cost'] = lambda L, k: (-(-k * L.n // chunk)) * chunk
    exec(pick(path, names), env)
    if q is not None:
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(q, f)
        os.environ['KT_KV_HOT_LAYER_QUOTA'] = f.name
    else:
        os.environ.pop('KT_KV_HOT_LAYER_QUOTA', None)
    return env['_plan'](budget)


rng = random.Random(1017)
same = 0
for case in range(400):
    layers = [L(i, rng.choice((383, 383, 383, 40, 3)), rng.choice((17694720, 18000000, 16 << 20))) for i in range(40)]
    budget = rng.randrange(0, 18 << 30)
    chunk = rng.choice((1 << 21, 8 << 20, 32 << 20))
    a = run(OLD, ('_plan',), layers, budget, chunk)
    b = run(NEW, ('_plan', '_quota'), layers, budget, chunk)
    assert a == b, (case, a, b)
    same += 1
q = json.load(open(sys.argv[3]))
layers = [L(i, 383, 17694720) for i in range(40)]
k = run(NEW, ('_plan', '_quota'), layers, int(952 * 17694720), 1 << 21, q)
ratio = [x / y for x, y in zip(k, q)]
assert max(ratio) - min(ratio) < 2.5 / min(q), ratio
assert sum(k) >= 940
print('LAYER_QUOTA_PASS unset equals accepted on', same, 'cases; quota', sum(q), 'gives', sum(k), 'slots,', min(k), '-', max(k), 'a layer')
