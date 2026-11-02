"""Placement for a smaller arena spill (Claude, 31 Oct 2026). Local only.

Keeps the served hot-first order and moves as little as possible: the coldest experts of each layer swap onto the physical ids that still
have remote bytes under the given divisors. Combine order is keyed on logical expert ids, so a placement cannot change a bit.
argv: spill_div rebalance_div output.json"""
import hashlib, json, sys
import cold_placement_base as b

SPILL_DIV, REBALANCE_DIV, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
KEEP = 2  # physical ids 0 and 1 hold the banked resident rows


def remote_fraction(stride, count, local_node, spill_nodes, rebalance_node0):
    mib, size, regions = 1 << 20, stride * count, []
    stripe, hp = 64 * mib, 2 * mib
    for off in range(0, size, stripe):
        s = min(stripe, size - off)
        lb = ((s - s // SPILL_DIV + hp - 1) // hp) * hp
        if rebalance_node0 and local_node == 0:
            extra = (s // REBALANCE_DIV // hp) * hp
            for i in range(extra // hp):
                regions.append((off + lb - extra + i * hp, off + lb - extra + (i + 1) * hp))
        regions.append((off + lb, off + s))
    out = []
    for e in range(count):
        a, z, remote = e * stride, (e + 1) * stride, 0
        for x, y in regions:
            lo, hi = max(x, a), min(y, z)
            remote += max(0, hi - lo)
        out.append(remote / stride)
    return out


b.arena_remote_fraction = remote_fraction
_, rows = b.load_placement('served-placement-da56036d.json')
template = json.load(open('served-placement-da56036d.json'))
layers, n = len(rows), len(rows[0])
hist = b.load_histogram(['c1-identity-counts-rank0.pt', 'rtr1-route-counts.json'], layers, n, rows, 0, 'physical', None)
geom = b.geometry(count=n)
cost = geom['cost']
remote_ids = sorted((p for p in range(n) if cost[p] > 0), key=lambda p: (-cost[p], p))
print('hist keys', list(hist.keys()), 'tokens', hist['tokens'], 'remote ids', len(remote_ids), 'fully remote on every part', sum(c > 0.8 for c in cost), 'mean cost', round(sum(cost) / n, 4))
counts = hist['counts']
new_rows, before, after, total = [], 0.0, 0.0, 0.0
for layer in range(layers):
    row = list(rows[layer])
    count = counts[layer]
    protected = set(row[:KEEP])
    cold = sorted((e for e in range(n) if e not in protected), key=lambda e: (count[e], e))[:len(remote_ids)]
    where = {e: p for p, e in enumerate(row)}
    for p, e in zip(remote_ids, cold):
        q, other = where[e], row[p]
        row[p], row[q] = e, other
        where[e], where[other] = p, q
    assert sorted(row) == list(range(n)) and row[:KEEP] == rows[layer][:KEEP]
    new_rows.append(row)
    total += sum(count)
    before += sum(count[rows[layer][p]] * cost[p] for p in range(n))
    after += sum(count[row[p]] * cost[p] for p in range(n))
print(f'route-weighted remote bytes: same divisors with the served file {before / total:.4f}, after the swap {after / total:.4f}')
moved = sum(a != c for x, y in zip(rows, new_rows) for a, c in zip(x, y))
print('ids moved', moved, 'of', layers * n)
template['physical_to_logical_map'] = new_rows
with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(template, f, separators=(',', ':'))
    f.write('\n')
print('wrote', OUT, hashlib.sha256(open(OUT, 'rb').read()).hexdigest())
