"""Fifth replay of the slots (Claude, 2 Nov 2026): what more slots buy at the served quota rule, scored in exposed window time. Offline, read-only. argv: trace folder.

Betelgeuse's arrival-slack idea on our hardware. A layer's window is the slower of its GPU part and its CPU routes, so a slot only saves time on a layer whose CPU is last: with n CPU routes a
call takes about 65 + 125 n us (worker traces) against about 185 us of GPU work, so going from 1 route to 0 saves nothing and from 3 to 2 saves 125 us. A layer's n in a token is taken as
binomial over its six routes at the layer's CPU share in that round. Quotas are fitted leaving out the workload they are scored on.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_dyn_slots_r2 as R
from replay_dyn_slots_r1 import load

R.TOKEN_MS = 33.0
BASE_US, ROUTE_US, GPU_US, ROUTES = 65.0, 125.0, 185.0, 6
PMF = None


def window_us(p):
    """Expected window on a layer whose routes each go to the CPU with probability p."""
    p = np.clip(p, 0.0, 1.0)
    out = 0.0
    from math import comb
    for n in range(ROUTES + 1):
        out = out + comb(ROUTES, n) * p ** n * (1 - p) ** (ROUTES - n) * np.maximum(GPU_US, BASE_US + ROUTE_US * n)
    return out


def simulate_layers(seen, held, quota, first, stop):
    """R.simulate, with the hits kept per layer: the same rule 64/1.5, same copy latency."""
    half_life, hyst, budget = 64.0, 1.5, 8
    L = seen.shape[1]
    score = np.zeros((L, seen.shape[2]))
    current = [set(h[h >= 0].tolist()) for h in held[first]]
    size = [len(c) for c in current] if quota is None else list(quota)
    decay = 0.5 ** (R.EVERY / half_life)
    hits = np.zeros((len(seen), L))
    new = [dict() for _ in range(L)]
    done_at = -1.0
    for i in range(first, stop):
        counts = seen[i]
        for l in range(L):
            got = counts[l, list(current[l])].sum()
            for e, f in new[l].items():
                got -= counts[l, e] * (1 - f)
            hits[i, l] = got
        new = [dict() for _ in range(L)]
        score = score * decay + counts
        now = i * R.EVERY * R.TOKEN_MS
        if now < done_at:
            continue
        plan = []
        for l in range(L):
            sc, have = score[l], current[l]
            n, v = 0, 0
            victims = sorted(have, key=lambda e: (sc[e], e))
            for e in np.argsort(-sc, kind='stable').tolist():
                if n >= budget or sc[e] <= 0:
                    break
                if e in have:
                    continue
                if len(have) < size[l]:
                    have.add(e); plan.append((l, e)); n += 1
                    continue
                while v < len(victims) and victims[v] not in have:
                    v += 1
                if v >= len(victims):
                    break
                old = victims[v]
                if not sc[e] > sc[old] * hyst:
                    break
                have.discard(old); have.add(e); plan.append((l, e)); v += 1; n += 1
        if plan:
            ms = R.BASE_MS + R.COPY_MS * len(plan)
            done_at = now + ms
            fraction = max(0.0, 1 - ms / (R.EVERY * R.TOKEN_MS))
            for l, e in plan:
                new[l][e] = fraction
    return hits


def slack_quota(seen, slots, floor=8):
    """Slots by the window time each one saves: a greedy pass over each layer's sorted route counts."""
    total = seen.sum(axis=0).astype(float)
    L = total.shape[0]
    srt = -np.sort(-total, axis=1)
    cum = np.cumsum(srt, axis=1)
    routes = total.sum(axis=1)
    cost = lambda l, k: window_us(1 - cum[l, k - 1] / routes[l])
    quota = [floor] * L
    import heapq
    heap = [(-(cost(l, quota[l]) - cost(l, quota[l] + 1)), l) for l in range(L)]
    heapq.heapify(heap)
    left = slots - floor * L
    while left > 0:
        gain, l = heapq.heappop(heap)
        quota[l] += 1
        left -= 1
        heapq.heappush(heap, (-(cost(l, quota[l]) - cost(l, quota[l] + 1)), l))
    return quota


rounds, times, seen, held = load(sys.argv[1])
base = int((held[1279] >= 0).sum())
print('slots in the trace', base, flush=True)
EXTRA = (0, 35, 70, 140, 280)
total = {k: [0.0, 0.0, 0.0] for k in EXTRA}
for name in R.BREADTH:
    keep = np.ones(len(seen), bool)
    spans = [(a, b) for n, a, b in R.ROWS if n == name]
    for a, b in spans:
        keep[a:b] = False
    a, b = spans[-1]
    row = []
    for extra in EXTRA:
        quota = R.quota_from(seen[keep], base + extra)
        h = simulate_layers(seen, held, quota, a, b)
        routes = seen[a:b].sum(axis=(0, 2))
        share = h[a:b].sum(axis=0) / routes
        ms = float(window_us(1 - share).sum()) / 1000
        cpu = float((routes - h[a:b].sum(axis=0)).sum()) / ((b - a) * R.EVERY)
        row.append((extra, ms, cpu))
        total[extra][0] += ms
        total[extra][1] += cpu
        total[extra][2] += 1
    print(f'{name:18s}', '  '.join(f'+{e}: {ms:5.2f} ms {cpu:5.1f} routes' for e, ms, cpu in row), flush=True)
for k, v in total.items():
    print(f'+{k:3d} slots: mean windows {v[0] / v[2]:6.2f} ms a token, CPU routes {v[1] / v[2]:5.1f}', flush=True)
