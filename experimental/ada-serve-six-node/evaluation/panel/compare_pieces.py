"""Pair two nll.py files, position by position. argv: reference.json served.json [label]. Paired mean over the 24 pieces, as nll.py compare does, plus how far single positions move."""
import json
import math
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
assert [len(x) for x in a] == [len(x) for x in b]
flat = [(p, q) for u, v in zip(a, b) for p, q in zip(u, v)]
diff = sorted(abs(p - q) for p, q in flat)
n = len(diff)
pieces = [-(sum(v) - sum(u)) / len(u) for u, v in zip(a, b)]
mean = sum(pieces) / len(pieces)
se = math.sqrt(sum((x - mean) ** 2 for x in pieces) / (len(pieces) - 1) / len(pieces))
report = dict(label=sys.argv[3] if len(sys.argv) > 3 else '', positions=n, nll_a=-sum(sum(x) for x in a) / n, nll_b=-sum(sum(x) for x in b) / n, mean_nll_b_minus_a=mean, se=se,
              exact=sum(p == q for p, q in flat) / n, mean_abs=sum(diff) / n, median_abs=diff[n // 2], p90=diff[int(.9 * n)], p99=diff[int(.99 * n)], largest=diff[-1])
print(json.dumps(report, indent=1))
