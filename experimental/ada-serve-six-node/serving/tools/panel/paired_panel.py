"""Paired NLL between two panel captures, by conversation (Claude, 2 Nov 2026). The ledger rule: mean increase of B over A within one paired standard error. stdlib only. argv: capture A, capture B."""
import array
import json
import math
import struct
import sys
from pathlib import Path


def chosen(path):
    data = Path(path).read_bytes()
    n, top = struct.unpack('<qq', data[:16])
    v = array.array('f')
    v.frombytes(data[16 + 8 * n * top:])
    assert len(v) == n
    return v


def main():
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    names = sorted(p.name for p in a.glob('seq-*.bin'))
    assert names and names == sorted(p.name for p in b.glob('seq-*.bin'))
    means, weights, differ, positions, largest = [], [], 0, 0, 0.0
    for name in names:
        x, y = chosen(a / name), chosen(b / name)
        assert len(x) == len(y), name
        if not x:
            continue
        d = [-(q - p) for p, q in zip(x, y)]       # NLL of B minus NLL of A
        means.append(sum(d) / len(d))
        weights.append(len(d))
        differ += sum(p != q for p, q in zip(x, y))
        positions += len(d)
        largest = max(largest, max(abs(v) for v in d))
    n = len(means)
    plain = sum(means) / n
    se = math.sqrt(sum((m - plain) ** 2 for m in means) / (n - 1) / n)
    weighted = sum(m * w for m, w in zip(means, weights)) / sum(weights)
    print(json.dumps(dict(conversations=n, positions=positions, positions_that_differ=differ, mean_nll_increase_by_position=weighted, mean_nll_increase_by_conversation=plain,
                          paired_standard_error=se, within_one_se=plain <= se, largest_abs_change=largest)))


if __name__ == '__main__':
    main()
