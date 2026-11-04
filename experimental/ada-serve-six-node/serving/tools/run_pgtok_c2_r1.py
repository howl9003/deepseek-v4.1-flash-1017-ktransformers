"""C2 references for the plain-gate batch check (Claude, 4 Nov 2026). On served C2 (port 18209): the four greedy prompts of
client_ada_adadpb_perf_r1.py alone, staggered and all at once, then one, two and four chat sessions, two rounds. Records whether C2
itself gives the same tokens alone and batched. With --candidate (an adadpb tokens.json) it only compares.

argv: output dir [--candidate tokens.json]"""
import argparse
import json
from pathlib import Path

from client_ada_adadpb_perf_r1 import MODES, capture, compare, first_difference, together, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    if args.candidate:
        reference = json.loads((args.out / 'tokens.json').read_text(encoding='utf-8'))
        candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
        result = compare(reference, candidate)
        write(args.out / 'compare-adadpb.json', result)
        print(json.dumps(result, indent=1))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    tokens = capture()
    write(args.out / 'tokens.json', tokens)
    invariance = {f"{mode}/{a['name']}": first_difference(a['output_ids'], b['output_ids'])
                  for mode in MODES[1:] for a, b in zip(tokens['single'], tokens[mode])}
    result = dict(batch_invariant=all(v is None for v in invariance.values()), first_difference_vs_single=invariance,
                  seconds={mode: [round(r['seconds'], 2) for r in rows] for mode, rows in tokens.items()}, sessions=[])
    write(args.out / 'RESULT.json', result)
    for _ in range(2):
        for count in (1, 2, 4):
            result['sessions'].append(together(count, 512))
            write(args.out / 'RESULT.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'sessions'}, indent=1))
    for row in result['sessions']:
        print(row['sessions'], round(row['total_tok_s'], 2), [round(x, 2) for x in row['per_session_tok_s']])


if __name__ == '__main__':
    main()
