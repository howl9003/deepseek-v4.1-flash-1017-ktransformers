#!/usr/bin/env python3
"""Apply only to an isolated, pinned recipe checkout; never build or launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sglang', type=Path, required=True)
    ap.add_argument('--ktransformers', type=Path, required=True)
    ap.add_argument('--apply', action='store_true', help='default: check only')
    args = ap.parse_args()
    roots = {k: getattr(args, k).resolve() for k in ('sglang', 'ktransformers')}
    data = json.loads((HERE / 'SOURCE_STACK.json').read_text())
    operations = []
    # Validate every changed input and overlay before making any write.
    for row in data['changes']:
        rel = Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        dest = roots[row['component']] / rel
        assert dest.resolve().is_relative_to(roots[row['component']])
        if dest.is_symlink():
            assert row.get('before_link') == os.readlink(dest), dest
        src = HERE / 'overlays' / row['component'] / rel
        assert sha(src) == row['after_sha256'], f'Changed package file: {src}'
        current = sha(dest) if dest.is_file() else None
        assert current in (row['before_sha256'], row['after_sha256']), (
            f'Wrong input: {dest}; use pinned tier2 plus both optional policy patches')
        if current != row['after_sha256']:
            operations.append((src, dest))
    if args.apply:
        for src, dest in operations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                # The captured build used a physical copy of this exact header.
                # Remove only this checked link, never its referent.
                dest.unlink()
            shutil.copyfile(src, dest)
        for component, entries in data['snapshots'].items():
            for name, expected in entries.items():
                assert sha(roots[component] / name) == expected, (component, name)
    print(json.dumps({'status': 'APPLIED_AND_SOURCE_HASHES_VERIFIED' if args.apply else 'CHECK_ONLY',
                      'files_to_replace': len(operations), 'model_run': False}))


if __name__ == '__main__':
    main()
