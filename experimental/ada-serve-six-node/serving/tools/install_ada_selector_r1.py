"""Add the SGLang build to the model selector (Claude, 2 Nov 2026). The owner runs this with sudo; agents have no root on fea01.

Adds one llama-swap entry and one guard chat model, keeps the old llama.cpp entry and the default model, backs up both files first, and restarts the swap manager and gateway only while nothing is loaded.
Run: ssh -t aster@fea01 'sudo python3 -B /scratch/aster/bench/six-node-performance-r1/tools/install_ada_selector_r1.py'
Undo: the same command with --remove.
--pin keeps this entry loaded by refusing every other model at the gateway; --unpin restores them.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

STACK = Path('/etc/fea/model-stack')
SWAP, GUARD = STACK / 'llama-swap.json', STACK / 'guard.json'
NAME = 'deepseek-v4.1-flash-sglang'
PINNED, KEEP = STACK / 'pinned-ada.json', ('chat_models', 'default_model', 'comfy_enabled')
ENTRY = {
    'cmd': '/usr/bin/python3 -B /scratch/aster/bench/six-node-performance-r1/tools/swap_entry_ada_r1.py',
    'proxy': 'http://127.0.0.1:18209',
    'checkEndpoint': '/health',
    'ttl': 0,
    'unloadTimeout': 360,
    'env': ['FEA_LLM_DAEMON=1'],
    'timeouts': {'connect': 10, 'responseHeader': 3600, 'idleConn': 90},
    'useModelName': 'default',
    'concurrencyLimit': 36,
    'name': 'DeepSeek V4.1 Flash 1017 SGLang (1M, 4 sessions, text)',
    'capabilities': {'in': ['text'], 'out': ['text'], 'tools': True, 'context': 1048576, 'disableAuto': True},
}


def running():
    with urllib.request.urlopen('http://127.0.0.1:8082/running', timeout=5) as response:
        return json.load(response)['running']


def save(path, value):
    temp = path.with_suffix('.new')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    shutil.copymode(path, temp)
    shutil.chown(temp, 'root', 'root')
    os.replace(temp, path)


def pin(on):
    """Keep this entry loaded: the gateway refuses every other model instead of swapping to it. Comfy is off too, since it shares the swap group.
    Only the gateway restarts, so a loaded model stays loaded. --unpin puts the saved values back."""
    guard = json.loads(GUARD.read_text())
    if on:
        if PINNED.exists():
            sys.exit('already pinned')
        PINNED.write_text(json.dumps({k: guard[k] for k in KEEP}, indent=2) + '\n', encoding='utf-8')
        guard.update(chat_models=[NAME], default_model=NAME, comfy_enabled=False)
    else:
        if not PINNED.exists():
            sys.exit('not pinned')
        guard.update(json.loads(PINNED.read_text()))
    save(GUARD, guard)
    if not on:
        PINNED.unlink()
    subprocess.run(['systemctl', 'restart', 'fea-model-gateway.service'], check=True, timeout=120)
    print('pinned to' if on else 'unpinned from', NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--pin', action='store_true')
    parser.add_argument('--unpin', action='store_true')
    args = parser.parse_args()
    if os.getuid() != 0:
        sys.exit('run with sudo')
    if args.pin or args.unpin:
        pin(args.pin)
        return
    if running():
        sys.exit('a model is loaded; unload it first: curl http://127.0.0.1:8082/unload')
    swap, guard = json.loads(SWAP.read_text()), json.loads(GUARD.read_text())
    members = swap['routing']['router']['settings']['groups']['gpu']['members']
    backup = STACK / ('backup-ada-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    backup.mkdir(mode=0o700)
    shutil.copy2(SWAP, backup)
    shutil.copy2(GUARD, backup)
    if args.remove:
        swap['models'].pop(NAME, None)
        members[:] = [m for m in members if m != NAME]
        guard['chat_models'] = [m for m in guard['chat_models'] if m != NAME]
    else:
        swap['models'][NAME] = ENTRY
        if NAME not in members:
            members.insert(0, NAME)
        if NAME not in guard['chat_models']:
            guard['chat_models'].insert(1, NAME)
    save(SWAP, swap)
    save(GUARD, guard)
    subprocess.run(['systemctl', 'restart', 'fea-model-swap.service'], check=True, timeout=240)
    subprocess.run(['systemctl', 'restart', 'fea-model-gateway.service'], check=True, timeout=120)
    print('done, backup in', backup)


if __name__ == '__main__':
    main()
