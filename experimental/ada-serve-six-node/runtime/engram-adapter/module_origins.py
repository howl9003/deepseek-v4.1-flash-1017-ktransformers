"""Observe staged module origins without replacing any finder or loader.

Stdlib only. The controller supplies a hash-bound config and an owned receipt
directory. Install after the unchanged Engram sitecustomize registers _Finder.
The wrapper delegates to CPython's existing _find_and_load, returns the same
object, and scans preloaded modules (including .pth imports). A violation exits
the process: sitecustomize's exception handling cannot turn it into a warning.

This proves observed module paths and on-disk bytes, not semantic correctness or
arbitrary dynamic-code provenance. Wrong .pth code may have executed before this
observer detects it. Manual exec/loader calls or later sys.modules reassignment
outside the import machinery are not continuously intercepted. Stage no pycache
and set PYTHONDONTWRITEBYTECODE=1; __file__ cannot prove which pyc was executed.
Gate again before requests and verify sources again after the owned trial.
"""
import hashlib
import importlib._bootstrap as bootstrap
from importlib.machinery import (NamespaceLoader, FileFinder, SourceFileLoader,
    SourcelessFileLoader, ExtensionFileLoader, SOURCE_SUFFIXES, BYTECODE_SUFFIXES,
    EXTENSION_SUFFIXES, all_suffixes)
from importlib.util import module_from_spec
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import stat as stat_module
import time

EXIT_CODE = 86
MANDATORY = (
    'sglang.srt.managers.scheduler',
    'sglang.srt.session.session_controller',
    'sglang.srt.session.streaming_session',
    'sglang.srt.mem_cache.base_prefix_cache',
    'sglang.srt.mem_cache.unified_radix_cache',
)
NATIVE = ('kt_kernel.kt_kernel_ext', 'kt_kernel.experts_base')
ENGRAM_HOOK_TARGETS = ('sglang.srt.layers.engram',
                      'sglang.srt.layers.attention.dsv4.metadata',
                      'sglang.srt.managers.scheduler')
_installed = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def namespace_directory(root, name, relative, resolve=None):
    """One physical, importable namespace directory, never a regular package."""
    require(name.startswith('sglang.') and name not in MANDATORY and name not in NATIVE
            and all(part.isidentifier() for part in name.split('.')), 'invalid namespace name: ' + name)
    require(relative == 'python/' + name.replace('.', '/'), 'namespace name/path differs: ' + name)
    root = Path(root)
    path = root / relative
    require(path.is_dir() and (resolve(path) if resolve else path.resolve(strict=True)) == path,
            'namespace directory is not physical: ' + name)
    require(not any((path / ('__init__' + suffix)).exists() or
                    (path / ('__init__' + suffix)).is_symlink() for suffix in all_suffixes()),
            'namespace has package initializer: ' + name)
    # FileFinder operates on this directory's parent without importing a parent
    # package, executing __init__, or relying on sys.modules parent stubs.
    finder = FileFinder(str(path.parent), (ExtensionFileLoader, EXTENSION_SUFFIXES),
        (SourceFileLoader, SOURCE_SUFFIXES), (SourcelessFileLoader, BYTECODE_SUFFIXES))
    spec = finder.find_spec(name)
    require(spec is not None and spec.origin is None and spec.loader is None
            and list(spec.submodule_search_locations or []) == [str(path)],
            'physical directory does not resolve as namespace: ' + name)
    return path, spec


def derive_namespace_packages(root, files):
    """Derive candidates from the complete copied file inventory, not selected patches."""
    candidates = set()
    for relative in files:
        p = PurePosixPath(relative)
        require(not p.is_absolute() and '..' not in p.parts, 'unsafe inventory path')
        if p.parts[:2] != ('python', 'sglang'):
            continue
        for parent in p.parents:
            parts = parent.parts
            if len(parts) > 2 and parts[:2] == ('python', 'sglang') and all(x.isidentifier() for x in parts[1:]):
                candidates.add(parent.as_posix())
    result = {}
    for relative in sorted(candidates):
        directory = Path(root) / relative
        if any((directory / ('__init__' + s)).exists() for s in all_suffixes()):
            continue
        name = relative[len('python/'):].replace('/', '.')
        namespace_directory(root, name, relative)
        result[name] = relative
    require(result, 'no physical SGLang namespaces found')
    return result


def namespace_item(config, name, resolve=None):
    require(name in config['namespace_packages'], 'unapproved no-file module: ' + name)
    relative = config['namespace_packages'][name]
    path, _ = namespace_directory(config['source_root'], name, relative, resolve=resolve)
    return dict(kind='namespace', relative_path=relative, search_locations=[str(path)],
        module_path=[str(path)], origin=None, spec_name=name, has_location=False,
        loader='importlib.machinery.NamespaceLoader',
        namespace_inventory_sha256=config['namespace_source_inventory_sha256'])


def topology_smoke(config):
    """Exercise real stdlib namespace objects; execute no SGLang package code."""
    probe = Observer.__new__(Observer)
    probe.config, probe.root = config, Path(config['source_root'])
    probe.python, probe.hashes, probe.modules = probe.root / 'python', {}, {}
    for name, relative in config['namespace_packages'].items():
        _, spec = namespace_directory(probe.root, name, relative)
        module = module_from_spec(spec)
        module.__spec__.loader.exec_module(module)
        probe.observe(name, module)
    return dict(status='PASS', operation='stdlib namespace topology only; no SGLang code execution',
        namespace_count=len(probe.modules), namespace_source_inventory_sha256=config['namespace_source_inventory_sha256'],
        modules=probe.modules)


def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    proc = Path('/proc') / str(pid)
    tail = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
    rows = [x.partition('::')[2] for x in (proc / 'cgroup').read_text().splitlines()
            if x.startswith('0::')]
    require(len(rows) == 1 and tail[0] != 'Z', 'no live unique cgroup identity')
    return dict(pid=pid, ppid=int(tail[1]), start_ticks=int(tail[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cgroup=str(Path('/sys/fs/cgroup') / rows[0].lstrip('/')))


def load_config(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    require(re.fullmatch('[0-9a-f]{64}', expected_sha256 or ''), 'invalid config pin')
    require(sha(path) == expected_sha256, 'origin config changed')
    config = json.loads(path.read_text())
    require(config.get('schema_version') == 1 and config.get('arm') in ('A', 'B'), 'bad origin config')
    require(re.fullmatch('[A-Za-z0-9_-]{8,80}', config.get('trial_id', '')), 'bad trial identity')
    for key in ('source_root', 'receipt_dir', 'pycache_prefix'):
        value = Path(config[key])
        require(value.is_absolute() and value.is_dir() and value.resolve() == value,
                key + ' must be an absolute physical directory')
    require(sha(__file__) == config.get('observer_sha256'), 'observer source differs')
    require(set(config['native_modules']) == set(NATIVE), 'exact native payloads required')
    root = Path(config['source_root'])
    require(isinstance(config.get('namespace_packages'), dict) and config['namespace_packages'],
            'namespace map missing')
    require(re.fullmatch('[0-9a-f]{64}', config.get('namespace_source_inventory_sha256', '')),
            'namespace inventory pin missing')
    for name in config['namespace_packages']:
        namespace_item(config, name)
    require(isinstance(config['source_sha256'], dict) and config['source_sha256'], 'missing source pins')
    for rel, expected in config['source_sha256'].items():
        name = PurePosixPath(rel)
        require(not name.is_absolute() and '..' not in name.parts and rel.startswith('python/sglang/'),
                'source pin escaped staged SGLang')
        target = (root / rel).resolve(strict=True)
        require(target.is_relative_to(root / 'python') and sha(target) == expected, 'staged source changed: ' + rel)
    for name, pin in config['native_modules'].items():
        target = Path(pin['path'])
        require(target.is_absolute() and target.resolve(strict=True) == target and sha(target) == pin['sha256'],
                'native payload changed: ' + name)
    return config


def sitecustomize_bytes(original, expected_sha256):
    """Append only; caller physically copies this plus module_origins.py privately."""
    require(hashlib.sha256(original).hexdigest() == expected_sha256, 'canonical sitecustomize changed')
    suffix = b'''\n# Owned feature-trial provenance observer; canonical prefix above is unchanged.
try:
    import module_origins as _feature_module_origins
    _feature_module_origins.install_from_env()
except BaseException as _feature_origin_error:
    import os as _feature_origin_os
    _feature_origin_os.write(2, ("feature origin observer failed: " + repr(_feature_origin_error) + "\\n").encode())
    _feature_origin_os._exit(86)
'''
    return original + suffix


class Observer:
    def __init__(self, config, config_sha256):
        self.config, self.config_sha256 = config, config_sha256
        self.root = Path(config['source_root'])
        self.python = (self.root / 'python').resolve(strict=True)
        self.output = Path(config['receipt_dir'])
        self.lock = threading.RLock()
        self.local = threading.local()
        self.process = identity()
        require(sys.pycache_prefix is not None and
                Path(sys.pycache_prefix).resolve(strict=True) == Path(config['pycache_prefix']),
                'runtime pycache prefix differs from private fresh prefix')
        require(sys.dont_write_bytecode is True, 'bytecode writes must be disabled')
        self.modules = {}
        self.hashes = {}
        self.failed = None
        self.original = None

    def watched(self, name):
        return name == 'sglang' or name.startswith('sglang.') or name in NATIVE

    def file_hash(self, path):
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        old = self.hashes.get(str(path))
        if old is None or old[0] != stamp:
            self.hashes[str(path)] = (stamp, sha(path))
        return self.hashes[str(path)][1]

    def resolve(self, path):
        path = Path(path)
        cache = getattr(getattr(self, 'local', None), 'parent_paths', None)
        if not self.config.get('fast_path_resolution', False) or cache is None or not path.is_absolute() or '..' in path.parts:
            return path.resolve(strict=True)
        # Each scan starts fresh. Recheck the file itself on every call; share
        # only parent canonicalization within that single locked scan.
        info = path.lstat()
        if stat_module.S_ISLNK(info.st_mode):
            return path.resolve(strict=True)
        parent = path.parent
        if parent not in cache:
            cache[parent] = parent.resolve(strict=True)
        return cache[parent] / path.name

    def observe(self, name, module):
        require(module is not None, 'module missing: ' + name)
        if getattr(module, '__file__', None) is None:
            spec = getattr(module, '__spec__', None)
            require(spec is not None and type(spec.loader) is NamespaceLoader
                    and getattr(module, '__loader__', None) is spec.loader,
                    'no-file module lacks namespace loader: ' + name)
            item = namespace_item(self.config, name, resolve=self.resolve)
            require(spec.name == name and spec.origin is None and spec.has_location is False,
                    'namespace spec identity/origin differs: ' + name)
            require(list(spec.submodule_search_locations or []) == item['search_locations']
                    and list(getattr(module, '__path__', [])) == item['module_path'],
                    'namespace search paths differ: ' + name)
            changed = self.modules.get(name) != item
            self.modules[name] = item
            return changed
        require(getattr(module, '__file__', None), 'module lacks file: ' + name)
        path = self.resolve(module.__file__)
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        require(origin and self.resolve(origin) == path, 'module spec/file differs: ' + name)
        if name in NATIVE:
            pin = self.config['native_modules'][name]
            require(path == Path(pin['path']), 'native module from wrong runtime: ' + name)
            expected = pin['sha256']
            rel = None
        else:
            if self.config.get('fast_path_resolution', False) and os.name == 'posix':
                # Both paths are canonical absolute POSIX paths. A slash-delimited
                # prefix is equivalent to walking every pathlib parent repeatedly.
                text, python_root, source_root = str(path), str(self.python), str(self.root)
                require(text == python_root or text.startswith(python_root + '/'), 'SGLang module outside staged source: ' + name)
                rel = text[len(source_root) + 1:]
            else:
                require(path.is_relative_to(self.python), 'SGLang module outside staged source: ' + name)
                rel = path.relative_to(self.root).as_posix()
            expected = self.config['source_sha256'].get(rel)
        actual = self.file_hash(path)
        require(expected is None or expected == actual, 'loaded module bytes differ: ' + name)
        item = dict(path=str(path), sha256=actual, selected_pin=expected is not None,
                    relative_path=rel)
        changed = self.modules.get(name) != item
        self.modules[name] = item
        return changed

    def save(self):
        row = dict(schema_version=1, status='FAIL' if self.failed else 'OBSERVED',
            config_sha256=self.config_sha256, trial_id=self.config['trial_id'], arm=self.config['arm'],
            observer_sha256=self.config['observer_sha256'], process=self.process,
            bytecode_policy=dict(pycache_prefix=str(Path(sys.pycache_prefix).resolve()),
                                 dont_write_bytecode=sys.dont_write_bytecode,
                                 fresh_prefix_required_at_staging=True),
            modules=self.modules, error=self.failed, observed_time_ns=time.time_ns())
        target = self.output / (str(self.process['pid']) + '.json')
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        tmp.replace(target)

    def fail(self, exc):
        self.failed = repr(exc)
        try:
            self.save()
        except BaseException:
            pass
        try:
            os.write(2, ('feature module origin violation: ' + repr(exc) + '\n').encode())
        finally:
            os._exit(EXIT_CODE)
        raise SystemExit(EXIT_CODE)  # Also stops control flow when _exit is mocked.

    def scan(self):
        with self.lock:
            previous = getattr(self.local, 'parent_paths', None)
            self.local.parent_paths = {}
            try:
                changed = False
                for name, module in list(sys.modules.items()):
                    if self.watched(name):
                        changed = self.observe(name, module) or changed
                if changed:
                    self.save()
            finally:
                self.local.parent_paths = previous

    def after_fork(self):
        self.lock = threading.RLock()
        self.local = threading.local()
        try:
            self.process = identity()
            self.modules, self.hashes = {}, {}
            self.scan()
            self.save()
        except BaseException as exc:
            self.fail(exc)

    def install(self):
        require(any(type(f).__name__ == '_Finder' and type(f).__module__ == 'sitecustomize'
                    for f in sys.meta_path), 'Engram finder must be registered before observer')
        self.scan()  # Detect wrong .pth/pre-sitecustomize imports before accepting new ones.
        require(not any(name in sys.modules for name in ENGRAM_HOOK_TARGETS),
                'Engram hook target was preloaded before observer installation')
        self.save()
        self.original = bootstrap._find_and_load

        def observed_import(name, import_):
            module = self.original(name, import_)
            if self.watched(name) and not getattr(self.local, 'scanning', False):
                self.local.scanning = True
                try:
                    if os.getpid() != self.process['pid']:
                        self.after_fork()
                    self.scan()
                except BaseException as exc:
                    self.fail(exc)
                finally:
                    self.local.scanning = False
            return module

        bootstrap._find_and_load = observed_import
        if hasattr(os, 'register_at_fork'):
            os.register_at_fork(after_in_child=self.after_fork)
        return self


def install_from_env():
    global _installed
    try:
        wanted = os.environ['FEATURE_ORIGINS_CONFIG_SHA256']
        config = load_config(os.environ['FEATURE_ORIGINS_CONFIG'], wanted)
        if _installed is not None:
            require(_installed.config_sha256 == wanted, 'observer reconfigured in one process')
            return _installed
        observer = Observer(config, wanted)
        try:
            _installed = observer.install()
        except BaseException as exc:
            observer.fail(exc)
        return _installed
    except BaseException as exc:
        os.write(2, ('feature origin installation failed: ' + repr(exc) + '\n').encode())
        os._exit(EXIT_CODE)
        raise SystemExit(EXIT_CODE)


def gate_ready(config_path, *, expected_sha256, owned_python_pids, scheduler_pid, cgroup):
    """Caller supplies fresh cgroup-derived Python PID set and exact scheduler PID.

    Every live supplied Python process must have an observer receipt. Mandatory
    origins and both native payloads must coexist in the scheduler's receipt.
    This function does not discover ownership, send requests or terminate PIDs.
    """
    config = load_config(config_path, expected_sha256)
    pids = set(owned_python_pids)
    require(pids and scheduler_pid in pids and all(type(p) is int and p > 0 for p in pids), 'bad owned PID set')
    rows = {}
    for pid in sorted(pids):
        path = Path(config['receipt_dir']) / (str(pid) + '.json')
        require(not path.is_symlink(), 'origin receipt is a symlink')
        row = json.loads(path.read_text())
        require(row.get('status') == 'OBSERVED' and row.get('error') is None, 'observer failed or incomplete')
        require(row['config_sha256'] == expected_sha256 and row['trial_id'] == config['trial_id']
                and row['arm'] == config['arm'] and row['observer_sha256'] == config['observer_sha256'], 'receipt belongs to another trial')
        require(row.get('bytecode_policy') == dict(pycache_prefix=config['pycache_prefix'],
                dont_write_bytecode=True, fresh_prefix_required_at_staging=True), 'bytecode policy differs')
        current = identity(pid)
        require(row['process'] == current and current['cgroup'] == str(cgroup), 'receipt PID/boot/cgroup identity differs')
        probe = Observer.__new__(Observer)
        probe.config, probe.root = config, Path(config['source_root'])
        probe.python, probe.hashes, probe.modules = probe.root / 'python', {}, {}
        for name, item in row['modules'].items():
            require(probe.watched(name), 'unexpected recorded module')
            if item.get('kind') == 'namespace':
                require(namespace_item(config, name) == item, 'namespace origin receipt changed')
                probe.modules[name] = item
                continue
            # Recompute policy and bytes using only recorded paths, never import modules.
            from types import SimpleNamespace
            probe.observe(name, SimpleNamespace(__file__=item['path'], __spec__=SimpleNamespace(origin=item['path'])))
            require(probe.modules[name] == item, 'module origin/hash receipt changed')
        rows[pid] = dict(path=str(path), sha256=sha(path), process=current, modules=row['modules'])
    required = set(MANDATORY) | set(NATIVE)
    require(required.issubset(rows[scheduler_pid]['modules']), 'scheduler mandatory origins incomplete')
    return dict(status='PASS', config_sha256=expected_sha256, scheduler_pid=scheduler_pid,
                cgroup=str(cgroup), processes=rows, mandatory_modules=sorted(required))
