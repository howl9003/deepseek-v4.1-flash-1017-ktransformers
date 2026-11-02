"""Pinned prebuilt G1p extension. No compiler or arithmetic lives here.

The streamed-prefill wrapper binds to the same hashed dual-target module the path A dense runtime already loaded, so one
process holds one copy of the native library and one flag word. Refuses to run without path A dense.
"""
import hashlib
import json
import os
from pathlib import Path

_bound = None


def bind_native(pa, environ):
    global _bound
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('G1p native binding must precede graph capture')
    path = Path(environ['KT_GPU_STREAM_PATHA_NATIVE_PIN']).resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != environ['KT_GPU_STREAM_PATHA_NATIVE_PIN_SHA256']:
        raise RuntimeError('G1p native pin changed')
    if _bound is not None:
        if _bound != (str(path), digest) or pa._EXT is None:
            raise RuntimeError('G1p native binding changed during process lifetime')
        return
    pin = json.loads(raw)
    if pin['status'] != 'G1P_QUALIFIED_NATIVE_PIN':
        raise RuntimeError('G1p requires a qualified component pin')
    root = Path(pa.__file__).resolve().parent
    for name, wanted in pin['source_sha256'].items():
        source = (root / name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != wanted:
            raise RuntimeError('G1p native source changed: ' + name)
    from sglang.srt.layers.quantization.patha.runtime import _backends
    native = _backends(False)[1]._EXT
    binary = Path(native.__file__).resolve()
    if native.__name__ != pin['module_name'] or hashlib.sha256(binary.read_bytes()).hexdigest() != pin['binary_sha256']:
        raise RuntimeError('G1p native binary differs from the path A dense module')
    if pa._EXT is not None:
        raise RuntimeError('G1p extension was loaded before binding')
    pa._EXT = native
    receipt = dict(pid=os.getpid(), binary=str(binary), binary_sha256=pin['binary_sha256'], pin_sha256=digest,
                   source_sha256=pin['source_sha256'], loader='shared with path A dense; no JIT',
                   component_review_sha256=pin['component_review_sha256'])
    out = Path(environ['KT_GPU_STREAM_PATHA_ORIGIN_DIR']).resolve()
    if not out.is_dir():
        raise RuntimeError('G1p native origin directory was not prepared')
    with (out / (str(os.getpid()) + '.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    _bound = (str(path), digest)
