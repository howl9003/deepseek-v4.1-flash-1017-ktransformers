"""Pinned loader of the wo_a CUDA kernel (wo_a_parts.cu, BPC 4, 3 stages); no compiler in the server (Claude, 1 Nov 2026).

parts(q, qs, w, scale, p): q e4m3 [M*G, K], qs fp32 [M*G, K/32], w e4m3 [G, N, K], scale fp32 [G, N/32, K/32], p fp32 [4, M, G, N].
Bit for bit fixed_wo_a_fp8._parts; the unchanged _merge reduces p.
"""
import hashlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_BINARY = Path(__file__).resolve().parent / 'wo_a_parts_b4s3.so'
_SHA256 = 'e42dc74f1695994c3e0b698813288a15e8d5af5ffef6af0cd3b279c63ecbb5de'
_EXT = None


def _load():
    global _EXT
    if _EXT is None:
        if hashlib.sha256(_BINARY.read_bytes()).hexdigest() != _SHA256:
            raise RuntimeError('wo_a CUDA kernel binary differs from its pin')
        spec = importlib.util.spec_from_file_location(_BINARY.stem, _BINARY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[spec.name] = module
        _EXT = module
        logger.warning('SGLANG_DSV41_FIXED_WO_A_CUDA: loaded %s %s', _BINARY, _SHA256)
    return _EXT


def parts(q, qs, w, scale, p):
    _load().parts(q, qs, w, scale, p)
