"""Bounded, lossless GGUF MXFP4 expert views for KT's native loader interface.

Uses KT's existing GGUF header reader. Source identities are supplied separately;
this reader never trusts a freshly computed digest as its expected digest.
Manifest completeness and original-checkpoint provenance are external gates.
Copyright 2026 Strata adaptation contributors. SPDX-License-Identifier: Apache-2.0
"""
import hashlib
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import re

import numpy as np
import torch
from kt_kernel.utils.loader import GGUFLoader


@lru_cache(maxsize=1)
def _native_repacker(path, expected_sha256):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('native MXFP4 repacker requires a pinned SHA256')
    library_path = Path(path).resolve(strict=True)
    if hashlib.sha256(library_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('native MXFP4 repacker digest mismatch')
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_uint8)
    library.repack_avx2.argtypes = [pointer, pointer, pointer, ctypes.c_size_t]
    library.repack_avx2.restype = ctypes.c_size_t
    return library


def split_to_adjacent(blocks):
    """ggml 17-byte [E8, split-half nibbles] to original E2M1/E8 arrays.

    Returns copied arrays. No floating-point conversion or requantization.
    Signed zero's nibble is preserved, as are every finite scale's raw bits.
    """
    b = np.asarray(blocks)
    if b.dtype != np.uint8 or b.ndim != 2 or b.shape[1] != 17:
        raise ValueError('expected [blocks,17] uint8 MXFP4 blocks')
    library_path = os.environ.get('KT_GGUF_REPACK_AVX2_LIBRARY')
    if library_path:
        library = _native_repacker(library_path,
            os.environ.get('KT_GGUF_REPACK_AVX2_SHA256', ''))
        source = np.ascontiguousarray(b)
        packed = np.empty((len(b),16), dtype=np.uint8)
        scales = np.empty(len(b), dtype=np.uint8)
        pointer = ctypes.POINTER(ctypes.c_uint8)
        rejected = library.repack_avx2(source.ctypes.data_as(pointer),
            packed.ctypes.data_as(pointer), scales.ctypes.data_as(pointer), len(b))
        if rejected:
            raise ValueError('MXFP4 UE8M0 contains reserved NaN code 255')
        return packed, scales
    if np.any(b[:, 0] == 255):
        raise ValueError('MXFP4 UE8M0 contains reserved NaN code 255')
    packed = np.empty((len(b),16), dtype=np.uint8)
    # Original order: low nibble contains even element, high nibble odd.
    q = b[:, 1:]
    packed[:, :8] = (q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)
    packed[:, 8:] = (q[:, 0::2] >> 4) | ((q[:, 1::2] >> 4) << 4)
    return packed, b[:, 0].copy()


def file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class NativeMXFP4GGUFReader(GGUFLoader):
    """One-layer-at-a-time adapter; returned tensors own their converted storage.

    At most one expert projection is read/repacked at once. The full layer's
    returned tensor lists remain live until KT's owned C++ load completes.
    This does not claim constant memory for a full layer or full-model fit.
    """
    def __init__(self, path, manifest_path, expected_manifest_sha256):
        path = Path(path).resolve(strict=True)
        manifest_bytes = Path(manifest_path).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
            raise ValueError('manifest digest mismatch')
        self.manifest = json.loads(manifest_bytes)
        if self.manifest.get('schema') != 'kt-native-mxfp4-ranges-v1':
            raise ValueError('unsupported native range manifest')
        self.fd = os.open(path, os.O_RDONLY)
        try:
            self.identity = file_identity(os.fstat(self.fd))
            if self.identity[2] != self.manifest['file_size']:
                raise ValueError('GGUF size mismatch')
            header_bytes = self.manifest['header_bytes']
            if type(header_bytes) is not int or not 0 < header_bytes <= 16 * 1024 * 1024:
                raise ValueError('invalid header bound')
            header = os.pread(self.fd, header_bytes, 0)
            if len(header) != header_bytes or hashlib.sha256(header).hexdigest() != self.manifest['header_sha256']:
                raise ValueError('GGUF header digest mismatch')
            super().__init__(str(path))
            if file_identity(path.stat()) != self.identity:
                raise ValueError('GGUF changed during header read')
            self.records = {}
            for row in self.manifest['ranges']:
                key = (row['tensor'], row['expert'])
                if key in self.records:
                    raise ValueError('duplicate expert range')
                self.records[key] = row
            self._validate_ranges()
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def _validate_ranges(self):
        for (name, expert), record in self.records.items():
            if not re.fullmatch(r'blk\.\d+\.ffn_(gate|up|down)_exps\.weight', name):
                raise ValueError('unsupported native expert tensor name')
            info = self.tensor_info[name]
            if info['dtype'].name != 'MXFP4' or len(info['shape']) != 3:
                raise ValueError('requires original MXFP4 expert stack')
            count, n, k = info['shape']
            if type(expert) is not int or not 0 <= expert < count or k % 32:
                raise ValueError('bad expert/group geometry')
            size = n*k//32*17
            offset = int(info['offset']) + expert*size
            if size > 64*1024*1024:
                raise ValueError('expert projection exceeds bounded reader capacity')
            if record['offset'] != offset or record['bytes'] != size or record['shape'] != [n,k]:
                raise ValueError('range/header geometry mismatch')
            if offset < self.manifest['header_bytes'] or offset+size > self.identity[2]:
                raise ValueError('range outside source')
            if not re.fullmatch('[0-9a-f]{64}', record['sha256']):
                raise ValueError('missing expert range digest')

    def read_projection_arrays(self, name, expert):
        if self.fd is None:
            raise RuntimeError('reader closed')
        if file_identity(os.fstat(self.fd)) != self.identity:
            raise ValueError('source identity changed')
        record = self.records[(name, expert)]  # Missing identity refuses before read.
        raw = os.pread(self.fd, record['bytes'], record['offset'])
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('expert payload digest mismatch or short read')
        if file_identity(os.fstat(self.fd)) != self.identity:
            raise ValueError('source changed during read')
        n, k = record['shape']
        packed, scales = split_to_adjacent(np.frombuffer(raw,dtype=np.uint8).reshape(-1,17))
        return packed.reshape(n,k//2), scales.reshape(n,k//32)

    def read_projection(self, name, expert):
        packed, scales = self.read_projection_arrays(name, expert)
        w = torch.from_numpy(packed)
        codes = torch.from_numpy(scales).to(torch.int32)
        # UE8M0 code0 is exactly2^-127, BF16 subnormal0x0040; not zero.
        bits = torch.where(codes == 0, 0x0040, codes << 7)
        s = bits.to(torch.int16).view(torch.bfloat16).contiguous()
        return w, s

    def load_experts(self, base_key, device='cpu'):
        if device != 'cpu':
            raise ValueError('native GGUF loader only supports CPU staging')
        match = re.fullmatch(r'(?:model\.)?layers\.(\d+)', base_key)
        if not match:
            raise ValueError('unsupported model key')
        names = {p:f'blk.{int(match[1])}.ffn_{p}_exps.weight' for p in ('gate','up','down')}
        counts = {int(self.tensor_info[name]['shape'][0]) for name in names.values()}
        if len(counts) != 1:
            raise ValueError('inconsistent expert count')
        count = counts.pop()
        gate_shape = [int(x) for x in self.tensor_info[names['gate']]['shape']]
        up_shape = [int(x) for x in self.tensor_info[names['up']]['shape']]
        down_shape = [int(x) for x in self.tensor_info[names['down']]['shape']]
        if gate_shape != up_shape or down_shape != [count,gate_shape[2],gate_shape[1]]:
            raise ValueError('inconsistent gate/up/down geometry')
        # Refuse partial manifests before materializing any layer tensors.
        if any((name,e) not in self.records for name in names.values() for e in range(count)):
            raise ValueError('manifest does not authenticate the entire requested layer')
        result = {key:[] for p in names for key in (p,p+'_scale')}
        for expert in range(count):
            for projection,name in names.items():
                w,s = self.read_projection(name, expert)
                result[projection].append(w)
                result[projection+'_scale'].append(s)
        return result

    def close_all_handles(self):
        self.file_data_map.clear()
        # GGUFLoader metadata can own NumPy views backed by GGUFReader's mmap.
        self.metadata.clear()
        self.tensor_file_map.clear()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
