"""Explicit benchmark-only binding; no automatic compiler or runtime installation."""
import ctypes
from pathlib import Path
import numpy as np

LIB=ctypes.CDLL(str(Path(__file__).parent/'librepack_avx2.so'))
PTR=ctypes.POINTER(ctypes.c_uint8)
LIB.repack_avx2.argtypes=[PTR,PTR,PTR,ctypes.c_size_t]
LIB.repack_avx2.restype=ctypes.c_size_t

def avx2(blocks):
    b=np.asarray(blocks)
    if b.dtype!=np.uint8 or b.ndim!=2 or b.shape[1]!=17:
        raise ValueError('expected [blocks,17] uint8 MXFP4 blocks')
    source=np.ascontiguousarray(b)
    packed=np.empty((len(b),16),dtype=np.uint8)
    scales=np.empty(len(b),dtype=np.uint8)
    rejected=LIB.repack_avx2(source.ctypes.data_as(PTR),packed.ctypes.data_as(PTR),
        scales.ctypes.data_as(PTR),len(b))
    if rejected:raise ValueError('MXFP4 UE8M0 contains reserved NaN code 255')
    return packed,scales
