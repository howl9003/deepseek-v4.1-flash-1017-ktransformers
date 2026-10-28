"""Byte-only MXFP4 layout candidates; no weight decoding or requantization.

Copyright 2026 Strata adaptation contributors. SPDX-License-Identifier: Apache-2.0
"""
import numpy as np


def checked(blocks):
    b=np.asarray(blocks)
    if b.dtype!=np.uint8 or b.ndim!=2 or b.shape[1]!=17:
        raise ValueError('expected [blocks,17] uint8 MXFP4 blocks')
    if np.any(b[:,0]==255):
        raise ValueError('MXFP4 UE8M0 contains reserved NaN code 255')
    return b


def planes(blocks):
    """Copy alternate bytes into contiguous planes before NumPy bit operations."""
    b=checked(blocks)
    even,odd=np.ascontiguousarray(b[:,1:].reshape(len(b),8,2).transpose(2,0,1))
    packed=np.empty((len(b),16),dtype=np.uint8)
    packed[:,:8]=(even&15)|((odd&15)<<4)
    packed[:,8:]=(even>>4)|(odd&240)
    return packed,b[:,0].copy()


# For little-endian pair a,b, map to packed even/odd low nibbles and high
# nibbles. Every possible 16-bit input is representable; no float operation.
_words=np.arange(65536,dtype=np.uint16)
_low=(_words&15)|((_words>>4)&240)
_high=((_words>>4)&15)|((_words>>8)&240)
PAIR_LUT=(_low|(_high<<8)).astype('<u2')
PAIR_LUT.flags.writeable=False
del _words,_low,_high


def pair_lut(blocks):
    """One 128KiB table maps two source bytes to two packed output bytes."""
    b=checked(blocks)
    q=b[:,1:]
    if q.strides[-1]!=1:
        q=np.ascontiguousarray(q)
    # A 17-byte row stride is legal; NumPy handles unaligned uint16 indices.
    mapped=PAIR_LUT[q.view('<u2')].view(np.uint8)
    packed=np.empty((len(b),16),dtype=np.uint8)
    packed[:,:8]=mapped[:,0::2]
    packed[:,8:]=mapped[:,1::2]
    return packed,b[:,0].copy()


VARIANTS={'planes':planes,'pair_lut':pair_lut}
