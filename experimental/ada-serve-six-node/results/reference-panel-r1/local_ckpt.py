"""DeepSeek's reference forward on fea01's weights (Claude, 1 Nov 2026). The HF shards are gone from this box, so this stands in for sudoingX's dsv41q.refmodel.Checkpoint: routed experts come from
the GGUF, everything else from the local original-* safetensors. Bits, not values: the GGUF's expert blocks are a lossless relayout of the HF bytes and invert exactly (verify_ckpt.py checks samples
against the Hub). DiskEngramEmbedding is replaced because it needs vLLM's reader; the arithmetic is unchanged. Import this before dsv41q.stream builds a Streamer."""
import json
import mmap
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'third_party'))
from dsv41q import refmodel, stream  # noqa: E402

CKPT = '/scratch/aster/bench/05yuki-baseline-3991/checkpoint'
GGUF = '/scratch/aster/models/gguf/DeepSeek-V4.1-Flash-1017-MXFP4.gguf'
CACHE = os.path.join(HERE, 'gguf_tensors.json')
PROJ = {'w1': 'gate', 'w2': 'down', 'w3': 'up'}
ROWS = {'w1': 2304, 'w2': 5120, 'w3': 2304}
BLOCKS = {'w1': 160, 'w2': 72, 'w3': 160}
EXPERTS = 384
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _string(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n)


def _skip_value(f, kind):
    if kind in SIZES:
        f.seek(SIZES[kind], 1)
    elif kind == 8:
        n = struct.unpack('<Q', f.read(8))[0]
        f.seek(n, 1)
    elif kind == 9:
        inner, n = struct.unpack('<IQ', f.read(12))
        if inner in SIZES:
            f.seek(n * SIZES[inner], 1)
        else:
            for _ in range(n):
                _skip_value(f, inner)
    else:
        raise ValueError(kind)


def gguf_tensors(path=GGUF):
    """{name: absolute data offset}, read from the header. Cached: the header walks every tokenizer string."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    with open(path, 'rb', buffering=1 << 20) as f:
        assert f.read(4) == b'GGUF'
        version, tensors, kvs = struct.unpack('<IQQ', f.read(20))
        assert version == 3
        alignment = 32
        for _ in range(kvs):
            key = _string(f)
            kind = struct.unpack('<I', f.read(4))[0]
            if key == b'general.alignment':
                alignment = struct.unpack('<I', f.read(4))[0]
            else:
                _skip_value(f, kind)
        info = []
        for _ in range(tensors):
            name = _string(f).decode()
            dims = struct.unpack('<I', f.read(4))[0]
            f.seek(8 * dims, 1)
            kind, offset = struct.unpack('<IQ', f.read(12))
            info.append((name, offset))
        start = (f.tell() + alignment - 1) // alignment * alignment
    out = {name: start + offset for name, offset in info}
    json.dump(out, open(CACHE, 'w'))
    return out


def expert_tensor_name(layer, proj):
    return f'blk.{layer}.ffn_{PROJ[proj]}_exps.weight'


def invert_blocks(raw, rows, blocks):
    """raw uint8 [n, rows, blocks * 17]: (one scale byte, 16 bytes holding element j low and j + 16 high) per group. Returns (HF packed [n, rows, blocks * 16], scales [n, rows, blocks])."""
    x = raw.view(raw.size(0), rows, blocks, 17)
    scale = x[..., 0]
    q = x[..., 1:]
    vals = torch.cat((q & 0x0f, q >> 4), -1)      # elements 0..15 then 16..31
    packed = vals[..., 0::2] | (vals[..., 1::2] << 4)
    return packed.flatten(2), scale


class LocalCheckpoint(refmodel.Checkpoint):
    """Same interface as refmodel.Checkpoint. Every routed expert of a layer reads from the GGUF, including the 21 a layer that also sit in the local original-gpu-experts files."""

    def __init__(self, ckpt_dir=CKPT):
        super().__init__(ckpt_dir)
        assert os.path.realpath(ckpt_dir) == os.path.realpath(CKPT)
        self.offsets = gguf_tensors()
        self.gguf = open(GGUF, 'rb', buffering=0)
        self._expert = re.compile(r'^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$')
        for layer in range(40):
            for e in range(EXPERTS):
                for proj in PROJ:
                    for part in ('weight', 'scale'):
                        self.weight_map[f'layers.{layer}.ffn.experts.{e}.{proj}.{part}'] = '@gguf'

    def _pinned(self, nbytes):
        if torch.cuda.is_available():
            return super()._pinned(nbytes)
        b = getattr(self, '_buf', None)            # verify_ckpt.py runs with no GPU
        if b is None or b.numel() < nbytes:
            self._buf = b = torch.empty(max(nbytes, 1 << 30), dtype=torch.uint8, device='cpu')
        return b

    def _read(self, offset, nbytes, device):
        buf = self._pinned(nbytes)
        mv, got = memoryview(buf.numpy()), 0
        self.gguf.seek(offset)
        while got < nbytes:
            r = self.gguf.readinto(mv[got: nbytes])
            if not r:
                raise IOError(f'short read at {offset + got}')
            got += r
        os.posix_fadvise(self.gguf.fileno(), offset, nbytes, os.POSIX_FADV_DONTNEED)
        t = buf[:nbytes].to(device, non_blocking=True)
        if str(device) != 'cpu':
            torch.cuda.current_stream().synchronize()
        return t.clone() if str(device) == 'cpu' else t

    def expert_projection(self, layer, proj, device):
        """All experts of one projection: (packed uint8 [384, rows, cols // 2], scales uint8 [384, rows, blocks])."""
        rows, blocks = ROWS[proj], BLOCKS[proj]
        nbytes = EXPERTS * rows * blocks * 17
        raw = self._read(self.offsets[expert_tensor_name(layer, proj)], nbytes, device)
        return invert_blocks(raw.view(EXPERTS, rows, blocks * 17), rows, blocks)

    def get_many(self, names, device='cpu', gap=64 << 20):
        plain, routed = [], {}
        for n in names:
            m = self._expert.match(n)
            if m:
                routed.setdefault((int(m[1]), m[3]), []).append((n, int(m[2]), m[4]))
            else:
                plain.append(n)
        out = super().get_many(plain, device, gap) if plain else {}
        for (layer, proj), members in routed.items():
            # Unpacked on the host: the parameters already sit on the card, and a panel's streams leave no room for a second copy there (first panel run: out of memory in layer 0).
            packed, scales = self.expert_projection(layer, proj, 'cpu')
            for n, e, part in members:
                if part == 'weight':
                    out[n] = packed[e].view(torch.float4_e2m1fn_x2)
                else:
                    out[n] = scales[e].view(torch.float8_e8m0fnu)
            del packed, scales
        return out


class MemmapEngram(torch.nn.Module):
    """refmodel.DiskEngramEmbedding with a memory map in place of vLLM's reader. Slower, same bytes, same dequantization."""

    def __init__(self, ckpt, layer_id, block_size=32, threads=0):
        super().__init__()
        wpath, woff, wdt, wshape = ckpt.locate(f'layers.{layer_id}.engram.embed.weight')
        spath, soff, sdt, sshape = ckpt.locate(f'layers.{layer_id}.engram.embed.scale')
        assert wdt == 'F8_E4M3' and sdt == 'F8_E8M0', (wdt, sdt)
        self.rows, self.dim = wshape
        self.sdim = sshape[1]
        self.block_size = block_size
        self.w = np.memmap(wpath, np.uint8, 'r', offset=woff, shape=tuple(wshape))
        self.s = np.memmap(spath, np.uint8, 'r', offset=soff, shape=tuple(sshape))
        # Rows are scattered: readahead around each fault read 764 GB for three layers of 24 pieces on the first try.
        self.w._mmap.madvise(mmap.MADV_RANDOM)
        self.s._mmap.madvise(mmap.MADV_RANDOM)

    @staticmethod
    def _rows(table, ids):
        with ThreadPoolExecutor(16) as pool:
            return np.concatenate(list(pool.map(lambda c: np.array(table[c]), np.array_split(ids, 64))))

    @torch.inference_mode()
    def forward(self, indices):
        dev = indices.device
        uniq, inv = torch.unique(indices.reshape(-1), return_inverse=True)
        ids = uniq.cpu().numpy()
        w = torch.from_numpy(self._rows(self.w, ids)).to(dev).view(torch.float8_e4m3fn)
        s = torch.from_numpy(self._rows(self.s, ids)).to(dev).view(torch.float8_e8m0fnu)
        vals = w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)
        if MXFP4_ENGRAM:
            vals = mxfp4_row(vals)
        vals = vals.flatten(-2).to(torch.bfloat16)
        return vals[inv].view(*indices.shape, self.dim)


def mxfp4_row(vals):
    """NVIDIA's lossy Engram tables from the FP8 rows: scale 2^ceil(log2(amax / 6)) per 32, then E2M1. verify_engram_fp4.py checks this rule against their bytes."""
    import w4a4
    m, e = torch.frexp(vals.abs().amax(-1, keepdim=True).clamp_min(1e-30).double() / 6)
    scale = torch.exp2((e - (m == 0.5).long()).double()).float()
    y = vals / scale
    return w4a4.to_e2m1(y.abs().clamp_max(6.0), True) * y.sign() * scale


MXFP4_ENGRAM = False
refmodel.Checkpoint = LocalCheckpoint
stream.Checkpoint = LocalCheckpoint
refmodel.DiskEngramEmbedding = MemmapEngram
