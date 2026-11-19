"""Block-ordered expert image.

One file a layer, L<nn>.img. For each projection (gate, up, down) and each block b of 32 intermediate columns, one
chunk holding that block of every expert, experts in logical (checkpoint) order:
  gate, up  : 32 rows x hidden/2 bytes an expert (rows 32b..32b+31, the arena's nibble pairs, low nibble = even k)
  down      : hidden rows x 16 bytes an expert (columns 32b..32b+31 of every output row)
Weight chunks first (gate b0..b71, up b0..b71, down b0..b71), then scale chunks in the same order:
  gate, up  : 32 rows x hidden/32 E8M0 bytes an expert
  down      : hidden rows x 1 E8M0 byte an expert
These are byte for byte the expert arenas' bytes with KT_ARENA_SCALE_BYTES=1: a NUMA part holding
blocks [b0, b1) reads one contiguous range of chunks per projection and section, for any split on 32-column blocks.

Arena bytes from the GGUF (the loader through the reader, mxfp4-moe.hpp load_weights): a GGUF MXFP4 block is
1 scale byte + 16 bytes, element j in the low nibble and j+16 in the high nibble of byte j; the arena packs element 2i
in the low and 2i+1 in the high nibble of byte i; the scale byte is the E8M0 code (kt_bf16_scales_to_e8m0 of the BF16
the reader makes from it gives the same byte for codes 0..254; 255 is refused by kt-kernel and here).
"""
import hashlib
import json
import os
import threading

import numpy as np

SCHEMA = "pathb-expert-image-v1"
PROJ = ("gate", "up", "down")
BLOCK = 32          # intermediate columns a block (the part granule)
GROUP = 32          # MXFP4 group
ALIGN = 4096


class Geometry:
    def __init__(self, experts=384, hidden=5120, inter=2304):
        if hidden % GROUP or inter % BLOCK or hidden % 8:
            raise ValueError("geometry not on 32-blocks")
        self.E, self.H, self.I = int(experts), int(hidden), int(inter)
        self.nblk = self.I // BLOCK
        self.w_chunk = self.E * BLOCK * self.H // 2          # == E * H * BLOCK / 2 for down
        self.s_chunk = self.E * BLOCK * self.H // GROUP       # == E * H * BLOCK / 32 for down
        self.w_per = self.w_chunk // self.E                   # one expert's bytes in a weight chunk
        self.s_per = self.s_chunk // self.E
        if self.w_chunk % ALIGN or self.s_chunk % ALIGN:
            raise ValueError("chunks not 4 KiB multiples")
        self.layer_bytes = 3 * self.nblk * (self.w_chunk + self.s_chunk)

    def w_off(self, p, b):
        return (p * self.nblk + b) * self.w_chunk

    def s_off(self, p, b):
        return 3 * self.nblk * self.w_chunk + (p * self.nblk + b) * self.s_chunk

    def as_dict(self):
        return {"experts": self.E, "hidden": self.H, "intermediate": self.I, "block": BLOCK, "group": GROUP,
                "projections": list(PROJ), "w_chunk_bytes": self.w_chunk, "s_chunk_bytes": self.s_chunk,
                "layer_file_bytes": self.layer_bytes, "chunk_order": "weights (gate,up,down) x blocks, then scales"}


def proj_shape(g, p):
    """(rows N, columns K) of one expert's matrix in the checkpoint: gate/up [I, H], down [H, I]."""
    return (g.I, g.H) if p < 2 else (g.H, g.I)


def gguf_to_arena(raw, n, k):
    """One expert's GGUF MXFP4 bytes ([n rows][k/32 blocks][17]) -> (packed [n, k/2] uint8, e8 [n, k/32] uint8)."""
    r = np.frombuffer(raw, np.uint8).reshape(n, k // 32, 17)
    q = r[..., 1:]
    codes = np.concatenate((q & 15, q >> 4), axis=-1)            # element j low nibble, j + 16 high
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(n, k // 2)
    e8 = np.ascontiguousarray(r[..., 0])
    if (e8 == 255).any():
        raise ValueError("E8M0 code 255 (kt-kernel refuses it)")
    return np.ascontiguousarray(packed), e8


def arena_to_gguf(packed, e8):
    """The inverse (tests): (packed [n, k/2], e8 [n, k/32]) -> GGUF bytes."""
    n, kh = packed.shape
    k = kh * 2
    lo, hi = packed & 15, packed >> 4
    el = np.empty((n, k), np.uint8)
    el[:, 0::2], el[:, 1::2] = lo, hi
    el = el.reshape(n, k // 32, 32)
    q = (el[..., :16] | (el[..., 16:] << 4)).astype(np.uint8)
    out = np.empty((n, k // 32, 17), np.uint8)
    out[..., 0] = e8
    out[..., 1:] = q
    return out.tobytes()


def chunk_from_full(g, p, scales, b, full):
    """Chunk (p, b) from the projection's full arrays: full = W [E, N, K/2] or S [E, N, K/32] (checkpoint layout)."""
    if p < 2:
        a = full[:, BLOCK * b:BLOCK * (b + 1), :]                              # 32 rows of every expert
    else:
        w = BLOCK // 2 if not scales else BLOCK // GROUP
        a = full[:, :, w * b:w * (b + 1)]                                      # the block's columns of every row
    return np.ascontiguousarray(a).reshape(-1)


def expert_from_chunks(g, p, scales, chunks, lid):
    """One expert's full matrix back from the 72 chunks of (p, section): W [N, K/2] or S [N, K/32]."""
    n, k = proj_shape(g, p)
    per = g.s_per if scales else g.w_per
    parts = [np.frombuffer(c, np.uint8)[lid * per:(lid + 1) * per] for c in chunks]
    if p < 2:
        cols = k // 2 if not scales else k // GROUP
        return np.concatenate([x.reshape(BLOCK, cols) for x in parts], axis=0)
    w = BLOCK // 2 if not scales else BLOCK // GROUP
    return np.concatenate([x.reshape(n, w) for x in parts], axis=1)


def part_slice(g, p, scales, full_expert, row0, ip):
    """The bytes the loader puts in a part's arena for one expert (rows row0..row0+ip of gate/up; columns of down)."""
    if p < 2:
        return np.ascontiguousarray(full_expert[row0:row0 + ip])
    if not scales:
        return np.ascontiguousarray(full_expert[:, row0 // 2:(row0 + ip) // 2])
    return np.ascontiguousarray(full_expert[:, row0 // GROUP:(row0 + ip) // GROUP])


def part_rows(g, table):
    """[(row0, rows)] of the parts for a block table (blocks a part), e.g. (10, 10, 11, 10, 10, 11, 10)."""
    if sum(table) != g.nblk or any(int(t) <= 0 for t in table):
        raise ValueError(f"block table {table} does not make {g.nblk} blocks")
    out, r = [], 0
    for t in table:
        out.append((r, BLOCK * int(t)))
        r += BLOCK * int(t)
    return out


def equal_table(g, parts):
    if g.nblk % parts:
        raise ValueError(f"{g.nblk} blocks do not split into {parts} equal parts")
    return (g.nblk // parts,) * parts


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def digests_blob(layer_rec):
    """The layer's chunk digests as kt-kernel's image loader takes them: 32 bytes each, weights (gate, up, down) x
    blocks, then scales in the same order."""
    out = bytearray()
    for sec in ("w", "s"):
        for p in range(3):
            for h in layer_rec[sec][p]:
                out += bytes.fromhex(h)
    return bytes(out)


O_BINARY = getattr(os, "O_BINARY", 0)     # Windows (local tests): no text-mode translation
O_DIRECT = getattr(os, "O_DIRECT", 0)     # 0 where the platform has none (local tests)
_io_lock = threading.Lock()               # the lseek + read form below (platforms without pread) is not thread-safe


def _pread_into(fd, mv, off):
    if hasattr(os, "preadv"):
        return os.preadv(fd, [mv], off)
    with _io_lock:
        os.lseek(fd, off, os.SEEK_SET)
        b = os.read(fd, len(mv))
    mv[:len(b)] = b
    return len(b)


def pwrite_all(fd, mv, off):
    n, done = len(mv), 0
    while done < n:
        if hasattr(os, "pwrite"):
            done += os.pwrite(fd, mv[done:], off + done)
        else:
            with _io_lock:
                os.lseek(fd, off + done, os.SEEK_SET)
                done += os.write(fd, mv[done:])


def read_aligned(fd, off, n, direct_buf=None):
    """n bytes at off (pread loop); with buf (an aligned mmap, needed for O_DIRECT) the bytes land in it and a view
    of them is returned."""
    mv = memoryview(direct_buf)[:n] if direct_buf is not None else memoryview(bytearray(n))
    got = 0
    while got < n:
        r = _pread_into(fd, mv[got:], off + got)
        if r <= 0:
            raise OSError(f"short read at {off + got}")
        got += r
    return mv if direct_buf is not None else bytes(mv)


def load_manifest(image_dir, pin=None):
    raw = open(os.path.join(image_dir, "manifest.json"), "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    if pin is not None and h != pin:
        raise ValueError(f"manifest sha256 {h} is not the pinned {pin}")
    m = json.loads(raw)
    if m.get("schema") != SCHEMA:
        raise ValueError("manifest schema " + str(m.get("schema")))
    return m, h
