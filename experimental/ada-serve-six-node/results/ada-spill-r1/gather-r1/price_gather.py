"""Price the SM89 dual-scope gather on the card (Claude, 1 Nov 2026), no model. A captured graph of 40 calls on distinct queries and indices, replayed with synchronize, the way the served
token runs it. Finds the token counts that reproduce the capture's 11, 33 and 46 us, then the unsplit kernel against forced split-K. argv: none."""
import os
os.environ.setdefault('SGLANG_SM89_ATTN_SPLITK', '0')
import statistics as st
import time

import torch

from sglang.kernels.ops.attention import sm89_dual_scope as S

CALLS = 40
HEADS, DIM, BYTES = 64, 512, 584
dev = 'cuda'


def kv(blocks, block_size, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    t = torch.empty(blocks, block_size, BYTES, dtype=torch.uint8, device=dev)
    data = t.view(blocks, -1)
    # E4M3 codes well inside the range, rope BF16 small, E8M0 scales near 2^-4
    nope = (torch.randint(0x20, 0x48, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8)
            | (torch.randint(0, 2, (blocks, block_size, 448), generator=g, device=dev, dtype=torch.uint8) << 7))
    rope = (torch.randn(blocks, block_size, 64, generator=g, device=dev) * 0.05).to(torch.bfloat16).view(torch.uint8).reshape(blocks, block_size, 128)
    row = torch.cat((nope, rope), dim=2).reshape(blocks, -1)
    scales = torch.randint(120, 128, (blocks, block_size, 8), generator=g, device=dev, dtype=torch.uint8).reshape(blocks, -1)
    data.copy_(torch.cat((row, scales), dim=1))
    return t


def case(main, extra, cache):
    """Arguments of one call; distinct queries and indices so nothing repeats."""
    q = (torch.randn(1, 1, HEADS, DIM, device=dev) * 0.3).to(torch.bfloat16)
    k_main, k_extra = cache
    tokens = k_main.shape[0] * k_main.shape[1]
    idx = torch.randperm(tokens, device=dev)[:main].to(torch.int32).reshape(1, 1, -1)
    tl = torch.tensor([main], dtype=torch.int32, device=dev)
    sink = torch.randn(HEADS, device=dev).to(torch.float32)
    if extra:
        tokens = k_extra.shape[0] * k_extra.shape[1]
        idy = torch.randperm(tokens, device=dev)[:extra].to(torch.int32).reshape(1, 1, -1)
        return q, k_main, idx, tl, sink, DIM, None, k_extra, idy, torch.tensor([extra], dtype=torch.int32, device=dev)
    return q, k_main, idx, tl, sink, DIM, None, None, None, None


def price(main, extra, cache, mode):
    S._SPLITK_MODE = mode
    args = [case(main, extra, cache) for _ in range(CALLS)]
    S.sm89_dual_scope_decode(*args[0])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    keep = []
    with torch.cuda.graph(g):
        for a in args:
            keep.append(S.sm89_dual_scope_decode(*a))
    torch.cuda.synchronize()
    runs = []
    for _ in range(30):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) * 1e6 / CALLS)
    return st.median(runs)


cache = (kv(2048, 256, 1), kv(512, 256, 2))
print('model', S.__file__, 'mode', S._SPLITK_MODE, flush=True)
for main, extra in ((256, 0), (128, 128), (512, 0), (512, 128), (512, 256), (768, 0), (768, 128), (768, 256), (1024, 0), (1024, 256)):
    print(f'main {main:5d} extra {extra:4d}: unsplit {price(main, extra, cache, "0"):6.1f} us, split2 {price(main, extra, cache, "2"):6.1f}, split4 {price(main, extra, cache, "4"):6.1f}, split8 {price(main, extra, cache, "8"):6.1f}', flush=True)
