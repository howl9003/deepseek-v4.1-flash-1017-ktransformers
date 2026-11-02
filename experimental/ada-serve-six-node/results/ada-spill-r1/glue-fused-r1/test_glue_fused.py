"""Bitwise test of KT_CONTRACT_V2_GLUE_FUSED on the card, no model (Claude, 31 Oct 2026). Run with the trial's python and tree on PYTHONPATH.
argv: trial root. The fused kernels against the adapter's own torch ops and kernels: every route output, the counts, the combine, the two GPU
passes, and the flag bits on broken tables and weights. Unowned slots hold junk in the fused buffer, to prove they are never read."""
import os
import sys
import tempfile
from pathlib import Path

T = Path(sys.argv[1])
os.environ.update(KT_KV_HOT_LEND='1', KT_KV_HOT_EXTRA='auto', KT_KV_HOT_DYNAMIC_SLOTS='1', KT_KV_HOT_DYNAMIC_RESIDENT='1', KT_GPU_MASK_ZERO='1',
                  KT_CONTRACT_V2='1', KT_CONTRACT_V2_FAST_ADAPTER='1', KT_CONTRACT_V2_STATIC_GPU='1', KT_CONTRACT_V2_GPU_MODE='int8',
                  SGLANG_FP8_PATHA_EXPERT_STATIC='1', SGLANG_FP8_PATHA_EXPERT_STATIC_DIR=str(T / 'packages'),
                  SGLANG_FP8_PATHA_EXPERT_STATIC_BINARY=str(T / 'packages/patha_expert_static_sm89_e0_s3_b8_t0.so'),
                  SGLANG_FP8_PATHA_EXPERT_STATIC_ORIGIN_DIR=tempfile.mkdtemp(), PATHA_EXPERT_LUT='0', PATHA_STAGES='3', PATHA_EXPERT_BPC='8', PATHA_EXPERT_TC='0')
import torch

import sglang.srt.layers.moe.contract_v2 as contract_v2
contract_v2.__path__.insert(0, str(Path(__file__).resolve().parent))   # the overlay beside this test, on the unmodified tree
from sglang.srt.layers.moe import kt_kv_hot_lend as hot
from sglang.srt.layers.moe.contract_v2 import fast_combine, glue_fused, moe, patha_static_runtime as static

DEV = torch.device('cuda', 0)
E, NRES, NSLOT, H = 384, 2, 24, 5120


class L:
    idx = 0


def tables(rng):
    tbl = torch.full((E,), -1, dtype=torch.int32)
    rtbl = torch.full((E,), -1, dtype=torch.int32)
    perm = torch.randperm(E, generator=rng)
    tbl[perm[:NSLOT]] = torch.arange(NSLOT, dtype=torch.int32)
    rtbl[perm[NSLOT - 2:NSLOT + NRES - 2]] = torch.arange(NRES, dtype=torch.int32)   # two ids in both tables: the row wins
    hot.STATE['tbl'], hot.STATE['rtbl'] = [tbl.to(DEV)], [rtbl.to(DEV)]
    hot.STATE['cnt'] = [torch.zeros(E, device=DEV)]


def routes_in(rng, t, k, dtype):
    ids = torch.randint(-1, E, (t, k), generator=rng).to(dtype)
    w = torch.rand((t, k), generator=rng)
    w[torch.rand((t, k), generator=rng) < .2] = 0
    return ids.to(DEV), w.to(DEV).float()


def old_route(ids, w):
    """The adapter's own derivation from kt_kv_hot_lend's kernel."""
    cid, cw, sid, sw, rid = hot._route_fused(L, ids, w)
    cpu_owned = (cid >= 0) & (cw != 0)
    res_owned = (rid >= 0) & (rid < NRES) & (w != 0)
    slot_owned = (sid >= 0) & (sid < NSLOT) & (sw != 0)
    owners = cpu_owned.to(torch.int32) + res_owned.to(torch.int32) + slot_owned.to(torch.int32)
    assert torch.equal(owners, ((ids >= 0) & (w != 0)).to(torch.int32)), 'reference routes are not exactly once'
    active_r = torch.where((rid >= 0) & (rid < NRES) & (w != 0), rid, -1)
    active_s = torch.where((sid >= 0) & (sid < NSLOT) & (sw != 0), sid, -1)
    return cid, cw, active_r, active_s, sw, cpu_owned, res_owned | slot_owned


def equal(a, b):
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    return torch.equal(a.view(-1).contiguous().view(torch.uint8), b.view(-1).contiguous().view(torch.uint8))


pa = static._module()
static.warmup(DEV)
flag = static._flag
rng = torch.Generator().manual_seed(1017)
order = torch.randperm(E, generator=rng).to(DEV)
routes_checked = combines = 0
for trial in range(400):
    t, k = (1 + trial % 8), 6
    dtype = torch.int64 if trial % 3 == 0 else torch.int32
    tables(rng)
    ids, w = routes_in(rng, t, k, dtype)
    if trial % 7 == 0:
        ids[:, 1] = ids[:, 0]    # a token routed twice to one expert
    cid, cw, rid, sid, sw, cpu_owned, gpu_owned = old_route(ids, w)
    cnt_old = hot.STATE['cnt'][0].clone()
    hot.STATE['cnt'][0].zero_()
    fid, fw, frid, fsid, own = glue_fused.route(hot, L, ids, w, NRES, NSLOT, flag)
    torch.cuda.synchronize()
    assert int(flag.item()) == 0
    assert equal(fid, cid) and equal(fw, cw), trial
    assert torch.equal(frid.long(), rid.long()) and torch.equal(fsid.long(), sid.long()), trial
    assert frid.dtype == rid.dtype and fsid.dtype == sid.dtype
    assert equal(hot.STATE['cnt'][0], cnt_old), trial
    assert torch.equal(own == glue_fused.OWN_CPU, cpu_owned) and torch.equal(own == glue_fused.OWN_GPU, gpu_owned), trial
    # sw is the slots' weight only: at an active route it equals w, and the activation reads no other weight
    assert torch.equal(sw[fsid >= 0], w[fsid >= 0])
    routes_checked += t * k

    gpu = (torch.randn((t, k, H), generator=rng) * 3).to(DEV).to(torch.bfloat16)
    cpu = (torch.randn((t, k, H), generator=rng) * 3).to(DEV).to(torch.bfloat16)
    gpu[0, 0, :8] = torch.tensor([0., -0., 1e-30, -1e-30, 1e30, -1e30, 1., -1.], dtype=torch.bfloat16)
    zeroed = torch.where(gpu_owned[:, :, None], gpu, torch.zeros_like(gpu))
    old = fast_combine.combine(torch.where(cpu_owned[:, :, None], cpu, zeroed), ids, order=order)
    both = torch.cat([gpu, cpu]).view(2 * t * k, H).contiguous()
    both.view(torch.int16)[torch.cat([own.view(-1) != glue_fused.OWN_GPU, own.view(-1) != glue_fused.OWN_CPU])] = 0x7fc1   # junk: a NaN if ever read
    new = glue_fused.combine(both, ids, own, order)
    assert equal(old, new), trial
    combines += 1
torch.cuda.synchronize()
assert int(flag.item()) == 0

# the flag bits, then clear them by hand: nothing else may touch the word
tables(torch.Generator().manual_seed(2))
ids = torch.arange(6, dtype=torch.int32, device=DEV).view(1, 6) + 100
w = torch.full((1, 6), .25, device=DEV)


def bits(ids, w, nres=NRES, nslot=NSLOT):
    flag.zero_()
    glue_fused.route(hot, L, ids, w, nres, nslot, flag)
    torch.cuda.synchronize()
    return int(flag.item())


assert bits(ids, w) == 0
hot.STATE['tbl'][0][ids[0, 2]] = NSLOT + 5
hot.STATE['rtbl'][0][ids[0, 2]] = -1
assert bits(ids, w) == glue_fused.FLAG_OWNERS, 'slot beyond the shadow'
hot.STATE['tbl'][0][ids[0, 2]] = -1
hot.STATE['rtbl'][0][ids[0, 3]] = NRES
assert bits(ids, w) == glue_fused.FLAG_OWNERS, 'row beyond the residents'
hot.STATE['rtbl'][0][ids[0, 3]] = -1
assert bits(ids, w) == 0
for value in (float('nan'), float('inf'), float('-inf')):
    bad = w.clone(); bad[0, 4] = value
    assert bits(ids, bad) & glue_fused.FLAG_WEIGHT, value
pa.FLAG_NAMES.update({glue_fused.FLAG_OWNERS: 'route ownership not exactly once', glue_fused.FLAG_WEIGHT: 'non-finite MoE weights'})
assert 'route ownership' in pa.describe_flags(64) and 'non-finite' in pa.describe_flags(128)
flag.zero_()

# the two GPU passes: expert_routes_patha twice against the pair, on synthetic experts
pass_rng = torch.Generator().manual_seed(3)


def experts(n):
    w13 = torch.randint(0, 256, (n, pa.TWO_I, H // 2), generator=pass_rng, dtype=torch.uint8).to(DEV)
    s13 = torch.randint(118, 124, (n, pa.TWO_I, H // 32), generator=pass_rng, dtype=torch.uint8).to(DEV)
    w2 = torch.randint(0, 256, (n, H, pa.INTER // 2), generator=pass_rng, dtype=torch.uint8).to(DEV)
    s2 = torch.randint(118, 124, (n, H, pa.INTER // 32), generator=pass_rng, dtype=torch.uint8).to(DEV)
    return w13, s13, w2, s2


resident, shadow = experts(NRES), experts(5)
passes = 0
for t, dtype in ((1, torch.int32), (1, torch.int64), (2, torch.int32), (4, torch.int32), (8, torch.int32)):
    k = 6
    x = (torch.randn((t, H), generator=pass_rng) * 2).to(DEV).to(torch.bfloat16)
    w = torch.rand((t, k), generator=pass_rng).to(DEV).float()
    w[torch.rand((t, k), generator=pass_rng).to(DEV) < .2] = 0
    rid = torch.full((t, k), -1, dtype=dtype, device=DEV)
    sid = torch.full((t, k), -1, dtype=torch.int64, device=DEV)
    for row in range(t):
        rid[row, 0], sid[row, 1], sid[row, 2], rid[row, 3] = 0, row % 5, (row + 1) % 5, 1
    sw = torch.where(sid >= 0, w, torch.zeros_like(w))
    rid_active = torch.where(w != 0, rid, -1)
    sid_active = torch.where(w != 0, sid, -1)
    old = torch.zeros((t, k, H), dtype=torch.bfloat16, device=DEV)
    pa.expert_routes_patha(x, *resident, rid, w, out=old)
    pa.expert_routes_patha(x, *shadow, sid, sw, out=old)
    new = torch.full((t, k, H), 7., dtype=torch.bfloat16, device=DEV)
    glue_fused.expert_pair(pa, x, resident, shadow, rid_active, sid_active, w, new)
    torch.cuda.synchronize()
    written = (rid_active >= 0) | (sid_active >= 0)
    assert written.any()
    assert equal(old[written], new[written]), (t, dtype)
    assert (old[~written] == 0).all()
    passes += 1
assert int(flag.item()) == 0, pa.describe_flags(int(flag.item()))
print('GLUE_FUSED_ALL_PASS', dict(routes=routes_checked, combines=combines, pass_cases=passes))
