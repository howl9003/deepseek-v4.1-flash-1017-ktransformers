# Generated from the private reviewed gpu_v2.py; only table import differs.
"""Default-off Triton counterpart of contract_v2.py. SOURCE ONLY: not compiled or executed in the
session that wrote it (no GPU); run validate_gpu_v2.py under the fea01 guard once the benchmark
campaign has released the GPU.

Three projection kernels share the contract's int64 accumulation and differ only in how the
exact 32-wide group dot D is produced:
  gemv   : int32 multiply/sum on CUDA cores (decode; any GPU from SM61, no tensor cores)
  int8   : three signed 6-bit planes through tl.dot int8 -> int32 (SM75+/SM80+; exact by ISA)
  fp8hl  : E4M3 operands split into |v|>=2 / |v|<2 halves, two tl.dot float8 -> float32
           (SM120/SM100: exact under the measured 25-bit alignment; NOT exact on SM89, 13 bits)
"""
import weakref
import numpy as np
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .tables import FP8, SILU, N_OF_CODE, W_INT, MARGIN as _MARGIN
MARGIN = tl.constexpr(_MARGIN)   # Triton kernels may only read globals that are tl.constexpr

@tr.jit
def _bf(x):
    b = x.to(tl.uint32, bitcast=True)
    return ((b + 32767 + ((b >> 16) & 1)) & 0xffff0000).to(tl.float32, bitcast=True)

@tr.jit
def _quant_v2(X, Q, CODES, EA, HI, LO, P0, P1, P2, FTAB, NTAB):
    block = tl.program_id(0)
    j = tl.arange(0, 32)
    v = tl.load(X + block * 32 + j)
    am = tl.maximum(tl.max(tl.abs(v), 0), 1.e-4)
    b = (am * (1. / 448.)).to(tl.uint32, bitcast=True)
    ex = ((b >> 23) & 255).to(tl.int32) + ((b & 0x7fffff) != 0).to(tl.int32) - 127
    scale = ((ex + 127) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
    z = v / scale
    a = tl.abs(z); ab = a.to(tl.uint32, bitcast=True)
    normal = ((ab + 0x7ffff + ((ab >> 20) & 1)) >> 20).to(tl.int32) - 960
    sub = a * 512.; flo = tl.floor(sub); frac = sub - flo
    rounded = flo.to(tl.int32) + ((frac > .5) | ((frac == .5) & ((flo.to(tl.int32) & 1) != 0))).to(tl.int32)
    code = tl.where(a >= 448., 126, tl.where(a < .015625, rounded, normal))
    sign = ((z.to(tl.uint32, bitcast=True) >> 31) & 1).to(tl.int32)
    code = code | (sign << 7)
    n = tl.load(NTAB + code)
    sg = 1 - 2 * sign
    tl.store(CODES + block * 32 + j, code.to(tl.uint8))
    tl.store(Q + block * 32 + j, tl.load(FTAB + code))
    tl.store(HI + block * 32 + j, (sg * (n >> 9)).to(tl.int16))
    tl.store(LO + block * 32 + j, (sg * (n & 511)).to(tl.int16))
    tl.store(P0 + block * 32 + j, (sg * (n & 63)).to(tl.int8))
    tl.store(P1 + block * 32 + j, (sg * ((n >> 6) & 63)).to(tl.int8))
    tl.store(P2 + block * 32 + j, (sg * (n >> 12)).to(tl.int8))
    tl.store(EA + block, ex.to(tl.int8))

@tr.jit
def _finish(acc, eat, ewc):
    # acc int64 [BM, BN]; one RNE in the int64->fp32 conversion, then an exact power-of-two scale.
    B = eat[:, None] + ewc[None, :] - MARGIN
    return _bf(libdevice.ldexp(acc.to(tl.float32), B - 10))

@tr.jit
def _gemv_v2(HI, LO, EAG, EAT, W, WS, EW, WLUT, Y, N, K: tl.constexpr, BN: tl.constexpr):
    t = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN); nm = n < N
    j = tl.arange(0, 32)
    G: tl.constexpr = K // 32
    ewc = tl.load(EW + n, nm, 0).to(tl.int32)
    eat = tl.load(EAT + t).to(tl.int32)
    acc = tl.zeros((BN,), tl.int64)
    for g in range(G):
        kk = g * 32 + j
        byte = tl.load(W + n[:, None] * (K // 2) + kk[None, :] // 2, nm[:, None], 0).to(tl.int32)
        wi = tl.load(WLUT + ((byte >> ((kk[None, :] & 1) * 4)) & 15))
        a = (tl.load(HI + t * K + kk).to(tl.int32) << 9) + tl.load(LO + t * K + kk).to(tl.int32)
        D = tl.sum(a[None, :] * wi, axis=1)                      # exact int32, any order
        wsg = tl.load(WS + n * G + g, nm, 127).to(tl.int32)
        sh = (tl.load(EAG + t * G + g).to(tl.int32) - eat) + (wsg - 127 - ewc) + MARGIN
        D64 = D.to(tl.int64)
        # floor(D 2^sh): the right shift is clamped to 63 (gives 0 or -1), the left shift is at most MARGIN;
        # a shift by 64 or more is undefined at the LLVM level even though PTX would clamp it.
        acc += tl.where(sh >= 0, D64 << tl.maximum(sh, 0), D64 >> tl.minimum(0 - sh, 63))
    B = eat + ewc - MARGIN
    tl.store(Y + t * N + n, _bf(libdevice.ldexp(acc.to(tl.float32), B - 10)), nm)

@tr.jit
def _gemm_v2(A0, A1, A2, Q, EAG, EAT, W, WS, EW, WLUT8, WLUTF, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, FP8_SPLIT: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM); rm = rows < M
    cols = tl.program_id(1) * BN + tl.arange(0, BN); cm = cols < N
    j = tl.arange(0, 32)
    G: tl.constexpr = K // 32
    eat = tl.load(EAT + rows, rm, 0).to(tl.int32); ewc = tl.load(EW + cols, cm, 0).to(tl.int32)
    acc = tl.zeros((BM, BN), tl.int64)
    for g in range(G):
        kk = g * 32 + j
        byte = tl.load(W + cols[None, :] * (K // 2) + kk[:, None] // 2, cm[None, :], 0).to(tl.int32)
        code = (byte >> ((kk[:, None] & 1) * 4)) & 15                                   # [32, BN]
        if FP8_SPLIT:
            q = tl.load(Q + rows[:, None] * K + kk[None, :], rm[:, None], 0.)         # exact E4M3 values
            b = tl.load(WLUTF + code).to(tl.float8e4nv)                                 # FP4 values are exact in E4M3
            hi = tl.where(tl.abs(q) >= 2., q, 0.).to(tl.float8e4nv); lo = tl.where(tl.abs(q) < 2., q, 0.).to(tl.float8e4nv)
            dot = tl.dot(hi, b, out_dtype=tl.float32) + tl.dot(lo, b, out_dtype=tl.float32)
            D = (dot * 1024.).to(tl.int64)                                              # exact: dot is a multiple of 2^-10
        else:
            b = tl.load(WLUT8 + code)                                                   # int8 [32, BN]
            a0 = tl.load(A0 + rows[:, None] * K + kk[None, :], rm[:, None], 0)
            a1 = tl.load(A1 + rows[:, None] * K + kk[None, :], rm[:, None], 0)
            a2 = tl.load(A2 + rows[:, None] * K + kk[None, :], rm[:, None], 0)
            d0 = tl.dot(a0, b, out_dtype=tl.int32); d1 = tl.dot(a1, b, out_dtype=tl.int32); d2 = tl.dot(a2, b, out_dtype=tl.int32)
            D = ((d2 << 12) + (d1 << 6) + d0).to(tl.int64)
        eag = tl.load(EAG + rows * G + g, rm, 0).to(tl.int32); wsg = tl.load(WS + cols * G + g, cm, 127).to(tl.int32)
        sh = (eag[:, None] - eat[:, None]) + (wsg[None, :] - 127 - ewc[None, :]) + MARGIN
        acc += tl.where(sh >= 0, D << tl.maximum(sh, 0), D >> tl.minimum(0 - sh, 63))   # clamped floor shift, as in _gemv_v2
    tl.store(Y + rows[:, None] * N + cols[None, :], _finish(acc, eat, ewc), rm[:, None] & cm[None, :])

@tr.jit
def _activate(GU, R, TABLE, H, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(1); j = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(GU + row * D * 2 + j, j < D, 0); u = tl.load(GU + row * D * 2 + D + j, j < D, 0)
    si = tl.load(TABLE + (g.to(tl.uint32, bitcast=True) >> 16)); w = tl.load(R + row)
    tl.store(H + row * D + j, _bf((si * tl.minimum(10., tl.maximum(-10., u))) * w), j < D)

@tr.jit
def _combine(X, ORDER, SHARED, Y, T: tl.constexpr, H: tl.constexpr, B: tl.constexpr, HAS_SHARED: tl.constexpr):
    row = tl.program_id(1); j = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0., tl.float32)
    for e in range(T):
        slot = tl.load(ORDER + row * T + e)
        acc = acc + tl.load(X + (row * T + slot) * H + j, j < H, 0.)
    if HAS_SHARED:
        acc = acc + tl.load(SHARED + row * H + j, j < H, 0.)
    tl.store(Y + row * H + j, _bf(acc), j < H)

class GPU_V2:
    def __init__(self, *, enabled=False, device='cuda', mode='int8', fast_checks=False):
        if enabled is not True: raise RuntimeError('Research prototype requires explicit enabled=True')
        self.device = torch.device(device)
        if self.device.type != 'cuda': raise ValueError('CUDA required')
        if mode not in ('gemv', 'int8', 'fp8hl'): raise ValueError(mode)
        self.mode = mode
        self.fast_checks = fast_checks
        self._weight_metadata = {}
        self.ftab = torch.from_numpy(FP8.copy()).to(self.device); self.silu = torch.from_numpy(SILU.copy()).to(self.device)
        self.ntab = torch.from_numpy(N_OF_CODE.astype(np.int32)).to(self.device)
        self.wlut = torch.from_numpy(W_INT.astype(np.int32)).to(self.device); self.wlut8 = torch.from_numpy(W_INT.astype(np.int8)).to(self.device)
        self.wlutf = torch.from_numpy((W_INT * 0.5).astype(np.float32)).to(self.device)

    def _input(self, x, rank, bf16=False):
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.device != self.ftab.device: raise ValueError('tensor must be on adapter CUDA device')
        if x.dtype != torch.float32 or x.ndim != rank or any(d == 0 for d in x.shape): raise ValueError('nonempty float32 tensor with expected rank required')
        if self.fast_checks:
            torch._assert_async(torch.isfinite(x).all(), 'finite values required')
            if bf16: torch._assert_async((x.view(torch.int32) & 65535 == 0).all(), 'BF16-representable input required')
        else:
            if not torch.isfinite(x).all().item(): raise ValueError('finite values required')
            if bf16 and torch.any(x.view(torch.int32) & 65535).item(): raise ValueError('BF16-representable input required')
        return x.contiguous()

    def quant(self, x):
        x = self._input(x, 2); m, k = x.shape
        if k % 32: raise ValueError('width must be divisible by 32')
        q = torch.empty_like(x); codes = torch.empty((m, k), dtype=torch.uint8, device=x.device); ea = torch.empty((m, k // 32), dtype=torch.int8, device=x.device)
        hi = torch.empty((m, k), dtype=torch.int16, device=x.device); lo = torch.empty_like(hi)
        p = [torch.empty((m, k), dtype=torch.int8, device=x.device) for _ in range(3)]
        _quant_v2[(m * k // 32,)](x, q, codes, ea, hi, lo, p[0], p[1], p[2], self.ftab, self.ntab, enable_fp_fusion=False, num_warps=1)
        EA = ea.view(m, k // 32).max(dim=1).values.to(torch.int8)
        return dict(q=q, codes=codes, ea=ea, hi=hi, lo=lo, planes=p, EA=EA)

    def prepare_weights(self, ws):
        if not self.fast_checks:
            if torch.any(ws == 255).item() or torch.any(ws < 32).item(): raise ValueError('weight scale code outside the contract range')
            return (ws.to(torch.int32).max(dim=1).values - 127).to(torch.int8)
        # Cache only versioned tensors. Views share their root's mutation counter;
        # retaining a weak reference avoids extending the weight storage lifetime.
        root=ws
        while root._base is not None: root=root._base
        try: version=root._version
        except RuntimeError: version=None
        key=(id(root),ws.data_ptr(),tuple(ws.shape),tuple(ws.stride()))
        old=self._weight_metadata.get(key)
        if version is not None and old is not None and old[0]() is root and old[1]==version:
            return old[2]
        torch._assert_async(((ws >= 32) & (ws < 255)).all(), 'weight scale code outside the contract range')
        result=(ws.to(torch.int32).max(dim=1).values - 127).to(torch.int8)
        if version is not None:
            cache=self._weight_metadata
            cache[key]=(weakref.ref(root,lambda _:cache.pop(key,None)),version,result)
        return result

    def linear(self, qd, w, ws, EW=None):
        m, k = qd['hi'].shape; n = w.shape[0]
        if w.ndim != 2 or k % 32 or w.shape[1] != k // 2 or ws.shape != (n, k // 32): raise ValueError('packed weight shape mismatch')
        if EW is None: EW = self.prepare_weights(ws)
        y = torch.empty((m, n), dtype=torch.float32, device=w.device); G = k // 32
        if self.mode == 'gemv':
            _gemv_v2[(tr.cdiv(n, 64), m)](qd['hi'], qd['lo'], qd['ea'], qd['EA'], w.contiguous(), ws.contiguous(), EW, self.wlut, y, n, k, 64, enable_fp_fusion=False, num_warps=4)
        else:
            _gemm_v2[(tr.cdiv(m, 32), tr.cdiv(n, 64))](qd['planes'][0], qd['planes'][1], qd['planes'][2], qd['q'], qd['ea'], qd['EA'], w.contiguous(), ws.contiguous(), EW,
                                                      self.wlut8, self.wlutf, y, m, n, k, 32, 64, self.mode == 'fp8hl', enable_fp_fusion=False, num_warps=4)
        return y

    def activate(self, gu, weight):
        gu = self._input(gu, 2, True); weight = self._input(weight, 1); m, width = gu.shape
        if width % 2 or weight.shape != (m,): raise ValueError('activation shape mismatch')
        d = width // 2; h = torch.empty((m, d), dtype=torch.float32, device=gu.device)
        _activate[(tr.cdiv(d, 128), m)](gu, weight, self.silu, h, d, 128, enable_fp_fusion=False)
        return h

    def expert(self, x, w13, s13, w2, s2, weight, *, stages=False):
        x = self._input(x, 2, True)
        qd = self.quant(x); gu = self.linear(qd, w13, s13); h = self.activate(gu, weight); hd = self.quant(h); out = self.linear(hd, w2, s2)
        return (out, dict(input_codes=qd['codes'], gate_up=gu, activation=h, down_codes=hd['codes'])) if stages else out

    def combine(self, experts, expert_ids, shared=None):
        experts = self._input(experts, 3, True); m, topk, hidden = experts.shape
        order = torch.argsort(expert_ids.to(torch.int64), dim=1, stable=True).to(torch.int32).contiguous()
        y = torch.empty((m, hidden), dtype=torch.float32, device=experts.device)
        sh = experts if shared is None else self._input(shared, 2, True)
        _combine[(tr.cdiv(hidden, 128), m)](experts, order, sh, y, topk, hidden, 128, shared is not None, enable_fp_fusion=False)
        return y
