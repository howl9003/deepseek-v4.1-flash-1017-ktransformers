"""NVFP4 W4A4 expert activations in DeepSeek's reference forward (Claude, 1 Nov 2026). An emulation of the arithmetic, not NVIDIA's kernels: neither their inference/ code, which is DeepSeek's byte for byte, nor
a Blackwell tensor core is involved. Only routed experts change, as in nvidia/DeepSeek-V4.1-Flash-1017-NVFP4.

Rule, from vLLM's cvt_warp_fp16_to_fp4 with SFScale 1 (input_scale = 1.0, AtomicChat's uncalibrated cast): per 16 activations, scale = E4M3(amax / 6), zero when that rounds to zero; x / scale rounds to E2M1, ties to
even, saturating at 6 (cvt.rn.satfinite.e2m1x2). Weights: the same E2M1 codes and scales as the reference, so the same values NVFP4 stores. The product runs in FP32 here, against Blackwell's FP4 MMA with its 25 fractional
bits. One BF16 at the output, as the reference's GEMM."""
import torch
import torch.nn.functional as F

VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
ODD = (0.75, 1.75, 3.5)                  # a tie here rounds up to the even code, as in ModelOpt's _cast_fp4
GROUP = 16


def to_e2m1(a, ties_to_zero=False):
    """|a| in [0, 6] to its E2M1 value, ties to even code, or toward zero as NVIDIA's Engram tables do."""
    bounds = torch.tensor(BOUNDS, dtype=torch.float32, device=a.device)
    values = torch.tensor(VALUES, dtype=torch.float32, device=a.device)
    code = torch.searchsorted(bounds, a)
    if not ties_to_zero:
        for b in ODD:
            code = code + (a == b).long()
    return values[code]


def activation(x):
    """x [.., K] to NVFP4 activation values, FP32: E2M1 x E4M3, exact."""
    v = x.float()
    groups = v.unflatten(-1, (-1, GROUP))
    amax = groups.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    inverse = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    y = groups * inverse
    q = to_e2m1(y.abs().clamp_max(6.0)) * y.sign()
    return (q * scale).flatten(-2)


def weight_values(weight):
    """float4_e2m1fn_x2 [N, K / 2] with E8M0 scales per 32 to FP32 [N, K]: the values NVFP4's lossless transcode keeps."""
    table = torch.tensor(VALUES + tuple(-v for v in VALUES), dtype=torch.float32, device=weight.device)
    codes = weight.view(torch.uint8)
    pair = torch.stack((table[(codes & 15).long()], table[(codes >> 4).long()]), -1).flatten(-2)
    return pair * weight.scale.float().repeat_interleave(32, dim=-1)


def install(model):
    ref_linear = model.linear

    def linear_w4a4(x, weight, bias=None):
        if weight.dtype != torch.float4_e2m1fn_x2:
            return ref_linear(x, weight, bias)
        assert bias is None
        return F.linear(activation(x), weight_values(weight)).to(torch.bfloat16)

    model.linear = linear_w4a4
