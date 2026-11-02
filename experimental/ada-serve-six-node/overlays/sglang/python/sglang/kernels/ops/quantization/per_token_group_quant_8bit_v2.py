"""JIT port of ``sgl_per_token_group_quant_8bit_v2``.

``sglang.kernels.ops.quantization.per_token_group_quant`` is the default CUDA
path. This kernel still serves the row-major UE8M0 layout stored as fp32
power-of-two scales, which per_token_group_quant does not produce (see
``fp8_kernel._run_per_token_group_quant_8bit_kernel``), and is the perf baseline
for the per_token_group_quant benchmarks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.layers.quantization.patha.policy import (
    AMAX_FLOOR as _PATHA_AMAX_FLOOR, enabled as _patha_enabled,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(in_dtype: torch.dtype, out_dtype: torch.dtype, use_pdl: bool, strict_patha: bool = False) -> Module:
    args = make_cpp_args(in_dtype, out_dtype, use_pdl)
    return load_jit(
        "per_token_group_quant_8bit_v2_patha_floor1em4_r2" if strict_patha else "per_token_group_quant_8bit_v2",
        *args,
        cuda_files=["gemm/per_token_group_quant_8bit_v2.cuh"],
        cuda_wrappers=[
            (
                "per_token_group_quant_8bit_v2",
                f"PerTokenGroupQuant8bitV2Kernel<{args}>::run",
            )
        ],
        # Match the AOT sgl-kernel build (-use_fast_math) so the FP8 scale
        # division/rounding is bit-identical to sgl_per_token_group_quant_8bit_v2.
        extra_cuda_cflags=["-fmad=false", "-ftz=false", "-DSGLANG_PATHA_QUANTIZER=1"] if strict_patha else ["--use_fast_math"],
    )


@register_custom_op(
    op_name="per_token_group_quant_8bit_v2",
    mutates_args=["output_q", "output_s"],
)
def _per_token_group_quant_8bit_v2_custom_op(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """Opaque custom-op boundary around the JIT v2 kernel.

    Registering this as a custom op (instead of calling the tvm-ffi module
    directly) keeps torch.compile / piecewise-CUDA-graph from tracing into the
    tvm-ffi ``Function.__call__`` (which Dynamo cannot trace). All shape-derived
    scalars are computed here and passed to the kernel.

    Layouts (matching the AOT v2):
      vanilla:                   input (num_tokens, hidden), output_q (num_tokens, hidden)
      fuse_silu_and_mul:         input (num_tokens, hidden*2), output_q (num_tokens, hidden)
      fuse_silu_and_mul+masked:  input (num_experts, tokens_pad, hidden*2),
                                 output_q (num_experts, tokens_pad, hidden), masked_m (num_experts,)
    """
    masked_layout = masked_m is not None
    numel = input.numel()
    num_groups = numel // group_size // (2 if fuse_silu_and_mul else 1)
    if num_groups == 0:  # empty input -> grid 0 -> cudaErrorInvalidConfiguration
        return
    num_local_experts = input.shape[0] if masked_layout else 1
    last = output_q.dim() - 1
    is_column_major = output_s.stride(last - 1) < output_s.stride(last)
    hidden_dim_num_groups = output_q.shape[last] // group_size
    num_tokens_per_expert = output_q.shape[last - 1]
    scale_expert_stride = output_s.stride(0) if masked_layout else 0
    scale_hidden_stride = output_s.stride(last)

    strict_patha = _patha_enabled() and eps == _PATHA_AMAX_FLOOR
    expected_eps = _PATHA_AMAX_FLOOR if strict_patha else 1e-10
    if eps != expected_eps:
        raise ValueError(f"v2 quantizer native floor is {expected_eps}, got {eps}")
    if strict_patha:
        if not (group_size == 32 and scale_ue8m0 and output_s.dtype == torch.float32
                and input.dtype == torch.bfloat16 and output_q.dtype == torch.float8_e4m3fn
                and not is_column_major and not fuse_silu_and_mul and masked_m is None
                and (min_8bit, max_8bit) == (-448.0, 448.0)):
            raise ValueError("Path A requires BF16 -> E4M3, group32, row-major FP32 UE8M0 scales")
    module = _jit_module(input.dtype, output_q.dtype, is_arch_support_pdl(), strict_patha)
    module.per_token_group_quant_8bit_v2(
        input,
        output_q,
        output_s,
        masked_m if masked_layout else input,  # unused (nullptr) when not masked
        int(group_size),
        bool(scale_ue8m0),
        bool(fuse_silu_and_mul),
        bool(masked_layout),
        int(num_groups),
        int(num_local_experts),
        bool(is_column_major),
        int(hidden_dim_num_groups),
        int(num_tokens_per_expert),
        int(scale_expert_stride),
        int(scale_hidden_stride),
    )


@debug_kernel_api
def per_token_group_quant_8bit_v2(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """JIT port of sgl_per_token_group_quant_8bit_v2 (full feature parity).

    Wraps the registered custom op so torch.compile / piecewise CUDA graph treat
    the tvm-ffi kernel call as an opaque boundary.
    """
    _per_token_group_quant_8bit_v2_custom_op(
        input=input,
        output_q=output_q,
        output_s=output_s,
        group_size=group_size,
        eps=eps,
        min_8bit=min_8bit,
        max_8bit=max_8bit,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )
    if _patha_enabled() and eps == _PATHA_AMAX_FLOOR and input.numel() > 0:
        # Registration is after the custom-op call so tracked version counters
        # include its mutation. In a CUDA graph, this op is recorded before the
        # consumer; replay recomputes scales on the strict floor on every input.
        from sglang.srt.layers.quantization.patha.runtime import register_quantized
        register_quantized(output_q, output_s)
