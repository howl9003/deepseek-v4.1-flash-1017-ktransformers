"""DSpark draft routed experts in the KT CPU pool (KT_DSPARK_CPU_DRAFT=1, default off; Claude, 3 Nov 2026).

Installed as sglang/srt/layers/moe/kt_dspark_draft.py by apply_dspark_cpu.py and imported only when the selector is on.
The three DSpark stages' MoE layers are built inside draft_model_build_scope (flags.moe.in_speculative_scope). With the
selector on, each gets a KT wrapper of its own:

- KT layer id KT_DSPARK_DRAFT_LAYER_BASE + stage (40, 41, 42), in KT_DRAFT_LAYERS, never in the target's KT_LAYERS;
- no GPU-resident expert (--kt-num-gpu-experts does not apply): no hot-lend shadow, no dynamic slot, no streamed
  prefill, no target loader, no target expert-location map (drafts have no placement: physical id == logical id);
- its own loader on the draft GGUF (KT_DSPARK_DRAFT_GGUF) with its own range manifest (KT_DSPARK_DRAFT_MANIFEST,
  pinned by KT_DSPARK_DRAFT_MANIFEST_SHA256), through kt_kernel.utils.amx._OWN_LOADER (overlaid kt_kernel);
- the same CPU backend and pool as the target: kt-kernel's contract v2 MoE (KT_MXFP4_BACKEND=contract_v2) on the
  shared CPUInfer, K-partitioned over the target's KT_NUMA_NODES parts;
- its own pinned handoff buffers keyed by (rows, hidden, top-k): kt-kernel's KExpertsCPUBuffer is keyed by rows only
  and sized by the first caller's top-k (the target's 6), so a top-3 draft at the same row count would get [T, 6] id
  buffers;
- the apply_tolerant.py kt-kernel build with KT_CONTRACT_V2_TOLERANT_FROM = the draft's first KT layer id, checked at
  construction (check_tolerant): stage 2 reaches the contract's truncation regime on ordinary inputs, where the served
  build refuses and poisons a process-wide counter for the target too.

The forward is the contract adapter's CPU-only case: every route to the CPU per-route API, the result buffer as in
adapter.apply, one FP32 combine in ascending expert id (combine order None: the draft ids are logical).
"""
import dataclasses
import functools
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import KTConfig

STAGES = 3
_HOST = {}


@dataclasses.dataclass
class DraftKTConfig(KTConfig):
    draft_stage: int = -1
    manifest: Optional[str] = None
    manifest_sha256: Optional[str] = None


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"KT_DSPARK_CPU_DRAFT=1 needs {name}")
    return value


def layer_base(num_target_layers: int) -> int:
    base = int(os.environ.get("KT_DSPARK_DRAFT_LAYER_BASE", str(num_target_layers)))
    if base < num_target_layers:
        raise RuntimeError(f"KT_DSPARK_DRAFT_LAYER_BASE {base} overlaps the target's KT layers 0..{num_target_layers - 1}")
    return base


def draft_kt_config(server_args, layer_idx: int) -> Optional[DraftKTConfig]:
    """create_kt_config_from_server_args for a MoE layer built inside the draft scope, else None (the target's)."""
    from sglang.srt.arg_groups.overrides import model_config_of
    from sglang.srt.layers.moe import kt_ep_wrapper as w
    from sglang.srt.runtime_context import get_exec, get_flags, get_schedule

    if not get_flags().moe.in_speculative_scope:
        return None
    if not 0 <= layer_idx < STAGES:
        raise RuntimeError(f"KT_DSPARK_CPU_DRAFT: draft MoE layer id {layer_idx} is not a DSpark stage 0..{STAGES - 1}")
    moe = get_exec().moe
    if moe.kt_method != "MXFP4":
        raise RuntimeError("KT_DSPARK_CPU_DRAFT: the draft experts need --kt-method MXFP4 (contract v2 CPU backend)")
    num_target_layers = int(getattr(model_config_of(server_args).hf_config, "num_hidden_layers"))
    manifest = _required("KT_DSPARK_DRAFT_MANIFEST")
    pin = _required("KT_DSPARK_DRAFT_MANIFEST_SHA256")
    return DraftKTConfig(
        layer_idx=layer_base(num_target_layers) + layer_idx,
        num_gpu_experts=0,
        cpuinfer_threads=moe.kt_cpuinfer,
        threadpool_count=moe.kt_threadpool_count,
        weight_path=_required("KT_DSPARK_DRAFT_GGUF"),
        chunked_prefill_size=w._kt_cpu_moe_max_len(get_schedule().chunked_prefill_size),
        max_deferred_experts_per_token=0,
        method=moe.kt_method,
        num_layers=None,  # never the target's last layer: no stream init, no deferral rule, no page-cache drop
        draft_stage=layer_idx,
        manifest=manifest,
        manifest_sha256=pin,
    )


def is_draft(method) -> bool:
    return isinstance(getattr(method, "kt_config", None), DraftKTConfig)


def _reader(path: str, manifest: str, pin: str):
    from kt_kernel.utils.recipe_gguf_reader import NativeMXFP4GGUFReader

    return NativeMXFP4GGUFReader(path, manifest, pin)


def loader_factory(cfg: DraftKTConfig):
    return functools.partial(_reader, cfg.weight_path, cfg.manifest, cfg.manifest_sha256)


def check_tolerant(cfg: DraftKTConfig) -> None:
    """Refuse at load, not mid-serving: stage 2's down projection reaches the contract's truncation regime on ordinary
    inputs, and the served kt-kernel then refuses the call and sets a process-wide, never-reset counter that stops every
    later target call too. The draft needs the apply_tolerant.py build with KT_CONTRACT_V2_TOLERANT_FROM = the draft's
    first KT layer id (target layers stay strict). KT_DSPARK_DRAFT_STRICT=1 skips this, for experiments only."""
    if os.environ.get("KT_DSPARK_DRAFT_STRICT", "0") == "1":
        return
    from kt_kernel import kt_kernel_ext

    base = cfg.layer_idx - cfg.draft_stage
    tol = os.environ.get("KT_CONTRACT_V2_TOLERANT_FROM", "")
    if not hasattr(kt_kernel_ext, "contract_v2_tolerant_events") or tol != str(base):
        raise RuntimeError(f"KT_DSPARK_CPU_DRAFT needs the apply_tolerant.py kt-kernel build and KT_CONTRACT_V2_TOLERANT_FROM={base} "
                           f"(got {tol!r}, tolerant build: {hasattr(kt_kernel_ext, 'contract_v2_tolerant_events')})")


def create_wrapper(method, layer, num_experts: int, num_experts_per_tok: int, hidden_size: int,
                   intermediate_size_full: int) -> None:
    """KTEPWrapperMethod.create_weights after the GPU method made its (zero) resident experts."""
    from sglang.srt.layers.moe import contract_v2
    from sglang.srt.layers.moe import kt_ep_wrapper as w

    cfg = method.kt_config
    if not contract_v2.ENABLED:
        raise RuntimeError("KT_DSPARK_CPU_DRAFT needs KT_CONTRACT_V2=1: the draft runs the contract adapter's per-route path")
    if method.num_gpu_experts != 0 or method._fp32_subtotals:
        raise RuntimeError("KT_DSPARK_CPU_DRAFT: draft layers hold no GPU expert and no FP32 subtotal mode")
    if cfg.layer_idx in w.KT_DRAFT_LAYERS or cfg.layer_idx in w.KT_LAYERS:
        raise RuntimeError(f"KT_DSPARK_CPU_DRAFT: KT layer {cfg.layer_idx} registered twice")
    if method.tp_rank != 0:
        return
    from kt_kernel.utils import amx

    if not hasattr(amx, "_OWN_LOADER"):
        raise RuntimeError("KT_DSPARK_CPU_DRAFT needs the overlaid kt_kernel (per-wrapper loader, apply_dspark_cpu.py)")
    check_tolerant(cfg)
    token = amx._OWN_LOADER.set(loader_factory(cfg))
    try:
        method.wrapper = w.KTMoEWrapper(
            layer_idx=cfg.layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size_full,
            gpu_experts_mask=torch.zeros(num_experts, dtype=torch.bool),
            cpuinfer_threads=cfg.cpuinfer_threads,
            threadpool_count=cfg.threadpool_count,
            weight_path=cfg.weight_path,
            chunked_prefill_size=cfg.chunked_prefill_size,
            method=cfg.method,
            max_deferred_experts_per_token=0,
            **w._kt_swiglu_kwargs(layer),
            numa_nodes=(
                [int(x) for x in os.environ["KT_NUMA_NODES"].split(",")]
                if os.environ.get("KT_NUMA_NODES")
                else None
            ),
        )
    finally:
        amx._OWN_LOADER.reset(token)
    # the draft GGUF names its stages blk.0..2, whatever the KT layer id
    method.wrapper.weight_base_key = f"model.layers.{cfg.draft_stage}"
    w.KT_DRAFT_LAYERS[cfg.layer_idx] = method


def _drop_page_cache(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def load(method, layer) -> None:
    """KTEPWrapperMethod.process_weights_after_loading for a draft layer: the CPU experts only."""
    if method.tp_rank != 0 or method.wrapper is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    identity = torch.arange(method.global_num_experts, dtype=torch.int64)
    method.wrapper.load_weights(identity)
    if os.environ.get("KT_DSPARK_DRAFT_DROP_CACHE", "1") == "1":
        _drop_page_cache(method.kt_config.weight_path)


def _host_buffers(x: torch.Tensor, k: int):
    from kt_kernel.experts_base import KExpertsCPUBuffer

    key = (x.shape[0], x.shape[1], k)
    if key not in _HOST:
        t, h, depth = x.shape[0], x.shape[1], KExpertsCPUBuffer.buffer_depth
        pin = torch.cuda.is_available()
        _HOST[key] = (
            [torch.zeros((t, h), dtype=torch.bfloat16, pin_memory=pin) for _ in range(depth)],
            [torch.zeros((t, k), dtype=torch.int64, pin_memory=pin) for _ in range(depth)],
            [torch.zeros((t, k), dtype=torch.float32, pin_memory=pin) for _ in range(depth)],
            [torch.full((1,), t, dtype=torch.int32, pin_memory=pin) for _ in range(depth)],
        )
    return _HOST[key]


def submit_routes(native, x, ids, weights, packer=None) -> None:
    """adapter.submit_routes with the draft's own (rows, hidden, top-k) pinned buffers."""
    from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer
    from sglang.srt.layers.moe.contract_v2 import adapter

    if native.max_deferred_experts_per_token or not hasattr(native.moe, "forward_routes_task"):
        raise RuntimeError("contract v2 routes require the native route API (AVX2ContractV2_MOE) and zero deferred experts")
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("contract v2 expects CUDA BF16 hidden states")
    if not 1 <= x.shape[0] <= native.chunked_prefill_size:
        raise ValueError(f"contract v2 rows {x.shape[0]} exceed native capacity {native.chunked_prefill_size}")
    if ids.shape != (x.shape[0], native.num_experts_per_tok) or weights.shape != ids.shape:
        raise ValueError("contract v2 route shapes mismatch")
    if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
        raise ValueError("contract v2 route IDs/weights require int32 or int64 IDs and float32 weights")
    host_x, host_ids, host_weights, host_bsz = _host_buffers(x, ids.shape[1])
    slot = native.layer_idx % KExpertsCPUBuffer.buffer_depth
    host_out, _ = adapter._route_buffers(native, x, ids.shape[1])
    packed = bool(packer is not None and packer(x, weights, ids, host_x[slot], host_weights[slot], host_ids[slot]))
    if not packed:
        host_x[slot].copy_(x, non_blocking=True)
        host_ids[slot].copy_(ids, non_blocking=True)
        host_weights[slot].copy_(weights, non_blocking=True)
    stream = torch.cuda.current_stream(x.device)
    task = native.moe.forward_routes_task(host_bsz[slot].data_ptr(), ids.shape[1], host_ids[slot].data_ptr(),
                                          host_weights[slot].data_ptr(), host_x[slot].data_ptr(), host_out.data_ptr())
    native.cpu_infer.submit_with_cuda_stream(stream.cuda_stream, task)
    BaseMoEWrapper._layer_has_pending_deferred[native.layer_idx] = False


def apply(wrapper, layer, dispatch):
    """adapter.apply for a draft layer: every route on the CPU, then the contract combine."""
    from sglang.srt.layers.moe import kt_ep_wrapper
    from sglang.srt.layers.moe.contract_v2 import adapter
    from sglang.srt.layers.moe.contract_v2.moe import combine_routes
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    if wrapper.tp_rank != 0 or layer.moe_tp_size != 1 or layer.moe_ep_size != 1:
        raise RuntimeError("KT_DSPARK_CPU_DRAFT: draft experts on the CPU require one GPU rank")
    if wrapper.num_gpu_experts != 0:
        raise RuntimeError("KT_DSPARK_CPU_DRAFT: a draft layer holds no GPU expert")
    x = dispatch.hidden_states
    ids = dispatch.topk_output.topk_ids
    weights = dispatch.topk_output.topk_weights
    cid = torch.where(ids >= 0, ids, -1)
    cpu_owned = (cid >= 0) & (weights != 0)
    routes = torch.zeros((*ids.shape, x.shape[1]), dtype=torch.bfloat16, device=x.device)
    submit_routes(wrapper.wrapper, x, cid, weights,
                  kt_ep_wrapper.pack_to_host if getattr(kt_ep_wrapper, "_KT_HANDOFF_PACK", False) else None)
    if getattr(kt_ep_wrapper, "_KT_SHARED_AFTER_SUBMIT", False):
        wrapper._kt_post_submit_event = torch.cuda.current_stream(x.device).record_event()
    cpu = adapter.sync_routes(wrapper.wrapper, x, ids.shape[1])
    torch.where(cpu_owned[:, :, None], cpu, routes, out=routes)
    torch._assert_async(torch.isfinite(weights).all(), "non-finite MoE weights")
    return StandardCombineInput(hidden_states=combine_routes(routes, ids, order=None))
