"""Default-off TP2 route that keeps TP1's bits (SGLANG_PATHA_TP2_EXACT=1).

Two row-parallel layers, wo_b and the shared expert's down, change to column-only
under this selector: each rank gathers the whole K input, computes its half of N
over the whole K, and the outputs are gathered. Every output row of an integer
path A projection is a function of the whole-K activation row, that row's weights
and one row exponent, so an N split changes no bit and a K split does.

Transports (SGLANG_PATHA_TP2_TRANSPORT):
  all_gather  the guide's form: copies only, no arithmetic on any value.
  reduce      each rank fills its peer's positions with -0.0 and all-reduces, so
              decode and prefill can use the two-GPU LL and host-staged all-reduces.
              Bit-exact only if the collective adds x + (-0.0) = x for every x,
              signed zeros and subnormals included; qualify on the cards first.

Everything here runs on the forward's main stream. The LL kernel's sequence and
NCCL's ordering both assume one issue order per group, so the shared expert does
not run on the alternate stream under this selector (deepseek_v2.DeepseekV2MoE).
"""
import os

SELECTOR = "SGLANG_PATHA_TP2_EXACT"
TRANSPORT_SELECTOR = "SGLANG_PATHA_TP2_TRANSPORT"

ENABLED = os.environ.get(SELECTOR, "0") == "1"
TRANSPORT = os.environ.get(TRANSPORT_SELECTOR, "all_gather")
if TRANSPORT not in ("all_gather", "reduce"):
    raise ValueError(TRANSPORT_SELECTOR + " must be all_gather or reduce")
# reduce leaves the output on the shards; project_column gathers it with the filler.
COLUMN_GATHER_OUTPUT = TRANSPORT == "all_gather"
_NEG_ZERO = -0.0

_comms = {}


def validate_parallel(parallel):
    """Refuse every layout this route is not defined for, before weights load."""
    wrong = [
        name
        for name, value in (
            ("tp_size", 2),
            ("attn_tp_size", 2),
            ("attn_dp_size", 1),
            ("attn_cp_size", 1),
            ("moe_ep_size", 1),
            ("moe_dp_size", 1),
        )
        if getattr(parallel, name) != value
    ]
    if wrong:
        raise RuntimeError(SELECTOR + " requires two attention-TP ranks; check: " + ", ".join(wrong))
    if getattr(parallel, "enable_attn_tp_input_scattered", False):
        raise RuntimeError(SELECTOR + " excludes attention-TP input scattering")
    if os.environ.get("SGLANG_SHARED_EXPERT_TP1", "0") not in ("", "0"):
        raise RuntimeError(SELECTOR + " excludes SGLANG_SHARED_EXPERT_TP1")
    if os.environ.get("KT_CONTRACT_V2") != "1":
        raise RuntimeError(SELECTOR + " requires KT_CONTRACT_V2=1")


def prepare(group):
    """Warm-up, outside capture: the two-GPU collectives for the reduce transport.

    Built here rather than in GroupCoordinator, so the selector-off tree is untouched.
    """
    if TRANSPORT != "reduce" or id(group) in _comms:
        return
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(SELECTOR + " transports must be prepared outside graph capture")
    from sglang.srt.distributed.device_communicators.host_staged_allreduce import (
        HostStagedAllReduce,
    )
    from sglang.srt.distributed.device_communicators.ll_allreduce import LLAllReduce

    _comms[id(group)] = (
        HostStagedAllReduce(group.cpu_group, group.rank_in_group, group.world_size, group.device),
        LLAllReduce(group.cpu_group, group.rank_in_group, group.world_size, group.device),
    )


def _reduce(group, t):
    prepare(group)
    host, ll = _comms[id(group)]
    if host.should_use(t):
        return host.all_reduce(t)
    if ll.should_use(t):
        return ll.all_reduce(t)
    return group.all_reduce(t)


def gather_last_dim(group, shard):
    """Whole rows from equal shards, ranks in order along the last dimension."""
    shard = shard.contiguous()
    if TRANSPORT == "all_gather":
        return group.all_gather(shard, -1)
    world, rank, width = group.world_size, group.rank_in_group, shard.shape[-1]
    full = shard.new_full((*shard.shape[:-1], world * width), _NEG_ZERO)
    full[..., rank * width : (rank + 1) * width] = shard
    return _reduce(group, full)


def project_column(layer, group, x):
    """The column-only form of a row-parallel projection: x is this rank's K/2."""
    y, _ = layer(gather_last_dim(group, x))
    return y if layer.gather_output else gather_last_dim(group, y)


def exchange_rows(group, packet, rank1, active):
    """Whole route rows on both ranks. packet [T, K, H] holds this rank's finished
    rows and -0.0 elsewhere; rank1 [T, K] marks the rows rank 1 owns, the same on
    both ranks; active [T, K] the routes that have an owner. Rows are BF16 values,
    so they travel as BF16: FP32 would add no bit and leaves LL's dtype."""
    import torch

    if TRANSPORT == "all_gather":
        both = group.all_gather(packet.contiguous(), 0)
        rows = packet.shape[0]
        out = torch.where(rank1[:, :, None], both[rows:], both[:rows])
    else:
        out = _reduce(group, packet)
    # TP1's route buffer starts at +0: a slot nobody fills stays canonical.
    return torch.where(active[:, :, None], out, torch.zeros_like(out))


def copy_from_rank0(group, value):
    """Rank 0's FP32 fold on both ranks, bit for bit (streamed prefill only: the
    route rows of 35K tokens are too large to exchange). Rank 1 passes a buffer of
    -0.0 of the same shape."""
    if TRANSPORT == "all_gather":
        group.broadcast(value, src=0)
        return value
    return _reduce(group, value)
