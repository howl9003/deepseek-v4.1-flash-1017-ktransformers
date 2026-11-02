"""Rank-aware contract v2 routes under SGLANG_PATHA_TP2_EXACT=1.

Each rank owns whole experts; a route is computed once, by its expert's owner, or
by the CPU from rank 0. Finished rows are exchanged, then both ranks run the same
logical-order FP32 fold. Rank subtotals are never reduced: FP32 addition does not
associate, so the fold has to see the rows in the one order TP1 uses.

Rank 0 owns everything by default, the plan's first step: the CPU service, the
resident rows and the hot lend all stay there. SGLANG_PATHA_TP2_OWNERS names a JSON
file {"rank1": {"*": [ids], "<layer>": [ids]}} of physical experts rank 1 takes.
Rank 1 holds resident rows 0..N-1 whole (the loader gives every rank the same rows
once moe_tp_size is 1) and has no lend of its own, so its ids must be below N.

A streamed prefill call stays whole on rank 0, as G1c asks: its fold goes to rank 1
as a bit copy. Both ranks reach the same mode from the same row count.
"""
import json
import os

import torch

from .moe import combine_routes, expert_routes

OWNERS_SELECTOR = "SGLANG_PATHA_TP2_OWNERS"
_file = None
_tables = {}


def _rank1_ids(layer_idx):
    global _file
    path = os.environ.get(OWNERS_SELECTOR, "")
    if not path:
        return []
    if _file is None:
        with open(path) as f:
            _file = json.load(f)["rank1"]
        if any(k != "*" and not k.isdigit() for k in _file):
            raise ValueError(OWNERS_SELECTOR + ": layers are '*' or a layer index")
    return sorted({int(e) for e in _file.get("*", []) + _file.get(str(layer_idx), [])})


def rank1_table(wrapper, device):
    """bool [experts] of this layer's rank 1 experts, or None when it has none.
    Made in eager warm-up, like the logical-order table; graphs keep its storage."""
    layer_idx = wrapper.kt_config.layer_idx
    key = (str(device), layer_idx)
    if key in _tables:
        return _tables[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("TP2 expert owners must be prepared in eager warm-up")
    ids = _rank1_ids(layer_idx)
    if any(not 0 <= e < wrapper.num_gpu_experts for e in ids):
        raise ValueError("rank 1 holds resident rows only: owners must be below --kt-num-gpu-experts")
    table = None
    if ids:
        table = torch.zeros((wrapper.global_num_experts,), dtype=torch.bool, device=device)
        table[torch.tensor(ids, dtype=torch.int64, device=device)] = True
    _tables[key] = table
    return table


def stream_call(rows):
    """stream_prefill_for's decision without its instance: rank 1 builds none."""
    from sglang.srt.layers.moe import kt_stream_prefill as source

    if source.STREAM_THRESHOLD <= 0 or source._failed:
        return False
    if rows < source.STREAM_THRESHOLD and not (source.IN_WINDOW and source._in_lend_window()):
        return False
    return not torch.cuda.is_current_stream_capturing()


class Plan:
    """One MoE call's ownership on this rank."""

    def __init__(self, wrapper, layer, dispatch, order):
        from sglang.srt.layers.quantization.patha.tp2 import prepare
        from sglang.srt.runtime_context import get_parallel

        if layer.moe_tp_size != 1 or layer.moe_ep_size != 1:
            raise RuntimeError("TP2 routes require whole experts: moe_tp_size and moe_ep_size 1")
        self.group = get_parallel().tp_group
        prepare(self.group)
        self.rank = wrapper.tp_rank
        self.layer, self.order = layer, order
        topk = dispatch.topk_output
        self.x = dispatch.hidden_states
        self.all_ids, self.all_weights = topk.topk_ids, topk.topk_weights
        self.active = (self.all_ids >= 0) & (self.all_weights != 0)
        self.streaming = stream_call(self.x.shape[0])
        table = None if self.streaming else rank1_table(wrapper, self.x.device)
        if table is None:
            self.rank1 = torch.zeros_like(self.active)
            self.has_rank1 = False
            self.dispatch, self.ids, self.weights = dispatch, self.all_ids, self.all_weights
            return
        self.rank1 = table[self.all_ids.clamp(min=0)] & self.active
        self.has_rank1 = True
        # Rank 0 sees rank 1's routes as skipped: -1 at weight 0, as the lend leaves them.
        self.ids = torch.where(self.rank1, torch.full_like(self.all_ids, -1), self.all_ids)
        self.weights = torch.where(self.rank1, torch.zeros_like(self.all_weights), self.all_weights)
        self.dispatch = dispatch._replace(
            topk_output=topk._replace(topk_ids=self.ids, topk_weights=self.weights)
        )

    @property
    def peer_only(self):
        return self.rank != 0

    def check_stream(self, streamer):
        if (streamer is not None) != self.streaming:
            raise RuntimeError("TP2 ranks disagree on streaming this call")

    def _combined(self, routes):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        return StandardCombineInput(hidden_states=combine_routes(routes, self.all_ids, order=self.order))

    def _fold_copied(self, fold):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.quantization.patha.tp2 import copy_from_rank0

        return StandardCombineInput(hidden_states=copy_from_rank0(self.group, fold))

    def _exchanged(self, packet):
        from sglang.srt.layers.quantization.patha.tp2 import exchange_rows

        return self._combined(exchange_rows(self.group, packet, self.rank1, self.active))

    def finish(self, routes):
        """Rank 0, after its routes are done."""
        if self.streaming:
            return self._fold_copied(combine_routes(routes, self.all_ids, order=self.order))
        mine = self.active & ~self.rank1
        return self._exchanged(torch.where(mine[:, :, None], routes, torch.full_like(routes, -0.0)))

    def run_peer(self):
        """Rank 1's whole call: its own rows, the exchange, the same fold."""
        from .adapter import _payload

        x, ids = self.x, self.all_ids
        if self.streaming:
            fold = torch.full(x.shape, -0.0, dtype=torch.float32, device=x.device)
            return self._fold_copied(fold)
        packet = torch.full((*ids.shape, x.shape[1]), -0.0, dtype=torch.bfloat16, device=x.device)
        if self.has_rank1:
            rid = torch.where(self.rank1, ids, torch.full_like(ids, -1))
            expert_routes(x, *_payload(self.layer), rid.contiguous(), self.all_weights.contiguous().float(), out=packet)
        return self._exchanged(packet)
