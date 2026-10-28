"""Pinned 9c0f6d6 SGLang flat /get_server_info profile extraction.

The endpoint merges ServerArgs.resolved_dict(), launch_command, scheduler_info,
and HTTP metadata. Keep exact key-shape checks so new metadata cannot silently
be mistaken for a server argument or silently ignored.
"""

from __future__ import annotations

import math
import json
from typing import Any, Callable


TOP_SUFFIX = (
    "launch_command", "status", "max_total_num_tokens", "max_req_input_len",
    "startup_time", "kv_events", "internal_states", "version", "frontend",
)
INTERNAL_EXTRA = frozenset({
    "world_size", "startup_time", "effective_max_running_requests_per_dp",
    "last_gen_throughput", "memory_usage", "decode_cuda_graph_bs",
    "decode_cuda_graph_memory_gb",
})
INTERNAL_ABSENT_ARGS = frozenset({"custom_sigquit_handler"})
MEMORY_KEYS = frozenset({
    "weight", "kvcache", "startup_available", "token_capacity",
    "token_capacity_swa", "graph",
})
GRAPH_MEMORY_KEYS = frozenset({
    "prefill", "decode", "target_verify", "draft_prefill", "draft_decode",
    "draft_extend",
})
STARTUP_KEYS = frozenset({
    "load_weight", "kv_cache_allocation", "scheduler_e2e", "cuda_graph",
    "tokenizer_e2e",
})


def _finite_nonnegative(value: Any, label: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid nonnegative numeric telemetry: {label}")


def _startup(value: Any, label: str, tokenizer: bool) -> None:
    keys = STARTUP_KEYS if tokenizer else STARTUP_KEYS - {"tokenizer_e2e"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"unknown startup timing shape: {label}")
    graph = value["cuda_graph"]
    if not isinstance(graph, dict) or set(graph) != GRAPH_MEMORY_KEYS:
        raise ValueError(f"unknown startup graph timing shape: {label}")
    for key, item in value.items():
        if key == "cuda_graph":
            for phase, duration in graph.items():
                _finite_nonnegative(duration, f"{label}.cuda_graph.{phase}")
        else:
            _finite_nonnegative(item, f"{label}.{key}")


def split_server_info(info: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(info, dict):
        raise ValueError("server info must be an object")
    keys = list(info)
    if len(keys) <= len(TOP_SUFFIX) or tuple(keys[-len(TOP_SUFFIX):]) != TOP_SUFFIX:
        raise ValueError("unknown flat server-info shape or metadata tail")
    args = {k: info[k] for k in keys[:-len(TOP_SUFFIX)]}
    if not {"model_path", "context_length", "chunked_prefill_size",
            "random_seed", "cuda_graph_config", "init_expert_location"} <= set(args):
        raise ValueError("essential resolved server arguments absent")
    if info["status"] != "ready" or info["frontend"] != "python":
        raise ValueError("unexpected server readiness/frontend")
    if not isinstance(info["launch_command"], str) or not info["launch_command"]:
        raise ValueError("missing launch command")
    if not isinstance(info["version"], str) or not info["version"]:
        raise ValueError("missing server version")
    _startup(info["startup_time"], "http", tokenizer=True)
    for key in ("max_total_num_tokens", "max_req_input_len"):
        if type(info[key]) is not int or info[key] <= 0:
            raise ValueError(f"invalid effective capacity: {key}")
    internal = info["internal_states"]
    if not isinstance(internal, list) or len(internal) != 1 or not isinstance(internal[0], dict):
        raise ValueError("expected one scheduler internal-state record")
    state = internal[0]
    expected = (set(args) - INTERNAL_ABSENT_ARGS) | INTERNAL_EXTRA
    if set(state) != expected:
        raise ValueError("unknown scheduler internal-state shape")
    memory = state["memory_usage"]
    if not isinstance(memory, dict) or set(memory) != MEMORY_KEYS:
        raise ValueError("unknown scheduler memory-usage shape")
    if not isinstance(memory["graph"], dict) or set(memory["graph"]) != GRAPH_MEMORY_KEYS:
        raise ValueError("unknown scheduler graph-memory shape")
    for key in ("weight", "kvcache", "startup_available"):
        _finite_nonnegative(memory[key], f"memory_usage.{key}")
    for key, value in memory["graph"].items():
        _finite_nonnegative(value, f"memory_usage.graph.{key}")
    for key in ("token_capacity", "token_capacity_swa"):
        if type(memory[key]) is not int or memory[key] <= 0:
            raise ValueError(f"invalid scheduler memory capacity: {key}")
    _startup(state["startup_time"], "scheduler", tokenizer=False)
    _finite_nonnegative(state["last_gen_throughput"], "last_gen_throughput")
    _finite_nonnegative(state["decode_cuda_graph_memory_gb"], "decode_cuda_graph_memory_gb")
    for key in ("world_size", "effective_max_running_requests_per_dp"):
        if type(state[key]) is not int or state[key] <= 0:
            raise ValueError(f"invalid scheduler value: {key}")
    graph_bs = state["decode_cuda_graph_bs"]
    if not isinstance(graph_bs, list) or any(type(n) is not int or n <= 0 for n in graph_bs) or (not graph_bs and args['cuda_graph_config']['decode']['backend']!='disabled'):
        raise ValueError("invalid graph batch sizes")
    return {
        "server_args": args,
        # Scheduler may resolve/override some argument values. Compare this
        # separate effective view across arms, not against top-level args.
        "scheduler_args": {k: state[k] for k in args if k in state},
        "effective": {
            "max_total_num_tokens": info["max_total_num_tokens"],
            "max_req_input_len": info["max_req_input_len"],
            "kv_events": info["kv_events"],
            "version": info["version"],
            "frontend": info["frontend"],
            "world_size": state["world_size"],
            "effective_max_running_requests_per_dp": state["effective_max_running_requests_per_dp"],
            "decode_cuda_graph_bs": graph_bs,
            "token_capacity": memory["token_capacity"],
            "token_capacity_swa": memory["token_capacity_swa"],
        },
        "observed_only": {
            "launch_command": info["launch_command"],
            "startup_time": info["startup_time"],
            "internal_startup_time": state["startup_time"],
            "last_gen_throughput": state["last_gen_throughput"],
            "memory_usage": memory,
            "decode_cuda_graph_memory_gb": state["decode_cuda_graph_memory_gb"],
        },
    }


def compare_profiles(a_info: dict[str, Any], b_info: dict[str, Any],
                     normalize: Callable[[Any], Any] = lambda x: x) -> dict[str, Any]:
    a, b = split_server_info(a_info), split_server_info(b_info)
    differences = {}
    for section in ("server_args", "scheduler_args", "effective"):
        left, right = normalize(a[section]), normalize(b[section])
        differences[section] = {
            k: {"A": left.get(k), "B": right.get(k),
                "missing_in": [arm for arm, row in (("A", left), ("B", right)) if k not in row]}
            for k in sorted(set(left) | set(right))
            if (k not in left or k not in right or
                json.dumps(left[k], sort_keys=True, allow_nan=False) !=
                json.dumps(right[k], sort_keys=True, allow_nan=False))
        }
    return {
        "pass_gate": all(not rows for rows in differences.values()),
        "differences": differences,
        "observed_only": {"A": a["observed_only"], "B": b["observed_only"]},
    }
