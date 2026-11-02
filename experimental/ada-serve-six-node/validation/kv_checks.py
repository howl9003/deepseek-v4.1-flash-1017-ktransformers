"""Bounded prefix TTL/host evidence collection through an owner's HTTP context.

No sockets, process control, model imports or runtime mutations here. The owner
must supply an exclusive, idle server and a transport enforcing its absolute
deadline. PASS means the tier protocol was observed and numerical data recorded;
numerical equivalence is reported separately, never inferred from cache hits.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_profile(profile, remaining):
    """Require explicit effective settings; the parent verifies their provenance."""
    remaining = _positive(remaining, "remaining deadline")
    args, env, kv = profile["server_args"], profile["environment"], profile["kv"]
    for name, expected in {"disable_radix_cache": False, "enable_hierarchical_cache": True,
                           "enable_cache_report": True, "hicache_write_policy": "write_through",
                           "tp_size": 1, "pp_size": 1}.items():
        if name not in args or args[name] != expected or type(args[name]) is not type(expected):
            raise ValueError(f"effective server_args.{name} must be {expected!r}")
    for name, expected in {"SGLANG_CACHE_TTL": "1", "KT_HICACHE_SWA_TAIL_ONLY": "1",
                           "KT_PREFILL_SWA_LEND": "0", "SGLANG_CACHE_TTL_RELEND_TOKENS": "0"}.items():
        if env.get(name) != expected:
            raise ValueError(f"effective environment.{name} must be {expected!r}")
    for setting, name in (("ttl_agent_s", "SGLANG_CACHE_TTL_AGENT_S"),
                          ("ttl_user_s", "SGLANG_CACHE_TTL_USER_S"),
                          ("sweep_s", "SGLANG_CACHE_TTL_SWEEP_S")):
        if _positive(kv[setting], setting) != _positive(env[name], name):
            raise ValueError(f"test TTL differs from effective {name}")
    count = _integer(kv["prefix_tokens"], "prefix_tokens", 1)
    page = _integer(args["page_size"], "page_size", 1)
    reply = _integer(kv.get("reply_tokens", 8), "reply_tokens", 1)
    if reply > 16 or count < 2 * page:
        raise ValueError("KV fixture requires >=2 pages and <=16 output tokens")
    if count + reply > _integer(args["context_length"], "context_length", 1):
        raise ValueError("KV fixture exceeds context length")
    pool = args.get("max_total_num_tokens", args.get("max_total_tokens"))
    if _integer(pool, "max_total_num_tokens", 1) < 2 * (count + reply):
        raise ValueError("device pool cannot retain both fixture prefixes")
    margin = _positive(kv.get("margin_s", 2), "margin_s")
    wait = float(kv["ttl_agent_s"]) + 2 * float(kv["sweep_s"]) + margin
    if wait >= remaining:
        raise ValueError("TTL wait alone exhausts workflow deadline")
    if float(kv["ttl_user_s"]) <= wait + margin:
        raise ValueError("user TTL must exceed the planned agent TTL wait plus margin")
    vocab = _integer(kv.get("vocab_size", 129280), "vocab_size", 1001)
    seed = _integer(kv.get("seed", 3676), "seed")
    return dict(prefix_tokens=count, reply_tokens=reply, page_size=page, wait_s=wait,
                vocab_size=vocab, seed=seed)


def parse_completion(result, *, prompt_tokens, reply_tokens, vocab_size):
    if result["status"] != 200:
        raise ValueError(f"completion returned HTTP {result['status']}")
    body = result["value"]
    choices = body["choices"]
    if len(choices) != 1 or choices[0].get("finish_reason") != "length":
        raise ValueError("expected one complete, length-limited choice")
    choice, usage = choices[0], body["usage"]
    if type(usage["prompt_tokens"]) is not int or usage["prompt_tokens"] != prompt_tokens:
        raise ValueError("server prompt count differs from exact input IDs")
    if type(usage["completion_tokens"]) is not int or usage["completion_tokens"] != reply_tokens:
        raise ValueError("server output count differs from bounded request")
    ids = choice["token_ids"]
    logs = choice["logprobs"]["token_logprobs"]
    if len(ids) != reply_tokens or len(logs) != reply_tokens:
        raise ValueError("missing or partial output IDs/logprobs")
    if any(type(x) is not int or not 0 <= x < vocab_size for x in ids):
        raise ValueError("invalid output token ID")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or x > 1e-3 for x in logs):
        raise ValueError("invalid output logprob")
    cached = _integer(usage["prompt_tokens_details"]["cached_tokens"], "cached_tokens")
    # Pinned SGLang returns no tier object when every cache counter is zero,
    # then removes its None-valued sglext during OpenAI serialization. This
    # special case does not attribute a positive aggregate hit to any tier.
    if "sglext" not in body:
        if cached != 0:
            raise ValueError("positive cached_tokens requires explicit tier details")
        device = host = storage = 0
        tier_evidence = "upstream_zero_cache_omission"
    else:
        extension = body["sglext"]
        if not isinstance(extension, dict) or "cached_tokens_details" not in extension:
            raise ValueError("malformed explicit sglext")
        tiers = extension["cached_tokens_details"]
        if not isinstance(tiers, dict) or not {"device", "host"}.issubset(tiers):
            raise ValueError("explicit cache tiers require device and host")
        if set(tiers) - {"device", "host", "storage"}:
            raise ValueError("unknown cache tier")
        device, host = (_integer(tiers[name], name) for name in ("device", "host"))
        storage = _integer(tiers.get("storage", 0), "storage")
        tier_evidence = "explicit_sglext"
    if storage or device + host != cached or cached > prompt_tokens:
        raise ValueError("cache accounting is inconsistent or uses untested storage tier")
    wall = float(result["wall_s"])
    if not math.isfinite(wall) or wall < 0:
        raise ValueError("invalid request wall time")
    return dict(prompt_tokens=prompt_tokens, cached_tokens=cached, device_tokens=device,
                host_tokens=host, cache_tier_evidence=tier_evidence,
                output_ids=ids, output_logprobs=logs, wall_s=wall)


def compare_outputs(left, right):
    same_ids = left["output_ids"] == right["output_ids"]
    logs_same = left["output_logprobs"] == right["output_logprobs"]
    return dict(output_ids_equal=same_ids, selected_logprobs_equal=logs_same if same_ids else None,
                max_selected_logprob_delta=(max(abs(a-b) for a, b in zip(left["output_logprobs"], right["output_logprobs"])) if same_ids else None),
                same_cached_prefix_length=left["cached_tokens"] == right["cached_tokens"],
                scope="Selected-token logprobs only; differing output histories are not aligned")


def run_kv_checks(ctx, *, profile, output_dir):
    """Six sequential requests, one bounded wait, no eviction-pressure loop.

    ctx.request(label, route, body, allow_reject=False, headers=None) must persist
    raw exchanges and enforce ctx.remaining(). ctx.flush() is owned/idle only.
    ctx.wait(seconds) enforces the same absolute deadline.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", rows=[], comparisons={}, checks={},
                  scope="Short prefix TTL/device/host evidence, not 35K retention or full numerical acceptance")

    def save():
        temporary = out / "KV_CHECKS.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(out / "KV_CHECKS.json")

    try:
        settings = validate_profile(profile, ctx.remaining())
        report.update(settings=settings, profile=profile, profile_sha256=_digest(profile))
        inputs = {}
        for cls, seed in (("agent", settings["seed"]), ("user", settings["seed"] + 1)):
            rng = random.Random(seed)
            inputs[cls] = [rng.randrange(1000, settings["vocab_size"]) for _ in range(settings["prefix_tokens"])]
        report["inputs"] = {cls: dict(input_ids=ids, input_ids_sha256=_digest(ids)) for cls, ids in inputs.items()}
        save()
        ctx.record_event("kv-profile-and-inputs", dict(profile=profile, settings=settings, inputs=report["inputs"]))
        ctx.flush()

        def send(label, cls):
            if ctx.remaining() <= 0:
                raise TimeoutError("KV deadline exhausted before request")
            body = dict(model=ctx.model, prompt=inputs[cls], max_tokens=settings["reply_tokens"],
                        temperature=0, seed=settings["seed"], ignore_eos=True, stream=False,
                        echo=False, logprobs=1, return_token_ids=True, return_cached_tokens_details=True)
            headers = {"X-Cache-Class": "agent"} if cls == "agent" else {}
            if cls == "user":
                body["cache_class"] = "user"
            result = ctx.request(label, "/v1/completions", body, headers=headers)
            row = parse_completion(result, prompt_tokens=settings["prefix_tokens"],
                                   reply_tokens=settings["reply_tokens"], vocab_size=settings["vocab_size"])
            row.update(label=label, cache_class=cls, input_ids_sha256=_digest(inputs[cls]))
            report["rows"].append(row)
            save()
            return row

        agent_cold = send("kv-agent-cold", "agent")
        agent_device = send("kv-agent-device", "agent")
        agent_touch_upper = ctx.remaining()  # response end is later than the touch
        user_cold = send("kv-user-cold", "user")
        user_touch_lower = ctx.remaining()  # request start is earlier than the touch
        user_device = send("kv-user-device", "user")
        elapsed_since_agent = agent_touch_upper - ctx.remaining()
        wait_s = max(0.0, settings["wait_s"] - elapsed_since_agent)
        if wait_s >= ctx.remaining():
            raise TimeoutError("not enough deadline remaining for TTL wait")
        ctx.record_event("kv-ttl-wait", dict(seconds=wait_s, seconds_since_agent_response=elapsed_since_agent))
        ctx.wait(wait_s)
        agent_after = send("kv-agent-after-ttl", "agent")
        if user_touch_lower - ctx.remaining() >= float(profile["kv"]["ttl_user_s"]):
            raise TimeoutError("user control TTL expired before retained-prefix observation")
        user_after = send("kv-user-after-agent-ttl", "user")
        if user_touch_lower - ctx.remaining() >= float(profile["kv"]["ttl_user_s"]):
            raise TimeoutError("user control observation exceeded its TTL window")
        checks = dict(
            agent_cold_uncached=agent_cold["cached_tokens"] == 0,
            user_cold_uncached=user_cold["cached_tokens"] == 0,
            agent_device_hit=agent_device["device_tokens"] >= settings["page_size"] and agent_device["host_tokens"] == 0,
            user_device_hit=user_device["device_tokens"] >= settings["page_size"] and user_device["host_tokens"] == 0,
            agent_device_expired=agent_after["device_tokens"] == 0,
            agent_host_reload=agent_after["host_tokens"] >= settings["page_size"],
            user_device_retained=user_after["device_tokens"] >= user_device["device_tokens"] and user_after["host_tokens"] == 0,
        )
        report["checks"] = checks
        for name, left, right in (("agent_cold_vs_device", agent_cold, agent_device),
                                  ("agent_device_vs_host", agent_device, agent_after),
                                  ("user_cold_vs_device", user_cold, user_device),
                                  ("user_device_vs_retained", user_device, user_after)):
            report["comparisons"][name] = compare_outputs(left, right)
        report["status"] = "PASS" if all(checks.values()) else "INCOMPLETE"
        report["incomplete_reasons"] = [name for name, passed in checks.items() if not passed]
        report["numerical_acceptance"] = "REVIEW_REQUIRED"
        report["interpretation"] = "TTL evicts device entries; a surviving host copy and nonzero total cached_tokens are valid. Cold/hit reductions can differ; raw comparisons do not set a new tolerance."
    except Exception as exc:
        report.update(status="FAIL", error=dict(type=type(exc).__name__, message=str(exc)))
    save()
    ctx.record_event("kv-result", report)
    return report
