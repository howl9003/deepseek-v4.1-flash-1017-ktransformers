"""Bounded FIFO/queue-full witness for one caller-owned, idle FCFS server.

The context owns HTTP/SSE threads and enforces socket deadlines. A stream's
snapshot() returns latched first_token/finished booleans, HTTP status and errors;
first_token means an actual output token, never merely headers or an SSE event.
This module never infers admission order from request completion order.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import uuid


class Incomplete(RuntimeError):
    """The controlled observation could not be completed."""


def _positive(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be finite, positive and <= {maximum}")
    return float(value)


def validate_profile(profile):
    args, cfg = profile["server_args"], profile["queue"]
    for name, expected in {"max_running_requests": 4, "max_queued_requests": 4,
                           "schedule_policy": "fcfs", "enable_priority_scheduling": False,
                           "tp_size": 1, "pp_size": 1}.items():
        if type(args.get(name)) is not type(expected) or args[name] != expected:
            raise ValueError(f"effective {name} must equal {expected!r}")
    ids = cfg["input_ids"]
    if not isinstance(ids, list) or not 1 <= len(ids) <= 256 or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("queue.input_ids must contain 1..256 exact token IDs")
    tokens = cfg.get("hold_tokens", 1024)
    if type(tokens) is not int or not 1 <= tokens <= 1024:
        raise ValueError("hold_tokens must be an integer in 1..1024")
    if type(args.get("context_length")) is not int or len(ids) + tokens > args["context_length"]:
        raise ValueError("holder request exceeds effective context")
    return dict(input_ids=list(ids), hold_tokens=tokens,
                stage_timeout_s=_positive(cfg.get("stage_timeout_s", 10), "stage_timeout_s", 30),
                total_timeout_s=_positive(cfg.get("total_timeout_s", 90), "total_timeout_s", 180),
                cleanup_timeout_s=_positive(cfg.get("cleanup_timeout_s", 15), "cleanup_timeout_s", 30))


def _snapshot(handle):
    value = handle.snapshot()
    if not isinstance(value, dict) or type(value.get("first_token")) is not bool or type(value.get("finished")) is not bool:
        raise ValueError("invalid stream snapshot contract")
    if value.get("status") is not None and type(value["status"]) is not int:
        raise ValueError("invalid stream HTTP status")
    if value["first_token"] and value["status"] != 200:
        raise ValueError("first-token observation requires HTTP200")
    if not isinstance(value.get("errors"), list):
        raise ValueError("stream snapshot must expose errors")
    return value


def _load(raw):
    rows = raw["loads"]
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("FIFO witness requires one rank load snapshot")
    row = rows[0]
    stamp = row["timestamp"]
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        raise ValueError("invalid load timestamp")
    for key in ("num_running_reqs", "num_waiting_reqs"):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError(f"invalid {key}")
    if row["num_running_reqs"] > 4 or row["num_waiting_reqs"] > 4:
        raise AssertionError("effective 4-running/4-waiting limit exceeded")
    return row


def run_queue_checks(ctx, *, profile, output_dir):
    """Hold three slots, rotate the fourth, observe Q0..Q3 first-token order.

    Required context methods: request(label,route,body,allow_reject=False),
    start_stream(label,body), loads(), abort(rid), remaining(), wait(seconds),
    record_event(label,data). Loads use time.time() on this same loopback host.
    The parent must close/join its tracked reader threads even on cleanup failure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", requests=[], loads=[], transitions=[], cleanup=[],
                  scope="One controlled FCFS 4+4 first-token-order and queue-full witness; not a throughput benchmark")
    owned, handles = {}, {}
    settings = None
    floor = None

    def save():
        tmp = out / "QUEUE_CHECKS.json.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(out / "QUEUE_CHECKS.json")

    def states():
        return {label: _snapshot(handle) for label, handle in handles.items()}

    def assert_live(current, labels):
        for label in labels:
            state = current[label]
            if state["errors"] or state["status"] not in (None, 200):
                raise RuntimeError(f"{label}: unexpected stream error: {state}")
            if state["finished"]:
                raise Incomplete(f"{label} ended before its owned release; holder budget/transport insufficient")

    def poll(label, after, condition, cleanup=False):
        left = ctx.remaining()
        limit = left - (settings["cleanup_timeout_s"] if cleanup else settings["stage_timeout_s"])
        if not cleanup:
            limit = max(limit, floor)
        while ctx.remaining() > limit:
            current = states()
            raw = ctx.loads()
            row = _load(raw)
            if len(report["loads"]) >= 2000:
                raise Incomplete("load observation cap reached")
            report["loads"].append(dict(stage=label, snapshot=raw, streams=current))
            save()
            if row["timestamp"] >= after and condition(current, row):
                report["transitions"].append(dict(stage=label, timestamp=row["timestamp"], streams=current))
                save()
                return
            ctx.wait(min(0.1, max(0, ctx.remaining() - limit)))
        raise Incomplete(f"deadline reached without fresh observation: {label}")

    def launch(label, stream=True):
        rid = f"fifo-{uuid.uuid4().hex}-{label}"
        body = dict(rid=rid, input_ids=settings["input_ids"], stream=stream,
                    sampling_params=dict(temperature=0, max_new_tokens=settings["hold_tokens"], ignore_eos=True))
        # Register before dispatch, including an exception after the server accepts.
        owned[label] = rid
        report["requests"].append(dict(label=label, rid=rid, body=body))
        save()
        if stream:
            handles[label] = ctx.start_stream(label, body)
            return handles[label]
        return ctx.request(label, "/generate", body, allow_reject=True)

    try:
        settings = validate_profile(profile)
        left = _positive(ctx.remaining(), "remaining deadline", 86400)
        if left <= settings["cleanup_timeout_s"] + settings["stage_timeout_s"]:
            raise Incomplete("insufficient budget for witness and reserved cleanup")
        floor = max(left - settings["total_timeout_s"], settings["cleanup_timeout_s"])
        report.update(settings=settings, profile=profile)
        save()
        poll("initial-owned-idle", time.time(), lambda s, r: r["num_running_reqs"] == r["num_waiting_reqs"] == 0)
        mark = time.time()
        for i in range(4):
            launch(f"H{i}")

        def all_holders(s, r):
            assert_live(s, [f"H{i}" for i in range(4)])
            return all(s[f"H{i}"]["first_token"] for i in range(4)) and r["num_running_reqs"] == 4 and r["num_waiting_reqs"] == 0
        poll("four-running-holders", mark, all_holders)
        for i in range(4):
            mark = time.time()
            launch(f"Q{i}")
            def enqueued(s, r, count=i+1):
                assert_live(s, [f"H{j}" for j in range(4)] + [f"Q{j}" for j in range(count)])
                if any(s[f"Q{j}"]["first_token"] for j in range(count)):
                    raise AssertionError("queued request produced a token before any running slot was released")
                return r["num_running_reqs"] == 4 and r["num_waiting_reqs"] == count
            poll(f"waiting-{i+1}", mark, enqueued)

        overflow = launch("overflow", stream=False)
        report["overflow_response"] = overflow
        if overflow["status"] != 503 or "request queue is full" not in json.dumps(overflow["value"]).lower():
            raise AssertionError("overflow did not return HTTP503 with the queue-full reason")
        poll("queue-still-full-after-503", time.time(), enqueued)
        active = "H3"
        for i in range(4):
            mark = time.time()
            ctx.abort(owned[active])
            expected = f"Q{i}"
            def admitted(s, r, index=i):
                live = ["H0", "H1", "H2"] + [f"Q{j}" for j in range(index, 4)]
                assert_live(s, live)
                if any(s[f"Q{j}"]["first_token"] for j in range(index+1, 4)):
                    raise AssertionError(f"FIFO violation: a later queued request produced a token before Q{index} was released")
                return s[f"Q{index}"]["first_token"] and r["num_running_reqs"] == 4 and r["num_waiting_reqs"] == 3-index
            poll(f"first-token-{expected}", mark, admitted)
            report.setdefault("first_token_order", []).append(expected)
            active = expected
        report["status"] = "PASS"
    except Incomplete as exc:
        report.update(status="INCOMPLETE", error=str(exc))
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["protocol_status_before_cleanup"] = report["status"]
        # Every owned ID gets an independent attempt; a failed abort never skips
        # later IDs. No prefix abort, abort-all or unowned ID is ever sent.
        for label, rid in owned.items():
            try:
                result = ctx.abort(rid)
                report["cleanup"].append(dict(label=label, rid=rid, attempted=True, result=result))
            except Exception as exc:
                report["cleanup"].append(dict(label=label, rid=rid, attempted=True, error=f"{type(exc).__name__}: {exc}"))
                report["status"] = "FAIL"
        if owned:
            try:
                poll("owned-cleanup-drained", time.time(),
                     lambda s, r: all(v["finished"] for v in s.values()) and r["num_running_reqs"] == r["num_waiting_reqs"] == 0,
                     cleanup=True)
                report["cleanup_drained"] = True
            except Exception as exc:
                report.update(status="FAIL", cleanup_drained=False, cleanup_error=f"{type(exc).__name__}: {exc}")
        else:
            report["cleanup_drained"] = True
        save()
    ctx.record_event("queue-result", report)
    return report
