"""CPU tests for KT_DSPARK_PLAIN_GATE (dspark_plain_gate.py, apply_dspark_plain_gate.py). Claude, 4 Nov 2026; r3 adds the
confidence signal (config, helpers, the lagged ring, confidence probes, calibration, accept mode step for step as r2,
the signal reading logits only, and the sources it reads); r4 adds the draft-only probe (config, the probe ring, the
match count, probe scheduling and the switch rule, batches, accept and confidence modes step for step as r3, the
probe's call sites and the window plan against the pinned sources).

No torch, no GPU, no sglang import: the gate is pure Python, and the tree-facing parts are checked against the pinned
sources by AST. Run:

    python -m unittest -v test_dspark_plain_gate.py

PLAIN_GATE_SGLANG=<sglang package dir with the pinned files> also runs the overlay and source checks (an unmodified
copy: the overlay test patches a temporary copy of it, never the directory itself)."""

import ast
import difflib
import hashlib
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dspark_plain_gate", HERE / "dspark_plain_gate.py")
pg = importlib.util.module_from_spec(_spec)
sys.modules["dspark_plain_gate"] = pg  # dataclasses resolve string annotations through sys.modules
_spec.loader.exec_module(pg)

SGLANG = os.environ.get("PLAIN_GATE_SGLANG")


def cfg(**kw):
    c = pg.GateConfig(**kw)
    c.validate()
    return c


class Req:
    """The scheduler's CPU counters for one request (Req.spec_verify_ct, Req.spec_num_correct_drafts)."""

    def __init__(self, rid):
        self.rid = rid
        self.spec_verify_ct = 0
        self.spec_num_correct_drafts = 0


class OverlapLoop:
    """The overlap scheduler as the gate sees it: a step's result reaches the request's counters `lag` steps after
    the step was issued (run_batch(t) runs before the result of t-1 is processed, so a decision at t sees results
    through t-2). Speculative steps commit `accept()` tokens; plain steps commit one."""

    def __init__(self, gate, req, accept, lag=2, eligible=lambda step: True):
        self.gate, self.req, self.accept, self.lag, self.eligible = gate, req, accept, lag, eligible
        self.inflight = deque()
        self.kinds = []
        self.tokens = 0

    def step(self):
        while len(self.inflight) >= self.lag:
            self._process(self.inflight.popleft())
        n = len(self.kinds)
        plain = self.gate.decide(
            self.req.rid,
            verify_ct=self.req.spec_verify_ct,
            correct_drafts=self.req.spec_num_correct_drafts,
            eligible=self.eligible(n),
        )
        kind = pg.PLAIN if plain else pg.SPEC
        self.kinds.append(kind)
        self.inflight.append(1 if plain else self.accept(n))
        return kind

    def _process(self, committed):
        self.req.spec_verify_ct += 1
        self.req.spec_num_correct_drafts += committed - 1
        self.tokens += committed

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return "".join(self.kinds)


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        c = pg.GateConfig.from_env({})
        self.assertEqual((c.threshold, c.probe_every, c.alpha, c.min_spec), (2.2, 16, 0.25, 4))
        self.assertEqual((c.force, c.graph, c.log_every, c.max_bs), ("", True, 0, 1))
        self.assertEqual((c.signal, c.conf), ("accept", 0.0))
        self.assertEqual((c.draft_every, c.draft_alpha, c.draft_min), (8, 0.5, 2))

    def test_env(self):
        c = pg.GateConfig.from_env(
            {
                "KT_DSPARK_PLAIN_GATE_THRESHOLD": "1.9",
                "KT_DSPARK_PLAIN_GATE_PROBE": "8",
                "KT_DSPARK_PLAIN_GATE_ALPHA": "0.5",
                "KT_DSPARK_PLAIN_GATE_MIN_SPEC": "2",
                "KT_DSPARK_PLAIN_GATE_FORCE": " sspp ",
                "KT_DSPARK_PLAIN_GATE_GRAPH": "0",
                "KT_DSPARK_PLAIN_GATE_LOG": "100",
                "KT_DSPARK_PLAIN_GATE_MAX_BS": "4",
                "KT_DSPARK_PLAIN_GATE_SIGNAL": " Confidence ",
                "KT_DSPARK_PLAIN_GATE_CONF": "0.62",
                "KT_DSPARK_PLAIN_GATE_DRAFT_EVERY": "4",
                "KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA": "0.3",
                "KT_DSPARK_PLAIN_GATE_DRAFT_MIN": "3",
            }
        )
        self.assertEqual((c.threshold, c.probe_every, c.alpha, c.min_spec), (1.9, 8, 0.5, 2))
        self.assertEqual((c.force, c.graph, c.log_every, c.max_bs), ("SSPP", False, 100, 4))
        self.assertEqual((c.signal, c.conf), ("confidence", 0.62))
        self.assertEqual((c.draft_every, c.draft_alpha, c.draft_min), (4, 0.3, 3))
        self.assertEqual(pg.GateConfig.from_env({"KT_DSPARK_PLAIN_GATE_SIGNAL": "DRAFT"}).signal, "draft")

    def test_invalid(self):
        for key, value in (
            ("KT_DSPARK_PLAIN_GATE_THRESHOLD", "0"),
            ("KT_DSPARK_PLAIN_GATE_PROBE", "0"),
            ("KT_DSPARK_PLAIN_GATE_ALPHA", "0"),
            ("KT_DSPARK_PLAIN_GATE_ALPHA", "1.5"),
            ("KT_DSPARK_PLAIN_GATE_MIN_SPEC", "0"),
            ("KT_DSPARK_PLAIN_GATE_FORCE", "SPX"),
            ("KT_DSPARK_PLAIN_GATE_LOG", "-1"),
            ("KT_DSPARK_PLAIN_GATE_MAX_BS", "0"),
            ("KT_DSPARK_PLAIN_GATE_MAX_BS", "17"),
            ("KT_DSPARK_PLAIN_GATE_SIGNAL", "shadow"),
            ("KT_DSPARK_PLAIN_GATE_DRAFT_EVERY", "0"),
            ("KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA", "0"),
            ("KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA", "1.1"),
            ("KT_DSPARK_PLAIN_GATE_DRAFT_MIN", "0"),
            ("KT_DSPARK_PLAIN_GATE_CONF", "-0.1"),
            ("KT_DSPARK_PLAIN_GATE_CONF", "1.5"),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                pg.GateConfig.from_env({key: value})

    def test_refusals(self):
        ok = dict(verify_mode="static", attn_dp=False, pp_size=1, simulate_acc_len=0.0, draft_is_moe=True,
                  target_type="deepseek_v41")
        self.assertEqual(pg.refusal_reasons(**ok), [])
        for change in (
            {"verify_mode": "compact"},
            {"attn_dp": True},
            {"pp_size": 2},
            {"simulate_acc_len": 1.0},
            {"draft_is_moe": False},
            {"target_type": "deepseek_v4"},
        ):
            with self.subTest(change=change):
                self.assertEqual(len(pg.refusal_reasons(**{**ok, **change})), 1)

    def test_eligibility(self):
        ok = dict(bs=1, max_bs=1, has_grammar=False, return_logprob=False, return_hidden_states=False,
                  sampling_masks=False)
        self.assertTrue(pg.step_eligible(**ok))
        for change in ({"bs": 2}, {"bs": 0}, {"has_grammar": True}, {"return_logprob": True},
                       {"return_hidden_states": True}, {"sampling_masks": True}):
            with self.subTest(change=change):
                self.assertFalse(pg.step_eligible(**{**ok, **change}))


class GateTest(unittest.TestCase):
    def test_new_request_speculates_until_min_spec(self):
        loop = OverlapLoop(pg.PlainGate(cfg(min_spec=4)), Req("a"), accept=lambda n: 1)
        kinds = loop.run(8)
        # Four speculative results must be counted first; a decision at step t sees results through t - 2.
        self.assertEqual(kinds[:5], "SSSSS")
        self.assertEqual(kinds[5:], "PPP")

    def test_prose_runs_plain_with_a_probe_every_n(self):
        loop = OverlapLoop(pg.PlainGate(cfg(probe_every=16)), Req("a"), accept=lambda n: 2 if n % 3 else 1)
        kinds = loop.run(200)
        steady = kinds[kinds.index("P"):]
        runs = steady.split("S")
        # Exactly 16 plain steps between probes, and one speculative step per probe.
        self.assertTrue(all(len(r) == 16 for r in runs[:-1]), runs)
        self.assertNotIn("SS", steady)
        self.assertLess(loop.gate.ema("a"), 2.2)

    def test_code_stays_speculative(self):
        loop = OverlapLoop(pg.PlainGate(cfg()), Req("a"), accept=lambda n: 4 if n % 2 else 3)
        self.assertEqual(loop.run(100), "S" * 100)
        self.assertGreater(loop.gate.ema("a"), 3.0)

    def test_threshold_is_the_switch(self):
        for accept, expect_plain in ((2, True), (3, False)):
            with self.subTest(accept=accept):
                loop = OverlapLoop(pg.PlainGate(cfg(threshold=2.5)), Req("a"), accept=lambda n: accept)
                self.assertEqual("P" in loop.run(40), expect_plain)

    def test_probe_switches_back_when_the_draft_recovers(self):
        # Prose for 60 steps, then code-like acceptance: the next probe brings the request back to speculation.
        loop = OverlapLoop(pg.PlainGate(cfg()), Req("a"), accept=lambda n: 1 if n < 60 else 4)
        kinds = loop.run(140)
        self.assertIn("P" * 16, kinds[:60])
        tail = kinds[100:]
        self.assertEqual(tail, "S" * len(tail))

    def test_plain_results_do_not_move_the_ema(self):
        gate = pg.PlainGate(cfg(min_spec=1))
        self.assertFalse(gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True))
        # The speculative result: 2 tokens (1 draft); below the threshold, so plain steps follow.
        self.assertTrue(gate.decide("a", verify_ct=1, correct_drafts=1, eligible=True))
        ct = 1
        for _ in range(10):
            ct += 1  # the previous plain step's result: one more verify count, no draft
            self.assertTrue(gate.decide("a", verify_ct=ct, correct_drafts=1, eligible=True))
        # Counted as speculative steps they would have pulled the EMA towards 1.
        self.assertEqual(gate.ema("a"), 2.0)

    def test_ema_is_tokens_per_spec_step(self):
        gate = pg.PlainGate(cfg(alpha=0.5, min_spec=1))
        gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True)  # S
        gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True)  # S
        gate.decide("a", verify_ct=2, correct_drafts=4, eligible=True)  # two results, 2 drafts each: 3 tokens a step
        self.assertEqual(gate.ema("a"), 3.0)
        gate.decide("a", verify_ct=3, correct_drafts=4, eligible=True)  # one more result, 0 drafts: 1 token
        self.assertEqual(gate.ema("a"), 2.0)

    def test_new_request_and_forget_and_prune_reset(self):
        gate = pg.PlainGate(cfg(min_spec=1))
        loop = OverlapLoop(gate, Req("a"), accept=lambda n: 1)
        self.assertIn("P", loop.run(10))
        gate.forget("a")
        self.assertNotIn("a", gate)
        self.assertFalse(gate.decide("a", verify_ct=10, correct_drafts=0, eligible=True))
        gate.prune(["b"])
        self.assertNotIn("a", gate)
        # A different request starts speculating regardless of another's history.
        self.assertFalse(gate.decide("b", verify_ct=0, correct_drafts=0, eligible=True))

    def test_counter_restart_is_a_new_request(self):
        gate = pg.PlainGate(cfg(min_spec=1))
        loop = OverlapLoop(gate, Req("a"), accept=lambda n: 1)
        loop.run(10)
        self.assertFalse(gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True))
        self.assertIsNone(gate.ema("a"))

    def test_ineligible_steps_speculate_and_still_count(self):
        # Batches of two (ineligible) for the first 30 steps: all speculative, and their results feed the EMA, so
        # the first eligible step can already go plain.
        loop = OverlapLoop(pg.PlainGate(cfg()), Req("a"), accept=lambda n: 1, eligible=lambda n: n >= 30)
        kinds = loop.run(40)
        self.assertEqual(kinds[:30], "S" * 30)
        self.assertEqual(kinds[30], "P")

    def test_force_pattern(self):
        loop = OverlapLoop(pg.PlainGate(cfg(force="SSPPP")), Req("a"), accept=lambda n: 4)
        self.assertEqual(loop.run(15), "SSPPP" * 3)
        loop = OverlapLoop(pg.PlainGate(cfg(force="SP")), Req("a"), accept=lambda n: 1,
                           eligible=lambda n: n % 4 != 1)
        # Ineligible steps speculate; the pattern position still advances.
        self.assertEqual(loop.run(8), "SSSPSSSP")

    def test_untracked_results_count_as_spec(self):
        gate = pg.PlainGate(cfg(min_spec=1))
        gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True)
        # Three results arrive for one issued step (a request the gate lost track of): treated as speculative.
        gate.decide("a", verify_ct=3, correct_drafts=3, eligible=True)
        self.assertEqual(gate.ema("a"), 2.0)
        self.assertEqual(gate._reqs["a"].samples, 3)


class BatchLoop:
    """OverlapLoop for a batch: every request votes, the batch runs batch_kind(votes), every request records it.
    `accept[rid](n)` is the tokens a speculative step commits for that request at step n."""

    def __init__(self, gate, rids, accept, lag=2, max_bs=4):
        self.gate, self.accept, self.lag, self.max_bs = gate, accept, lag, max_bs
        self.reqs = {rid: Req(rid) for rid in rids}
        self.inflight = deque()
        self.kinds = []

    def step(self, rids):
        while len(self.inflight) >= self.lag:
            for rid, committed in self.inflight.popleft():
                if rid in self.reqs:
                    self.reqs[rid].spec_verify_ct += 1
                    self.reqs[rid].spec_num_correct_drafts += committed - 1
        self.gate.prune(rids)
        n = len(self.kinds)
        eligible = 0 < len(rids) <= self.max_bs
        votes = [self.gate.vote(rid, verify_ct=self.reqs[rid].spec_verify_ct,
                                correct_drafts=self.reqs[rid].spec_num_correct_drafts, eligible=eligible)
                 for rid in rids]
        kind = pg.batch_kind(votes)
        for rid in rids:
            self.gate.record(rid, kind)
        self.kinds.append(kind)
        self.inflight.append([(rid, 1 if kind == pg.PLAIN else self.accept[rid](n)) for rid in rids])
        return kind

    def run(self, steps, rids):
        return "".join(self.step(rids) for _ in range(steps))


class BatchRuleTest(unittest.TestCase):
    def test_batch_kind(self):
        self.assertEqual(pg.batch_kind(["S"]), "S")
        self.assertEqual(pg.batch_kind(["P"]), "P")
        self.assertEqual(pg.batch_kind(["S", "S", "S"]), "S")
        self.assertEqual(pg.batch_kind(["S", "P", "S"]), "P")

    def test_one_request_matches_decide(self):
        # bs = 1 through vote/record gives the r1 decide() sequence step for step.
        accept = lambda n: 1 if n < 60 else 4
        r1 = OverlapLoop(pg.PlainGate(cfg()), Req("a"), accept=accept).run(140)
        r2 = BatchLoop(pg.PlainGate(cfg()), ["a"], {"a": accept}).run(140, ["a"])
        self.assertEqual(r1, r2)

    def test_two_code_requests_speculate(self):
        loop = BatchLoop(pg.PlainGate(cfg()), ["a", "b"], {"a": lambda n: 4, "b": lambda n: 3})
        self.assertEqual(loop.run(60, ["a", "b"]), "S" * 60)

    def test_one_prose_request_makes_the_batch_plain(self):
        loop = BatchLoop(pg.PlainGate(cfg()), ["a", "b"], {"a": lambda n: 4, "b": lambda n: 1})
        kinds = loop.run(60, ["a", "b"])
        # b's EMA falls below the threshold after min_spec samples (plus the lag); from then on plain, except when
        # b probes while a still wants to speculate (the batch speculates only when both vote S).
        self.assertEqual(kinds[:6], "S" * 5 + "P")  # 4 samples seen at step 5 (lag 2)
        self.assertGreater(kinds[6:].count("P"), 50)

    def test_votes_and_records_differ_without_losing_attribution(self):
        # a votes S (high EMA) but runs plain steps with b; its EMA is unmoved by them and it speculates again alone.
        gate = pg.PlainGate(cfg())
        loop = BatchLoop(gate, ["a", "b"], {"a": lambda n: 4, "b": lambda n: 1})
        loop.run(40, ["a", "b"])
        ema_a = gate.ema("a")
        self.assertGreaterEqual(ema_a, 3.9)
        tail = loop.run(20, ["a"])  # b finished
        self.assertEqual(tail, "S" * 20)
        self.assertGreaterEqual(gate.ema("a"), 3.9)

    def test_oversized_batch_speculates(self):
        loop = BatchLoop(pg.PlainGate(cfg()), list("abcde"), {r: (lambda n: 1) for r in "abcde"}, max_bs=4)
        self.assertEqual(loop.run(30, list("abcde")), "S" * 30)

    def test_new_request_joins_plain_batch(self):
        # A new request votes S, but the batch stays plain while the other votes P; the new one samples only on the
        # other's probes (both vote S) and measures once it runs alone.
        gate = pg.PlainGate(cfg())
        loop = BatchLoop(gate, ["a", "b"], {"a": lambda n: 1, "b": lambda n: 4})
        loop.run(20, ["a"])
        kinds = loop.run(10, ["a", "b"])
        self.assertGreaterEqual(kinds.count("P"), 8)
        self.assertLessEqual(gate._reqs["b"].samples, kinds.count("S"))
        self.assertEqual(loop.run(3, ["b"]), "S" * 3)


# ---------------------------------------------------------------------------------------------------------------------
# r3: the confidence signal

_spec2 = importlib.util.spec_from_file_location("dspark_plain_gate_r2", HERE.parent / "dspark_plain_gate.py")
pg2 = importlib.util.module_from_spec(_spec2)
sys.modules["dspark_plain_gate_r2"] = pg2
_spec2.loader.exec_module(pg2)


class ConfLoop(OverlapLoop):
    """OverlapLoop that also feeds the gate a confidence sample per step, `lag` steps late like the counters."""

    def __init__(self, gate, req, accept, conf, lag=2, eligible=lambda step: True):
        super().__init__(gate, req, accept, lag=lag, eligible=eligible)
        self.conf = conf
        self.samples = deque()

    def step(self):
        n = len(self.kinds)
        while self.samples and self.samples[0][0] <= n - self.lag:
            _, value = self.samples.popleft()
            self.gate.note_confidence(self.req.rid, value)
        kind = super().step()
        self.samples.append((n, self.conf(n)))
        return kind


class SignalHelpersTest(unittest.TestCase):
    def test_conf_bucket(self):
        self.assertEqual(pg.conf_bucket(None), -1)
        self.assertEqual([pg.conf_bucket(x) for x in (0.0, 0.05, 0.1, 0.55, 0.999, 1.0)], [0, 0, 1, 5, 9, 9])

    def test_anchor_stride(self):
        self.assertEqual(pg.anchor_stride(1, 1, 4), 1)  # plain, one request
        self.assertEqual(pg.anchor_stride(3, 3, 4), 1)  # plain batch
        self.assertEqual(pg.anchor_stride(12, 3, 4), 4)  # verify: request-major rows
        self.assertIsNone(pg.anchor_stride(10, 3, 4))
        self.assertIsNone(pg.anchor_stride(4, 0, 4))
        self.assertEqual(pg.anchor_stride(4, 1, 4), 4)


class FakeEvent:
    def __init__(self):
        self.done = True  # an event never recorded reports complete, as CUDA's does
        self.records = 0

    def query(self):
        return self.done


class FakeBackend:
    def host(self, slots, width):
        return [[0.0] * width for _ in range(slots)]

    def event(self):
        return FakeEvent()

    def copy(self, host, slot, values, n, event):
        host[slot][:n] = list(values[:n])
        event.done = False
        event.records += 1

    def read(self, host, slot, n):
        return list(host[slot][:n])


class RingTest(unittest.TestCase):
    def test_lagged_in_order(self):
        ring = pg.ConfidenceRing(slots=4, width=4, backend=FakeBackend())
        for step in range(3):
            self.assertTrue(ring.push([f"r{step}", "b"], [0.1 * step, 0.5]))
        self.assertEqual(ring.drain(), [])  # nothing completed: nothing read, nothing waited for
        ring.events[1].done = True  # completes out of order: still blocked behind slot 0
        self.assertEqual(ring.drain(), [])
        ring.events[0].done = True
        self.assertEqual(ring.drain(), [("r0", 0.0), ("b", 0.5), ("r1", 0.1), ("b", 0.5)])
        ring.events[2].done = True
        self.assertEqual(ring.drain(), [("r2", 0.2), ("b", 0.5)])
        self.assertEqual((ring.pushed, ring.dropped), (3, 0))

    def test_full_ring_drops_instead_of_waiting(self):
        ring = pg.ConfidenceRing(slots=2, width=4, backend=FakeBackend())
        self.assertTrue(ring.push(["a"], [0.9]))
        self.assertTrue(ring.push(["a"], [0.8]))
        self.assertFalse(ring.push(["a"], [0.7]))  # both rows in flight
        self.assertEqual(ring.dropped, 1)
        ring.events[0].done = True
        self.assertEqual(ring.drain(), [("a", 0.9)])
        self.assertTrue(ring.push(["a"], [0.6]))  # the freed row is reused
        ring.events[1].done = ring.events[0].done = True
        self.assertEqual(ring.drain(), [("a", 0.8), ("a", 0.6)])

    def test_oversized_or_empty_step_is_dropped(self):
        ring = pg.ConfidenceRing(slots=2, width=2, backend=FakeBackend())
        self.assertFalse(ring.push(["a", "b", "c"], [0.1, 0.2, 0.3]))
        self.assertFalse(ring.push([], []))
        self.assertEqual((ring.pushed, ring.dropped), (0, 2))


class AcceptModeUnchangedTest(unittest.TestCase):
    """'accept' (the default) issues r2's steps exactly, with or without confidence samples arriving."""

    ACCEPTS = {
        "prose": lambda n: 1,
        "code": lambda n: 4,
        "mixed": lambda n: 1 if n < 60 else 4,
        "near_threshold": lambda n: 2 if n % 3 else 3,
        "random": lambda n: random.Random(n).choice((1, 1, 2, 3, 4)),
    }

    def test_single_request_matches_r2(self):
        for name, accept in self.ACCEPTS.items():
            for eligible in (lambda n: True, lambda n: n % 7 != 3):
                with self.subTest(case=name):
                    r2 = OverlapLoop(pg2.PlainGate(pg2.GateConfig()), Req("a"), accept=accept, eligible=eligible).run(200)
                    r3 = OverlapLoop(pg.PlainGate(cfg()), Req("a"), accept=accept, eligible=eligible).run(200)
                    fed = ConfLoop(pg.PlainGate(cfg()), Req("a"), accept=accept, eligible=eligible,
                                   conf=lambda n: 0.99 if n % 5 else 0.1).run(200)
                    self.assertEqual(r3, r2)
                    self.assertEqual(fed, r2)

    def test_batches_match_r2(self):
        accept = {"a": lambda n: 1, "b": lambda n: 4, "c": lambda n: 1 if n < 30 else 4}
        r2 = BatchLoop(pg2.PlainGate(pg2.GateConfig()), list("abc"), accept)
        r3 = BatchLoop(pg.PlainGate(cfg()), list("abc"), accept)
        self.assertEqual(r3.run(40, ["a", "b"]) + r3.run(40, ["a", "b", "c"]) + r3.run(40, ["c"]),
                         r2.run(40, ["a", "b"]) + r2.run(40, ["a", "b", "c"]) + r2.run(40, ["c"]))


class ConfidenceModeTest(unittest.TestCase):
    def test_no_timer_probe_and_one_probe_on_the_rise(self):
        gate = pg.PlainGate(cfg(signal="confidence", conf=0.6))
        conf = lambda n: 0.3 if n < 60 else 0.9
        kinds = ConfLoop(gate, Req("a"), accept=lambda n: 1, conf=conf).run(140)
        self.assertEqual(kinds[:5], "S" * 5)
        self.assertNotIn("S", kinds[5:60])  # no timer probes (r2 would probe after 16 plain steps)
        tail = kinds[60:]
        self.assertEqual(tail.count("S"), 1)  # one probe once the EMA reaches 0.6; it accepts 1, so plain again

    def test_probe_again_after_a_dip(self):
        gate = pg.PlainGate(cfg(signal="confidence", conf=0.6))
        conf = lambda n: 0.3 if n < 40 or 80 <= n < 110 else 0.9
        kinds = ConfLoop(gate, Req("a"), accept=lambda n: 1, conf=conf).run(160)
        self.assertEqual(kinds[40:80].count("S"), 1)
        self.assertEqual(kinds[110:].count("S"), 1)

    def test_probe_that_finds_the_draft_good_keeps_speculating(self):
        gate = pg.PlainGate(cfg(signal="confidence", conf=0.6))
        conf = lambda n: 0.3 if n < 40 else 0.9
        accept = lambda n: 1 if n < 40 else 4
        kinds = ConfLoop(gate, Req("a"), accept=accept, conf=conf).run(120)
        self.assertNotIn("S", kinds[5:40])
        # One probe: its 4 tokens replace the EMA of 1.0 (blended, 1.75 would stay plain with no further probe).
        first = kinds.index("S", 5)
        self.assertEqual(kinds[first:first + 5], "SPSSS")  # the probe's result is seen two decisions later
        self.assertEqual(kinds[70:], "S" * 50)
        self.assertEqual(gate.ema("a"), 4.0)

    def test_conf_zero_never_probes(self):
        gate = pg.PlainGate(cfg(signal="confidence", conf=0.0))
        kinds = ConfLoop(gate, Req("a"), accept=lambda n: 1, conf=lambda n: 0.99).run(100)
        self.assertEqual(kinds, "S" * 5 + "P" * 95)

    def test_code_still_speculates(self):
        gate = pg.PlainGate(cfg(signal="confidence", conf=0.6))
        self.assertEqual(ConfLoop(gate, Req("a"), accept=lambda n: 4, conf=lambda n: 0.2).run(50), "S" * 50)

    def test_unknown_request_sample_is_ignored(self):
        gate = pg.PlainGate(cfg(signal="confidence", conf=0.6))
        gate.note_confidence("ghost", 0.9)
        self.assertNotIn("ghost", gate)
        self.assertIsNone(gate.conf_ema("ghost"))

    def test_confidence_ema(self):
        gate = pg.PlainGate(cfg(alpha=0.5))
        gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True)
        gate.note_confidence("a", 0.8)
        gate.note_confidence("a", 0.4)
        self.assertAlmostEqual(gate.conf_ema("a"), 0.6)


class CalibrationTest(unittest.TestCase):
    def test_buckets_by_confidence_at_issue(self):
        gate = pg.PlainGate(cfg(min_spec=1))
        gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True)  # S, no confidence yet
        gate.note_confidence("a", 0.55)
        gate.decide("a", verify_ct=0, correct_drafts=0, eligible=True)  # S issued at 0.55
        gate.decide("a", verify_ct=1, correct_drafts=1, eligible=True)  # first result: 2 tokens, bucket 'none'
        gate.decide("a", verify_ct=2, correct_drafts=4, eligible=True)  # second result: 4 tokens, bucket 0.5-0.6
        self.assertEqual(gate.calibration(), {-1: [1, 2.0], 5: [1, 4.0]})
        self.assertEqual(gate.calibration_line(), "none 1x2.00, 0.5-0.6 1x4.00")

    def test_plain_results_are_not_counted(self):
        gate = pg.PlainGate(cfg(min_spec=1))
        loop = ConfLoop(gate, Req("a"), accept=lambda n: 1, conf=lambda n: 0.35)
        kinds = loop.run(60)
        counted = sum(n for n, _ in gate.calibration().values())
        # Results counted so far: every speculative step but those still in flight (lag 2).
        self.assertLessEqual(counted, kinds.count("S"))
        self.assertGreaterEqual(counted, kinds.count("S") - 2)
        self.assertTrue(all(mean == 1.0 for _, mean in gate.calibration().values()))
        self.assertIn(3, gate.calibration())  # issued while the EMA sat at 0.35

    def test_empty(self):
        self.assertEqual(pg.PlainGate(cfg()).calibration_line(), "no speculative results yet")


class SignalReadsOnlyTest(unittest.TestCase):
    """The signal code writes no tensor it is given: no in-place tensor method (name ending in '_') on anything but the
    ring's host row, no item or slice assignment, no attribute assignment outside self."""

    FUNCS = ("top1_probability", "_note_logits", "note_verify_logits")

    def test_signal_functions(self):
        tree = ast.parse((HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))
        found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in self.FUNCS]
        self.assertEqual(sorted(n.name for n in found), sorted(self.FUNCS))
        for fn in found:
            with self.subTest(fn=fn.name):
                for n in ast.walk(fn):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                        self.assertFalse(n.func.attr.endswith("_") and not n.func.attr.startswith("_"),
                                         ast.unparse(n))
                    if isinstance(n, (ast.Assign, ast.AugAssign)):
                        targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                        for t in targets:
                            self.assertNotIsInstance(t, ast.Subscript, ast.unparse(n))
                            if isinstance(t, ast.Attribute):
                                self.assertEqual(ast.unparse(t.value), "self", ast.unparse(n))

    def test_only_the_ring_copies_in_place(self):
        tree = ast.parse((HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))
        backend = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "_TorchRingBackend")
        copy = next(n for n in backend.body if isinstance(n, ast.FunctionDef) and n.name == "copy")
        src = ast.unparse(copy)
        self.assertIn("host[slot, :n].copy_(", src)
        self.assertIn("non_blocking=True", src)
        self.assertNotIn("synchronize", (HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))
        self.assertNotIn(".item()", (HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))

    def test_plain_step_reads_the_sampled_buffer_after_the_forward(self):
        tree = ast.parse((HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))
        run_plain = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_plain")
        src = ast.unparse(run_plain)
        fwd = src.index("target_worker.forward_batch_generation(")
        note = src.index("self._note_logits(batch, logits_output.next_token_logits, 1, sampled=True)")
        self.assertLess(fwd, note)



class BookkeepingModelTest(unittest.TestCase):
    """The state a mixed run of plain and speculative steps leaves, with the commit rules copied from the tree:
    engram history (EngramHasher.commit_after_verify's gather for verify; the MODE_DECODE in-forward commit for a
    plain step), draft and target KV positions, seq lens, and the scheduler's token slicing
    (batch_result_processor._resolve_spec_v2_tokens). Checks the invariants the plain step must keep: after any
    interleaving, the history holds the last n-1 committed tokens, every committed position has target and draft
    KV, and the emitted tokens are the reference stream."""

    STRIDE = 4

    @staticmethod
    def verify_commit(history, verify_ids, commit_len):
        n1 = len(history)
        window = history + verify_ids
        return [window[commit_len + i] for i in range(n1)]

    @staticmethod
    def decode_commit(history, input_id):
        n1 = len(history)
        tokens = [input_id] + history[::-1]
        return tokens[:n1][::-1]

    def run_mix(self, kinds, n1, seed):
        rng = random.Random(seed)
        ref = [rng.randrange(1000) for _ in range(4000)]
        prompt = 7
        history = ref[prompt - n1:prompt]
        seq_len = prompt
        target_kv = set(range(prompt))
        draft_kv = set(range(prompt))
        bonus = ref[prompt]
        emitted = []
        for kind in kinds:
            if kind == pg.SPEC:
                # Drafts: right for a random prefix, then wrong.
                good = rng.randrange(self.STRIDE)
                drafts = [ref[seq_len + 1 + i] if i < good else -1 for i in range(self.STRIDE - 1)]
                verify_ids = [bonus] + drafts
                commit_len = good + 1
                out_row = drafts[:good] + [ref[seq_len + commit_len]] + [0] * (self.STRIDE - commit_len)
                history = self.verify_commit(history, verify_ids, commit_len)
                positions = range(seq_len, seq_len + commit_len)
            else:
                commit_len = 1
                out_row = [ref[seq_len + 1]] + [0] * (self.STRIDE - 1)
                history = self.decode_commit(history, bonus)
                positions = [seq_len]
            target_kv.update(positions)
            draft_kv.update(positions)
            # The scheduler's slice of the flattened [bs, stride] output (bs = 1).
            accepted = out_row[0: commit_len]
            emitted += accepted
            seq_len += commit_len
            bonus = accepted[-1]
            self.assertEqual(history, ref[seq_len - n1:seq_len])
            self.assertEqual(target_kv, set(range(seq_len)))
            self.assertEqual(draft_kv, set(range(seq_len)))
            self.assertEqual(bonus, ref[seq_len])
        self.assertEqual(emitted, ref[prompt + 1:seq_len + 1])

    def test_interleavings(self):
        rng = random.Random(1)
        patterns = ["S" * 50, "P" * 50, "SP" * 25, "SSPPP" * 10, "SPP" * 17]
        patterns += ["".join(rng.choice("SP") for _ in range(200)) for _ in range(20)]
        for n1 in (1, 2, 3):
            for i, kinds in enumerate(patterns):
                with self.subTest(n1=n1, pattern=i):
                    self.run_mix(kinds, n1, seed=i)

    def test_decode_and_verify_commits_agree(self):
        # A verify that commits one position (the anchor) leaves the history a plain step would.
        rng = random.Random(2)
        for n1 in (1, 2, 3, 4):
            history = [rng.randrange(100) for _ in range(n1)]
            anchor = rng.randrange(100)
            drafts = [rng.randrange(100) for _ in range(self.STRIDE - 1)]
            self.assertEqual(self.verify_commit(history, [anchor] + drafts, 1), self.decode_commit(history, anchor))


class RunnerMechanicsTest(unittest.TestCase):
    class FakeBase:
        """The assignments DecodeCudaGraphRunner.__init__ and capture_prepare make, in their order."""

        def __init__(self, model_runner):
            self.capture_forward_mode = "DECODE"
            self.capture_hidden_mode = "NULL"
            self.captured_req_width = 4
            if model_runner.speculative:
                self.capture_forward_mode = "TARGET_VERIFY"
            self.capture_bs, self.compile_bs = list(model_runner.bs), []
            self.max_bs = max(self.capture_bs)
            self.max_num_token = self.max_bs * self.captured_req_width
            model_runner.kt_sizes = self.capture_bs
            spec_info = self.get_spec_info(self.max_num_token)
            self.capture_hidden_mode = "FULL" if spec_info is not None else self.capture_hidden_mode
            self.spec_info = spec_info

        def get_spec_info(self, num_tokens):
            return "DFlashVerifyInput"

    class MR:
        speculative = True
        bs = (1, 2, 4)

    def test_properties_pin_decode(self):
        cls = pg.make_plain_runner_cls(self.FakeBase, decode_mode="DECODE", full_mode="FULLMODE")
        runner = cls(self.MR())
        self.assertEqual(runner.capture_forward_mode, "DECODE")
        self.assertEqual(runner.captured_req_width, 1)
        self.assertEqual(runner.capture_hidden_mode, "FULLMODE")
        self.assertEqual(runner.capture_bs, [1])
        self.assertEqual((runner.max_bs, runner.max_num_token), (1, 1))
        self.assertIsNone(runner.spec_info)
        self.assertNotIn("capture_forward_mode", vars(runner))

    def test_capture_bs_filter(self):
        for plain_bs, sizes, want in (((1,), (1, 2, 4), [1]), ((1, 2), (1, 2, 4), [1, 2]), ((1,), (2, 4), [2])):
            with self.subTest(plain_bs=plain_bs, sizes=sizes):
                cls = pg.make_plain_runner_cls(self.FakeBase, decode_mode="D", full_mode="F", plain_bs=plain_bs)
                mr = self.MR()
                mr.bs = sizes
                self.assertEqual(cls(mr).capture_bs, want)

    def test_installed_runner_restores(self):
        class MRun:
            decode_cuda_graph_runner = "verify"

        mr = MRun()
        with pg.installed_runner(mr, None):
            self.assertIsNone(mr.decode_cuda_graph_runner)
        self.assertEqual(mr.decode_cuda_graph_runner, "verify")
        with self.assertRaises(KeyError), pg.installed_runner(mr, "plain"):
            self.assertEqual(mr.decode_cuda_graph_runner, "plain")
            raise KeyError
        self.assertEqual(mr.decode_cuda_graph_runner, "verify")

    def test_kept_graph_state(self):
        class Backend:
            def __init__(self):
                self.cuda_graph_metadata_of_bucket_and_bs = {"decode_or_idle": {}, "target_verify": {1: "meta"}}

            def init_cuda_graph_state(self, max_bs, max_num_tokens):
                self.cuda_graph_metadata_of_bucket_and_bs = {"decode_or_idle": {}, "target_verify": {}}

        backend = Backend()
        with pg.kept_graph_state(backend):
            backend.init_cuda_graph_state(1, 1)
            with self.assertRaises(RuntimeError), pg.kept_graph_state(backend):
                pass
        self.assertEqual(backend.cuda_graph_metadata_of_bucket_and_bs["target_verify"], {1: "meta"})
        self.assertNotIn("init_cuda_graph_state", vars(backend))
        backend.init_cuda_graph_state(4, 16)  # the class method is back
        self.assertEqual(backend.cuda_graph_metadata_of_bucket_and_bs["target_verify"], {})
        with self.assertRaises(RuntimeError), pg.kept_graph_state(object()):
            pass


# ---------------------------------------------------------------------------------------------------------------------
# r4: the draft-only probe

_spec3 = importlib.util.spec_from_file_location("dspark_plain_gate_r3", HERE.parent / "plain-gate-r3" / "dspark_plain_gate.py")
pg3 = importlib.util.module_from_spec(_spec3)
sys.modules["dspark_plain_gate_r3"] = pg3
_spec3.loader.exec_module(pg3)


def dcfg(**kw):
    return cfg(signal="draft", **kw)


class DraftLoop(OverlapLoop):
    """OverlapLoop with draft probes as the controller runs them: after a step is recorded plain, a probe precedes it
    when the gate wants one; the probe's sample, `probe(n)` tokens a step, settles `settle` steps later (the probe
    lands, then the committed tokens decide it) and is handed to note_probe before the next vote."""

    def __init__(self, gate, req, accept, probe, settle=4, lag=2, eligible=lambda step: True):
        super().__init__(gate, req, accept, lag=lag, eligible=eligible)
        self.probe, self.settle = probe, settle
        self.landing = deque()
        self.probes = []  # steps preceded by a probe
        self.taken = 0

    def step(self):
        n = len(self.kinds)
        while self.landing and self.landing[0][0] <= n:
            _, value = self.landing.popleft()
            self.taken += bool(self.gate.note_probe(self.req.rid, value))
        kind = super().step()
        if kind == pg.PLAIN and self.gate.wants_probe(self.req.rid):
            self.probes.append(n)
            self.landing.append((n + self.settle, self.probe(n)))
        return kind


class ProbeSampleTest(unittest.TestCase):
    def test_full_match_partial_and_mismatch(self):
        origin, out = [10, 11, 12], [20, 21, 22, 23]  # positions 0-2 prompt, 3-6 output
        # bonus at p = 3 (token 20): drafts guess positions 4, 5, 6
        self.assertEqual(pg.probe_sample(origin, out, 3, [21, 22, 23]), 4)
        self.assertEqual(pg.probe_sample(origin, out, 3, [21, 99, 23]), 2)
        self.assertEqual(pg.probe_sample(origin, out, 3, [99, 22, 23]), 1)
        # positions not committed yet: undecided while every committed one matches, decided at a mismatch
        self.assertIsNone(pg.probe_sample(origin, out, 5, [23, 24, 25]))
        self.assertEqual(pg.probe_sample(origin, out, 5, [99, 24, 25]), 1)
        self.assertIsNone(pg.probe_sample(origin, out, 6, [1, 2, 3]))
        # positions inside the prompt (p before its end) read the prompt
        self.assertEqual(pg.probe_sample(origin, out, 0, [11, 12, 20]), 4)
        self.assertEqual(pg.probe_sample(origin, out, 1, [12, 20, 0]), 3)

    def test_probe_bucket(self):
        self.assertEqual(pg.probe_bucket(None), -1)
        self.assertEqual([pg.probe_bucket(x) for x in (0.5, 1.0, 1.49, 1.5, 2.2, 3.9, 4.0, 9.0)], [0, 0, 0, 1, 2, 5, 5, 5])


class IntBackend(FakeBackend):
    def host(self, slots, width):
        return [[0] * width for _ in range(slots)]


class ProbeRingTest(unittest.TestCase):
    def test_rows_lagged_in_order(self):
        ring = pg.ProbeRing(row=4, slots=3, width=2, backend=IntBackend())
        self.assertTrue(ring.push(["a", "b"], [100, 1, 2, 3, 200, 4, 5, 6]))
        self.assertTrue(ring.push(["a"], [101, 7, 8, 9]))
        self.assertEqual(ring.drain(), [])
        ring.events[1].done = True  # out of order: blocked behind slot 0
        self.assertEqual(ring.drain(), [])
        ring.events[0].done = True
        self.assertEqual(ring.drain(), [("a", 100, [1, 2, 3]), ("b", 200, [4, 5, 6]), ("a", 101, [7, 8, 9])])
        self.assertEqual((ring.pushed, ring.dropped), (2, 0))

    def test_full_or_oversized_drops(self):
        ring = pg.ProbeRing(row=2, slots=1, width=1, backend=IntBackend())
        self.assertTrue(ring.push(["a"], [5, 6]))
        self.assertFalse(ring.push(["a"], [7, 8]))  # the only row is in flight
        self.assertFalse(ring.push(["a", "b"], [1, 2, 3, 4]))
        self.assertEqual(ring.dropped, 2)


class DraftModeTest(unittest.TestCase):
    def test_prose_never_takes_a_timer_probe(self):
        loop = DraftLoop(pg.PlainGate(dcfg()), Req("a"), accept=lambda n: 1, probe=lambda n: 1.0)
        kinds = loop.run(300)
        self.assertEqual(kinds[:5], "S" * 5)
        self.assertEqual(kinds[5:], "P" * 295)  # r2 would probe every 16 plain steps
        # a probe before every 8th plain step
        plain_steps = [n for n, k in enumerate(kinds) if k == "P"]
        self.assertEqual(loop.probes, [plain_steps[i] for i in range(7, len(plain_steps), 8)])
        self.assertGreater(loop.taken, 30)

    def test_switch_when_the_draft_starts_to_pay(self):
        good = lambda n: n >= 100
        loop = DraftLoop(pg.PlainGate(dcfg()), Req("a"), accept=lambda n: 4 if good(n) else 1,
                         probe=lambda n: 4.0 if good(n) else 1.0)
        kinds = loop.run(220)
        first = kinds.index("S", 10)
        # two good probes (the 8th-plain-step cadence) plus their settling, then speculation that stays
        self.assertGreater(first, 100)
        self.assertLess(first, 100 + 2 * 8 + 4 + 8)
        self.assertEqual(kinds[first:], "S" * (220 - first))

    def test_awaiting_then_back_to_plain_when_the_draft_was_wrong(self):
        gate = pg.PlainGate(dcfg())
        # probes say 4 tokens, but a verify accepts nothing: the switch runs until its first result, then plain
        loop = DraftLoop(gate, Req("a"), accept=lambda n: 1, probe=lambda n: 4.0)
        kinds = loop.run(120)
        runs = [len(r) for r in kinds[5:].split("P") if r]
        self.assertTrue(runs and all(r == 2 for r in runs), runs)  # lag 2: two speculative steps per switch
        # each switch consumed the probe EMA: the next one needs draft_min fresh samples, 2 x 8 plain steps
        gaps = [len(r) for r in kinds[5:].split("S") if r][1:-1]
        self.assertTrue(gaps and min(gaps) >= 16, gaps)

    def test_samples_while_speculating_are_ignored(self):
        gate = pg.PlainGate(dcfg())
        self.assertFalse(gate.note_probe("x", 4.0))  # unknown request
        loop = OverlapLoop(gate, Req("a"), accept=lambda n: 4)
        loop.run(20)
        self.assertEqual(gate._reqs["a"].plain_run, 0)
        self.assertFalse(gate.note_probe("a", 1.0))
        self.assertIsNone(gate.draft_ema("a"))

    def test_draft_ema_and_min_samples(self):
        gate = pg.PlainGate(dcfg(draft_min=3))
        loop = OverlapLoop(gate, Req("a"), accept=lambda n: 1)
        loop.run(10)  # plain by now
        self.assertTrue(gate.note_probe("a", 4.0))
        self.assertTrue(gate.note_probe("a", 2.0))
        self.assertEqual(gate.draft_ema("a"), 3.0)
        self.assertEqual(loop.step(), "P")  # 2 samples < draft_min 3
        self.assertTrue(gate.note_probe("a", 3.0))
        self.assertEqual(loop.step(), "S")

    def test_no_probes_outside_draft_mode(self):
        for signal in ("accept", "confidence"):
            gate = pg.PlainGate(cfg(signal=signal))
            loop = OverlapLoop(gate, Req("a"), accept=lambda n: 1)
            for _ in range(100):
                loop.step()
                self.assertFalse(gate.wants_probe("a"))

    def test_probe_calibration(self):
        gate = pg.PlainGate(dcfg())
        loop = DraftLoop(gate, Req("a"), accept=lambda n: 3 if n >= 60 else 1, probe=lambda n: 3.0 if n >= 60 else 1.0)
        loop.run(200)
        cal = gate.probe_calibration()
        # earlier 1-token samples, then two 3-token ones at alpha 0.5: the switch fires at EMA 2.5
        self.assertEqual(list(cal), [pg.probe_bucket(2.5)])
        self.assertAlmostEqual(cal[pg.probe_bucket(2.5)][1], 3.0)
        self.assertIn("2.5-3.0", gate.probe_calibration_line())
        self.assertEqual(pg.PlainGate(dcfg()).probe_calibration_line(), "no switches yet")

    def test_batch_speculates_only_when_every_request_wants_to(self):
        gate = pg.PlainGate(dcfg(max_bs=2))
        reqs = {r: Req(r) for r in "ab"}
        good = {"a": lambda n: True, "b": lambda n: n >= 120}
        inflight, kinds, landing = deque(), [], deque()
        for n in range(240):
            while len(inflight) >= 2:
                for rid, committed in inflight.popleft():
                    reqs[rid].spec_verify_ct += 1
                    reqs[rid].spec_num_correct_drafts += committed - 1
            while landing and landing[0][0] <= n:
                _, rid, value = landing.popleft()
                gate.note_probe(rid, value)
            votes = [gate.vote(r, verify_ct=reqs[r].spec_verify_ct, correct_drafts=reqs[r].spec_num_correct_drafts,
                               eligible=True) for r in "ab"]
            kind = pg.batch_kind(votes)
            for r in "ab":
                gate.record(r, kind)
            if kind == pg.PLAIN and any(gate.wants_probe(r) for r in "ab"):
                for r in "ab":
                    landing.append((n + 4, r, 4.0 if good[r](n) else 1.0))
            kinds.append(kind)
            inflight.append([(r, 1 if kind == pg.PLAIN else (4 if good[r](n) else 1)) for r in "ab"])
        kinds = "".join(kinds)
        # 'a' alone wants to speculate from early on: the batch stays plain until 'b' does too
        self.assertEqual(kinds[6:120], "P" * 114)
        first = kinds.index("S", 120)
        self.assertLess(first, 120 + 40)
        self.assertEqual(kinds[first:], "S" * (240 - first))


class ModesUnchangedFromR3Test(unittest.TestCase):
    """'accept' and 'confidence' issue r3's steps exactly; 'draft' is new."""

    def test_single_request(self):
        for name, accept in AcceptModeUnchangedTest.ACCEPTS.items():
            for eligible in (lambda n: True, lambda n: n % 7 != 3):
                with self.subTest(case=name):
                    r3 = OverlapLoop(pg3.PlainGate(pg3.GateConfig()), Req("a"), accept=accept, eligible=eligible).run(200)
                    r4 = OverlapLoop(pg.PlainGate(cfg()), Req("a"), accept=accept, eligible=eligible).run(200)
                    self.assertEqual(r4, r3)
                    conf = lambda n: 0.95 if (n // 9) % 2 else 0.2
                    c3 = ConfLoop(pg3.PlainGate(pg3.GateConfig(signal="confidence", conf=0.6)), Req("a"), accept=accept,
                                  eligible=eligible, conf=conf).run(200)
                    c4 = ConfLoop(pg.PlainGate(cfg(signal="confidence", conf=0.6)), Req("a"), accept=accept,
                                  eligible=eligible, conf=conf).run(200)
                    self.assertEqual(c4, c3)

    def test_batches(self):
        accept = {"a": lambda n: 1, "b": lambda n: 4, "c": lambda n: 1 if n < 30 else 4}
        r3 = BatchLoop(pg3.PlainGate(pg3.GateConfig()), list("abc"), accept)
        r4 = BatchLoop(pg.PlainGate(cfg()), list("abc"), accept)
        self.assertEqual(r4.run(40, ["a", "b"]) + r4.run(40, ["a", "b", "c"]) + r4.run(40, ["c"]),
                         r3.run(40, ["a", "b"]) + r3.run(40, ["a", "b", "c"]) + r3.run(40, ["c"]))

    def test_calibration_unchanged(self):
        accept = lambda n: random.Random(n).choice((1, 2, 3, 4))
        conf = lambda n: random.Random(1000 + n).random()
        g3, g4 = pg3.PlainGate(pg3.GateConfig()), pg.PlainGate(cfg())
        ConfLoop(g3, Req("a"), accept=accept, conf=conf).run(300)
        ConfLoop(g4, Req("a"), accept=accept, conf=conf).run(300)
        self.assertEqual(g4.calibration(), g3.calibration())
        self.assertEqual(g4.probe_calibration(), {})


class FullReq(Req):
    def __init__(self, rid, origin, output):
        super().__init__(rid)
        self.origin_input_ids, self.output_ids = origin, output


class CollectProbesTest(unittest.TestCase):
    """PlainStepController._collect_probes: landed rows settle against committed ids; undecided rows wait."""

    def test_settle_and_wait(self):
        ctl = pg.PlainStepController(worker=None, config=dcfg())
        ctl.probe_ring = pg.ProbeRing(row=4, slots=4, width=4, backend=IntBackend())
        req = FullReq("a", [1, 2, 3], [])
        loop = OverlapLoop(ctl.gate, req, accept=lambda n: 1)
        loop.run(10)  # plain mode
        req.output_ids = [7, 8, 9]  # positions 3, 4, 5
        # probe at p = 3: guesses 8, 9 (match), then position 6 not committed yet
        ctl.probe_ring.push(["a"], [3, 8, 9, 5])
        ctl.probe_ring.events[0].done = True
        ctl._collect_probes([req])
        self.assertEqual(ctl.num_probe_samples, 0)
        self.assertEqual(len(ctl._landed["a"]), 1)
        req.output_ids.append(6)  # position 6 commits 6, not 5: two drafts accepted
        ctl._collect_probes([req])
        self.assertEqual(ctl.num_probe_samples, 1)
        self.assertEqual(ctl.gate.draft_ema("a"), 3.0)

    def test_requests_the_gate_dropped(self):
        ctl = pg.PlainStepController(worker=None, config=dcfg())
        ctl.probe_ring = pg.ProbeRing(row=2, slots=2, width=2, backend=IntBackend())
        ctl.probe_ring.push(["gone"], [0, 1])
        ctl.probe_ring.events[0].done = True
        ctl._collect_probes([])
        self.assertNotIn("gone", ctl._landed)


# ---------------------------------------------------------------------------------------------------------------------
# Against the pinned tree (PLAIN_GATE_SGLANG)


def _src(rel):
    return (Path(SGLANG) / rel).read_text(encoding="utf-8")


def _find(tree, *names):
    """The def/class at the dotted path `names` in an AST."""
    node = tree
    for name in names:
        node = next(
            n for n in ast.iter_child_nodes(node)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name
        )
    return node


def _params(fn):
    a = fn.args
    return {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}


def _self_assigned(fn):
    out = set()
    for n in ast.walk(fn):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AnnAssign, ast.AugAssign)) else []
        for t in targets:
            for e in ast.walk(t):
                if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self":
                    out.add(e.attr)
    return out


def _attrs_read(fn, base="self"):
    return {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == base}


@unittest.skipUnless(SGLANG, "PLAIN_GATE_SGLANG not set")
class TreeTest(unittest.TestCase):
    def test_apis_the_module_calls(self):
        planner = ast.parse(_src("srt/speculative/dspark_components/dspark_planner.py"))
        self.assertLessEqual(
            {"batch", "bs", "device", "verify_num_draft_tokens", "block_pos_offsets", "model_runner", "seq_lens_cpu"},
            _params(_find(planner, "alloc_verify_window")),
        )
        inject = ast.parse(_src("srt/speculative/dspark_components/dspark_kv_inject.py"))
        self.assertLessEqual(
            {"target_hidden", "cache_loc", "cache_loc_2d", "positions", "commit_lens", "state_slot"},
            _params(_find(inject, "TargetHiddenKvInjector", "inject_target_hidden")),
        )
        _find(inject, "TargetHiddenKvInjector", "ids_for")
        _find(inject, "TargetHiddenKvInjector", "swa_ring_size")
        draft = ast.parse(_src("srt/speculative/dspark_components/dspark_draft.py"))
        self.assertLessEqual({"bonus_tokens", "new_seq_lens"}, _params(_find(draft, "make_next_draft_input")))
        fbi = ast.parse(_src("srt/model_executor/forward_batch_info.py"))
        self.assertLessEqual(
            {"capture_hidden_mode", "return_hidden_states_before_norm", "iteration_ids"},
            _params(_find(fbi, "ForwardBatch", "init_new")),
        )
        tpw = ast.parse(_src("srt/managers/tp_worker.py"))
        self.assertLessEqual({"batch", "forward_batch"}, _params(_find(tpw, "TpModelWorker", "forward_batch_generation")))
        utils = ast.parse(_src("srt/managers/utils.py"))
        fields = {n.target.id for n in _find(utils, "GenerationBatchResult").body if isinstance(n, ast.AnnAssign)}
        self.assertLessEqual(
            {"patha_static_flags", "logits_output", "next_token_ids", "accept_lens", "block_accept_lens", "cap_lens",
             "can_run_cuda_graph", "next_draft_input", "speculative_num_draft_tokens", "new_seq_lens"},
            fields,
        )
        deferred = ast.parse(_src("srt/speculative/dspark_components/dspark_deferred_commit.py"))
        self.assertIn("reason", _params(_find(deferred, "DeferredContextCommit", "flush")))
        sync = _src("srt/speculative/spec_tp_sync.py")
        self.assertIn("DSPARK_TARGET =", sync)
        self.assertIn("DSPARK_MEM =", sync)
        patha = ast.parse(_src("srt/layers/moe/contract_v2/patha_static_runtime.py"))
        _find(patha, "snapshot_after_forward")

    def test_schedule_batch_view_and_req_counters(self):
        tree = ast.parse(_src("srt/managers/schedule_batch.py"))
        batch = _find(tree, "ScheduleBatch")
        # A plain dataclass: copy.copy gives an independent view whose fields can be rebound.
        self.assertTrue(any("dataclass" in ast.unparse(d) and "slots" not in ast.unparse(d) for d in batch.decorator_list))
        fields = {n.target.id for n in batch.body if isinstance(n, ast.AnnAssign)}
        self.assertLessEqual(
            {"reqs", "forward_mode", "input_ids", "req_pool_indices", "seq_lens", "seq_lens_cpu", "seq_lens_sum",
             "out_cache_loc", "spec_info", "sampling_info", "has_grammar", "return_logprob", "return_hidden_states"},
            fields,
        )
        self.assertFalse(any(n.name in ("__copy__", "__setattr__") for n in batch.body if isinstance(n, ast.FunctionDef)))
        req_init = _self_assigned(_find(tree, "Req", "__init__"))
        self.assertLessEqual({"rid", "spec_verify_ct", "spec_num_correct_drafts"}, req_init)
        # The result processor advances those counters per decode result: +1 verify, + accepted - 1 drafts.
        proc = ast.unparse(_find(ast.parse(_src("srt/managers/scheduler_components/batch_result_processor.py")),
                                 "SchedulerBatchResultProcessor", "_resolve_spec_v2_tokens"))
        self.assertIn("req.spec_verify_ct += 1", proc)
        self.assertIn("req.spec_num_correct_drafts += num_correct_drafts", proc)
        self.assertIn("next_token_ids[i * stride:i * stride + accept_lens[i]]", proc)

    def test_worker_attributes_the_module_reads(self):
        worker = ast.parse(_src("srt/speculative/dspark_components/dspark_worker_v2.py"))
        cls = _find(worker, "DSparkWorkerV2")
        assigned = _self_assigned(_find(cls, "__init__")) | _self_assigned(_find(cls, "init_attention_backends"))
        module = ast.parse((HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))
        used = set()
        for fn in ast.walk(module):
            if isinstance(fn, ast.FunctionDef):
                used |= _attrs_read(fn, "worker")
        used -= {"target_worker"}  # BaseSpecWorker property
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertLessEqual(used, assigned | methods, used - assigned - methods)

    def test_runner_assignments_the_properties_override(self):
        runner = ast.parse(_src("srt/model_executor/runner/decode_cuda_graph_runner.py"))
        cls = _find(runner, "DecodeCudaGraphRunner")
        init = _find(cls, "__init__")
        self.assertLessEqual({"capture_forward_mode", "captured_req_width", "capture_hidden_mode", "capture_bs"},
                             _self_assigned(init))
        self.assertIn("init_cuda_graph_state", _src("srt/model_executor/runner/decode_cuda_graph_runner.py"))
        self.assertIn("get_spec_info", {n.name for n in cls.body if isinstance(n, ast.FunctionDef)})
        prep = ast.unparse(_find(cls, "capture_prepare"))
        self.assertIn("self.get_spec_info(num_tokens)", prep)
        self.assertFalse(any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__slots__" for t in n.targets)
                             for n in cls.body))
        # A DECODE batch passes the verify runner's can_run_graph when spec_info is None: the plain step must
        # install its own runner (or None) around the forward.
        can_run = ast.unparse(_find(cls, "can_run_graph"))
        self.assertIn("spec_info is not None", can_run)
        model_runner = ast.unparse(_find(ast.parse(_src("srt/model_executor/model_runner.py")), "ModelRunner", "_forward_raw"))
        self.assertIn("self.decode_cuda_graph_runner.can_run_graph(forward_batch)", model_runner)

    def test_backend_reset_and_buckets(self):
        backend = ast.parse(_src("srt/layers/attention/deepseek_v4_backend.py"))
        init = _find(backend, "DeepseekV4AttnBackend", "init_cuda_graph_state")
        self.assertIn("cuda_graph_metadata_of_bucket_and_bs", _self_assigned(init))
        bucket = ast.unparse(_find(backend, "_GraphBucket"))
        self.assertIn("DECODE_OR_IDLE", bucket)
        self.assertIn("TARGET_VERIFY", bucket)

    def test_decode_commits_engram_history_in_forward(self):
        hasher = ast.unparse(_find(ast.parse(_src("srt/layers/engram.py")), "EngramHasher", "forward"))
        self.assertIn("kmode = MODE_DECODE", hasher)
        self.assertIn("engram_hash_ids_and_commit(", hasher)

    def test_signal_sources(self):
        worker = _src("srt/speculative/dspark_components/dspark_worker_v2.py")
        self.assertEqual(worker.count("        logits_output = target_verify.logits_output\n"
                                      "        can_run_cuda_graph = target_verify.can_run_cuda_graph\n"), 1)
        # The verify target forward returns before sampling, so the hook reads raw logits.
        tpw = ast.unparse(_find(ast.parse(_src("srt/managers/tp_worker.py")), "TpModelWorker", "forward_batch_generation"))
        self.assertIn("if is_verify:\n            return batch_result", tpw)
        self.assertLess(tpw.index("if is_verify:"), tpw.index("self.model_runner.sample("))
        # The sampler: argmax only for an all-greedy batch; otherwise the buffer becomes softmax(logits / T) in place.
        sampler = ast.unparse(_find(ast.parse(_src("srt/layers/sampler.py")), "Sampler", "forward"))
        self.assertIn("logits = logits_output.next_token_logits", sampler)
        self.assertIn("batch_next_token_ids = torch.argmax(logits, -1)", sampler)
        self.assertIn("logits.div_(sampling_info.temperatures)", sampler)
        self.assertIn("logits[:] = torch.softmax(logits, dim=-1) if self.enable_deterministic else sampler_softmax(logits)",
                      sampler)
        logits = ast.parse(_src("srt/layers/logits_processor.py"))
        fields = {n.target.id for n in _find(logits, "LogitsProcessorOutput").body if isinstance(n, ast.AnnAssign)}
        self.assertIn("next_token_logits", fields)
        info = ast.parse(_src("srt/sampling/sampling_batch_info.py"))
        cls = _find(info, "SamplingBatchInfo")
        fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}
        self.assertIn("temperatures", fields)
        self.assertIn("is_all_greedy", fields | {n.name for n in cls.body if isinstance(n, ast.FunctionDef)})
        # The verify graph's own buffers stay the logits' storage until the next forward: the hook runs in the same
        # stream before it (the worker reads next_token_logits again for the accept, after the hook).
        decode = ast.unparse(_find(ast.parse(worker), "DSparkWorkerV2", "_forward_decode"))
        self.assertLess(decode.index("logits_output = target_verify.logits_output"),
                        decode.index("accept_and_finalize("))

    def test_probe_call_sites(self):
        """The draft-only probe calls what _forward_decode calls before a verify, the same way."""
        planner = ast.parse(_src("srt/speculative/dspark_components/dspark_planner.py"))
        self.assertLessEqual({"bonus", "anchor_ids"}, _params(_find(planner, "alloc_verify_window")))
        draft = ast.parse(_src("srt/speculative/dspark_components/dspark_draft.py"))
        proposer = _find(draft, "DraftBlockProposer")
        self.assertLessEqual(
            {"batch", "draft_input", "verify_window", "bs", "device", "target_model", "sampling_info", "anchor_prepared"},
            _params(_find(proposer, "propose")),
        )
        self.assertLessEqual({"bs", "device"}, _params(_find(proposer, "draft_block_ids_view")))
        fields = lambda cls: {n.target.id for n in _find(draft, cls).body if isinstance(n, ast.AnnAssign)}
        self.assertIn("draft_block", fields("DraftProposal"))
        self.assertIn("draft_tokens", fields("DraftBlockResult"))
        self.assertIn("draft_model_runner", _self_assigned(_find(proposer, "__init__")))
        run = ast.unparse(_find(proposer, "_run_forward"))
        # The probe skips exactly where the proposer would raise: no host seq lens and a backend that needs them.
        self.assertIn("if not self.draft_model_runner.attn_backend.needs_cpu_seq_lens:", run)
        self.assertIn("commit.before_draft_forward(", run)
        self.assertIn("commit.after_draft_forward(graph_ran=draft_out.can_run_graph)", run)
        self.assertIn("raise RuntimeError('DSpark decode expected batch.seq_lens_cpu, got None')", run)
        decode = ast.unparse(_find(ast.parse(_src("srt/speculative/dspark_components/dspark_worker_v2.py")),
                                   "DSparkWorkerV2", "_forward_decode"))
        for s in (
            "target_model = self.target_worker.model_runner.model",
            "anchor_ids = self._proposer.draft_block_ids_view(bs=bs, device=prefix_lens.device) if self._prepare_inputs_once else None",
            "verify_window = alloc_verify_window(batch=batch, bs=bs, device=device, verify_num_draft_tokens=self.verify_num_draft_tokens, "
            "block_pos_offsets=self._block_pos_offsets, model_runner=self.model_runner, seq_lens_cpu=window_seq_lens_cpu, "
            "bonus=draft_input.bonus_tokens.view(-1) if anchor_ids is not None else None, anchor_ids=anchor_ids)",
            "with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):",
            "proposal = self._proposer.propose(batch=batch, draft_input=draft_input, verify_window=verify_window, bs=bs, "
            "device=device, target_model=target_model, sampling_info=sampling_info, anchor_prepared=anchor_ids is not None)",
            "draft_tokens = draft_block.draft_tokens",
        ):
            with self.subTest(s=s[:60]):
                self.assertIn(s, decode)
        # The verify ids are the anchor then the draft tokens: draft token k guesses position p + k.
        self.assertIn("verify_ids_2d = torch.cat([draft_block_ids[:, :1], draft_tokens], dim=1).contiguous()", decode)

    def test_window_is_a_pure_plan(self):
        runner_ids = ast.parse(_src("srt/mem_cache/kv/alloc/runner_ids.py"))
        plan = _find(runner_ids, "RunnerIds", "plan_uniform_extend")
        item_writes = [ast.unparse(t) for n in ast.walk(plan) if isinstance(n, (ast.Assign, ast.AugAssign))
                       for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Subscript)]
        # Only the function's own local dicts (mappings, translated); nothing of self, no table.
        self.assertEqual(sorted(item_writes), ["mappings[kind]", "translated[kind]"])
        self.assertEqual({ast.unparse(t) for n in ast.walk(plan) if isinstance(n, ast.Assign) for t in n.targets
                          if isinstance(t, ast.Attribute)}, set())
        locals_ = {t.id for n in ast.walk(plan) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        self.assertLessEqual({"mappings", "translated"}, locals_)
        src = ast.unparse(plan)
        self.assertIn("req_to_token=self.req_to_token", src)
        self.assertIn("assign_extend_cache_locs_uniform_func(", src)
        self.assertIn("return self.plan(", src)
        self.assertIn("return IterationIds(", ast.unparse(_find(runner_ids, "RunnerIds", "plan")))
        kernels = ast.parse(_src("kernels/ops/speculative/cache_locs.py"))
        kernel = _find(kernels, "assign_extend_cache_locs_uniform")
        stored = {ast.unparse(c.args[0]).split(" ")[0] for c in ast.walk(kernel)
                  if isinstance(c, ast.Call) and ast.unparse(c.func) == "tl.store"}
        self.assertEqual(stored, {"out_cache_ptr", "positions_out", "anchor_out", "full_out", "swa_out"})
        self.assertNotIn("req_to_token[", ast.unparse(_find(kernels, "assign_extend_cache_locs_uniform_func")))
        bind = _find(ast.parse(_src("srt/mem_cache/kv/alloc/iteration_ids.py")), "IterationIds", "bind")
        assigned = {ast.unparse(t) for n in ast.walk(bind) if isinstance(n, ast.Assign) for t in n.targets}
        self.assertEqual(assigned, {"batch.iteration_ids", "batch.iteration_cols", "batch.out_cache_loc",
                                    "batch.out_cache_loc_virtual", "batch.out_cache_loc_is_physical"})

    def test_deferred_commit_without_a_verify(self):
        commit = ast.parse(_src("srt/speculative/dspark_components/dspark_deferred_commit.py"))
        cls = _find(commit, "DeferredContextCommit")
        # Every write disarms; a flush writes a pending staging; after a graph forward a pending staging counts as
        # written by it; the graph's head write with nothing staged writes zero commit lengths.
        self.assertIn("self.commit_lens.zero_()", ast.unparse(_find(cls, "write")))
        self.assertIn("if self.pending and (not _capturing()):", ast.unparse(_find(cls, "flush")))
        after = ast.unparse(_find(cls, "after_draft_forward"))
        self.assertIn("if graph_ran and self.pending:", after)
        self.assertEqual(ast.unparse(_find(cls, "pending")).count("self.staged_bs > 0"), 1)
        # The plain step flushes before it runs, so a probe that follows a plain step finds nothing staged.
        module = ast.parse((HERE / "dspark_plain_gate.py").read_text(encoding="utf-8"))
        run_plain = ast.unparse(_find(module, "PlainStepController", "_run_plain"))
        self.assertIn("commit.flush('plain')", run_plain)
        step = ast.unparse(_find(module, "PlainStepController", "maybe_plain_step"))
        self.assertLess(step.index("self._draft_probe(batch)"), step.index("self._run_plain(batch, on_publish)"))
        # Probes precede plain steps only (wants_probe needs this step plain); with DRAFT_EVERY >= 2 the step before
        # was plain too and flushed. At 1 a probe may follow a verify: _run_forward's hooks above write the staging.
        wants = ast.unparse(_find(module, "PlainGate", "wants_probe"))
        self.assertIn("state.plain_run > 0", wants)

    def test_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = importlib.util.spec_from_file_location("apply_dspark_plain_gate", HERE / "apply_dspark_plain_gate.py")
            mod = importlib.util.module_from_spec(overlay)
            overlay.loader.exec_module(mod)
            root = Path(tmp) / "sglang"
            for rel in mod.PINS:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(SGLANG) / rel, root / rel)
            run = subprocess.run([sys.executable, "-I", str(HERE / "apply_dspark_plain_gate.py"), str(root)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            printed = dict(line.split() for line in run.stdout.strip().splitlines())
            before = _src(mod.WORKER)
            after = (root / mod.WORKER).read_text(encoding="utf-8")
            self.assertEqual(printed[mod.WORKER], hashlib.sha256(after.encode()).hexdigest())
            module = (root / mod.MODULE_DST).read_bytes()
            self.assertEqual(module, (HERE / "dspark_plain_gate.py").read_bytes())
            for rel in mod.PINS:
                if rel != mod.WORKER:
                    self.assertEqual((root / rel).read_bytes(), (Path(SGLANG) / rel).read_bytes(), rel)

            # Insertions only: removing the inserted lines gives the served file byte for byte.
            a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
            ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
            self.assertTrue(all(op in ("equal", "insert") for op, *_ in ops), ops)
            inserted = [range(j1 + 1, j2 + 1) for op, _, _, j1, j2 in ops if op == "insert"]
            self.assertEqual(len(inserted), 7)
            kept = [line for op, i1, i2, _, _ in ops if op == "equal" for line in a[i1:i2]]
            self.assertEqual("".join(kept), before)

            # Every inserted statement is the selector constant, the None default, or behind a selector test.
            tree = ast.parse(after)
            lines = {n for r in inserted for n in r}
            stmts = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno in lines]
            outer = [n for n in stmts if not any(
                p is not n and p.lineno in lines and n in ast.walk(p) for p in stmts)]
            self.assertEqual(len(outer), 7, [ast.unparse(n).splitlines()[0] for n in outer])
            for node in outer:
                src = ast.unparse(node)
                with self.subTest(stmt=src.splitlines()[0]):
                    if isinstance(node, ast.Assign):
                        self.assertIn(src, ('_PLAIN_GATE_ON = os.environ.get(\'KT_DSPARK_PLAIN_GATE\', \'0\') == \'1\'',
                                            "_plain_gate = None"))
                    else:
                        self.assertIsInstance(node, ast.If)
                        self.assertTrue(ast.unparse(node.test).startswith(("_PLAIN_GATE_ON", "self._plain_gate is not None")))

            # A second run refuses: the worker no longer matches its pin.
            again = subprocess.run([sys.executable, "-I", str(HERE / "apply_dspark_plain_gate.py"), str(root)],
                                   capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)


if __name__ == "__main__":
    unittest.main()
