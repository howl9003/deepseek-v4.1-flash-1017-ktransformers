"""CPU tests for KT_DSPARK_PLAIN_GATE (dspark_plain_gate.py, apply_dspark_plain_gate.py). Claude, 4 Nov 2026; r3 adds the
confidence signal (config, helpers, the lagged ring, confidence probes, calibration, accept mode step for step as r2,
the signal reading logits only, and the sources it reads).

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
            }
        )
        self.assertEqual((c.threshold, c.probe_every, c.alpha, c.min_spec), (1.9, 8, 0.5, 2))
        self.assertEqual((c.force, c.graph, c.log_every, c.max_bs), ("SSPP", False, 100, 4))
        self.assertEqual((c.signal, c.conf), ("confidence", 0.62))

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
            ("KT_DSPARK_PLAIN_GATE_SIGNAL", "draft"),
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
