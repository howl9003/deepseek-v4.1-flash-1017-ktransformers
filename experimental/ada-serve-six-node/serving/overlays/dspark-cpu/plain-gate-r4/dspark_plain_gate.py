"""KT_DSPARK_PLAIN_GATE: per-request plain decode steps for DSpark (dspark-cpu-r1, Claude, 4 Nov 2026).

A DSpark step (draft + static 4-row verify) costs about 56-80 ms on fea01; a plain decode token about 30 ms. When a
request's draft accepts poorly (prose at T=1: 1.6-1.8 tokens a step) the step loses to plain decode. This module runs
the target's normal one-token DECODE forward for such a request instead, and keeps the draft's state as a verify that
committed one position would:

- the target forward is ForwardMode.DECODE at one token, with the DSpark aux hidden (layers 37-39) captured
  (CaptureHiddenMode.FULL); its token is sampled by the target worker exactly as plain decode samples it;
- the draft KV of that position is written from the captured hidden (TargetHiddenKvInjector.inject_target_hidden with
  commit length 1, the same call verify's commit_hidden makes);
- the engram history is committed by the DECODE forward itself (EngramHasher, MODE_DECODE), so no
  update_after_verify runs;
- the result has accept length 1, the sampled token as the next bonus, seq lens + 1 and the path A flag snapshot.

The DECODE forward replays its own CUDA graph (PlainDecodeCudaGraphRunner), captured beside the verify graphs on the
same attention backend. Everything here runs only with KT_DSPARK_PLAIN_GATE=1; the worker never imports this module
otherwise. Batches up to KT_DSPARK_PLAIN_GATE_MAX_BS requests (default 1) may take plain steps; larger ones speculate
as before. In a batch of two or more, every request votes and the batch runs plain when any request votes plain, so a
batch speculates only when all of its requests want to (r2, 4 Nov: a verify of b requests carries 4b rows' experts on
the CPU, so a mixed batch is costed as plain).

The gate (PlainGate) is pure Python: an EMA, per request, of tokens committed per speculative step, read from the
scheduler's own CPU counters (Req.spec_verify_ct, Req.spec_num_correct_drafts), so it never syncs the device.

r3 (4 Nov, owner: "the free signal ... first"): a confidence signal. Every decode step, plain or speculative, the
target's top-1 probability at the position that commits next (a plain step's DECODE row, read from the buffer the
sampler left: softmax(logits / T) in place for a sampled batch, raw logits for an all-greedy one; row 0 of each
request's verify rows, read before the accept) at the request's sampling temperature is computed on the device and
handed to the CPU through
a pinned host ring (ConfidenceRing: non-blocking copy, an event per row, read when the event has completed, never
waited for), so the gate sees it two or more steps late, like its counters. Per request it keeps an EMA of it.
KT_DSPARK_PLAIN_GATE_SIGNAL selects the probe rule: 'accept' (default) is r2's timer probe, step for step;
'confidence' replaces the timer: a request in plain mode takes one speculative step when its confidence EMA rises to
KT_DSPARK_PLAIN_GATE_CONF after being below it since its last speculative step. That probe re-measures the draft from
scratch: its tokens replace the acceptance EMA rather than blend into it (one blended good probe lifts an EMA of 1.0
only to 1.75 at alpha 0.25, under the 2.2 threshold, and the probe would not repeat). The acceptance EMA and the
threshold still decide whether it keeps speculating. KT_DSPARK_PLAIN_GATE_CONF defaults to 0.0, which disables confidence
probes; its value is to be calibrated from the calibration lines this module logs in both modes (confidence EMA
when a speculative step was issued, against the tokens that step committed). The signal only reads the logits: it
writes new tensors, never the logits or any buffer a step reads, so plain and verify steps keep their arithmetic.

r4 (4 Nov, step 2a of dspark-cpu-r1/shadow-lane-r1/SHADOW_LANE_DESIGN_r1.md): a draft-only probe.
KT_DSPARK_PLAIN_GATE_SIGNAL='draft': while a request runs plain steps, every KT_DSPARK_PLAIN_GATE_DRAFT_EVERY-th plain
step (default 8) is preceded by the draft's own proposal for the batch, with no verify: the full-width verify window
(a pure plan: plan_uniform_extend gathers the reserved slot ids from req_to_token into new tensors and writes no table)
and DraftBlockProposer.propose, called as DSparkWorkerV2._forward_decode calls them. The draft writes its block's KV
into those reserved slots, which the plain steps' injections overwrite as each position commits, as after a rejected
speculative step; the deferred context-KV commit has nothing pending after a plain step (_run_plain flushes it), so
the draft graph's head write is then a disarmed no-op (and a probe right after a verify, at DRAFT_EVERY 1, has the
staging written by the draft forward's own before/after hooks, as a speculative step's draft does). The draft tokens and the request's committed length go to a
pinned host ring (ProbeRing: non-blocking copy, an event per row, read only when complete). A probe is read when the
committed tokens settle it: m = the number of leading draft tokens equal to the committed tokens at positions p+1..,
known at the first mismatch or when all gamma are committed; its sample is 1 + m tokens a step, the units of the
acceptance EMA. Per request, an EMA of probe samples (KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA, default 0.5); a request in
plain mode votes to speculate once that EMA reaches the threshold after KT_DSPARK_PLAIN_GATE_DRAFT_MIN samples
(default 2), and keeps voting so until its first speculative results arrive; from then on the acceptance EMA decides,
as in r2. The timer probe is off in this mode: a plain request speculates only by the probe rule. The probe EMA is
consumed by the switch and cleared again when the request drops back to plain. Calibration: the probe EMA at each
switch against the tokens the switch's speculative steps committed, in 0.5-token buckets. Costs: one draft forward a
probe (about 3.3 ms on fea01), no device sync, no committed token changes (the probe only reads the draft); at T=1
the draft samples with its own Philox counter, which a probe advances, so later sampled drafts differ from a run
without probes, and m is a proxy for the acceptance a verify would draw.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Speculative ('S') and plain ('P') step kinds.
SPEC = "S"
PLAIN = "P"

# Steps a request may have issued whose results the scheduler has not counted yet; the overlap loop lags by two.
_PENDING_MAX = 64

# Free GPU memory the plain graph capture requires, as the draft graph capture does (dspark_worker_v2.init_cuda_graphs).
MIN_FREE_GB = 1.0

# Confidence signal (r3): host ring rows and their width (requests a step); calibration buckets of width 0.1.
RING_SLOTS = 8
RING_WIDTH = 64
CONF_BUCKETS = 10
SIGNALS = ("accept", "confidence", "draft")

# Draft probes (r4): host ring rows, requests a row, probe calibration buckets of 0.5 tokens from 1.0, and the most
# probes a request may have landed but not yet settled.
PROBE_SLOTS = 8
PROBE_BUCKET = 0.5
PROBE_BUCKETS = 6
_LANDED_MAX = 16


# ---------------------------------------------------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class GateConfig:
    threshold: float = 2.2  # tokens per speculative step below which a request runs plain steps
    probe_every: int = 16  # plain steps between two probe speculative steps
    alpha: float = 0.25  # EMA weight of a new speculative step
    min_spec: int = 4  # speculative steps a request runs before the gate may choose plain
    force: str = ""  # cyclic per-request pattern of S/P (identity runs); overrides the EMA rule
    graph: bool = True  # capture the plain DECODE graph (False: eager plain steps)
    log_every: int = 0  # log the step counts every N decode steps (0: off)
    max_bs: int = 1  # batch sizes that may take plain steps
    signal: str = "accept"  # probe rule in plain mode: 'accept' (timer, r2), 'confidence' (r3) or 'draft' (r4)
    conf: float = 0.0  # confidence EMA that triggers a probe in 'confidence' mode; 0.0 = never (to be calibrated)
    draft_every: int = 8  # 'draft': a draft-only probe before every this-many-th plain step
    draft_alpha: float = 0.5  # 'draft': EMA weight of a new probe sample
    draft_min: int = 2  # 'draft': probe samples before the probe EMA may switch a request to speculation

    @classmethod
    def from_env(cls, environ=None) -> "GateConfig":
        env = os.environ if environ is None else environ
        config = cls(
            threshold=float(env.get("KT_DSPARK_PLAIN_GATE_THRESHOLD", "2.2")),
            probe_every=int(env.get("KT_DSPARK_PLAIN_GATE_PROBE", "16")),
            alpha=float(env.get("KT_DSPARK_PLAIN_GATE_ALPHA", "0.25")),
            min_spec=int(env.get("KT_DSPARK_PLAIN_GATE_MIN_SPEC", "4")),
            force=env.get("KT_DSPARK_PLAIN_GATE_FORCE", "").strip().upper(),
            graph=env.get("KT_DSPARK_PLAIN_GATE_GRAPH", "1") == "1",
            log_every=int(env.get("KT_DSPARK_PLAIN_GATE_LOG", "0")),
            max_bs=int(env.get("KT_DSPARK_PLAIN_GATE_MAX_BS", "1")),
            signal=env.get("KT_DSPARK_PLAIN_GATE_SIGNAL", "accept").strip().lower(),
            conf=float(env.get("KT_DSPARK_PLAIN_GATE_CONF", "0.0")),
            draft_every=int(env.get("KT_DSPARK_PLAIN_GATE_DRAFT_EVERY", "8")),
            draft_alpha=float(env.get("KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA", "0.5")),
            draft_min=int(env.get("KT_DSPARK_PLAIN_GATE_DRAFT_MIN", "2")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.threshold > 0:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_THRESHOLD must be > 0, got {self.threshold}")
        if self.probe_every < 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_PROBE must be >= 1, got {self.probe_every}")
        if not 0 < self.alpha <= 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_ALPHA must be in (0, 1], got {self.alpha}")
        if self.min_spec < 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_MIN_SPEC must be >= 1, got {self.min_spec}")
        if set(self.force) - {SPEC, PLAIN}:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_FORCE takes S and P only, got {self.force!r}")
        if self.log_every < 0:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_LOG must be >= 0, got {self.log_every}")
        if not 1 <= self.max_bs <= 16:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_MAX_BS must be in [1, 16], got {self.max_bs}")
        if self.signal not in SIGNALS:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_SIGNAL must be one of {SIGNALS}, got {self.signal!r}")
        if not 0 <= self.conf <= 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_CONF must be in [0, 1], got {self.conf}")
        if self.draft_every < 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_DRAFT_EVERY must be >= 1, got {self.draft_every}")
        if not 0 < self.draft_alpha <= 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_DRAFT_ALPHA must be in (0, 1], got {self.draft_alpha}")
        if self.draft_min < 1:
            raise ValueError(f"KT_DSPARK_PLAIN_GATE_DRAFT_MIN must be >= 1, got {self.draft_min}")


def conf_bucket(conf: Optional[float]) -> int:
    """Calibration bucket of a confidence EMA: 0..CONF_BUCKETS-1 by tenths, -1 when the request had none yet."""
    if conf is None:
        return -1
    return min(max(int(conf * CONF_BUCKETS), 0), CONF_BUCKETS - 1)


def probe_bucket(ema: Optional[float]) -> int:
    """Calibration bucket of a probe EMA (tokens a step): 0.5-token buckets from 1.0, -1 for none."""
    if ema is None:
        return -1
    return min(max(int((ema - 1.0) / PROBE_BUCKET), 0), PROBE_BUCKETS - 1)


def probe_sample(origin_ids, output_ids, p: int, draft_tokens) -> Optional[int]:
    """Tokens a step a verify of this probe would have committed (greedy): 1 + the number of leading draft tokens equal
    to the committed tokens at positions p+1, p+2, ... (draft token k guesses position p+k; the bonus sits at p). None
    while undecided: decided at the first mismatch among the committed positions, or when all are committed."""
    n_origin = len(origin_ids)
    have = n_origin + len(output_ids)
    m = 0
    for k, token in enumerate(draft_tokens, start=1):
        at = p + k
        if at >= have:
            return None
        committed = origin_ids[at] if at < n_origin else output_ids[at - n_origin]
        if int(committed) != int(token):
            return 1 + m
        m += 1
    return 1 + m


def anchor_stride(num_rows: int, bs: int, verify_width: int) -> Optional[int]:
    """Rows between two requests' anchors in a step's logits: 1 for a plain step's (bs, vocab), the verify width for
    a verify's request-major (bs * width, vocab); None for any other shape (the step's signal is skipped)."""
    if bs <= 0:
        return None
    if num_rows == bs:
        return 1
    if verify_width > 1 and num_rows == bs * verify_width:
        return verify_width
    return None


def top1_probability(logits, temperatures=None):
    """Per row, the softmax top-1 probability at the sampling temperature: exp(max - logsumexp) of logits / T.
    Reads `logits` and `temperatures`; every result is a new tensor."""
    import torch

    x = logits.float()
    if temperatures is not None:
        x = x / temperatures.reshape(-1, 1).float().clamp_min(1e-6)
    return torch.exp(x.amax(dim=-1) - torch.logsumexp(x, dim=-1))


def refusal_reasons(
    *, verify_mode: str, attn_dp: bool, pp_size: int, simulate_acc_len: float, draft_is_moe: bool, target_type
) -> List[str]:
    """Why the plain gate cannot run on this server; empty when it can."""
    reasons = []
    if verify_mode != "static":
        reasons.append(f"needs SGLANG_RAGGED_VERIFY_MODE=static, got {verify_mode!r}")
    if attn_dp:
        reasons.append("attention DP (idle ranks would miss the plain step's collectives)")
    if pp_size > 1:
        reasons.append("pipeline parallelism")
    if simulate_acc_len > 0:
        reasons.append("SGLANG_SIMULATE_ACC_LEN")
    if not draft_is_moe:
        reasons.append("a dense draft (built and read for the DeepSeek-V4 MoE draft only)")
    if target_type != "deepseek_v41":
        reasons.append(f"target model type {target_type!r} (built for deepseek_v41)")
    return reasons


def step_eligible(
    *, bs: int, max_bs: int, has_grammar: bool, return_logprob: bool, return_hidden_states: bool, sampling_masks: bool
) -> bool:
    """A decode batch the plain step serves exactly as today's result path expects; others speculate."""
    return 0 < bs <= max_bs and not (has_grammar or return_logprob or return_hidden_states or sampling_masks)


def batch_kind(votes: List[str]) -> str:
    """The step a batch runs from its requests' votes: one request, its vote; two or more, plain when any votes plain
    (an ineligible batch's votes are all SPEC)."""
    return PLAIN if PLAIN in votes else SPEC


# ---------------------------------------------------------------------------------------------------------------------
# The gate (pure Python)


@dataclass
class _ReqState:
    last_ct: int
    last_drafts: int
    ema: Optional[float] = None
    samples: int = 0  # speculative steps folded into the EMA
    plain_run: int = 0  # plain steps since the last speculative step
    steps: int = 0  # decode steps decided for this request
    # (kind, confidence EMA when issued, probe vote, probe EMA at a draft switch) of each step whose result the counters
    # have not shown yet
    pending: deque = field(default_factory=lambda: deque(maxlen=_PENDING_MAX))
    conf_ema: Optional[float] = None  # EMA of the target's top-1 probability (r3)
    armed: bool = False  # the confidence EMA was below the probe level since the last speculative step
    probe_vote: bool = False  # the last vote was a confidence probe, or a draft-probe switch or its wait (r4)
    draft_ema: Optional[float] = None  # EMA of draft-probe samples, tokens a step (r4)
    draft_samples: int = 0  # draft-probe samples in draft_ema
    awaiting: bool = False  # switched to speculation by the draft probes; its first speculative result is not in yet


class PlainGate:
    """Per request: an EMA of tokens committed per speculative step; below the threshold, plain steps, with one
    speculative probe after every `probe_every` plain steps ('accept'), or when the confidence EMA rises to `conf`
    ('confidence'). A new request speculates.

    The inputs are the scheduler's CPU counters for the request: `verify_ct` counts every decode result it processed
    (speculative or plain), `correct_drafts` the accepted draft tokens. A plain result adds 1 to the first and 0 to
    the second. The gate remembers the kind of each step it issued, so the counters' advance since the last call is
    split into speculative and plain results without any device read. Under the overlap scheduler the counters lag
    the issued steps by two; that only delays the EMA. The confidence samples (note_confidence) arrive as late.

    Calibration: every speculative result is counted in the bucket of the confidence EMA its step was issued with,
    with the tokens it committed (calibration(), calibration_line()). In 'draft' mode the results of a switch's
    speculative step are also counted in the bucket of the probe EMA that made the switch (probe_calibration()).

    'draft' (r4): draft-probe samples (note_probe, as late as the probes settle) feed a per-request probe EMA; a plain
    request votes to speculate when it reaches the threshold after `draft_min` samples, and keeps voting so until the
    first result of that speculation is in (awaiting); then the acceptance EMA decides as usual. wants_probe() says
    which plain steps the controller precedes with a probe."""

    def __init__(self, config: GateConfig):
        self.config = config
        self._reqs: Dict[str, _ReqState] = {}
        self._calibration: Dict[int, List[float]] = {}  # bucket -> [speculative results, tokens committed]
        self._probe_calibration: Dict[int, List[float]] = {}  # probe EMA bucket -> [switch results, tokens committed]

    def __contains__(self, rid: str) -> bool:
        return rid in self._reqs

    def ema(self, rid: str) -> Optional[float]:
        state = self._reqs.get(rid)
        return None if state is None else state.ema

    def conf_ema(self, rid: str) -> Optional[float]:
        state = self._reqs.get(rid)
        return None if state is None else state.conf_ema

    def note_confidence(self, rid: str, prob: float) -> None:
        """A (lagged) top-1 probability of the target at one of the request's steps; unknown requests are ignored."""
        state = self._reqs.get(rid)
        if state is None:
            return
        alpha = self.config.alpha
        state.conf_ema = prob if state.conf_ema is None else state.conf_ema + alpha * (prob - state.conf_ema)
        if state.conf_ema < self.config.conf:
            state.armed = True

    def draft_ema(self, rid: str) -> Optional[float]:
        state = self._reqs.get(rid)
        return None if state is None else state.draft_ema

    def note_probe(self, rid: str, tokens: float) -> bool:
        """A settled draft-probe sample (tokens a step) of a request in plain mode. Ignored, False, for an unknown
        request or one speculating (its probes were issued before it switched)."""
        state = self._reqs.get(rid)
        if state is None or state.awaiting or state.plain_run == 0:
            return False
        alpha = self.config.draft_alpha
        state.draft_ema = tokens if state.draft_ema is None else state.draft_ema + alpha * (tokens - state.draft_ema)
        state.draft_samples += 1
        return True

    def wants_probe(self, rid: str) -> bool:
        """'draft' mode, after record(): this request's current step is a plain step that a draft probe precedes
        (every `draft_every`-th plain step since it last speculated)."""
        if self.config.signal != "draft":
            return False
        state = self._reqs.get(rid)
        return state is not None and state.plain_run > 0 and state.plain_run % self.config.draft_every == 0

    def probe_calibration(self) -> Dict[int, List[float]]:
        """probe EMA bucket at a draft switch -> [the switch steps' speculative results, mean tokens committed]."""
        return {b: [n, total / n] for b, (n, total) in sorted(self._probe_calibration.items()) if n}

    def probe_calibration_line(self) -> str:
        parts = []
        for b, (n, mean) in self.probe_calibration().items():
            lo = 1.0 + b * PROBE_BUCKET
            label = f"{lo:.1f}-{lo + PROBE_BUCKET:.1f}" if b < PROBE_BUCKETS - 1 else f"{lo:.1f}+"
            parts.append(f"{label} {int(n)}x{mean:.2f}")
        return ", ".join(parts) if parts else "no switches yet"

    def calibration(self) -> Dict[int, List[float]]:
        """bucket (-1: no confidence yet, else tenths) -> [speculative results, mean tokens committed]."""
        return {b: [n, total / n] for b, (n, total) in sorted(self._calibration.items()) if n}

    def calibration_line(self) -> str:
        parts = []
        for b, (n, mean) in self.calibration().items():
            label = "none" if b < 0 else f"{b / CONF_BUCKETS:.1f}-{(b + 1) / CONF_BUCKETS:.1f}"
            parts.append(f"{label} {int(n)}x{mean:.2f}")
        return ", ".join(parts) if parts else "no speculative results yet"

    def forget(self, rid: str) -> None:
        self._reqs.pop(rid, None)

    def prune(self, live_rids: Iterable[str]) -> None:
        """Drop requests that left the batch; one that comes back starts speculating again."""
        live = set(live_rids)
        for rid in [rid for rid in self._reqs if rid not in live]:
            del self._reqs[rid]

    def _absorb(self, rid: str, verify_ct: int, correct_drafts: int) -> _ReqState:
        state = self._reqs.get(rid)
        if state is None or verify_ct < state.last_ct or correct_drafts < state.last_drafts:
            # New request (or counters restarted): no history, speculate.
            state = _ReqState(last_ct=verify_ct, last_drafts=correct_drafts)
            self._reqs[rid] = state
            return state
        new_results = verify_ct - state.last_ct
        if new_results > 0:
            # Results arrive in issue order; a result the gate has no record of counts as speculative.
            issued = [state.pending.popleft() if state.pending else (SPEC, None, False, None) for _ in range(new_results)]
            spec = [(conf, probe, switch) for kind, conf, probe, switch in issued if kind == SPEC]
            num_spec = len(spec)
            if num_spec:
                tokens = 1.0 + (correct_drafts - state.last_drafts) / num_spec
                alpha = self.config.alpha
                for conf, probe, switch in spec:
                    # A probe re-measures the draft from scratch (the text changed); others blend in.
                    if probe or state.ema is None:
                        state.ema = tokens
                    else:
                        state.ema = state.ema + alpha * (tokens - state.ema)
                    cell = self._calibration.setdefault(conf_bucket(conf), [0, 0.0])
                    cell[0] += 1
                    cell[1] += tokens
                    if switch is not None:
                        cell = self._probe_calibration.setdefault(probe_bucket(switch), [0, 0.0])
                        cell[0] += 1
                        cell[1] += tokens
                    if probe:
                        state.awaiting = False  # a draft switch's speculation has its first result
                state.samples += num_spec
        state.last_ct, state.last_drafts = verify_ct, correct_drafts
        return state

    def vote(self, rid: str, *, verify_ct: int, correct_drafts: int, eligible: bool) -> str:
        """The kind of step this request wants now. Absorbs its counters; the step is not issued until record()."""
        config = self.config
        state = self._absorb(rid, verify_ct, correct_drafts)
        state.probe_vote = False
        if not eligible:
            return SPEC
        if config.force:
            return config.force[state.steps % len(config.force)]
        if state.samples < config.min_spec or state.ema is None or state.ema >= config.threshold:
            return SPEC
        if config.signal == "draft":
            # Speculate once the draft probes say it pays, and until that speculation's first result is in.
            if state.awaiting or (
                state.draft_samples >= config.draft_min
                and state.draft_ema is not None
                and state.draft_ema >= config.threshold
            ):
                state.probe_vote = True
                return SPEC
            return PLAIN
        if config.signal == "confidence":
            # Probe once when the target's confidence rises to the calibrated level after being below it.
            if config.conf > 0 and state.armed and state.conf_ema is not None and state.conf_ema >= config.conf:
                state.probe_vote = True
                return SPEC
            return PLAIN
        if state.plain_run >= config.probe_every:
            return SPEC  # probe: re-measure the draft
        return PLAIN

    def record(self, rid: str, kind: str) -> None:
        """The kind of step issued for this request (its batch's, which may differ from its vote)."""
        state = self._reqs[rid]
        probe = kind == SPEC and state.probe_vote
        switch = None
        if self.config.signal == "draft":
            if probe and not state.awaiting:
                # The draft switch: the probe EMA is consumed; the speculation runs until its first result is in.
                switch = state.draft_ema
                state.awaiting = True
                state.draft_ema, state.draft_samples = None, 0
            elif kind == PLAIN and state.plain_run == 0:
                state.draft_ema, state.draft_samples = None, 0  # back to plain: probes start afresh
        state.plain_run = state.plain_run + 1 if kind == PLAIN else 0
        state.steps += 1
        state.pending.append((kind, state.conf_ema, probe, switch))
        state.probe_vote = False
        if kind == SPEC:
            state.armed = False

    def decide(self, rid: str, *, verify_ct: int, correct_drafts: int, eligible: bool) -> bool:
        """True: run a plain step for this request now (one request: vote, then record it). Every decode step of a
        request must be voted and recorded, in order, so the gate can attribute its results."""
        kind = self.vote(rid, verify_ct=verify_ct, correct_drafts=correct_drafts, eligible=eligible)
        self.record(rid, kind)
        return kind == PLAIN


# ---------------------------------------------------------------------------------------------------------------------
# The confidence ring (r3): device values to the CPU, late, without a sync


class _TorchRingBackend:
    """Pinned host rows, non-blocking copies on the current stream, one CUDA event per row."""

    def __init__(self):
        import torch

        self.torch = torch

    def host(self, slots: int, width: int):
        return self.torch.zeros((slots, width), dtype=self.torch.float32, pin_memory=True)

    def event(self):
        return self.torch.cuda.Event()

    def copy(self, host, slot: int, values, n: int, event) -> None:
        host[slot, :n].copy_(values[:n].to(self.torch.float32), non_blocking=True)
        event.record()

    def read(self, host, slot: int, n: int) -> List[float]:
        return host[slot, :n].tolist()


class ConfidenceRing:
    """Per-step device vectors (one value a request) to the CPU two or more steps late. push() copies the vector into
    a free host row with a non-blocking copy and records the row's event; drain() returns the rows whose events have
    completed, oldest first, and stops at the first row still in flight: it never waits. With no free row (every row
    still in flight), or more requests than the row holds, a push is dropped and counted."""

    def __init__(self, slots: int = RING_SLOTS, width: int = RING_WIDTH, backend=None):
        self.backend = _TorchRingBackend() if backend is None else backend
        self.width = width
        self.host = self.backend.host(slots, width)
        self.events = [self.backend.event() for _ in range(slots)]
        self.rids: List[Optional[List[str]]] = [None] * slots
        self._free = deque(range(slots))
        self._inflight: deque = deque()
        self.pushed = 0
        self.dropped = 0

    def push(self, rids: List[str], values) -> bool:
        n = len(rids)
        if n == 0 or n > self.width or not self._free:
            self.dropped += 1
            return False
        slot = self._free.popleft()
        self.backend.copy(self.host, slot, values, n, self.events[slot])
        self.rids[slot] = list(rids)
        self._inflight.append(slot)
        self.pushed += 1
        return True

    def drain(self) -> List[tuple]:
        out = []
        while self._inflight and self.events[self._inflight[0]].query():
            slot = self._inflight.popleft()
            rids = self.rids[slot]
            out.extend(zip(rids, self.backend.read(self.host, slot, len(rids))))
            self.rids[slot] = None
            self._free.append(slot)
        return out


class _TorchProbeBackend:
    """Pinned int64 host rows, non-blocking copies on the current stream, one CUDA event per row."""

    def __init__(self):
        import torch

        self.torch = torch

    def host(self, slots: int, width: int):
        return self.torch.zeros((slots, width), dtype=self.torch.int64, pin_memory=True)

    def event(self):
        return self.torch.cuda.Event()

    def copy(self, host, slot: int, values, n: int, event) -> None:
        host[slot, :n].copy_(values.reshape(-1)[:n].to(self.torch.int64), non_blocking=True)
        event.record()

    def read(self, host, slot: int, n: int) -> List[int]:
        return host[slot, :n].tolist()


class ProbeRing:
    """Draft probes (r4) to the CPU without a sync: per probe, one row of `row` int64 values a request (the request's
    committed length on the device, then the draft tokens), pushed with a non-blocking copy and an event, read by
    drain() once the event has completed, oldest first, never waited for. Pushes that find every row in flight, or
    more requests than a row holds, are dropped and counted."""

    def __init__(self, row: int, slots: int = PROBE_SLOTS, width: int = RING_WIDTH, backend=None):
        self.backend = _TorchProbeBackend() if backend is None else backend
        self.row = row
        self.width = width
        self.host = self.backend.host(slots, width * row)
        self.events = [self.backend.event() for _ in range(slots)]
        self.rids: List[Optional[List[str]]] = [None] * slots
        self._free = deque(range(slots))
        self._inflight: deque = deque()
        self.pushed = 0
        self.dropped = 0

    def push(self, rids: List[str], values) -> bool:
        n = len(rids)
        if n == 0 or n > self.width or not self._free:
            self.dropped += 1
            return False
        slot = self._free.popleft()
        self.backend.copy(self.host, slot, values, n * self.row, self.events[slot])
        self.rids[slot] = list(rids)
        self._inflight.append(slot)
        self.pushed += 1
        return True

    def drain(self) -> List[tuple]:
        """(rid, committed length p, draft tokens) of each landed probe row."""
        out = []
        while self._inflight and self.events[self._inflight[0]].query():
            slot = self._inflight.popleft()
            rids = self.rids[slot]
            flat = self.backend.read(self.host, slot, len(rids) * self.row)
            for i, rid in enumerate(rids):
                row = flat[i * self.row:(i + 1) * self.row]
                out.append((rid, int(row[0]), [int(t) for t in row[1:]]))
            self.rids[slot] = None
            self._free.append(slot)
        return out


# ---------------------------------------------------------------------------------------------------------------------
# The plain DECODE graph runner


def make_plain_runner_cls(base_cls, *, decode_mode, full_mode, plain_bs=(1,)):
    """A decode graph runner that captures DECODE graphs at one token a request beside a speculative target's verify
    graphs. The base __init__ sets the capture mode to TARGET_VERIFY whenever speculation is on and the width to the
    verify width (decode_cuda_graph_runner.py, "capture mode + tokens-per-bs"); here those attributes are properties
    the base's assignments cannot change. The capture hidden mode is FULL (the DSpark aux hidden), the capture batch
    sizes are those in `plain_bs`, and a capture batch carries no spec_info (a plain decode batch has none)."""

    class PlainDecodeCudaGraphRunner(base_cls):
        @property
        def capture_forward_mode(self):
            return decode_mode

        @capture_forward_mode.setter
        def capture_forward_mode(self, value):
            pass

        @property
        def captured_req_width(self):
            return 1

        @captured_req_width.setter
        def captured_req_width(self, value):
            pass

        @property
        def capture_hidden_mode(self):
            return full_mode

        @capture_hidden_mode.setter
        def capture_hidden_mode(self, value):
            pass

        @property
        def capture_bs(self):
            return self.__dict__["_plain_capture_bs"]

        @capture_bs.setter
        def capture_bs(self, value):
            sizes = [bs for bs in value if bs in plain_bs]
            self.__dict__["_plain_capture_bs"] = sizes or [min(value)]

        def get_spec_info(self, num_tokens):
            return None

    return PlainDecodeCudaGraphRunner


def _keep_graph_state(max_bs: int, max_num_tokens: int) -> None:
    """Stands in for the backend's init_cuda_graph_state while the plain runner is built."""


@contextlib.contextmanager
def kept_graph_state(attn_backend):
    """Build a second decode runner on a backend that already serves captured graphs. DeepseekV4AttnBackend's
    init_cuda_graph_state replaces its per-bucket metadata dicts; the verify graphs read the metadata objects in
    those dicts, so a reset would leave them replaying stale metadata. DECODE and TARGET_VERIFY metadata live in
    separate buckets, so the plain graphs need no reset."""
    if not hasattr(attn_backend, "cuda_graph_metadata_of_bucket_and_bs"):
        raise RuntimeError(
            "KT_DSPARK_PLAIN_GATE: the target attention backend "
            f"{type(attn_backend).__name__} keeps no per-bucket graph metadata; only DeepseekV4AttnBackend is read"
        )
    if "init_cuda_graph_state" in vars(attn_backend):
        raise RuntimeError("KT_DSPARK_PLAIN_GATE: init_cuda_graph_state is already shadowed on the backend")
    attn_backend.init_cuda_graph_state = _keep_graph_state
    try:
        yield
    finally:
        del attn_backend.init_cuda_graph_state


@contextlib.contextmanager
def installed_runner(model_runner, runner):
    """The plain runner (or None: eager) as the target's decode graph runner for one forward. The target's own
    decode runner captures TARGET_VERIFY graphs only, and its can_run_graph accepts a DECODE batch (no spec_info, so
    no width check), so a plain forward must never reach it."""
    saved = model_runner.decode_cuda_graph_runner
    model_runner.decode_cuda_graph_runner = runner
    try:
        yield
    finally:
        model_runner.decode_cuda_graph_runner = saved


def _kt_capture_bs():
    try:
        from kt_kernel import KTMoEWrapper
    except ImportError:
        return None
    return list(KTMoEWrapper.get_capture_batch_sizes())


def _set_kt_capture_bs(sizes) -> None:
    if sizes is None:
        return
    from kt_kernel import KTMoEWrapper

    KTMoEWrapper.set_capture_batch_sizes(sizes)


# ---------------------------------------------------------------------------------------------------------------------
# The plain step


class PlainStepController:
    def __init__(self, *, worker, config: GateConfig):
        self.worker = worker
        self.config = config
        self.gate = PlainGate(config)
        self.runner = None  # PlainDecodeCudaGraphRunner; None: eager plain steps
        self._verify_runner = None  # the target's decode runner the plain graphs were captured beside
        self._prepared = False
        self.num_plain = 0
        self.num_spec = 0
        self.num_plain_graph = 0
        self._logged_first = False
        self.ring = None  # ConfidenceRing (r3), made with the plain graphs
        self.num_signal_skipped = 0  # steps whose logits had no anchor layout
        self.probe_ring = None  # ProbeRing (r4), made at the first probe (its row width is the draft's)
        self._landed: Dict[str, deque] = {}  # rid -> (p, draft tokens) of landed probes not yet settled
        self.num_probes = 0  # draft-only probes run
        self.num_probe_samples = 0  # probe samples settled and taken by the gate
        self.num_probe_skipped = 0  # probes not run: no host seq lens where the draft's backend needs them

    @classmethod
    def from_env(cls, *, worker, environ=None) -> "PlainStepController":
        config = GateConfig.from_env(environ)
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.speculative.dspark_components.dspark_verify import target_model_type

        parallel = get_parallel()
        reasons = refusal_reasons(
            verify_mode=worker._verify_planner.mode_value,
            attn_dp=parallel.attn_dp_enabled,
            pp_size=parallel.pp_size,
            simulate_acc_len=worker._simulate_acc_len,
            draft_is_moe=worker._draft_is_moe,
            target_type=target_model_type(worker.target_worker),
        )
        if reasons:
            raise RuntimeError("KT_DSPARK_PLAIN_GATE=1 refused: " + "; ".join(reasons))
        logger.info(
            "DSpark plain gate on: threshold %.2f tokens a step, probe every %d plain steps, alpha %.2f, "
            "min %d spec steps, force %r, graph %s, bs <= %d, signal %s, conf %.2f, draft probe every %d plain "
            "steps, alpha %.2f, min %d.",
            config.threshold,
            config.probe_every,
            config.alpha,
            config.min_spec,
            config.force,
            config.graph,
            config.max_bs,
            config.signal,
            config.conf,
            config.draft_every,
            config.draft_alpha,
            config.draft_min,
        )
        return cls(worker=worker, config=config)

    # -- startup ------------------------------------------------------------------------------------------------------

    def capture_plain_graph(self) -> None:
        """From DSparkWorkerV2.init_cuda_graphs, after the target's verify graphs and before the draft's."""
        worker = self.worker
        model_runner = worker.target_worker.model_runner
        if worker._need_mamba_verify_commit:
            raise RuntimeError("KT_DSPARK_PLAIN_GATE=1 refused: a mamba-state target (no plain-step state commit)")
        self._verify_runner = model_runner.decode_cuda_graph_runner
        self._prepared = True
        self.ring = ConfidenceRing()
        if not self.config.graph:
            logger.warning("DSpark plain gate: KT_DSPARK_PLAIN_GATE_GRAPH=0, plain steps run eagerly.")
            return
        if (
            not worker._decode_graph_allowed
            or self._verify_runner is None
            or self._verify_runner is getattr(model_runner, "eager_runner", None)
        ):
            logger.warning("DSpark plain gate: decode CUDA graphs are off; plain steps run eagerly.")
            return

        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
        from sglang.srt.runtime_context import get_observability
        from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

        if get_observability().forward_hooks:
            # Registered after the first capture; a later capture would trace them into the plain graph.
            raise RuntimeError("KT_DSPARK_PLAIN_GATE=1 refused: forward hooks are registered on the model")
        free_gb = worker._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM, worker.device, worker.gpu_id, group=worker._draft_graph_group
        )
        if free_gb < MIN_FREE_GB:
            raise RuntimeError(
                f"KT_DSPARK_PLAIN_GATE: {free_gb:.2f} GB free before the plain graph capture "
                f"(needs {MIN_FREE_GB}); set KT_DSPARK_PLAIN_GATE_GRAPH=0 for eager plain steps"
            )
        runner_cls = make_plain_runner_cls(
            DecodeCudaGraphRunner,
            decode_mode=ForwardMode.DECODE,
            full_mode=CaptureHiddenMode.FULL,
            plain_bs=tuple(range(1, self.config.max_bs + 1)),
        )
        kt_sizes = _kt_capture_bs()
        tic = time.perf_counter()
        try:
            with kept_graph_state(model_runner.attn_backend):
                runner = runner_cls(model_runner)
        finally:
            # The verify runner's sizes; the plain runner's constructor set its own.
            _set_kt_capture_bs(kt_sizes)
        after_gb = worker._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM, worker.device, worker.gpu_id, group=worker._draft_graph_group
        )
        self.runner = runner
        logger.info(
            "DSpark plain gate: captured plain DECODE graphs bs=%s in %.2f s, %.2f GB (free %.2f GB).",
            runner.capture_bs,
            time.perf_counter() - tic,
            free_gb - after_gb,
            after_gb,
        )

    # -- per step -----------------------------------------------------------------------------------------------------

    def forget(self, rid: str) -> None:
        self.gate.forget(rid)
        self._landed.pop(rid, None)

    def _collect_probes(self, reqs) -> None:
        """'draft': take landed probe rows off the ring and settle each against its request's committed tokens; a
        probe still undecided waits for more commits; probes of requests no longer known to the gate are dropped."""
        ring = self.probe_ring
        if ring is not None:
            for rid, p, tokens in ring.drain():
                self._landed.setdefault(rid, deque(maxlen=_LANDED_MAX)).append((p, tokens))
        for rid in [rid for rid in self._landed if rid not in self.gate]:
            del self._landed[rid]
        for req in reqs:
            landed = self._landed.get(req.rid)
            while landed:
                p, tokens = landed[0]
                sample = probe_sample(req.origin_input_ids, req.output_ids, p, tokens)
                if sample is None:
                    break
                landed.popleft()
                if self.gate.note_probe(req.rid, float(sample)):
                    self.num_probe_samples += 1

    def _draft_probe(self, batch) -> None:
        """'draft': the draft's proposal for the batch with no verify, as DSparkWorkerV2._forward_decode makes it
        before a verify: the full-width verify window (a pure plan), then the proposer under the draft context. Its
        tokens, with each request's committed length on the device, go to the probe ring; nothing is read back now."""
        import torch

        from sglang.srt.speculative.dspark_components.dspark_planner import alloc_verify_window

        worker = self.worker
        draft_input = batch.spec_info
        proposer = worker._proposer
        if batch.seq_lens_cpu is None and proposer.draft_model_runner.attn_backend.needs_cpu_seq_lens:
            # Where DraftBlockProposer._run_forward would raise: skip the probe, never fail the step.
            self.num_probe_skipped += 1
            return
        bs = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        anchor_ids = (
            proposer.draft_block_ids_view(bs=bs, device=prefix_lens.device) if worker._prepare_inputs_once else None
        )
        window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=worker.device,
            verify_num_draft_tokens=worker.verify_num_draft_tokens,
            block_pos_offsets=worker._block_pos_offsets,
            model_runner=worker.model_runner,
            seq_lens_cpu=(batch.seq_lens_cpu if batch.seq_lens_cpu is not None else draft_input.nxt_kv_lens_cpu),
            bonus=draft_input.bonus_tokens.view(-1) if anchor_ids is not None else None,
            anchor_ids=anchor_ids,
        )
        with worker._draft_context():
            proposal = proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=window,
                bs=bs,
                device=worker.device,
                target_model=worker.target_worker.model_runner.model,
                sampling_info=batch.sampling_info,
                anchor_prepared=anchor_ids is not None,
            )
        draft_tokens = proposal.draft_block.draft_tokens
        values = torch.cat([prefix_lens.view(bs, 1).to(torch.int64), draft_tokens.view(bs, -1).to(torch.int64)], dim=1)
        if self.probe_ring is None:
            self.probe_ring = ProbeRing(row=int(values.shape[1]))
        self.probe_ring.push([req.rid for req in batch.reqs], values)
        self.num_probes += 1

    def _eligible(self, batch) -> bool:
        sampling_info = batch.sampling_info
        draft_input = batch.spec_info
        bs = len(batch.reqs)
        return (
            step_eligible(
                bs=bs,
                max_bs=self.config.max_bs,
                has_grammar=bool(batch.has_grammar),
                return_logprob=bool(batch.return_logprob),
                return_hidden_states=bool(batch.return_hidden_states),
                sampling_masks=(
                    sampling_info is not None
                    and getattr(sampling_info, "sampling_mask_batch_indices", None) is not None
                ),
            )
            and len(batch.seq_lens) == bs
            and draft_input is not None
            and draft_input.bonus_tokens is not None
            and draft_input.bonus_tokens.numel() == bs
        )

    def maybe_plain_step(self, batch, on_publish):
        """From DSparkWorkerV2._forward_decode, before the draft: a plain step's result, or None to speculate."""
        if not self._prepared:
            raise RuntimeError("KT_DSPARK_PLAIN_GATE: a decode step before init_cuda_graphs prepared the gate")
        reqs = batch.reqs
        if self.ring is not None:
            for rid, prob in self.ring.drain():
                self.gate.note_confidence(rid, prob)
        self.gate.prune(req.rid for req in reqs)
        draft_mode = self.config.signal == "draft"
        if draft_mode:
            self._collect_probes(reqs)
        eligible = self._eligible(batch)
        votes = [
            self.gate.vote(
                req.rid,
                verify_ct=int(getattr(req, "spec_verify_ct", 0)),
                correct_drafts=int(getattr(req, "spec_num_correct_drafts", 0)),
                eligible=eligible,
            )
            for req in reqs
        ]
        kind = batch_kind(votes)
        plain = kind == PLAIN
        for req in reqs:
            self.gate.record(req.rid, kind)
        if not plain:
            self.num_spec += 1
            self._maybe_log(reqs)
            return None
        if draft_mode and any(self.gate.wants_probe(req.rid) for req in reqs):
            self._draft_probe(batch)
        result = self._run_plain(batch, on_publish)
        self.num_plain += 1
        self._maybe_log(reqs)
        return result

    def note_verify_logits(self, batch, next_token_logits) -> None:
        """From DSparkWorkerV2._forward_decode, right after the target verify and before any accept, grammar mask or
        sampling touches the logits: the confidence signal of the step (row 0 of each request's verify rows)."""
        self._note_logits(batch, next_token_logits, int(self.worker.verify_num_draft_tokens), sampled=False)

    def _note_logits(self, batch, logits, verify_width: int, *, sampled: bool) -> None:
        """Push each request's top-1 probability at the position the step commits first into the ring. Stream-ordered
        after the forward that wrote `logits`, so it reads them before the next forward reuses the buffer; it writes
        only new tensors and the ring's host row.

        `sampled`: the target worker has already sampled from this buffer (a plain step). Its Sampler.forward leaves an
        all-greedy batch's logits as they were (argmax) but turns any other batch's buffer into softmax(logits / T) in
        place (logits.div_(temperatures); logits[:] = softmax), so the row maximum is then the probability itself."""
        ring = self.ring
        if ring is None or logits is None:
            return
        import torch

        if torch.cuda.is_current_stream_capturing():
            return
        reqs = batch.reqs
        bs = len(reqs)
        stride = anchor_stride(int(logits.shape[0]), bs, verify_width)
        if stride is None:
            self.num_signal_skipped += 1
            return
        rows = logits[::stride] if stride > 1 else logits
        sampling_info = batch.sampling_info
        if sampled and sampling_info is not None and not sampling_info.is_all_greedy:
            probs = rows.float().amax(dim=-1)
        else:
            temperatures = getattr(sampling_info, "temperatures", None) if sampling_info is not None else None
            if temperatures is not None and int(temperatures.shape[0]) != bs:
                temperatures = None
            probs = top1_probability(rows, None if sampled else temperatures)
        ring.push([req.rid for req in reqs], probs)

    def _maybe_log(self, reqs) -> None:
        if not self._logged_first and self.num_plain:
            self._logged_first = True
            logger.info(
                "DSpark plain gate: first plain step (%s).", "CUDA graph" if self.runner is not None else "eager"
            )
        every = self.config.log_every
        steps = self.num_plain + self.num_spec
        if every and steps % every == 0:
            emas = [self.gate.ema(req.rid) for req in reqs]
            confs = [self.gate.conf_ema(req.rid) for req in reqs]
            logger.info(
                "DSpark plain gate: %d plain (%d graph) and %d speculative steps; EMA %s; confidence %s.",
                self.num_plain,
                self.num_plain_graph,
                self.num_spec,
                ", ".join("-" if e is None else f"{e:.2f}" for e in emas),
                ", ".join("-" if c is None else f"{c:.3f}" for c in confs),
            )
            ring = self.ring
            logger.info(
                "DSpark plain gate calibration (confidence EMA at issue: speculative steps x tokens a step): %s; "
                "signal %d pushed, %d dropped, %d skipped.",
                self.gate.calibration_line(),
                0 if ring is None else ring.pushed,
                0 if ring is None else ring.dropped,
                self.num_signal_skipped,
            )
            if self.config.signal == "draft":
                probes = self.probe_ring
                drafts = [self.gate.draft_ema(req.rid) for req in reqs]
                logger.info(
                    "DSpark plain gate draft probes: %d run, %d settled, %d skipped, %d dropped; probe EMA %s; "
                    "calibration (probe EMA at switch: speculative steps x tokens a step): %s.",
                    self.num_probes,
                    self.num_probe_samples,
                    self.num_probe_skipped,
                    0 if probes is None else probes.dropped,
                    ", ".join("-" if d is None else f"{d:.2f}" for d in drafts),
                    self.gate.probe_calibration_line(),
                )

    def _run_plain(self, batch, on_publish):
        import copy

        import torch

        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch, ForwardMode
        from sglang.srt.speculative.dspark_components.dspark_draft import make_next_draft_input
        from sglang.srt.speculative.dspark_components.dspark_planner import alloc_verify_window
        from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite

        worker = self.worker
        target_worker = worker.target_worker
        model_runner = target_worker.model_runner
        if model_runner.decode_cuda_graph_runner is not self._verify_runner:
            raise RuntimeError(
                "KT_DSPARK_PLAIN_GATE: the target's decode graphs changed after the plain graphs were captured"
            )
        draft_input = batch.spec_info
        bs = len(batch.seq_lens)
        device = worker.device
        # The draft KV this step writes may land where a staged verify commit writes (as before a prefill).
        commit = worker._deferred_commit()
        if commit is not None:
            commit.flush("plain")

        # One-column window at the committed length: the bonus token's slot, from the same reservation verify uses.
        prefix_lens = batch.seq_lens
        window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=1,
            block_pos_offsets=worker._block_pos_offsets[:1],
            model_runner=worker.model_runner,
            seq_lens_cpu=(batch.seq_lens_cpu if batch.seq_lens_cpu is not None else draft_input.nxt_kv_lens_cpu),
        )

        # The decode batch plain decode would run (ScheduleBatch.prepare_for_decode): input = the last sampled token,
        # seq lens counting it, one write slot. A copy, so the scheduler's batch keeps its spec-v2 fields.
        view = copy.copy(batch)
        view.forward_mode = ForwardMode.DECODE
        view.input_ids = draft_input.bonus_tokens.view(-1)
        view.seq_lens = prefix_lens + 1
        view.seq_lens_cpu = None if batch.seq_lens_cpu is None else batch.seq_lens_cpu + 1
        view.seq_lens_sum = None if batch.seq_lens_sum is None else int(batch.seq_lens_sum) + bs
        view.out_cache_loc = window.verify_cache_loc
        view.spec_info = None
        forward_batch = ForwardBatch.init_new(
            view,
            model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
            iteration_ids=window.iteration_ids,
        )
        with installed_runner(model_runner, self.runner):
            out = target_worker.forward_batch_generation(batch=None, forward_batch=forward_batch)
        logits_output = out.logits_output
        # r3: the confidence signal of this step (reads the buffer the sampler left).
        self._note_logits(batch, logits_output.next_token_logits, 1, sampled=True)
        next_token_ids = worker._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, out.next_token_ids).view(-1)
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("KT_DSPARK_PLAIN_GATE: the plain DECODE forward returned no DSpark aux hidden")

        new_seq_lens = prefix_lens + 1
        if on_publish is not None:
            on_publish(new_seq_lens)

        # Draft KV of the one committed position, as verify's commit_hidden writes committed positions.
        commit_lens = torch.ones((bs,), dtype=torch.int32, device=device)
        injector = worker._kv_injector
        state_slot = None
        if injector.swa_ring_size is not None:
            state_slot = batch.req_pool_indices[:bs].view(-1, 1).expand(bs, 1).reshape(-1)
        cache_loc = injector.ids_for(window.iteration_ids)
        injector.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(bs, 1),
            positions=window.positions_2d.reshape(-1),
            commit_lens=commit_lens,
            state_slot=state_slot,
        )
        logits_output.hidden_states = None

        # The scheduler reads accept_lens tokens from each row of the verify-width output.
        stride = int(worker.verify_num_draft_tokens)
        out_tokens = torch.zeros((bs, stride), dtype=torch.int64, device=device)
        out_tokens[:, 0] = next_token_ids.to(torch.int64)
        next_draft_input = make_next_draft_input(bonus_tokens=next_token_ids, new_seq_lens=new_seq_lens)
        patha_static_flags = None
        if os.environ.get("SGLANG_FP8_PATHA_EXPERT_STATIC", "0") == "1":
            from sglang.srt.layers.moe.contract_v2.patha_static_runtime import snapshot_after_forward

            patha_static_flags = snapshot_after_forward()
        if out.can_run_cuda_graph and self.runner is not None:
            self.num_plain_graph += 1
        return GenerationBatchResult(
            patha_static_flags=patha_static_flags,
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            block_accept_lens=commit_lens,
            cap_lens=None,
            can_run_cuda_graph=out.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=stride,
            new_seq_lens=new_seq_lens,
        )
