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
otherwise. Only bs = 1 takes plain steps; larger batches speculate as before.

The gate (PlainGate) is pure Python: an EMA, per request, of tokens committed per speculative step, read from the
scheduler's own CPU counters (Req.spec_verify_ct, Req.spec_num_correct_drafts), so it never syncs the device.
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
    pending: deque = field(default_factory=lambda: deque(maxlen=_PENDING_MAX))


class PlainGate:
    """Per request: an EMA of tokens committed per speculative step; below the threshold, plain steps, with one
    speculative probe after every `probe_every` plain steps. A new request speculates.

    The inputs are the scheduler's CPU counters for the request: `verify_ct` counts every decode result it processed
    (speculative or plain), `correct_drafts` the accepted draft tokens. A plain result adds 1 to the first and 0 to
    the second. The gate remembers the kind of each step it issued, so the counters' advance since the last call is
    split into speculative and plain results without any device read. Under the overlap scheduler the counters lag
    the issued steps by two; that only delays the EMA."""

    def __init__(self, config: GateConfig):
        self.config = config
        self._reqs: Dict[str, _ReqState] = {}

    def __contains__(self, rid: str) -> bool:
        return rid in self._reqs

    def ema(self, rid: str) -> Optional[float]:
        state = self._reqs.get(rid)
        return None if state is None else state.ema

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
            kinds = [state.pending.popleft() if state.pending else SPEC for _ in range(new_results)]
            num_spec = kinds.count(SPEC)
            if num_spec:
                tokens = 1.0 + (correct_drafts - state.last_drafts) / num_spec
                alpha = self.config.alpha
                for _ in range(num_spec):
                    state.ema = tokens if state.ema is None else state.ema + alpha * (tokens - state.ema)
                state.samples += num_spec
        state.last_ct, state.last_drafts = verify_ct, correct_drafts
        return state

    def decide(self, rid: str, *, verify_ct: int, correct_drafts: int, eligible: bool) -> bool:
        """True: run a plain step for this request now. Every decode step of a request must be decided here, in
        order, so the gate can attribute its results."""
        config = self.config
        state = self._absorb(rid, verify_ct, correct_drafts)
        if not eligible:
            kind = SPEC
        elif config.force:
            kind = config.force[state.steps % len(config.force)]
        elif state.samples < config.min_spec or state.ema is None or state.ema >= config.threshold:
            kind = SPEC
        elif state.plain_run >= config.probe_every:
            kind = SPEC  # probe: re-measure the draft
        else:
            kind = PLAIN
        state.plain_run = state.plain_run + 1 if kind == PLAIN else 0
        state.steps += 1
        state.pending.append(kind)
        return kind == PLAIN


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
            "min %d spec steps, force %r, graph %s, bs <= %d.",
            config.threshold,
            config.probe_every,
            config.alpha,
            config.min_spec,
            config.force,
            config.graph,
            config.max_bs,
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
        self.gate.prune(req.rid for req in reqs)
        eligible = self._eligible(batch)
        plain = False
        for req in reqs:
            plain |= self.gate.decide(
                req.rid,
                verify_ct=int(getattr(req, "spec_verify_ct", 0)),
                correct_drafts=int(getattr(req, "spec_num_correct_drafts", 0)),
                eligible=eligible,
            )
        if not plain:
            self.num_spec += 1
            self._maybe_log(reqs)
            return None
        result = self._run_plain(batch, on_publish)
        self.num_plain += 1
        self._maybe_log(reqs)
        return result

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
            logger.info(
                "DSpark plain gate: %d plain (%d graph) and %d speculative steps; EMA %s.",
                self.num_plain,
                self.num_plain_graph,
                self.num_spec,
                ", ".join("-" if e is None else f"{e:.2f}" for e in emas),
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
