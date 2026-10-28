"""Opt-in CPU placement for the KT scheduler's service (non-expert) threads.

KT_SERVICE_CPUS=<list>, e.g. "a-b,c": once the Scheduler is built, every thread of this
process that is not a kt-kernel expert worker gets exactly these CPUs; expert workers
are never written. Unset: nothing happens. Independent of KT_GPU_GLUE.

  plan_service_affinity  before the model load: parse, refuse unsupported setups
  apply_service_affinity after the load: validate the expert topology, write, verify,
                         roll back on any failure, raise

Expert workers are recognised by the names kt-kernel's worker_pool.cpp gives them with
pthread_setname_np: numa_<node>_t_<i> (InNumaPool) and numa_<node>_m_<start>
(NumaJobDistributor), read from /proc/self/task/<tid>/stat. A thread's identity is
(TID, start time in clock ticks). Memory policy is never read or changed.
"""

import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

ENV = "KT_SERVICE_CPUS"
EXPERT_COMM = re.compile(r"numa_([0-9]+)_([tm])_([0-9]+)")


class ServiceAffinityError(RuntimeError):
    pass


def parse_cpu_list(text: str) -> FrozenSet[int]:
    """'0-3,8' -> {0,1,2,3,8}; anything else (empty, reversed, signs) is an error."""
    cpus = set()
    for part in text.split(","):
        m = re.fullmatch(r"\s*([0-9]+)(?:-([0-9]+))?\s*", part)
        if m is None:
            raise ServiceAffinityError(f"{ENV}={text!r}: bad entry {part!r}")
        first = int(m[1])
        last = int(m[2]) if m[2] is not None else first
        if last < first:
            raise ServiceAffinityError(f"{ENV}={text!r}: reversed range {part!r}")
        cpus.update(range(first, last + 1))
    return frozenset(cpus)


def fmt(cpus) -> str:
    """Compact CPU list such as 0-6,39."""
    cpus, parts = sorted(cpus), []
    while cpus:
        end = 1
        while end < len(cpus) and cpus[end] == cpus[end - 1] + 1:
            end += 1
        parts.append(str(cpus[0]) if end == 1 else f"{cpus[0]}-{cpus[end - 1]}")
        cpus = cpus[end:]
    return ",".join(parts)


@dataclass(frozen=True)
class Plan:
    service_cpus: FrozenSet[int]
    launch_cpus: FrozenSet[int]  # the process mask before the NUMA/GPU binding
    expert_threads: int  # --kt-cpuinfer: one named thread per CPU worker
    expert_pools: int  # --kt-threadpool-count: one numa_<n>_m_<i> thread per pool


def plan_service_affinity(
    server_args,
    parallel,
    *,
    launch_mask_known: bool,
    environ=None,
    getaffinity=None,
    platform: Optional[str] = None,
) -> Optional[Plan]:
    """None when KT_SERVICE_CPUS is unset (nothing else is read). Must run before
    configure_scheduler_process binds the process to its NUMA node."""
    text = (os.environ if environ is None else environ).get(ENV)
    if text is None:
        return None
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        raise ServiceAffinityError(
            f"{ENV} is set, but per-thread placement needs Linux /proc and "
            f"sched_setaffinity (platform {platform})"
        )
    ranks = {
        name: getattr(parallel, name, None)
        for name in ("tp_size", "pp_size", "num_dp_ranks")
    }
    if any(v != 1 for v in ranks.values()):
        raise ServiceAffinityError(
            f"{ENV} is set, but there is no per-rank CPU contract for more than one "
            f"scheduler process ({ranks})"
        )
    if not launch_mask_known:
        raise ServiceAffinityError(
            f"{ENV} is set, but the launcher's CPU mask is not visible: NUMA bind v2 "
            f"binds before Python starts (set SGLANG_NUMA_BIND_V2=0 or unset {ENV})"
        )
    service = parse_cpu_list(text)
    launch = frozenset((os.sched_getaffinity if getaffinity is None else getaffinity)(0))
    if not service <= launch:
        raise ServiceAffinityError(
            f"{ENV}={text}: CPUs {fmt(service - launch)} are outside the launcher's "
            f"CPU mask {fmt(launch)}"
        )
    experts = getattr(server_args, "kt_cpuinfer", None)
    pools = getattr(server_args, "kt_threadpool_count", None)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (experts, pools)) or not (
        0 < pools <= experts
    ):
        raise ServiceAffinityError(
            f"{ENV} needs kt-kernel CPU experts (kt_cpuinfer={experts}, "
            f"kt_threadpool_count={pools})"
        )
    logger.info(f"{ENV}: service CPUs {fmt(service)} within launcher mask {fmt(launch)}")
    return Plan(service, launch, experts, pools)


# ------------------------------------------------------------------ the OS boundary


class ProcThreads:
    """This process's threads: /proc/self/task and sched_{get,set}affinity on TIDs.

    A TID reused by another process is not under /proc/self/task, so identity() raises
    FileNotFoundError for it instead of describing a foreign thread."""

    def tids(self) -> List[int]:
        return sorted(int(t) for t in os.listdir("/proc/self/task"))

    def identity(self, tid: int):
        """(start time in clock ticks, comm)."""
        with open(f"/proc/self/task/{tid}/stat") as f:
            text = f.read()
        return int(text.rsplit(")", 1)[1].split()[19]), text[text.index("(") + 1 : text.rindex(")")]

    def get(self, tid: int) -> FrozenSet[int]:
        return frozenset(os.sched_getaffinity(tid))

    def set(self, tid: int, cpus) -> None:
        os.sched_setaffinity(tid, cpus)


@dataclass(frozen=True)
class Thread:
    tid: int
    start: int
    comm: str
    cpus: FrozenSet[int]

    @property
    def expert(self) -> bool:
        return EXPERT_COMM.fullmatch(self.comm) is not None


_GONE = (FileNotFoundError, ProcessLookupError)


def _read(threads) -> Optional[Dict[int, Thread]]:
    rows = {}
    for tid in threads.tids():
        try:
            start, comm = threads.identity(tid)
            rows[tid] = Thread(tid, start, comm, frozenset(threads.get(tid)))
        except _GONE:
            return None
    return rows


def _stable(threads, tries: int = 5) -> Dict[int, Thread]:
    """Two identical consecutive reads, so a pool half-way through creation is not used."""
    for _ in range(tries):
        first, second = _read(threads), _read(threads)
        if first is not None and first == second:
            return first
        time.sleep(0.05)
    raise ServiceAffinityError(f"the thread set was not stable across {tries} double reads")


def _expert_problems(rows: Dict[int, Thread], plan: Plan) -> List[str]:
    experts = [t for t in rows.values() if t.expert]
    pools = sum(EXPERT_COMM.fullmatch(t.comm)[2] == "m" for t in experts)
    cpus = [c for t in experts for c in t.cpus]
    problems = []
    if len(experts) != plan.expert_threads:
        problems.append(f"{len(experts)} expert threads, expected {plan.expert_threads} (kt_cpuinfer)")
    if pools != plan.expert_pools:
        problems.append(f"{pools} numa_*_m_* threads, expected {plan.expert_pools} (kt_threadpool_count)")
    problems += [f"expert {t.comm} (tid {t.tid}) mask {fmt(t.cpus)} is not one CPU" for t in experts if len(t.cpus) != 1]
    if len(set(cpus)) != len(cpus):
        problems.append("two expert threads share a CPU")
    if plan.service_cpus & set(cpus):
        problems.append(f"service CPUs {fmt(plan.service_cpus & set(cpus))} are expert CPUs")
    if len(experts) == len(rows):
        problems.append("no service threads")
    return problems


def _final_problems(before: Dict[int, Thread], after: Dict[int, Thread], plan: Plan) -> List[str]:
    """Experts exactly as before (identity, name, mask); every other thread now on the
    service CPUs. Service threads that exited are fine; new ones must have inherited the
    service mask from a moved thread."""
    experts_before = {t for t in before.values() if t.expert}
    experts_after = {t for t in after.values() if t.expert}
    problems = []
    if experts_before != experts_after:
        problems.append(
            "the native expert workers changed: "
            f"gone {sorted(t.comm for t in experts_before - experts_after)}, "
            f"new or rebound {sorted(t.comm for t in experts_after - experts_before)}"
        )
    problems += [
        f"service thread {t.tid} ({t.comm}) mask {fmt(t.cpus)}"
        for t in after.values()
        if not t.expert and t.cpus != plan.service_cpus
    ]
    return problems


def _rollback(threads, changed: List[Thread]) -> List[str]:
    """Each changed thread back to its own mask if it is still the same thread."""
    failed = []
    for t in changed:
        try:
            start, comm = threads.identity(t.tid)
            if start == t.start and not EXPERT_COMM.fullmatch(comm):
                threads.set(t.tid, t.cpus)
        except _GONE:
            pass  # exited: nothing to restore
        except Exception as exc:
            failed.append(f"{t.tid} ({t.comm}): {type(exc).__name__}: {exc}")
    return failed


def apply_service_affinity(plan: Plan, threads=None) -> dict:
    """Move this process's non-expert threads to plan.service_cpus. Raises
    ServiceAffinityError, after attempting to restore changed threads, if anything
    is unexpected. Any restoration failure is included in the error."""
    threads = ProcThreads() if threads is None else threads
    before = _stable(threads)
    problems = _expert_problems(before, plan)
    if problems:
        raise ServiceAffinityError(f"{ENV}: refusing, nothing changed: " + "; ".join(problems))
    changed = []
    try:
        for t in before.values():
            if t.expert:
                continue
            try:
                start, comm = threads.identity(t.tid)
            except _GONE:
                continue  # exited; the final check sees whatever replaced it
            if start != t.start or EXPERT_COMM.fullmatch(comm):
                raise ServiceAffinityError(f"thread {t.tid} ({t.comm}) changed identity before its write")
            if t.cpus != plan.service_cpus:
                try:
                    threads.set(t.tid, plan.service_cpus)
                except _GONE:
                    continue
                changed.append(t)
        after = _stable(threads)
        problems = _final_problems(before, after, plan)
        if problems:
            raise ServiceAffinityError("after the writes: " + "; ".join(problems))
    except BaseException as exc:
        failed = _rollback(threads, changed)
        if not isinstance(exc, Exception):
            raise
        note = f"; rolled back {len(changed)} changed threads"
        if failed:
            note += f", NOT restored: {failed}"
        raise ServiceAffinityError(f"{ENV}: {exc}{note}") from exc
    experts = [t for t in after.values() if t.expert]
    report = dict(
        service_cpus=fmt(plan.service_cpus),
        service_threads=len(after) - len(experts),
        changed=len(changed),
        expert_threads=len(experts),
        expert_cpus=fmt(c for t in experts for c in t.cpus),
    )
    logger.info(
        f"{ENV}: {report['service_threads']} service threads on {report['service_cpus']} "
        f"({report['changed']} moved); {report['expert_threads']} expert threads "
        f"unchanged on {report['expert_cpus']}"
    )
    return report
