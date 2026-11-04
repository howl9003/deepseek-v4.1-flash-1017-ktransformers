# DSpark shadow lane: design r1

Claude, 4 Nov 2026. Owner: "the lane is probably the arch we need in the end so build it". Selectors `KT_SHADOW_LANE=1`,
`KT_SHADOW_LANE_FROM=40`, default off.

## Goal

C2's plain decode stays the default path. A shadow DSpark draft runs on resources the main path leaves idle and
measures what speculation would gain. The gate switches to speculation only while it pays. If prediction is not
profitable, the shadow must not slow the main path; if it is, the shadow's cost must be small enough to keep the gain.

## What is idle (decode step about 28-30 ms)

| resource | idle | source |
|---|---|---|
| GPU compute | 2.8-7 ms a step, waiting for CPU experts | measured, 2 Nov decode Nsight (adaynsys, 32.98 build): 0.18-0.45 s idle per 64 steps |
| CPU expert workers and their memory bandwidth | about half the step: between layer windows and in the 2.2 ms head | estimate from the 2 Nov timeline (serial GPU 13.3, window GPU 7.4, CPU wait 6.9, head 2.2 ms) |
| PCIe | about 90% | estimate: hot-slot swaps move 45-75 MB a token |
| card memory | none | every spare MB becomes a hot slot; but 81 slots bought about 0.5% (adaswm), so slots are cheap |

The CPU window is the bottleneck (a ms of CPU lead is about a ms of token), so the shadow must not take CPU time inside
a window. Every expert is split into six parts, one per node, so there is no "cold node" to give the shadow; the idle
time is the same on every node and lies outside the windows.

## Part 1: the native lane (built here)

`apply_shadow_lane.py` on the DSpark tolerant tree (kt_ext 57e25283):

- **Lane CPUInfer** `CPUInfer(main, True)`: shares the main WorkerPool, own TaskQueue (thread marked
  `kt_lane::t_lane`), own memop poller. A task submitted through it (`submit_with_cuda_stream`, memops) runs on the
  lane queue thread, so the draft's forward orchestration never touches the main queue or the main memop slots, and no
  CUDA host function can hold up the main stream.
- **Routing**: on a lane thread, `NumaJobDistributor::do_numa_job` goes to a `LaneDistributor` (one sleeping leader per
  part) and `InNumaPool::do_work_stealing_job` posts the job to a second slot (`LaneJob`) and sleeps until done.
- **Execution**: pool workers and part leaders spin between jobs during decode (they sleep only after 50 ms idle).
  From that spin loop they run one lane item at a time, only while `kt_lane::main_active == 0` (no main TaskQueue task
  running). A main task starting during an item waits for that item at most (one expert slice, tens of us). If no
  item is taken for 1 ms while main is idle (threads asleep), the lane leader runs items itself.
- **Isolation** (from the shared-state audit, 4 Nov). The process-wide state is: the part scratch shared by every object
  of a part, the static route scratch, and the truncation counters. Objects at or above `KT_SHADOW_LANE_FROM` take their
  part scratch from `shared_mem_buffer_numa_lane` and their route scratch from a lane array. Truncation events on lane
  threads or inside lane items go to `g_lane_truncation_events`. So an overlapping lane item can change neither main's
  bits nor main's fail-closed check. Lane items are leaf calls: the single-job members of the pool and distributor, and
  `thread_local_id`, stay main's. Per-thread state is clean (audit: every thread_local scratch is written before read;
  MXCSR is saved and restored per scope; the only init is the empty `T::config()`).
- **Off**: no lane object and no LaneDistributor; slots stay empty. Main path cost: one relaxed load per spin pass, two
  atomic adds per main task, one thread_local read per posted job or truncation event.

`test_lane.cpp` (CPU, two cores, unbound threads) checks main outputs bit-equal with the lane idle and busy, lane
outputs correct, lane items not starting under a main task (one-item race allowed, overlap above 0.5% fails), and
exception propagation. `build_shadow_lane_r1.py` builds the test and the extension on fea01.

## Part 2: the shadow draft (next)

- **Draft objects**: the three DSpark stages' KT MoE layers (KT ids 40-42) are lane objects (`KT_SHADOW_LANE_FROM=40`).
  In spec mode they run on the main CPUInfer as today; in plain mode the shadow runs them through the lane CPUInfer.
  The two never overlap: a speculative step waits for the shadow's done event.
- **GPU side**: one CUDA graph of the draft forward (bs 1, block 3), captured with the draft wrappers pointed at the
  lane CPUInfer and its own pinned handoff buffers, replayed on a low-priority stream after every Nth plain step
  (start N = 4). Inputs come from the plain step's captured aux hidden and token by device copies on that stream after
  an event; the draft's context KV is already kept by the plain step's injection.
- **Signal**: draft tokens d1..d3 against the tokens plain decode then commits: m = leading matches, tokens a step
  estimate 1 + m. For greedy requests m is the acceptance verify would have; at T=1 it is a proxy (calibrate against
  speculative steps' own acceptance). Accumulated on the GPU, read lagged through the gate r3 pinned ring; the gate
  speculates when the shadow EMA reaches the threshold (2.2), instead of timer probes.
- **Cost per sample**: about one draft step, 3.3 ms, mostly GPU memory traffic (head over 3 rows, dense draft weights)
  plus 0.17-0.5 GB of CPU expert reads through the lane. On the low-priority stream it fills the GPU's idle gaps; what
  overlaps main kernels is the cost, bounded by 1/N of a draft step.

## Tests on the card

1. Identity: lane on and shadow running, exact vs C2 (no-logprob token check of adadpb plus the C2 identity client).
2. Plain-step speed with the shadow running against C2 (target: within 1%).
3. Timing: breadth, prose, code against C2 and static DSpark; decode Nsight for overlap.

## Risks

- GPU-side overlap: a low-priority stream does not preempt running kernels. Mitigated by N and small kernels.
- The draft's head over three rows is the largest GPU cost; if it overlaps main, raise N.
- `main_active` covers main TaskQueue tasks only. Pool jobs posted outside a TaskQueue task are not guarded; isolation
  keeps their bits, but a lane item could add latency to them.
