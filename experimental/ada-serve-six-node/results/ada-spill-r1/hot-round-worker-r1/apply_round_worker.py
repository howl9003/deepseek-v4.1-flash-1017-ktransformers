"""Hot-slot rounds on a worker thread (Claude, 31 Oct 2026), on tree r6's kt_kv_hot_lend.py. argv: base file, output file.

A dynamic round enqueues about 150 launches an expert on the scheduler's thread, and the tree r6 traces show it as a 25-33 ms step every eighth token.
Draft #8 carried the fix (22 Oct, ports/hot_slots) but never reached the served tree, so its KT_KV_HOT_DYN_DEFER and _ASYNC settings in the accepted
environment are read by nothing. This is the same job design on tree r6, behind a new selector so that those settings stay inert. KT_KV_HOT_DYN_WORKER=0
leaves every line of the old path as it was. No arithmetic: the tables say which experts the GPU takes, and under the integer contract either side gives the same result."""
import hashlib
import sys
from pathlib import Path

BASE_SHA = '09be733debc39f7597f83740a6d485963482f46ff44b01b6fea829dd3a9b2a55'

FLAG = '''DYN_LOG = int(os.environ.get("KT_KV_HOT_DYN_LOG", "512") or 512)  # rounds between log lines
# KT_KV_HOT_DYN_WORKER=1: a round's copies are enqueued on a worker thread and enter the tables at the first
# forward after they are done, so a round takes almost nothing from the scheduler's thread. GPU TP=1 only:
# ranks querying their own events could let rank 0's CPU skip routes another rank still lacks.
DYN_WORKER = os.environ.get("KT_KV_HOT_DYN_WORKER", "0") == "1"
'''

JOBS = '''# -- a round's copies as a job (KT_KV_HOT_DYN_WORKER) ----------------------------------
# items (layer, 0: slot / 1: resident row, its index, expert) are copied on DYN["stream"] after ev_unmap
# (recorded on the decode stream once the places' old experts left the tables), then ev is recorded;
# pub lists the table entries (layer, expert, place) that go in once ev is done.


def _job_start(items: list, pub: dict, dev) -> None:
    job = {"items": items, "pub": pub, "enq": False, "thread": None, "error": None,
           "ev_unmap": torch.cuda.Event(), "ev": torch.cuda.Event(), "t0": time.perf_counter(), "t_enq": 0.0}
    job["ev_unmap"].record(torch.cuda.current_stream(dev))
    side = DYN["stream"]
    # grad and inference mode are per thread: the worker runs as the scheduler's thread would
    grad, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()

    def run():
        t0 = time.perf_counter()
        try:
            with torch.inference_mode(inference), torch.set_grad_enabled(grad), torch.cuda.device(dev), torch.cuda.stream(side):
                side.wait_event(job["ev_unmap"])
                for L, w, i, e in job["items"]:
                    if w == 0:
                        dst = {key: L.slots[nm][i] for key, nm in DYN["roles"][L.idx].items()}
                    else:
                        dst = {key: t[i] for key, t in L.res_t.items()}
                    _arena_load(L, e, dst, DYN["order"])
                job["ev"].record(side)
        except BaseException as ex:  # raised again on the scheduler's thread (_job_check)
            job["error"] = ex
        finally:
            job["t_enq"] = time.perf_counter() - t0
            job["enq"] = True  # after the record: an unrecorded event reads as done

    DYN["job"] = job
    job["thread"] = threading.Thread(target=run, name="kt-kv-hot-copy", daemon=True)
    job["thread"].start()


def _job_check(job: dict) -> None:
    if job["error"] is not None:
        raise RuntimeError("kt_kv_hot_lend: a round's copies failed to enqueue") from job["error"]


def _job_join() -> None:
    """Wait until the job's copies are enqueued (not done): before this thread touches the
    slots' memory, the staging buffers or the tables."""
    job = DYN.get("job")
    if job is not None:
        job["thread"].join()
        _job_check(job)


def _tbl_write(table, rows: list, dev) -> None:
    """table[layer, expert] = value for rows [(layer, expert, value)], through pinned memory:
    no host sync (a pageable .to(dev) syncs the stream, and so does a boolean mask)."""
    if table is None or not rows:
        return
    t = torch.tensor(rows, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
    table[t[:, 0], t[:, 1]] = t[:, 2]


def _job_publish(dev) -> None:
    """The job's entries into the tables once its copies are done (an event query, no wait)."""
    job = DYN["job"]
    if not job["enq"]:
        return
    _job_check(job)
    if not job["ev"].query():
        return
    job["thread"].join()
    torch.cuda.current_stream(dev).wait_event(job["ev"])  # done already: ordering only
    _tbl_write(STATE["tbl"], job["pub"][0], dev)
    _tbl_write(STATE.get("rtbl"), job["pub"][1], dev)
    DYN["job"] = None
    st = DYN.setdefault("pubstat", {"n": 0, "lat": 0.0, "enq": 0.0, "items": 0})
    st["n"] += 1
    st["lat"] += time.perf_counter() - job["t0"]
    st["enq"] += job["t_enq"]
    st["items"] += len(job["items"])


def _job_note() -> str:
    """The worker's part of the round log line (appended: the line's start is parsed)."""
    if not DYN_WORKER:
        return ""
    st = DYN.pop("pubstat", None) or {"n": 0, "lat": 0.0, "enq": 0.0, "items": 0}
    d, DYN["deferred_logged"] = DYN.get("deferred", 0) - DYN.get("deferred_logged", 0), DYN.get("deferred", 0)
    n = max(st["n"], 1)
    return (f"; {st['n']} jobs ({st['items']} copies) in the tables {st['lat'] / n * 1e3:.0f} ms after issue "
            f"on average, enqueue {st['enq'] / n * 1e3:.1f} ms, {d} rounds deferred")


def _dyn_round(dev) -> None:
'''

SUBS = [
    ('import os\nimport time\nfrom typing import List, Optional\n', 'import os\nimport threading\nimport time\nfrom typing import List, Optional\n'),
    ('DYN_LOG = int(os.environ.get("KT_KV_HOT_DYN_LOG", "512") or 512)  # rounds between log lines\n', FLAG),
    ('''    DYN["on"] = lo >= 0 and lo == hi
    if DYN["on"]:
        DYN["order"] = order
''', '''    DYN["on"] = lo >= 0 and lo == hi
    if DYN["on"]:
        if DYN_WORKER and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError("KT_KV_HOT_DYN_WORKER requires GPU TP=1")
        DYN["order"] = order
'''),
    ('''    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    DYN["pending"] = None
    rtbl = STATE["rtbl"]
''', '''    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    DYN["pending"] = None
    _job_join()
    DYN["job"] = None
    rtbl = STATE["rtbl"]
'''),
    ('def _dyn_round(dev) -> None:\n', JOBS),
    ('''    for better-scored ones (out of the table now, copied in on the side stream)."""
    DYN["calls"] += 1
''', '''    for better-scored ones (out of the table now, copied in on the side stream).
    KT_KV_HOT_DYN_WORKER: every forward, a job whose copies are done enters the tables."""
    if DYN.get("job") is not None:
        _job_publish(dev)
    DYN["calls"] += 1
'''),
    ('''    budget = DYN_SWAPS * DYN_EVERY
    swaps = []  # (layer, 0: slot / 1: resident row, its index, expert out, expert in)
    for L in LAYERS:
        k = L.k
''', '''    budget = DYN_SWAPS * DYN_EVERY
    swaps = []  # (layer, 0: slot / 1: resident row, its index, expert out, expert in)
    # the last job's copies still running: this round plans nothing (its counts are in the scores), the next one does
    busy = DYN.get("job") is not None
    DYN["deferred"] = DYN.get("deferred", 0) + busy
    for L in () if busy else LAYERS:
        k = L.k
'''),
    ('''    if swaps:
        for which, table in ((0, tbl), (1, rtbl)):
            out = [(L.idx, old) for L, w, _, old, _ in swaps if w == which and old >= 0]
''', '''    if swaps and DYN_WORKER:
        # out of the tables now (pinned, no sync), in when the job's copies are done
        for which, table in ((0, tbl), (1, rtbl)):
            _tbl_write(table, [(L.idx, old, -1) for L, w, _, old, _ in swaps if w == which and old >= 0], dev)
        pub = {0: [], 1: []}
        for L, w, i, _, e in swaps:
            if w == 1:
                L.res_now[i] = e
            pub[w].append((L.idx, e, i))
        _job_start([(L, w, i, e) for L, w, i, _, e in swaps], pub, dev)
        DYN["swaps"] += len(swaps)
    elif swaps:
        for which, table in ((0, tbl), (1, rtbl)):
            out = [(L.idx, old) for L, w, _, old, _ in swaps if w == which and old >= 0]
'''),
    ('''             f"a forward and layer)")
        DYN["seen"], DYN["swaps_logged"] = None, DYN["swaps"]
''', '''             f"a forward and layer)" + _job_note())
        DYN["seen"], DYN["swaps_logged"] = None, DYN["swaps"]
'''),
    ('''        # the slots leave the table; a round's copies in flight finish in the sync,
        # and what they were for is in each layer's held list for the next lend
        STATE["tbl"].fill_(-1)
        DYN["pending"] = None
''', '''        # the slots leave the table; a round's copies in flight finish in the sync,
        # and what they were for is in each layer's held list for the next lend
        _job_join()
        STATE["tbl"].fill_(-1)
        DYN["pending"] = None
        DYN["job"] = None
'''),
]


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    text = base.read_bytes()
    assert hashlib.sha256(text).hexdigest() == BASE_SHA, 'base file differs'
    text = text.decode('utf-8')
    for old, new in SUBS:
        assert text.count(old) == 1, (old[:70], text.count(old))
        text = text.replace(old, new)
    compile(text, str(out), 'exec')
    out.write_bytes(text.encode('utf-8'))
    print('wrote', out, hashlib.sha256(text.encode('utf-8')).hexdigest())


if __name__ == '__main__':
    main()
