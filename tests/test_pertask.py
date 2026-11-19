"""Path B stagger r2 (concept-r1 STAGGER_PREREQ_r1.md section 5.1, STAGED_ARMS_r1.md section 20): compile and run
tests/test_pertask.cpp against this branch's cpu_backend (task_queue.cpp, task_queue.h, memop_detect.h, memop_per_task.h),
CPU only, no CUDA, no module. Run through freeze/cpu/guard_pbc.py (node 3). Writes <out>/test_pertask.json: every check,
its verdict, "ok" and "kind" (ok / finding / failed). Exit 0 for ok, 1 for a finding (a failed check), 3 when the harness
itself fails (compile error included: the compiler's words are in the JSON and on stdout).

  usage: test_pertask.py <out dir> <cpu_backend dir>

Hard checks: the counters stay 0 with KT_MEMOP_SYNC_PER_TASK unset and count with it set; the switch reads "1" only; a
sync pairs with the submit armed last on its stream (KT_MEMOP_PAIRED's Paired) and never with a second sync's or another
stream's; allow > 0 and unpaired syncs keep today's rule; no sync is released before its own task finished (two lanes at
c = 250 / 350 / 420 us, one lane, a reordered lane, unpaired syncs, a third thread enqueueing, two lanes on ring slots);
under today's rule and under KT_MEMOP_PAIRED alone no sync is released while the other lane has a task pending (coupled),
under the per-task rule some are (decoupled); a task's exception surfaces in the poller. Forced two-lane race (the poller
held after reading lane 0's submit flag as 0 until lane 0 raised its submit and sync, every 53rd hand-off, lane 1
running): today's rule releases early at least at every held hand-off (the control: the hold bites); KT_MEMOP_PAIRED and
the per-task rule release none early and defer at least every held sync; graph and ring slots; no ring collision.
Reported, not checked: step times against the CPU floor and today's rule; deferrals in unforced runs.
"""
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out, cb = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    here = Path(__file__).resolve().parent
    rep = dict(kind="incomplete", started=time.time(), checks={}, info={})
    save = lambda: (out / "test_pertask.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    try:
        exe = out / "test_pertask"
        cmd = ["g++", "-O2", "-std=c++17", "-pthread", "-I", str(cb), str(here / "test_pertask.cpp"), str(cb / "task_queue.cpp"),
               "-o", str(exe)]
        rep["compile"] = dict(cmd=cmd, gxx=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                              sha256={p.name: sha(p) for p in (here / "test_pertask.cpp", cb / "task_queue.cpp", cb / "task_queue.h",
                                                               cb / "memop_detect.h", cb / "memop_per_task.h", cb / "cpuinfer.h")})
        save()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        rep["compile"].update(rc=r.returncode, stderr=r.stderr[-6000:])
        if r.returncode:
            print(r.stderr[-6000:])
            raise RuntimeError("compile failed")
        save()
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
        rep["run"] = dict(rc=r.returncode, stderr=r.stderr[-4000:])
        data = json.loads(r.stdout)
        rep["raw"] = data
        u, runs = data["units"], {x["name"]: x for x in data["runs"]}
        ck = rep["checks"]
        ck["counters_off_stay_0"] = u["off_enqueued"] == 0 and u["off_finished"] == 0
        ck["counters_on_count"] = u["on_enqueued"] == 10 and u["on_finished"] == 10
        ck["switch_reads_1_only"] = (u["env_on"], u["env_10"], u["env_unset"]) == (True, False, False)
        ck["pairing"] = (u["submit_pair"], u["sync_pair"], u["second_sync_pair"], u["other_stream_pair"]) == (-1, 7, -1, -1)
        ck["takes_rule"] = (u["takes_allow1"], u["takes_unpaired"], u["takes_paired"]) == (False, False, True)
        clean = lambda x: x["violations"] == 0 and x["timeouts"] == 0 and x["exception"] == ""
        for c in (350, 250, 420):
            t, p = runs[f"today_c{c}"], runs[f"per_task_c{c}"]
            ck[f"today_c{c}_never_early"] = clean(t)
            ck[f"today_c{c}_coupled"] = t["released_other_pending"] == 0
            ck[f"per_task_c{c}_never_early"] = clean(p) and p["paired_syncs"] == 80 and p["waits"] == 2 * 40 * p["steps"]
            ck[f"per_task_c{c}_decoupled"] = p["released_other_pending"] > 0
            rep["info"][f"c{c}"] = dict(floor_ms=p["floor_ms"], today_ms=t["step_ms"], per_task_ms=p["step_ms"],
                                        per_task_over_floor=round(p["step_ms"] / p["floor_ms"], 3),
                                        today_over_per_task=round(t["step_ms"] / p["step_ms"], 3),
                                        released_other_pending=[t["released_other_pending"], p["released_other_pending"]],
                                        releases=[t["releases"], p["releases"]], per_task_deferred=p["deferred"],
                                        per_task_early=p["early"])
        pc = runs["paired_c350"]
        ck["paired_c350_never_early"] = clean(pc)
        ck["paired_c350_coupled"] = pc["released_other_pending"] == 0
        rep["info"]["paired_c350_ms"] = pc["step_ms"]
        one = runs["per_task_one_lane_c350"]
        ck["per_task_one_lane_never_early"] = clean(one)
        rep["info"]["one_lane_c350_ms"] = one["step_ms"]
        ro = runs["per_task_reorder"]
        ck["per_task_reorder_never_early"] = clean(ro) and ro["deferred"] > 0
        rt = runs["today_reorder"]
        rep["info"]["today_reorder"] = dict(violations=rt["violations"], timeouts=rt["timeouts"],
                                            note="today's rule releases a sync seen before its submit at once; "
                                                 "a stress case, not the served order")
        up = runs["per_task_unpaired"]
        ck["per_task_unpaired_keeps_today"] = (up["paired_syncs"] == 0 and up["waits"] == 0 and clean(up)
                                               and up["released_other_pending"] == 0)
        ex = runs["per_task_extra_enqueuer"]
        ck["per_task_extra_enqueuer_never_early"] = clean(ex)
        th = runs["per_task_task_throws"]
        ck["per_task_exception_surfaces"] = th["exception"] == "planted task failure"
        for slots in ("forced2", "ring2_forced"):
            t, pa, pt = runs[f"today_{slots}"], runs[f"paired_{slots}"], runs[f"per_task_{slots}"]
            ck[f"today_{slots}_control_bites"] = t["held"] > 0 and t["violations"] >= t["held"] and t["timeouts"] == 0
            ck[f"paired_{slots}_never_early"] = (clean(pa) and pa["held"] > 0 and pa["deferred"] >= pa["held"]
                                                 and pa["collisions"] == 0)
            ck[f"per_task_{slots}_never_early"] = (clean(pt) and pt["held"] > 0 and pt["deferred"] >= pt["held"]
                                                   and pt["collisions"] == 0)
            ck[f"per_task_{slots}_decoupled"] = pt["released_other_pending"] > 0
            rep["info"][slots] = {m: dict(held=x["held"], violations=x["violations"], deferred=x["deferred"],
                                          released_other_pending=x["released_other_pending"], step_ms=x["step_ms"])
                                  for m, x in (("today", t), ("paired", pa), ("per_task", pt))}
        rg = runs["per_task_ring2"]
        ck["per_task_ring2_never_early"] = clean(rg) and rg["collisions"] == 0
        ck["per_task_ring2_decoupled"] = rg["released_other_pending"] > 0
        rep["info"]["ring2"] = dict(step_ms=rg["step_ms"], floor_ms=rg["floor_ms"], deferred=rg["deferred"])
        rep["info"]["unforced_deferrals"] = {k: v["deferred"] for k, v in runs.items()
                                             if not v["forced"] and not v["reorder"] and v["mode"] != "today"}
        rep["ok"] = all(ck.values())
        rep["failed_checks"] = [k for k, v in ck.items() if not v]
        rep["kind"] = "ok" if rep["ok"] else "finding"
    except BaseException:
        rep["kind"], rep["error"] = "failed", traceback.format_exc()
    finally:
        rep["finished"] = time.time()
        save()
    print(json.dumps({k: rep.get(k) for k in ("kind", "ok", "failed_checks", "checks", "info", "error")})[:8000])
    sys.exit({"ok": 0, "finding": 1}.get(rep["kind"], 3))


if __name__ == "__main__":
    main()
