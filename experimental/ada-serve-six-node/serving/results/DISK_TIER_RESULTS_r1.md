# KV cache to disk on the 32.98 build, 2 Nov 2026

**The disk tier works and is bit-exact vs a cold run at every length tested, up to 131K tokens. A session's next turn after a 131K prompt starts in 1.9 s from disk, against 231 s cold.** Gate trial `ada-gate-disk-20261102-r5`: disk-aware KV check PASS, all replay and queue fixtures equal, logprob delta 0.0, memory guards unchanged.

SGLang's HiCache file backend, on `/scratch`. Each row is a real follow-up: a prompt runs, the GPU and host tiers are flushed, then the next turn (prompt, reply, 64 new tokens) is sent. The same turn is then computed cold for comparison: output ids, selected logprobs and top 8 all equal.

| Prompt, tokens | Stored on disk | Time to first token, cold | From disk | Read from disk | GPU hit, same turn |
|---|---|---:|---:|---:|---:|
| 1,024 | 33 MB, includes earlier fixtures | 17.1 s | 4.4 s | all 1,024 | 1.1 s |
| 8,192 | 19 MB | 23.3 s | 3.3 s | all 8,192 | 1.0 s |
| 40,960 | 79 MB | 73.7 s | 3.3 s | all 40,960 | 1.0 s |
| 131,072 | 244 MB | 231 s | 1.9 s | all 131,072 | 2.5 s |

About 1.9 KB a token on disk, so the 64 GiB cap holds about 35 million tokens of session history. Time to first token includes the decode start and slot re-lend, which is why short rows look long.

## What it took

Each of these stopped a run in its own namespace. None changes a kernel, a selector on the served path, or the frozen gate.

1. **Scheduler crash, real bug.** A streaming session's anchor is a virtual node the tree does not hold, and the L3 prefetch looked it up (`KeyError` in `is_backuped`). Fix in `scheduler.py`: skip the prefetch for session-owned KV. Only reachable with a disk tier.
2. **The frozen client's cold check.** "Fresh replay was cached" fired because the disk copy survived a flush, which is the feature. The client stays frozen. On gate chains only, `SGLANG_FLUSH_CLEARS_STORAGE=1` makes a flush clear the disk too, and the disk stage flushes with `keep_storage=true` (`http_server.py`). The serving chain does not set the selector.
3. **The frozen KV gate refuses a storage tier.** That is by design: "untested storage tier". `client-tools/kv_checks.py` and its hash `eb31f84a` are untouched and stay pinned for the no-disk profile. `disk-client-r1/kv_checks_disk.py` is a separate copy that accepts and counts the tier; nothing else changed.
4. **`server_info_profile_g0_r3.py`** accepts the file backend's form. r2 stays for the accepted profile.
5. **41K prompt ran the GPU out of memory, serving profile only.** This was not a disk problem. Serving keeps SWA mapped (`KT_PREFILL_SWA_LEND=0`), so a prefill window has about 3.5 GB less room than the performance profile, which the plan's cost model does not see. `KT_PREFILL_PLAN_MARGIN_MB` 128 to 4096 on serving chains. 41K and 131K now pass. Prompts above 36K had never run on this profile before.

## Limits

- **An identical replay does not read from disk.** Only the chain's last SWA window is stored (`KT_HICACHE_SWA_TAIL_ONLY`), and that window sits past the reply, so resending the same prompt misses and recomputes. The result is still right, just slow. A follow-up turn hits, which is the normal case in chat and in dsh.
- **A restart was tested, and it works.** Sessions saved on one load, server stopped, full reload, next turn sent. Three of three rows read from disk, equal to cold in ids, logprobs and top 8: 8K 1.9 s and 1.5 s against 25.6 s and 20.5 s; 41K 1.3 s against 77.0 s. An earlier attempt missed its one fair row (cached 0, reply still equal to cold); the cause is unknown, and its second row was wiped by that test's own clear. So a miss has been seen and costs time, not correctness.
- **`/scratch` is 95% full,** 177 GB free. Cap 64 GiB, and the store stops growing below 60 GiB free.
