# Serving the 32.98 build

Everything added to serve the Ada build on 2 Nov 2026. Six nodes, 42 workers, 320 W.

| Measure | Result |
|---|---|
| Breadth, one session | 33.15 tok/s, two rounds |
| Four sessions at once | 51.1 tok/s total, 13.5 each |
| Next turn from disk, 131K prompt | 1.9 s, against 231 s cold, bit-exact vs cold |
| Same, after a full server restart, 41K prompt | 1.3 s, against 77 s cold, bit-exact vs cold |
| Public panel, KL / top-1 against DeepSeek's reference on Ada | 0.0111 / 96.74%, 1.11x the reference's noise floor |

Results are in `results/`: `SERVING_RESULTS_r1.md`, `DISK_TIER_RESULTS_r1.md`, `SERVED_PANEL_RESULTS_r1.md`, `ROTATE_FREEZE_RESULTS_r1.md`.

## What is here

- `tools/` the disk gate chain (`*_adadk4_*`), the serving chain (`*_adasrr_*`), `serve_ada_r1.py` (one load under the same memory and provenance guards as a bench run, resident until stopped), `swap_entry_ada_r1.py` (llama-swap child), `install_ada_selector_r1.py` (owner's root step), `disk_restart_r1.py`, `server_info_profile_g0_r3.py`, `disk-client-r1/`.
- `overlays/` `scheduler.py` (a streaming session's virtual node no longer reaches the L3 prefetch) and `http_server.py` (`keep_storage` on `/flush_cache`, and `SGLANG_FLUSH_CLEARS_STORAGE` for gate chains). Both apply over the served tree. Neither changes behaviour without a disk tier.
- `profile/` the as-run launch profiles.

## Differences from the 32.98 performance profile

Radix cache on, HiCache host and file tiers on, four running requests, 32 queued, `KT_PREFILL_SWA_LEND=0`, `KT_PREFILL_PLAN_MARGIN_MB=4096`, real cache lifetimes. No kernel changes, and no selector on the served numeric path changes.

The frozen `client-tools/kv_checks.py`, hash `eb31f84a`, refuses a storage tier by design and stays pinned for the no-disk profile. `tools/disk-client-r1/kv_checks_disk.py` is a separate disk-aware copy.

Identity is exact vs C1, our accepted build, not vs DeepSeek's reference; the panel row above is the comparison with the reference.
