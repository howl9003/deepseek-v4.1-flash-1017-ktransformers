# Contract v2 + Route 3: six-node tested snapshot

This optional source snapshot records the build tested on 28 October 2026. It is based on upstream recipe main `7e199f2be41b601805c3d587bf2efbde8a8c2f23`. The inherited `setup.sh`, `serve.sh`, tier patches and defaults are unchanged. Installing this directory is an explicit opt-in; it is not an upstream PR or an automatic replacement for the default engine.

The candidate combines integer-exact contract v2 routed experts on CPU/GPU with Route 3's SM89 dense FP8 decode kernel. Its default kernel is build11, policy0, matching the accepted fixed-split-K Triton partials. The `ROUTE3_L2_POLICY=1` experiment remains available in the component package but is not selected by either model profile. The K4 owner-combine experiment is not selected. This is a distinct arithmetic context from DeepSeek's reference, not a claim of reference bit identity.

## Results

| Check or workload | Accepted contract K3 | Contract + Route 3 |
|---|---:|---:|
| Numerical gates | 26/26 | 28/28 |
| Performance campaign checks | 23/23 | 23/23 |
| Six-workload breadth median, tok/s | 16.55 | **17.80** |
| Japanese median, tok/s | 19.10 | 20.95 |
| Two 35K prefill times, seconds | 264.57 / 240.25 | 181.54 / 162.80 |

These are single matched campaigns, not a confidence interval for speedup. Python code decode was 16.5 versus 16.7 tok/s; the other five breadth rows improved. 30+ tok/s has not been demonstrated on this six-node contract build. The numerical profile and historical fast profile are deliberately different and must not be interchanged when interpreting results.

Route 3's paired NLL increase versus K3 was 0.000448332, below one paired-piece standard error of 0.003729299. Cold repeat, eight retained splits, rotation, feature replays, abort recovery and the unchanged host-reload check passed. The two scope cleanup operations each exceeded an initial 15-second wait before converging to empty; receipts retain those warnings.

The adopted read-only 256 MiB displacement component gate passed at 10.030080 ms per 208 calls. The prior read-modify-write displacement results, 12.764928 ms default and 12.732160 ms policy, remain recorded as failures of the original 12 ms gate. The method decision is preserved. Invalid policy validation9 and its explicit invalidation are retained for provenance; corrected build13/validation10 supersede it. Do not count validation9 as passing evidence.

## Contents and reconstruction

1. Start from this branch's recipe pin. Its SGLang pin is `656331295619a4d0267b40b7da0b5cd5b549fcbc`; KT is `95009ea6856c0799e517e93cb12be5e8494bc7ce`. Preserve the tier0, tier1 and tier2 patch order in `setup.sh`. KT's llama.cpp and pybind11 pins are in `SOURCE_STACK.json`.
2. In a separate checkout, apply the recipe's tier2 source stack. Then apply `optional/01-retained-session-admission.patch` and `optional/02-prefix-cache-policy.patch` to SGLang in that order. They remain separately selectable patches; this recorded candidate requires both for its numerical validation profile.
3. Run `python apply_overlay.py --sglang /path/to/sglang --ktransformers /path/to/ktransformers` to check the inputs. Add `--apply` only for the isolated candidate. It verifies all changed inputs before writing and verifies the resulting source snapshot afterward. It does not compile, install or launch anything. `SOURCE_STACK.json` marks each modified upstream file and retains its pre/post hash; upstream notices are unchanged.
4. Rebuild the native libraries with the flags in `BUILD.md`. Preserve the installed KT Python layer from `runtime/kt_kernel`, the Engram adapter and the Route 3 loader. These are exact tested files, not an inferred recreation from a different wheel.
5. The two `profiles/*-as-run.json` files preserve exact launch arguments and environment. `*-template.json` replaces external paths with explicit variables. A template is **not an executable launch authorization**: a rebuilt or relocated runtime requires new source/native-origin manifests, dependency/cache preparation and component checks before a guarded model run. Do not remove the pinned loader or origin checks to make it start.

The snapshot contains source, profiles and small review receipts. It excludes weights, checkpoint-derived tensor/logprob captures, raw traces, resident captures, compiled libraries, caches and evidence archives. `EXCLUSIONS.json` lists the size-triggered exclusions. Absolute host paths in immutable receipts and historical controllers identify the tested evidence; they are not portable defaults or files supplied by this repository.

## Resource and publication constraints

Compute uses NUMA nodes 0,1,2,4,5,6 and 42 workers. CPU nodes 3 and 7 are reserved for FEA; RAM is shared, with a 64 GiB FEA reserve and a 416 GiB no-swap model cap. The recorded power cap is 320 W. Preserve those guards, the settling checks, the shared benchmark lock, single-server ownership and unique receipt namespaces. Never kill a resident server merely to start this profile.

`validation/kv_checks.py` is unchanged: SHA256 `eb31f84a7ae96126bfa1fd97e8c3c98527c58d5f515610b15a03fe39c14597b4`. Historical controller sources are archived for review; they reference the original guard packages and are not a standalone launcher. No model runs were made for packaging. PR6 and upstream drafts remain held. DSpark was not run or enabled by this snapshot.

## Attribution

Reference: DeepSeek's official DeepSeek-V4.1-Flash-1017 inference code and released arithmetic contract. This recipe builds on 05yuki's integration, KTransformers/KVCache.AI, SGLang and its contributors, llama.cpp/ggml, and 0xSero's Engram adapter. NVIDIA's PTX instruction contract and CUDA tools informed the Route 3 implementation and generated-code inspection. See the repository `CREDITS.md`, inherited `LICENSE`/`NOTICE`, `licenses/`, and per-file notices. `PRECISION_LEDGER.md` is an historical record; the completed Route 3 report in `receipts/ROUTE3_STEP5_RESULTS_FOR_CLAUDE_r1.md` supplies the later outcome.
