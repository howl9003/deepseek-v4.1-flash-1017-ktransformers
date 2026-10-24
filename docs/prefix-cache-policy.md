# Optional request-prefix retention policy

This optional prerequisite integrates the request-prefix cache policy on recipe `e8608a2` and its
pinned SGLang `8b9f7241`. It changes cache residency and reuse, with all new switches off by
default. It does not alter attention, quantization, expert arithmetic, or the default launcher.
It is separate from `17-host-kv`, which stages a model KV pool when GPU capacity is insufficient.

The new tier-1 feature is `prefix-cache-policy`. Its declared prerequisites select the host-KV/lending stack; tier 2 remains applicable. Use `TIER=1 SGLANG_FEATURES=prefix-cache-policy ./setup.sh`
to select the policy and its prerequisites, or include it in an explicit feature list. A full default setup includes
the code, but `serve.sh` continues to disable radix caching unless `DSV41_RADIX=1` is supplied.
This draft has no SM89 code dependency. The base recipe includes SM89 attention support;
running it on Ada still requires the reviewed hardware/storage adapters.

## Request classes and expiry

`SGLANG_CACHE_TTL=1` enables two inactivity lifetimes. An OpenAI chat or completion request
can use `extra_body={"cache_class": "agent"}` or `X-Cache-Class: agent`. The header wins;
`user` is the default. Class names ignore case and surrounding whitespace. An invalid class
returns HTTP 400 when the policy is enabled. With the policy off the extra field is ignored.
The class uses the existing `kv_hints` envelope through tokenizer and scheduler requests.

Matched and inserted paths receive a monotonic timestamp for that class. A shared prefix
touched by both classes lives until the later of their two deadlines. Split fragments inherit
the relevant child timestamps before a new access stamps them. The rebase fixes an earlier
edge case where an agent access to a new split parent could erase the user's longer inherited
lifetime. Nested split fragments are resolved without inventing a user access for a new
agent-only leaf. Expired entries sort before the configured eviction policy,
and periodic sweeps use the existing cache eviction path. Active locks and session-held
entries remain protected. Host-only entries are not part of the device sweep.

**Expiry releases device residency, not all copies.** With HiCache, eviction may demote a
prefix to host memory. A later request can still hit that prefix and reload it. A nonzero
`cached_tokens` value after a sweep is therefore valid. This is not a deletion guarantee or
security boundary. It is not disk persistence, explicit save/restore, or compaction.

| Variable | Default | Meaning |
|---|---:|---|
| `SGLANG_CACHE_TTL` | `0` | Enable request-class policy and periodic sweeps |
| `SGLANG_CACHE_TTL_AGENT_S` | `1800` | Agent inactivity lifetime, seconds |
| `SGLANG_CACHE_TTL_USER_S` | `43200` | User inactivity lifetime, seconds |
| `SGLANG_CACHE_TTL_SWEEP_S` | `60` | Sweep cadence, seconds |
| `SGLANG_CACHE_TTL_RELEND_TOKENS` | `0` | Opt-in hot-lend release after a sweep frees at least this many tokens |

Lifetimes and cadence must be finite and positive; the relend threshold must be nonnegative.
Relending can pause service and requires the existing hot-lend stack. Leave it off during
initial acceptance. Sweeps are supported only by Python TreeCore with TP=PP=1; other cache,
Rust, or distributed configurations log a warning and do not enable this policy. The hooks
cover the normal and overlap scheduler loops; disaggregated/PP paths are not validated.
The first failed sweep disables subsequent periodic sweeps and logs the exception.

## Host-tier budgets and SWA backups

These options are independent of TTL and require DeepSeek V4 HiCache to be enabled:

| Variable | Default | Meaning |
|---|---:|---|
| `KT_HICACHE_FULL_HOST_TOKENS` | `0` | FULL host capacity in tokens; zero preserves `hicache_ratio × device pages` |
| `KT_HICACHE_SWA_HOST_TOKENS` | `0` | SWA host capacity in tokens; zero preserves the ratio |
| `KT_HICACHE_SWA_TAIL_ONLY` | `0` | With write-through, skip interior SWA backups when a child retains SWA |
| `KT_HICACHE_TRACE` | `0` | Log matches, backups, evictions and host reloads |

Token budgets must be nonnegative integers and round up to the corresponding page size.
They are allocation requests, not hard byte caps. FULL sidecars follow their existing sizing
rules; the independent NPU C128 rule is preserved. The code logs selected page counts.
Tail-only mode keeps the normal window walk when a leaf is shorter than one SWA window.
Write-back and host-buffer-only behavior remain unchanged. It can retain more chunk-end
checkpoints, while a mid-chunk divergence can reuse less history. This is an explicit tradeoff.

The trace wrapper is observational and default off. Enable it only in separate
correctness runs when needed; its timings are not performance measurements.
`KT_HICACHE_TRACE_MIN` is ignored while tracing is off and must be a nonnegative integer
when tracing is enabled.

## Validation

Run these checks against the patched source:

```sh
python3 -B tools/test_prefix_cache_policy.py third_party/sglang
python3 -B tools/test_prefix_policy_schema.py third_party/sglang
python3 -B tools/test_lend_expert_bundle.py third_party/sglang
```

The policy tests execute production definitions with CPU storage/transport doubles;
the schema tests use Pydantic. They do not establish live DMA correctness or model
equivalence. Full-model acceptance must compare cold execution, device prefix hits,
and positive host reloads after agent expiry while a user prefix remains retained.
The host-reload requirement is unchanged.

The current main-based validation uses a 4,224-token prefix and eight generated
tokens, agent/user lifetimes of 6/120 seconds, a one-second sweep, SWA lending off,
and a 3,072 MiB hot-lend spare floor. FULL/SWA host budgets are 1,048,576/65,536
tokens. These are test settings, not default or performance recommendations.

This prerequisite is included with the canonical MoE arithmetic candidate because
the unchanged cache acceptance test requires it. See [canonical MoE arithmetic](canonical-moe-math.md).
The guarded main-based attempt stopped on a missing FlashMLA runtime dependency
before cache or numerical checks. Full-model acceptance remains pending; source
and component checks alone do not authorize publication.
