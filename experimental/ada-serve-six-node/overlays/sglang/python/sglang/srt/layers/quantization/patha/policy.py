"""CPU-only policy shared by the draft dispatcher and its tests."""
import os

AMAX_FLOOR = 1e-4
_TP2_EXACT = os.environ.get("SGLANG_PATHA_TP2_EXACT", "0") == "1"
DECODE_SPLITS = {
    (1792, 5120): 4,
    (32768, 1280): 2,
    (5120, 8192): 1,
    (4608, 5120): 1,
    (5120, 2304): 1,
    (4096, 1280): 1,
}
if _TP2_EXACT:
    # Per-rank shapes at attn_tp=2: same K, half of N, the same splits as TP1.
    DECODE_SPLITS.update({(16384, 1280): 2, (2304, 5120): 1, (2560, 8192): 1, (2560, 2304): 1})
INDEXER = (4096, 1280)
PREFILL_SHAPES = frozenset(DECODE_SPLITS) - {INDEXER}
CONFLICTS = (
    "SGLANG_FP8_ROUTE3_CUDA", "SGLANG_FP8_FIXED_CUDA_CORE",
    "SGLANG_FP8_FIXED_DECODE_GEOMETRY", "SGLANG_FP8_FIXED_DECODE_TRANSPOSE",
    "SGLANG_FP8_PATHA_DECODE", "SGLANG_FP8_PATHA_PREFILL",
)


def enabled(environ=None):
    env = os.environ if environ is None else environ
    return env.get("SGLANG_FP8_PATHA_DENSE", "0") == "1"


def validate_policy(environ=None, floor=AMAX_FLOOR):
    env = os.environ if environ is None else environ
    if _TP2_EXACT and not enabled(env):
        raise RuntimeError("SGLANG_PATHA_TP2_EXACT requires SGLANG_FP8_PATHA_DENSE=1")
    if not enabled(env):
        return
    if floor != 1e-4:
        raise RuntimeError("Path A requires the configured amax floor 1e-4")
    bad = [name for name in CONFLICTS if env.get(name, "0") != "0"]
    if bad:
        raise RuntimeError("Path A excludes dense arithmetic selectors: " + ", ".join(bad))


def route_for(m, n, k, *, indexer_prefill_validated=False):
    """Unknown projections retain their old route; known projections never mix arithmetic."""
    if (n, k) not in DECODE_SPLITS or m == 0:
        return None
    if m < 0:
        raise ValueError("negative row count")
    if m <= 8:
        return "decode", DECODE_SPLITS[n, k]
    if (n, k) == INDEXER and not indexer_prefill_validated:
        raise RuntimeError("Path A indexer prefill identity receipt is required before dispatch")
    return "prefill", None


validate_policy()
