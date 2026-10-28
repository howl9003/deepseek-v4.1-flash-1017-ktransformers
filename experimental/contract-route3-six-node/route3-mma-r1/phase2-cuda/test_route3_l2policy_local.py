"""CPU-side checks of the ROUTE3_L2_POLICY build switch (no GPU, no torch). (1) With the macro at 0 the kernel source
reduces, token for token, to the reviewed r1 source archived in archive/route3_decode_r1.cu (sha256 952bb68e...):
the default build is the r1 kernel. (2) With the macro at 1 the only additions are the policy register and the two
hinted cp.async forms, written in the PTX ISA grammar (cp-size, src-size, cache-policy), and they are used for B and Bs
only; A and As keep the unhinted copies. Run from this folder: python -I test_route3_l2policy_local.py
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R1_SHA = "952bb68ea79c28caea889028ba1fcbe7cc79c77e22a5c42546b95388aade63d1"


def reduce_at(value, text):
    """Evaluate only the ROUTE3_L2_POLICY conditionals (the file has no other #if) and expand the R3_ macros."""
    out, stack = [], []   # stack entries: True when the current branch is kept
    for line in text.splitlines():
        s = line.strip()
        if s == "#ifndef ROUTE3_L2_POLICY":
            stack.append(("ndef", False)); continue
        if s == "#if ROUTE3_L2_POLICY":
            stack.append(("if", value == 1)); continue
        if s == "#else" and stack:
            kind, keep = stack.pop(); stack.append((kind, not keep)); continue
        if s == "#endif" and stack:
            stack.pop(); continue
        if any(not keep for _, keep in stack):
            continue
        if s.startswith("#define R3_"):
            continue
        out.append(line)
    text = "\n".join(out) + "\n"
    if value == 0:
        text = text.replace(" R3_POL_PARAM", "").replace(" R3_POL_ARG", "")
        text = re.sub(r"R3_CP16_B\(", "cp_async16(", text)
        text = re.sub(r"R3_CP4_B\(", "cp_async4(", text)
    else:
        text = text.replace(" R3_POL_PARAM", ", uint64_t pol").replace(" R3_POL_ARG", ", pol")
        text = re.sub(r"R3_CP16_B\(([^;]*?), ([^;]*?), ([^;]*?)\);", r"cp_async16_hint(\1, \2, \3, pol);", text)
        text = re.sub(r"R3_CP4_B\(([^;]*?), ([^;]*?), ([^;]*?)\);", r"cp_async4_hint(\1, \2, \3, pol);", text)
    return text


def tokens(text):
    return re.sub(r"\s+", "", text)


def test_default_build_is_r1():
    r1 = (HERE / "archive" / "route3_decode_r1.cu").read_bytes()
    assert hashlib.sha256(r1).hexdigest() == R1_SHA, "archived r1 source hash mismatch"
    cur = (HERE / "route3_decode.cu").read_text(encoding="utf-8")
    assert "#ifndef ROUTE3_L2_POLICY" in cur and cur.count("#if ROUTE3_L2_POLICY") == 2
    assert tokens(reduce_at(0, cur)) == tokens(r1.decode("utf-8")), "macro-0 reduction differs from r1"
    print("1. ROUTE3_L2_POLICY=0 reduces token for token to the r1 source (sha256 952bb68e...)")


def test_policy_build_forms():
    cur = (HERE / "route3_decode.cu").read_text(encoding="utf-8")
    one = reduce_at(1, cur)
    assert 'createpolicy.fractional.L2::evict_first.b64 %0, 1.0;' in one
    assert 'cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;' in one
    assert 'cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 4, %2, %3;' in one
    body = one[one.index("void load_chunk("):one.index("__global__")]
    a_line = [l for l in body.splitlines() if "st->A[row]" in l][0]
    as_line = [l for l in body.splitlines() if "st->As[row][b]" in l][0]
    b_line = [l for l in body.splitlines() if "st->B[w][r]" in l][0]
    bs_line = [l for l in body.splitlines() if "st->Bs[tid]" in l][0]
    assert a_line.strip().startswith("cp_async16(") and as_line.strip().startswith("cp_async4("), "A/As must stay unhinted"
    assert b_line.strip().startswith("cp_async16_hint(") and b_line.rstrip().endswith(", pol);"), b_line
    assert bs_line.strip().startswith("cp_async4_hint(") and bs_line.rstrip().endswith(", pol);"), bs_line
    kernel = one[one.index("__global__"):]
    assert kernel.count("make_evict_first_policy()") == 1 and kernel.count(", tid, pol);") == 2
    print("2. ROUTE3_L2_POLICY=1 adds one policy register and hinted copies for B and Bs only (ISA grammar: 16, src-size, policy)")


if __name__ == "__main__":
    test_default_build_is_r1()
    test_policy_build_forms()
    print("SELFTEST_PASS")
