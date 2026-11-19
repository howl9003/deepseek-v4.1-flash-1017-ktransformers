"""Select the integer contract in the installed host adapter."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
EDITS = [
    ('AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)\n',
     'AVX2MXFP4_MOE = getattr(_moe_mod, "AVX2MXFP4_MOE", None)\n'
     'AVX2ContractV3_MOE = getattr(_moe_mod, "AVX2ContractV3_MOE", None)\n'),
    ('    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()\n\n    if forced == "amx":\n',
     '    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()\n\n'
     '    if forced == "contract_v3":\n'
     '        if AVX2ContractV3_MOE is None:\n'
     '            raise RuntimeError("KT_MXFP4_BACKEND=contract_v3 requested, but AVX2ContractV3_MOE is not compiled in.")\n'
     '        return AVX2ContractV3_MOE\n'
     '    if forced == "amx":\n'),
]
for old, new in EDITS:
    if s.count(old) != 1:
        raise SystemExit(f"anchor not unique ({s.count(old)}): {old[:70]!r}")
    s = s.replace(old, new)
p.write_text(s)
print("patched", p)
