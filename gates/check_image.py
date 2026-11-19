"""Validate every served image receipt against the per-part GGUF comparison."""
import argparse
import json
from pathlib import Path
import re


def check(text, table):
    expected=json.loads(Path(__file__).with_name("image-"+table+".json").read_bytes())
    rows="288,288,288,288,288,288,288,288" if table=="A" else "320,320,320,320,192,192,320,320"
    fmt="pathb-arena-v1;blk=32;scale=e8m0;w=28cf128e48d2791b"
    digests={}
    adopted=set()
    for record in re.split(r"(?=\[)",text):
        line=record.partition("\n")[0]
        if line.startswith("[KT image digest]"):
            match=re.match(r"\[KT image digest\] layer (\d+) part (\d+) ([0-9a-f]{64})(?=\s|$)",line)
            if not match:
                raise ValueError("Malformed image digest")
            layer,part,value=match.groups()
            if (layer,part) in digests or expected.get(layer,{}).get(part)!=value:
                raise ValueError(f"Wrong or duplicate digest at layer {layer}, part {part}")
            digests[layer,part]=value
        elif line.startswith("[KT image]"):
            match=re.match(r"\[KT image\] layer (\d+):.*\(O_DIRECT\).*?parts ([\d,]+); format (\S+)",line)
            if not match or match[2]!=rows or match[3]!=fmt or match[1] in adopted:
                raise ValueError(f"Image adoption differs for table {table}")
            adopted.add(match[1])
    if adopted!=set(expected) or len(digests)!=320:
        raise ValueError(f"Incomplete image: {len(adopted)} layers and {len(digests)} part digests")
    print(f"PASS table {table}, 40 direct image loads and 320 part digests")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("log",type=Path)
    p.add_argument("--table",choices=("A","B"),default="A")
    args=p.parse_args()
    check(args.log.read_text(encoding="utf-8",errors="replace"),args.table)
