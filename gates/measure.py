"""Measure 512-token serving throughput and cold prompt latency."""
import json
import random
from pathlib import Path
import statistics
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parent))
import requests_c4 as client


def cold(length, seed, temperature):
    rng=random.Random(seed)
    body={"input_ids":[rng.randrange(1000,100000) for _ in range(length)],"cache_salt":uuid.uuid4().hex,
          "sampling_params":{"temperature":temperature,"max_new_tokens":1,"top_p":1.0,"ignore_eos":True}}
    start=time.perf_counter()
    meta=client.post("/generate",body)["meta_info"]
    seconds=time.perf_counter()-start
    if meta.get("cached_tokens")!=0 or meta.get("prompt_tokens")!=length or meta.get("completion_tokens")!=1:
        raise ValueError("Cold prompt token counts differ")
    return seconds


if __name__=="__main__":
    rates={n:[] for n in (1,2,3,4)}
    for _ in range(3):
        for n in rates:
            row=client.together(n,512)
            rates[n].append(row["total_tok_s"])
            print(json.dumps(row),flush=True)
    print("Mean aggregate tokens/s",json.dumps({n:statistics.mean(v) for n,v in rates.items()}))
    for size,seeds,temp in ((12288,(9100,9101),0),(35000,(35001,35002),1)):
        samples=[cold(size,seed,temp) for seed in seeds]
        print(f"{size} cold prompt + one output (s): {samples}, mean {statistics.mean(samples):.4f}")
