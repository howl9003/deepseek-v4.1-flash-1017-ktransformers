"""Check the served API against the release's recorded tokens and logits."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hashes
import requests_c4
from values import full_logprobs


def check(actual, expected, label):
    if actual != expected:
        raise SystemExit(f"FAIL {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20, help="repeated two- and four-request batches")
    parser.add_argument("--json-prompt",type=Path,help="original JSON prompt fixture, for all twelve reference comparisons")
    args = parser.parse_args()
    if args.json_prompt:
        prompt = args.json_prompt.read_text(encoding="utf-8").rstrip("\n")
        if hashlib.sha256(prompt.encode()).hexdigest() != requests_c4.ORIGINAL_JSON_PROMPT_SHA256:
            parser.error("JSON prompt differs from the measured fixture")
        requests_c4.TOKEN_PROMPTS = [(name,prompt if name=="json" else text) for name,text in requests_c4.TOKEN_PROMPTS]
    if args.repeats < 1:
        parser.error("at least one repetition is required")
    evidence = json.loads(Path(__file__).with_name("reference.json").read_bytes())
    tag = uuid.uuid4().hex
    url = requests_c4.URL
    hashes.I1_LANES = (*hashes.I1_LANES, ("i1-20000", 4109, 20000))
    result = hashes.i1_decode_vs_prefill(url, tag)
    check(result["status"], "PASS", "decode/prefill logits")
    actual = {lane["lane"]: [[p["decode_hl"], p["prefill_hl"]] for p in lane["points"]] for lane in result["lanes"]}
    check(actual, evidence["hashes"], "reference logits including the 20K context")
    full = full_logprobs(url, tag)
    check(full["status"], "PASS", "decode/prefill chosen-token probabilities")
    check({l["lane"]: {k:l[k] for k in ("decode","prefill")} for l in full["lanes"]}, evidence["values"], "256 full values")
    references = {}
    for mode in ("single", "staggered", "together"):
        expected = evidence["greedy"][mode]
        rows = requests_c4.run_mode(mode)
        actual = {r["name"]:r["output_ids"] for r in rows}
        if set(actual) != set(expected) or any(len(ids)!=320 for ids in actual.values()):
            raise SystemExit("Incomplete greedy reply")
        if mode == "single":
            references = actual.copy()
        expected = dict(expected)
        if not args.json_prompt:
            expected["json"] = references["json"]
        check(actual, expected, mode)
    print(f"PASS reference logits, 256 chosen-token values and {12 if args.json_prompt else 9} frozen greedy outputs", flush=True)

    def sample(item):
        name, prompt = item
        response = requests_c4.post("/generate", {"text":prompt, "cache_salt":uuid.uuid4().hex,
            "sampling_params":{"temperature":0, "max_new_tokens":320, "ignore_eos":True}})
        meta = response["meta_info"]
        check(meta["cached_tokens"], 0, "cold request")
        check(response["output_ids"], references[name], name)
        values = [meta["v3_hl"], meta["v3_hh"]]
        check([len(x) for x in values], [320,320], "hash coverage")
        return name, values

    prompts = requests_c4.TOKEN_PROMPTS
    solo = dict(map(sample, prompts))
    for width in (2, 4):
        with ThreadPoolExecutor(max_workers=width) as executor:
            for _ in range(args.repeats):
                for start in range(0,len(prompts),width):
                    for name, values in executor.map(sample, prompts[start:start+width]):
                        check(values, solo[name], f"{width} requests, {name}")
        print(f"PASS repeated {width}-request batches", flush=True)


if __name__ == "__main__":
    main()
