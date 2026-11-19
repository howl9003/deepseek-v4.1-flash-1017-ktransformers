import sys
from pathlib import Path

def full_logprobs(url, tag, n_new=32):
    """Retain every chosen-token value for comparison with the reference."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import identity_probes as ip
    sp = dict(temperature=0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0, frequency_penalty=0.0,
              presence_penalty=0.0)
    lanes = []
    for name, seed, n in ip.DVP_LANES:
        ids = ip._ids(seed, n)
        d = ip._post(url, dict(input_ids=ids, cache_salt=f"dvpf-{tag}-{name}-D", return_logprob=True,
                               sampling_params=dict(sp, max_new_tokens=n_new, ignore_eos=True)))
        dec = ip._lp(d["meta_info"]["output_token_logprobs"])
        toks = [t for t, _ in dec]
        p = ip._post(url, dict(input_ids=ids + toks, cache_salt=f"dvpf-{tag}-{name}-P", return_logprob=True,
                               logprob_start_len=n - 1, sampling_params=dict(sp, max_new_tokens=1)))
        pre = ip._lp(p["meta_info"]["input_token_logprobs"])[-len(toks):]
        same = [a == b for a, b in zip(dec, pre)]
        lanes.append(dict(lane=name, prompt_tokens=n, positions=len(same), equal=sum(same),
                          first_diff=next((i for i, s in enumerate(same) if not s), None),
                          decode_cached=d["meta_info"].get("cached_tokens"), prefill_cached=p["meta_info"].get("cached_tokens"),
                          decode=[list(x) for x in dec], prefill=[list(x) for x in pre]))
    ok = all(l["equal"] == l["positions"] == n_new for l in lanes)
    return dict(status="PASS" if ok else "DIFFERENCE", n_new=n_new, lanes=lanes)
