import json
import random
import urllib.request

SP = dict(temperature=0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0, frequency_penalty=0.0,
          presence_penalty=0.0, ignore_eos=True)


I1_LANES = (("i1-300", 4101, 300), ("i1-600", 4102, 600), ("i1-900", 4103, 900), ("i1-1000x", 4104, 1000),
            ("i1-1100", 4105, 1100), ("i1-1500", 4106, 1500), ("i1-3000", 4107, 3000),
            ("i1-90", 4108, 90))


T_POINTS = (0, 1, 8, 31)


N_NEW = 32


def _post(url, body, timeout=3600):
    req = urllib.request.Request(url.rstrip("/") + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ids(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]


def _gen(url, ids, salt, n_new, rid=None):
    body = dict(input_ids=ids, cache_salt=salt, sampling_params=dict(SP, max_new_tokens=n_new))
    if rid:
        body["rid"] = rid
    r = _post(url, body)
    m = r["meta_info"]
    hl, hh = m.get("v3_hl"), m.get("v3_hh")
    if hl is None:
        raise RuntimeError("no v3_hl in meta_info: start the server with SGLANG_V3_HASH=1")
    return dict(ids=r["output_ids"], hl=hl, hh=hh, cached=m.get("cached_tokens"))


def i1_decode_vs_prefill(url, tag):
    lanes = []
    for name, seed, n in I1_LANES:
        ids = _ids(seed, n)
        d = _gen(url, ids, f"h1-{tag}-{name}-D", N_NEW)
        pts = []
        for t in T_POINTS:
            p = _gen(url, ids + d["ids"][:t], f"h1-{tag}-{name}-P{t}", 1)
            pts.append(dict(t=t, position=n + t, decode_hl=d["hl"][t], prefill_hl=p["hl"][0], equal=d["hl"][t] == p["hl"][0],
                            hidden_equal=d["hh"][t] == p["hh"][0], token_equal=d["ids"][t] == p["ids"][0], cached=p["cached"]))
        lanes.append(dict(lane=name, prompt_tokens=n, points=pts, band=[pt["position"] for pt in pts if 513 <= pt["position"] <= 1024]))
    ok = all(pt["equal"] for l in lanes for pt in l["points"])
    band_fail = [(l["lane"], pt["position"]) for l in lanes for pt in l["points"] if not pt["equal"] and 513 <= pt["position"] <= 1024]
    return dict(status="PASS" if ok else "DIFFERENCE", band_failures=band_fail, lanes=lanes)
