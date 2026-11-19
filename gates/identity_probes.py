import json
import random
import urllib.request

def _post(url, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _ids(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]


DVP_LANES = (("dvp-300", 3001, 300), ("dvp-704", 3002, 704), ("dvp-1500", 3003, 1500), ("dvp-3000", 3004, 3000))


def _lp(entries):
    """ lp."""
    return [(int(e[1]), e[0]) for e in entries if e is not None and e[0] is not None]
