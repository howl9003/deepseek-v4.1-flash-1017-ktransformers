"""Qualify the two TP halves and the whole expert against the integer oracle."""
import hashlib
import sys
import json
from pathlib import Path
import time

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import partials as R
from sglang.srt.layers.moe import v3_his

RES = {"checks": {}, "notes": {}}


def record(name, got, want):
    got, want = np.asarray(got), np.asarray(want)
    c = RES["checks"].setdefault(name, {"mismatch": 0, "compared": 0})
    c["mismatch"] += int((got != want).sum())
    c["compared"] += int(want.size)


def sub(bank, sl):
    return [t[sl].contiguous() for t in bank]


def run_layer(layer, rng):
    xs, routes, rw, _ = R.E1.traces(layer)
    eids = [int(v) for v in rng.choice(384, 6, replace=False)]
    experts = [R.load_expert(layer, e) for e in eids]
    bL = v3_his.bl_of(layer)
    silu_np = np.load(R.E1.SILU).astype(np.int64).astype(np.uint32)
    for tierC in (1, 0):
        v3_his.TIER_C = bool(tierC)
        qmax = 32767 if tierC else 127
        tag = f"L{layer}{'C' if tierC else 'D'}"
        parts = {}
        inputs = {}   # one routing per T, shared by both halves (half0 + half1 must equal the whole expert)
        for T in (1, 2, 3, 4, 64):
            x = xs[:T] if xs.shape[0] >= T else np.concatenate((xs, R.E1.synth_rows(rng, T - xs.shape[0], R.H)))
            ids = np.stack([rng.permutation(6) for _ in range(T)]).astype(np.int64)
            wts = np.resize(rw, (T, 6)).astype(np.float32).copy()
            wts[rng.random((T, 6)) < 0.1] = 0.0
            inputs[T] = (x, ids, wts)
        for h in (0, 1):
            bank = R.bank(experts, h)
            res, hot = sub(bank, slice(0, 3)), sub(bank, slice(3, 6))
            v3_his.prepare(0, R.IH)
            for T in (1, 2, 3, 4, 64):
                x, ids, wts = inputs[T]
                ids_res = np.where(ids < 3, ids, -1)
                ids_hot = np.where(ids >= 3, ids - 3, -1)
                xt = R.E1.bf_t(x)
                acc = torch.zeros((T, R.H), dtype=torch.int64, device="cuda:0")
                v3_his.gpu_partial(xt, res, R.E1.to_t(ids_res), R.E1.to_t(wts), layer, acc)
                v3_his.gpu_partial(xt, hot, R.E1.to_t(ids_hot).to(torch.int32), R.E1.to_t(wts), layer, acc)
                got = acc.cpu().numpy()
                want = R.oracle_token_raw(x, ids, wts, experts, h, qmax, bL, silu_np)
                record(f"two_bank_sets_half{h}_{tag}_T{T if T <= 4 else 'prefill'}", got, want)
                if T == 4:
                    parts[h] = (x, ids, wts, got)
                if T == 64:
                    # memfix (r3b): the same prefill call with the device grid (sentinel launch after set_bl), as the
                    # streamer's shared group graph runs it
                    acc2 = torch.zeros((T, R.H), dtype=torch.int64, device="cuda:0")
                    v3_his.set_bl(layer)
                    with v3_his.bl_from_device():
                        v3_his.gpu_partial(xt, res, R.E1.to_t(ids_res), R.E1.to_t(wts), layer, acc2)
                        v3_his.gpu_partial(xt, hot, R.E1.to_t(ids_hot).to(torch.int32), R.E1.to_t(wts), layer, acc2)
                    record(f"device_bl_eq_explicit_half{h}_{tag}", acc2.cpu().numpy(), got)
                    v3_his.set_bl(layer)
        x, ids, wts, p0 = parts[0]
        p1 = parts[1][3]
        whole = R.oracle_token_raw(x, ids, wts, experts, None, qmax, bL, silu_np)
        record(f"half0_plus_half1_eq_whole_{tag}", p0 + p1, whole)
        S = torch.from_numpy(p0 + p1).cuda()
        got = v3_his.finish(S, layer).view(torch.int16).cpu().numpy()
        want = R.O.int64_finish(whole, bL).view(np.uint32) >> 16
        record(f"finish_{tag}", got.astype(np.int64) & 0xFFFF, want.astype(np.int64))
        v3_his.check(0)


def main():
    out, lib = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(exist_ok=True, parents=True)
    torch.cuda.set_device(0)
    blt, blf = R.E1.bl_table()
    v3_his._bind_for_test(lib, R.E1.SILU, blf)
    RES["notes"].update(lib=str(lib), lib_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(), gpu=torch.cuda.get_device_name(0),
                        gpu_uuid=v3_his._uuid(0), v3_his_sha256=hashlib.sha256(Path(v3_his.__file__).read_bytes()).hexdigest())
    rng = np.random.default_rng(20261110)
    t0 = time.time()
    for layer in (3, 20, 39):
        run_layer(layer, rng)
        print(f"layer {layer} done {time.time() - t0:.0f}s", flush=True)
    RES["total_mismatch"] = sum(c["mismatch"] for c in RES["checks"].values())
    RES["kind"] = "ok" if RES["total_mismatch"] == 0 else "mismatch"
    RES["lib_sha256"], RES["gpu_uuid"] = RES["notes"]["lib_sha256"], RES["notes"]["gpu_uuid"]   # v3_his.load review fields
    (out / (RES["gpu_uuid"] + ".json")).write_text(json.dumps(RES, indent=1) + "\n")
    print(json.dumps({"total_mismatch": RES["total_mismatch"], "checks": len(RES["checks"])}))
    sys.exit(3 if RES["total_mismatch"] else 0)


if __name__ == "__main__":
    main()
