import json
import os
from pathlib import Path

import numpy as np
import torch
import oracle as O

dev = torch.device("cuda:0")
D = Path(os.environ["C4_TRACES"])
GGUF = Path(os.environ["IMG_GGUF"])
input_dir = Path(os.environ["C4_INPUT"])
SILU = input_dir / "q6-silu-gpu.npy"
TRACES = ["cpu-trace-136-64", "cpu-trace-152-61-fp32", "cpu-trace-157-47"]
H, I = 5120, 2304
PROJ = {"w1":"gate", "w2":"down", "w3":"up"}
SHAPE = {"w1":(I,H),"w3":(I,H),"w2":(H,I)}
inventory_path = input_dir / "expert-ranges.json"
import hashlib
if hashlib.sha256(inventory_path.read_bytes()).hexdigest() != "9246e760c554a746662f3537cb2aadb626a90d827d2edbdda2e31438c7f30d12":
    raise ValueError("Wrong expert range inventory")
ranges = {(r["tensor"],r["expert"]):r for r in json.loads(inventory_path.read_bytes())["ranges"]}

def bl_table():
    path = input_dir / "bl_table.json"
    return json.loads(path.read_bytes())["bL"], path

def st_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


def st_read(path, name):
    hdr, base = st_header(path)
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    return info["dtype"], info["shape"], raw


def gguf_expert(layer, proj, e):
    """Gguf expert."""
    N, K = SHAPE[proj]
    G = K // 32
    n = N * G * 17
    with open(GGUF, "rb", buffering=0) as f:
        entry = ranges[(f"blk.{layer}.ffn_{PROJ[proj]}_exps.weight", e)]
        f.seek(entry["offset"])
        data = f.read(n)
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Wrong GGUF bytes for layer {layer}, {proj}, expert {e}")
        raw = np.frombuffer(data, dtype=np.uint8).reshape(N, G, 17)
    sc = raw[..., 0].copy()
    q = raw[..., 1:]
    codes = np.concatenate((q & 15, q >> 4), axis=-1)            # GGUF: byte j low = element j, high = element j+16
    hf = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)   # HF: byte i low = element 2i, high = 2i+1
    return hf.reshape(N, K // 2), sc


def traces(layer):
    """Traces."""
    X, R, Wt = [], [], []
    for tr in TRACES:
        p = D / (tr + ".safetensors")
        if not p.exists():
            continue
        hdr, _ = st_header(p)
        key = f"l{layer}.moe_in"
        if key not in hdr:
            continue
        dt, shp, raw = st_read(p, key)
        if dt == "F32":
            x = np.frombuffer(raw, np.float32).reshape(shp)
        elif dt == "BF16":
            x = O.bf16_bits_to_f32(np.frombuffer(raw, np.uint16)).reshape(shp)
        else:
            x = np.frombuffer(raw, np.float64).reshape(shp).astype(np.float32)
        meta = json.load(open(D / (tr + ".json")))["layers"][str(layer)]
        X.append(x); R += meta["route"]; Wt += meta["route_w"]
    x = np.concatenate(X).astype(np.float32)
    xb = O.bf16_rne(x)
    return xb, np.array(R, np.int64), np.array(Wt, np.float32), bool(np.array_equal(xb.view(np.uint32), x.view(np.uint32)))


def synth_rows(r, M, K):
    x = (r.standard_t(3, size=(M, K)) * np.exp2(r.integers(-8, 4, size=(M, 1)))).astype(np.float32)
    x[:, r.integers(0, K, size=4)] *= 64            # outlier channels
    if M > 2:
        x[1, :64] = 0.0                              # all-zero blocks (the 1e-30 floor)
        x[2, 32:64] = np.float32(1e-38)             # tiny block
    return O.bf16_rne(x)


def to_t(a, dtype=None):
    return torch.from_numpy(np.ascontiguousarray(a)).to(dev) if dtype is None else torch.from_numpy(np.ascontiguousarray(a)).to(dev, dtype)


def bf_t(xf32):
    return to_t(O.bf16_rne_bits(xf32).view(np.int16)).view(torch.bfloat16)
