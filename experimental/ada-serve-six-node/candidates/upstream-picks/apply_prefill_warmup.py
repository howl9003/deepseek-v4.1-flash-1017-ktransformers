"""05yuki's prefill warm-up (upstream 77021a7: tier 1 patch 22 and the engram-adapter hunk) as an overlay on the served
tree ada-serve-c2-20261103-r3 (Claude, 3 Nov 2026; upstream-picks-r1).

1. srt/entrypoints/warmup.py: `--warmups prefill_lengths` runs one fresh random prompt of each length in
   SGLANG_WARMUP_PREFILL_LENGTHS (comma separated) before the server reports ready, so the first request of a launch
   does not pay the first streamed prefill's kernel loads and buffer growth. Without `--warmups prefill_lengths` in
   argv the new function is only registered and never runs. One change from his: the token ids come from a
   per-launch seed (SGLANG_WARMUP_PREFILL_SEED=<int> fixes it). His fixed seed 0 gives the same prompt every launch;
   our HiCache file tier keeps a launch's chains on /scratch, so with a repeated prompt a later launch could match it
   from disk (more so with the swa-checkpoint-tail overlay) and warm nothing.
2. engram-adapter/engram_backend.py: DSV41_ENGRAM_RETIRE_EVENT=1 retires an eager staging set that has to grow behind
   a CUDA event instead of a stream synchronize (his change, made switchable here). The old set and its Works stay
   referenced until the event passes and are dropped at the next eager lookup. Unset: the synchronize, as before.

argv: the sglang package directory, then the engram-adapter directory; both patched in place."""
import hashlib
import sys
from pathlib import Path

WARMUP_PIN = '1c080ad3362b9131f344f31c03bed479a20e16507a9f53e31ec2bf80884e9610'
ENGRAM_PIN = '7e3eed1c65140b2cad9e716b5a86fc81ba7b66fec8542bdf3c1e7d427dbabb88'


def sub(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:80])
    return s.replace(old, new)


def patch_warmup(s):
    tail = ('    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):\n'
            '        generate_req_input = GenerateReqInput(\n'
            '            input_ids=(np.random.randint(2**16, size=[size])).tolist(),\n'
            '            sampling_params={\n'
            '                "max_new_tokens": 1,\n'
            '                "temperature": 0.0,\n'
            '            },\n'
            '        )\n'
            '        if disaggregation_mode != "null":\n'
            '            generate_req_input.bootstrap_room = 0\n'
            '            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST\n'
            '\n'
            '        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()\n')
    assert s.endswith(tail) and s.count(tail) == 1, 'warmup.py no longer ends with prefill_shapes'
    assert '"prefill_lengths"' not in s
    return s + (
        '\n'
        '\n'
        '@warmup("prefill_lengths")\n'
        'async def prefill_lengths(disaggregation_mode: str, tokenizer_manager: TokenizerManager):\n'
        '    """One fresh prompt of each length in SGLANG_WARMUP_PREFILL_LENGTHS (comma separated) before the server\n'
        '    reports ready (05yuki tier 1 patch 22, upstream 77021a7; overlay upstream-picks-r1). The first streamed\n'
        '    prefill of a launch loads every kernel variant its row count selects (each first load synchronises the\n'
        '    context behind a deep queue) and grows the staging a streamed prefill keeps; otherwise the first request\n'
        '    pays both. The ids come from a per-launch seed unless SGLANG_WARMUP_PREFILL_SEED fixes one: the HiCache\n'
        '    file tier outlives a launch, and a prompt repeated across launches could be read from it, not computed."""\n'
        '    import os\n'
        '\n'
        '    lengths = [\n'
        '        int(x) for x in os.environ.get("SGLANG_WARMUP_PREFILL_LENGTHS", "").split(",") if x.strip()\n'
        '    ]\n'
        '    if any(n <= 0 for n in lengths):\n'
        '        raise ValueError(f"SGLANG_WARMUP_PREFILL_LENGTHS takes positive token counts, got {lengths}")\n'
        '    seed = os.environ.get("SGLANG_WARMUP_PREFILL_SEED", "").strip()\n'
        '    rng = np.random.default_rng(int(seed) if seed else None)\n'
        '    for size in lengths:\n'
        '        logger.info("Warmup prefill of %d tokens", size)\n'
        '        generate_req_input = GenerateReqInput(\n'
        '            input_ids=rng.integers(1000, 2**16, size=[size]).tolist(),\n'
        '            sampling_params={"max_new_tokens": 1, "temperature": 0.0},\n'
        '        )\n'
        '        if disaggregation_mode != "null":\n'
        '            generate_req_input.bootstrap_room = 0\n'
        '            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST\n'
        '        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()\n')


def patch_engram(s):
    old = 'EAGER_ROWS = int(os.environ.get("DSV41_ENGRAM_EAGER_ROWS", str(1 << 14)))\n'
    s = sub(s, old, old +
            '# DSV41_ENGRAM_RETIRE_EVENT=1 (05yuki 77021a7; overlay upstream-picks-r1, default off): an eager set that\n'
            '# has to grow is retired behind a CUDA event instead of a stream synchronize, which waited for every lookup\n'
            '# queued ahead with the GPU idle; its Works and rows stay referenced until the stream passes that point.\n'
            'RETIRE_EVENT = os.environ.get("DSV41_ENGRAM_RETIRE_EVENT", "0") == "1"\n')
    s = sub(s, '        self._staging, self._works = {}, {}\n',
            '        self._staging, self._works = {}, {}\n'
            '        self._retired = []  # DSV41_ENGRAM_RETIRE_EVENT: (event, staging, works) of grown sets\n')
    old = ('            lent = _lend_prefill()\n'
           '            key = (dev, "eager-lend" if lent else "eager")\n')
    s = sub(s, old,
            '            if self._retired:\n'
            '                # DSV41_ENGRAM_RETIRE_EVENT: drop the grown sets whose lookups the stream has passed\n'
            '                self._retired = [r for r in self._retired if not r[0].query()]\n' + old)
    old = ('                if have is not None:\n'
           '                    # host callbacks still queued hold the old Work and pinned rows\n'
           '                    torch.cuda.current_stream().synchronize()\n'
           '                    old = have[0].data_ptr()\n'
           '                    self._works = {k: v for k, v in self._works.items() if v.ids != old}\n')
    new = ('                if have is not None:\n'
           '                    # host callbacks still queued hold the old Work and pinned rows\n'
           '                    if RETIRE_EVENT:\n'
           '                        # keep both referenced until the stream passes this point\n'
           '                        old = have[0].data_ptr()\n'
           '                        done = torch.cuda.Event()\n'
           '                        done.record()\n'
           '                        self._retired.append((done, have, {k: v for k, v in self._works.items() if v.ids == old}))\n'
           '                    else:\n'
           '                        torch.cuda.current_stream().synchronize()\n'
           '                        old = have[0].data_ptr()\n'
           '                    self._works = {k: v for k, v in self._works.items() if v.ids != old}\n')
    return sub(s, old, new)


def main(sglang, adapter):
    jobs = [(Path(sglang) / 'srt/entrypoints/warmup.py', WARMUP_PIN, patch_warmup),
            (Path(adapter) / 'engram_backend.py', ENGRAM_PIN, patch_engram)]
    out = []
    for p, pin, fn in jobs:
        s = p.read_bytes().decode('utf-8')
        assert hashlib.sha256(s.encode('utf-8')).hexdigest() == pin, f'{p} differs from the served pin'
        s = fn(s)
        compile(s, str(p), 'exec')
        out.append((p, s))
    for p, s in out:   # nothing is written unless both files patched and compiled
        p.write_bytes(s.encode('utf-8'))
        print(hashlib.sha256(s.encode('utf-8')).hexdigest(), p.name)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
