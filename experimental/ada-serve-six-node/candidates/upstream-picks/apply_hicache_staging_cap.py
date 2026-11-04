"""The V4 HiCache write-back staging cap from 05yuki's tier 1 patch 21 (upstream cef35da), and nothing else from that
patch, as an overlay on the served tree ada-serve-c2-20261103-r3 (Claude, 3 Nov 2026; upstream-picks-r1).

With --hicache-mem-layout page_first (ours, and SGLang's default here) every V4 host pool keeps a device staging
buffer of min(pages, 64) pages of all its layers for the jit write-back kernel. Our 'swa' pool is 40 layers x 135,168
bytes a page, so 64 pages are 330 MiB of VRAM that the hot lend then does not get. SGLANG_HICACHE_STAGING_MB=<MiB>
also caps each buffer at that much device memory (his value 32: 6 pages for 'swa'; the c1/c2 pools are under 8 MiB
and keep 64). transfer_hicache_all_layer_mla_staged_lf_pf (kernels/ops/kvcache/hicache.py) already copies in chunks
of the buffer's page capacity, so a smaller buffer moves the same bytes in more launches. Unset (the default): the
64-page buffer, as before.

The patch's other parts are not needed for the cap and are not taken: mapping the SWA pages before a load-back and
the host trim of the SWA budget both serve KT_PREFILL_SWA_LEND=1 (we serve 0), and the calibration change serves his
first-launch chunk calibration.

argv: the sglang package directory, patched in place."""
import hashlib
import sys
from pathlib import Path

PIN = 'c428d0353505c9b1a7f27303fc3e0686461d0281f20b80092bed3dd43eaabd76'


def sub(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:80])
    return s.replace(old, new)


def patch(s):
    old = ('logger = logging.getLogger(__name__)\n'
           '\n'
           '\n'
           'from sglang.srt.mem_cache.pool_host import HostKVCache\n')
    new = ('logger = logging.getLogger(__name__)\n'
           '\n'
           '\n'
           'def _staging_page_capacity(num_pages: int, page_bytes: int) -> int:\n'
           '    """V4 write-back staging pages: at most _WRITE_BACK_STAGING_PAGE_CHUNK, as before, and with\n'
           '    SGLANG_HICACHE_STAGING_MB=<MiB> (05yuki tier 1 patch 21, upstream cef35da; overlay upstream-picks-r1)\n'
           '    at most that much device memory. The staged kernel loops over the buffer\'s capacity, so a smaller\n'
           '    buffer moves the same bytes in more launches."""\n'
           '    import os\n'
           '\n'
           '    cap = os.environ.get("SGLANG_HICACHE_STAGING_MB", "").strip()\n'
           '    if not cap:\n'
           '        return min(num_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)\n'
           '    mb = int(cap)\n'
           '    if mb <= 0:\n'
           '        raise ValueError(f"SGLANG_HICACHE_STAGING_MB takes a positive MiB count, got {cap!r}")\n'
           '    return max(1, min(num_pages, _WRITE_BACK_STAGING_PAGE_CHUNK, (mb << 20) // max(1, page_bytes)))\n'
           '\n'
           '\n'
           'from sglang.srt.mem_cache.pool_host import HostKVCache\n')
    s = sub(s, old, new)
    for unit in ('self.item_bytes', 'self.state_page_bytes'):
        old = ('            element_size=' + unit + ' * self.dtype.itemsize,\n'
               '        )\n'
               '        staging_page_capacity = min(self.num_host_pages, _WRITE_BACK_STAGING_PAGE_CHUNK)\n')
        new = ('            element_size=' + unit + ' * self.dtype.itemsize,\n'
               '        )\n'
               '        staging_page_capacity = _staging_page_capacity(\n'
               '            self.num_host_pages, self.layer_num * ' + unit + ' * self.dtype.itemsize\n'
               '        )\n')
        s = sub(s, old, new)
    assert 'staging_page_capacity = min(' not in s
    return s


def main(root):
    p = Path(root) / 'srt/mem_cache/memory_pool_host.py'
    s = p.read_bytes().decode('utf-8')
    assert hashlib.sha256(s.encode('utf-8')).hexdigest() == PIN, 'memory_pool_host.py differs from the served pin'
    s = patch(s)
    compile(s, str(p), 'exec')
    p.write_bytes(s.encode('utf-8'))
    print(hashlib.sha256(s.encode('utf-8')).hexdigest(), 'srt/mem_cache/memory_pool_host.py')


if __name__ == '__main__':
    main(sys.argv[1])
