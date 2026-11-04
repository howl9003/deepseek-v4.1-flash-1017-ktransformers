"""Paired, authenticated projection benchmark of byte-only repacking candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
import unittest

ROOT=Path('/scratch/aster/bench/05yuki-baseline-3991')
GGUF=Path('/scratch/aster/models/gguf/DeepSeek-V4.1-Flash-1017-MXFP4.gguf')
MANIFEST=Path('/scratch/aster/bench/kt-native-pilot/evidence/expert-inventory/expert-ranges.json')
MANIFEST_SHA='9246e760c554a746662f3537cb2aadb626a90d827d2edbdda2e31438c7f30d12'
READER_SHA='624140d538f62533e618f7f3af69ffe10a7e342ce245ffa8dcc2c8c19344cc51'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if not a.run:
        print(json.dumps(dict(execute=False,layers=[0,20,39],experts=[0],
            projections=['gate','up','down'],repeats=3,variants=['baseline','planes','pair_lut','avx2'],
            maximum_projection_reads=108,deadline_s=90,no_source_changes=True)))
        return 0
    assert os.name=='posix' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert len(os.sched_getaffinity(0))==1
    assert os.environ.get('KT_GGUF_FAST_HEADER')=='1'
    assert a.output.resolve().is_relative_to((ROOT/'evidence').resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    def expired(*_):raise TimeoutError('90s loader repack probe')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,90)
    resource.setrlimit(resource.RLIMIT_CPU,(80,90))
    report=dict(status='RUNNING',rows=[],reader_sha256=READER_SHA,
        manifest_sha256=MANIFEST_SHA,caveats=[
            'Selected projections and warm-cache component work, not full startup.',
            'First-read cache state unknown; variants rotate order per projection/repeat.',
            'No inference, numerical approximation, source installation or production mutation.'])
    started=time.monotonic();reader=None;module=None;original=None
    try:
        import numpy as np
        import torch
        from kt_kernel.utils import recipe_gguf_reader as module
        from repack import VARIANTS
        from native import avx2
        from test_repack import RepackTests
        VARIANTS['avx2']=avx2
        report['native_library_sha256']=sha(Path(__file__).parent/'librepack_avx2.so')
        torch.set_num_threads(1)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(RepackTests)
        tests=unittest.TextTestRunner(verbosity=1).run(suite)
        assert tests.wasSuccessful()
        report['unit_tests']=tests.testsRun
        assert sha(Path(module.__file__))==READER_SHA
        assert sha(MANIFEST)==MANIFEST_SHA
        original=module.split_to_adjacent
        functions={'baseline':original,**VARIANTS};names=list(functions)
        reader=module.NativeMXFP4GGUFReader(GGUF,MANIFEST,MANIFEST_SHA)
        report['reader_init_and_tests_s']=time.monotonic()-started
        reference={}
        for repeat in range(3):
            for layer_index,layer in enumerate((0,20,39)):
                for projection_index,projection in enumerate(('gate','up','down')):
                    key=(layer,projection);tensor=f'blk.{layer}.ffn_{projection}_exps.weight'
                    shift=(repeat+layer_index+projection_index)%len(names)
                    order=names[shift:]+names[:shift]
                    for position,name in enumerate(order):
                        repack_ns=0
                        def timed(blocks):
                            nonlocal repack_ns
                            begin=time.perf_counter_ns()
                            try:return functions[name](blocks)
                            finally:repack_ns+=time.perf_counter_ns()-begin
                        module.split_to_adjacent=timed
                        begin=time.perf_counter_ns()
                        weight,scale=reader.read_projection(tensor,0)
                        elapsed=time.perf_counter_ns()-begin
                        checks=(hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                            hashlib.sha256(scale.view(torch.int16).numpy().tobytes()).hexdigest())
                        if key in reference:assert checks==reference[key],(name,key)
                        reference[key]=checks
                        report['rows'].append(dict(variant=name,layer=layer,expert=0,
                            projection=projection,repeat=repeat,order_position=position,
                            source_bytes=reader.records[(tensor,0)]['bytes'],
                            total_ns=elapsed,repack_ns=repack_ns,
                            weight_sha256=checks[0],scale_bits_sha256=checks[1]))
                        del weight,scale
        report['summaries']={}
        for name in names:
            rows=[r for r in report['rows'] if r['variant']==name]
            report['summaries'][name]={
                'all_projection_ms':sum(r['total_ns'] for r in rows)/1e6,
                'all_repack_ms':sum(r['repack_ns'] for r in rows)/1e6,
                'warm_projection_ms':sum(r['total_ns'] for r in rows if r['repeat']>0)/1e6,
                'warm_repack_ms':sum(r['repack_ns'] for r in rows if r['repeat']>0)/1e6}
        assert sha(Path(module.__file__))==READER_SHA
        assert not torch.cuda.is_initialized()
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2
        report.update(status='MEASURED',all_output_bytes_equal=True,
            installed_reader_unchanged=True,cuda_initialized=False,numpy_version=np.__version__)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
    finally:
        if module is not None and original is not None:module.split_to_adjacent=original
        if reader is not None:reader.close_all_handles()
        signal.setitimer(signal.ITIMER_REAL,0)
        report.update(wall_s=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
    return int(report['status']!='MEASURED')

if __name__=='__main__':raise SystemExit(main())
