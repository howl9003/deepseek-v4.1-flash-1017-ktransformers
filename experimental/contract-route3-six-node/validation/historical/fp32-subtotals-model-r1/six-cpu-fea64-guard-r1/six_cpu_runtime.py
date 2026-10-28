import ctypes,os,re
from pathlib import Path
from run_ab import now,save
NODES=(0,1,2,4,5,6)
CPUS=set(range(24))|set(range(32,56))
def cpu_set(mask):
    values=set()
    for part in mask.split(','):
        ab=list(map(int,part.split('-')))
        if len(ab)==1:values.add(ab[0])
        else:
            assert len(ab)==2 and ab[1]>=ab[0]
            values.update(range(ab[0],ab[1]+1))
    assert values
    return values
def require_allowed(mask):
    values=cpu_set(mask)
    assert values <= CPUS,('FEA CPU reservation violated',mask)
def policy():
    os.sched_setaffinity(0,CPUS)
    lib=ctypes.CDLL('libnuma.so.1',use_errno=True)
    mask=ctypes.c_ulong(255)
    assert lib.set_mempolicy(3,ctypes.byref(mask),64)==0
    got=ctypes.c_ulong();mode=ctypes.c_int()
    assert lib.get_mempolicy(ctypes.byref(mode),ctypes.byref(got),64,None,0)==0
    assert mode.value==3 and got.value==255

def runtime(cg, target):
    processes = []
    # Include child cgroups if the runtime ever creates them.
    pids = set()
    for path in [cg / 'cgroup.procs', *cg.glob('**/cgroup.procs')]:
        pids.update(path.read_text().split())
    for pid in sorted(pids, key=int):
        root = Path('/proc') / pid
        try:
            threads = []
            service_masks = {}
            for task in (root / 'task').iterdir():
                try:
                    comm = (task / 'comm').read_text().strip()
                    status = (task / 'status').read_text()
                    mask = re.search(r'Cpus_allowed_list:\s*(\S+)', status).group(1)
                    require_allowed(mask)
                    if comm.startswith('numa_'):
                        threads.append({'tid': int(task.name), 'comm': comm, 'cpus': mask})
                    else:
                        service_masks[mask] = service_masks.get(mask, 0) + 1
                except FileNotFoundError:
                    # A short-lived compiler/helper thread may disappear mid-read.
                    continue
            cmdline = (root / 'cmdline').read_bytes().decode(errors='replace').replace('\0', ' ')
            item = {'pid': int(pid), 'cmdline': cmdline,
                    'threads': threads, 'status': (root / 'status').read_text()}
            if threads:
                # numa_maps is process-wide here. Reading it for each of the
                # 42 expert threads made each runtime receipt about 46 MiB.
                item['numa_maps'] = (root / 'numa_maps').read_text()
                item['service_thread_affinity'] = {'count': sum(service_masks.values()),
                                                   'cpu_mask_counts': service_masks}
            if threads or '-m sglang.launch_server' in cmdline:
                raw_env = (root / 'environ').read_bytes().decode(errors='replace').split('\0')
                item['selected_env'] = {v.split('=', 1)[0]: v.split('=', 1)[1] for v in raw_env
                    if '=' in v and (v.startswith(('KT_', 'SGLANG_', 'DSV41_')) or
                        v.split('=', 1)[0] in ('MAX_JOBS', 'TORCHINDUCTOR_COMPILE_THREADS',
                        'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
                        'CUDA_CACHE_PATH', 'TILELANG_CACHE_DIR', 'FLASHINFER_WORKSPACE_BASE'))
                    and not re.search(r'SECRET|KEY|TOKEN|PASSWORD|CREDENTIAL|AUTH', v.split('=', 1)[0], re.I)}
                stat_tail = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
                item['environment_provenance'] = {
                    'source': str(root / 'environ'), 'captured_utc': now(),
                    'pid': int(pid), 'proc_starttime_ticks': int(stat_tail[19]),
                    'cgroup': (root / 'cgroup').read_text().strip(),
                    'selection': 'KT_*, SGLANG_*, DSV41_*, selected compiler/cache variables; secret/key/token names omitted'}
            processes.append(item)
        except FileNotFoundError:
            pass
    save(target, {'processes': processes})
    schedulers = [p for p in processes if len(p['threads']) == 42]
    assert len(schedulers) == 1, 'expected one scheduler with 42 named expert threads'
    launchers = [p for p in processes if '-m sglang.launch_server' in p['cmdline']]
    assert len(launchers) == 1 and launchers[0].get('selected_env'), 'launch-parent environment receipt missing'
    actual_t, actual_m = [], []
    for thread in schedulers[0]['threads']:
        assert re.fullmatch(r'numa_[012456]_[tm]_[0-9]+', thread['comm']), thread['comm']
        node = int(thread['comm'].split('_')[1])
        cpu = int(thread['cpus'])  # singleton mask required
        assert cpu // 8 == node and cpu % 8 < 7, (node, cpu)
        (actual_t if '_t_' in thread['comm'] else actual_m).append(cpu)
    # kt-kernel starts one distributor/main thread on core 0 and six
    # InNumaPool workers on cores 1..6 per node for a seven-thread pool.
    assert sorted(actual_t) == [n * 8 + i for n in NODES for i in range(1, 7)]
    assert sorted(actual_m) == [n * 8 for n in NODES]
    return schedulers[0]['pid']
