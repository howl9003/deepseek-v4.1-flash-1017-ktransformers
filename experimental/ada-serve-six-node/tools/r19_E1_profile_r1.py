"""Restore the accepted full KV/concurrency profile on qualified r19 arithmetic."""
import copy

POLICY_KEYS=('KT_HICACHE_FULL_HOST_TOKENS','KT_HICACHE_SWA_HOST_TOKENS','KT_HICACHE_SWA_TAIL_ONLY','KT_HICACHE_TRACE',
    'KT_STREAMING_SESSION_ADMISSION','SGLANG_CACHE_TTL','SGLANG_CACHE_TTL_AGENT_S','SGLANG_CACHE_TTL_USER_S',
    'SGLANG_CACHE_TTL_SWEEP_S','SGLANG_CACHE_TTL_RELEND_TOKENS','KT_PREFILL_SWA_LEND')
C1_KEYS=('KT_CONTRACT_V2_COMBINE_ORDER','KT_CONTRACT_V2_MERGE_POOLS','KT_EXPERT_PLACEMENT',
    'PATHA_EXPERT_BPC','PATHA_EXPERT_LUT','PATHA_EXPERT_TC','PATHA_STAGES','SGLANG_FP8_PATHA_EXPERT_STATIC',
    'SGLANG_FP8_PATHA_EXPERT_STATIC_BINARY','SGLANG_FP8_PATHA_EXPERT_STATIC_DIR','SGLANG_FP8_PATHA_EXPERT_STATIC_ORIGIN_DIR')
D2_KEYS=('KT_GPU_STREAM_PATHA','KT_GPU_STREAM_PATHA_DIR','KT_GPU_STREAM_PATHA_DOWN_PLANES','KT_GPU_STREAM_PATHA_MAX_ROWS',
    'KT_GPU_STREAM_PATHA_FLAGS_LOG','KT_GPU_STREAM_PATHA_NATIVE_PIN','KT_GPU_STREAM_PATHA_NATIVE_PIN_SHA256',
    'KT_GPU_STREAM_PATHA_ORIGIN_DIR','PATHA_EXT_NAME')

def build(numeric,fast,canonical_fast,numeric_root,fast_root,canonical_fast_root,target,origin_sha,selection):
    norm=lambda x,root:x.replace(root,'TRIAL') if isinstance(x,str) else x
    assert [norm(x,fast_root) for x in fast['argv']]==[norm(x,canonical_fast_root) for x in canonical_fast['argv']], 'Unexpected candidate argument change'
    a={k:norm(v,numeric_root) for k,v in numeric['environment'].items()}
    b={k:norm(v,fast_root) for k,v in fast['environment'].items()}
    difference={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    allowed=set(POLICY_KEYS+C1_KEYS)|{'FEATURE_ORIGINS_CONFIG_SHA256','KT_ROUTING_DUMP'}
    if selection['D2_sources_present']:allowed.update(D2_KEYS);allowed.add('KT_PREFILL_PLAN_COST')
    assert difference<=allowed,('Unexpected candidate environment changes',difference-allowed)
    assert set(C1_KEYS)<=set(b)
    assert b['KT_CONTRACT_V2_COMBINE_ORDER']=='logical'
    assert b['KT_CONTRACT_V2_MERGE_POOLS']==('all' if selection['B4_keep'] else 'off')
    assert b['SGLANG_FP8_PATHA_EXPERT_STATIC']==str(int(selection['C1_keep']))
    assert b['KT_HOT_DYNAMIC']=='0' and b['KT_CPU_THREADS']=='42' and b['KT_NUMA_NODES']=='0,1,2,4,5,6'
    assert b.get('KT_WORKER_TIMESTAMPS','0')=='0'
    if selection['D2_sources_present']:
        assert set(D2_KEYS)<=set(b)
        assert b['KT_GPU_STREAM_PATHA']==str(int(selection['D2_keep']))
        assert b['KT_GPU_STREAM_PATHA_FLAGS_LOG'] in ('0','1')
        if selection['D2_keep']:assert b['KT_PREFILL_PLAN_COST']=='2600,491840,280,820'
    else:assert not set(D2_KEYS)&set(b) and not selection['D2_keep']
    env=dict(b)
    for key in POLICY_KEYS:env[key]=a[key]
    env.pop('KT_ROUTING_DUMP',None)
    if selection['D2_sources_present']:env['KT_GPU_STREAM_PATHA_FLAGS_LOG']='0'
    env['FEATURE_ORIGINS_CONFIG_SHA256']=origin_sha
    assert env['KT_PREFILL_SWA_LEND']=='0'
    env={k:v.replace('TRIAL',target) if isinstance(v,str) else v for k,v in env.items()}
    result=copy.deepcopy(numeric)
    result.update(argv=[x.replace(numeric_root,target) for x in numeric['argv']],environment=env,
        purpose='r19 E1 combined full acceptance; frozen feature and host-reload protocols, qualified arithmetic, six nodes')
    return result
