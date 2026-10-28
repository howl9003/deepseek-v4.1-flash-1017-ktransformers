"""Small, fail-closed helpers for an isolated post-A/B feature programme."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

BASE=Path('/scratch/aster/bench')
GIB=1024**3

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

CANONICAL_METADATA_SYMLINKS={'sglang/multimodal_gen/.claude/skills':'../.agents/skills'}

def source_inventory(root,*,skip_bytecode=False,canonical_python=False):
    """Hash physical files; only canonical Python may omit the reviewed metadata link.

    The link itself is checked with readlink, never resolved or followed. Requiring
    the exact complete omission map also detects removal or replacement by a file.
    All private trees and other canonical inputs retain the no-symlink rule.
    """
    root=Path(root)
    require(not root.is_symlink(),'source root is a symlink: '+str(root))
    require(root.is_dir(),'source root is not a directory: '+str(root))
    result={};omitted={};total=0
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root).as_posix()
        if p.is_symlink():
            require(canonical_python and rel in CANONICAL_METADATA_SYMLINKS,
                    'source contains an unreviewed symlink: '+str(p))
            target=os.readlink(p)
            require(target==CANONICAL_METADATA_SYMLINKS[rel],
                    'canonical metadata symlink target changed: '+rel)
            omitted[rel]=target
            continue
        if skip_bytecode and ('__pycache__' in p.relative_to(root).parts or p.suffix in ('.pyc','.pyo')):continue
        if p.is_file():
            result[rel]=sha(p);total+=p.stat().st_size
        else:require(p.is_dir(),'source contains a non-regular entry: '+str(p))
    if canonical_python:
        require(omitted==CANONICAL_METADATA_SYMLINKS,'canonical metadata symlink identity changed')
    return {'files':len(result),'bytes':total,'sha256':result,'omitted_symlinks':omitted}

def inventory(root,*,skip_bytecode=False):
    return source_inventory(root,skip_bytecode=skip_bytecode)['sha256']

def checked_manifest(harness,manifest,expected_sha,version):
    harness,manifest=Path(harness).resolve(),Path(manifest)
    require(harness.name==version,'unexpected per-arm guard package')
    require(sha(manifest)==expected_sha,'guard manifest hash differs')
    data=read(manifest)
    require(data['status']=='PASS','guard package not successful')
    checked={}
    for rel,wanted in data['files'].items():
        p=Path(rel)
        require(p.parts[0]==version and '..' not in p.parts,'invalid guard manifest path')
        actual=harness.joinpath(*p.parts[1:])
        require(sha(actual)==wanted,'frozen guard source differs: '+rel)
        checked[p.relative_to(version).as_posix()]=wanted
    return checked


def load_frozen(harness,manifest,expected_sha,*,a_harness,a_manifest,a_manifest_sha256):
    """Load B-v9 guard code once; independently verify the accepted A-v7 tree."""
    pins=read(Path(__file__).with_name('HARNESS_PINS.json'))
    require(pins['status']=='REVIEWED_A_V7_B_V9','per-arm guard pins are not finalized')
    require(a_manifest_sha256==pins['A']['manifest_sha256'] and
            expected_sha==pins['B']['manifest_sha256'],'unreviewed per-arm manifests')
    harness=Path(harness).resolve();a_harness=Path(a_harness).resolve()
    checked=checked_manifest(harness,manifest,expected_sha,'retry_v9')
    a_checked=checked_manifest(a_harness,a_manifest,a_manifest_sha256,'retry_v7')
    changed={k:{'A':a_checked.get(k),'B':checked.get(k)}
             for k in set(a_checked)|set(checked) if a_checked.get(k)!=checked.get(k)}
    require(changed==pins['declared_package_delta'],'per-arm package delta differs')
    provenance={a:{'root':str(root),'manifest':str(Path(m).resolve()),
                    'manifest_sha256':h,'file_count':len(files)}
                for a,root,m,h,files in [('A',a_harness,a_manifest,a_manifest_sha256,a_checked),
                                        ('B',harness,manifest,expected_sha,checked)]}
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(harness/'pr_acceptance_b'),str(harness)]
    modules={name:importlib.import_module(name) for name in
        ('run_ab','scope_lifecycle','boot_gate','strict_native','cache_stabilization','baseline_gate')}
    for name,module in modules.items():
        require(Path(module.__file__).resolve().is_relative_to(harness),
                'guard module was already imported from another tree: '+name)
    return SimpleNamespace(**modules,harness=harness,manifest_sha256=expected_sha,files=checked,
                           files_by_arm={'A':a_checked,'B':checked},provenance=provenance)

def scope_empty(life,path,unit):
    cg=life.checked_scope_path(path,unit)
    if cg.exists():
        require('populated 0' in (cg/'cgroup.events').read_text(),'prior scope populated')
        require(not (cg/'cgroup.procs').read_text().split(),'prior scope has processes')
        require(int((cg/'memory.current').read_text())<GIB,'prior scope remains charged')
    return str(cg)

def verify_pair(a,b,mods,post_sources=None):
    a,b=Path(a).resolve(),Path(b).resolve()
    post_sources=post_sources or {}
    accepted=mods.baseline_gate.validate(a/'RUN.json')
    result={}
    for arm,campaign in [('A',a),('B',b)]:
        require(campaign.parent==BASE and campaign.name.startswith('pr-acceptance-'+arm+'-'),
                'unexpected canonical campaign')
        post_path=Path(post_sources.get(arm,campaign/'SOURCE_AFTER.json')).resolve()
        run=read(campaign/'RUN.json');post=read(post_path)
        root=mods.run_ab.ROOTS[arm]
        require((post.get('arm'),post.get('campaign'),post.get('root'))==
                (arm,str(campaign),str(root)),'independent post-source identity differs')
        require(run['status']=='TOOLS_COMPLETE_PENDING_NUMERICAL_REVIEW','canonical tools incomplete')
        require(run['owned_cleanup_verified'] is True and run['left_qwen_unloaded'] is True,
                'canonical cleanup incomplete')
        require(post.get('pass_gate') is True and post['scope_empty_at_verification'] is True,
                'independent post-source receipt missing')
        require(post.get('source_mismatches')=={} and post.get('native_source_matches_build') is True,
                'post-source/native audit failed')
        require(post.get('frozen_tool_sha256')==mods.files_by_arm[arm],
                'canonical run used another per-arm guard package')
        require(post.get('frozen_tools_root')==mods.provenance[arm]['root'],
                'canonical post verifier names another per-arm guard root')
        expected_kinds={'components','check','server'}|({'preflight','run'} if arm=='A' else set())
        require({s['kind'] for s in run['scopes']}==expected_kinds and len(run['scopes'])==len(expected_kinds),
                'canonical scope set differs')
        require(len(post['scopes'])==len(run['scopes']) and
                {s['unit'] for s in post['scopes']}=={s['unit'] for s in run['scopes']},
                'independent post-source scope set differs')
        for scope in post['scopes']:
            scope_empty(mods.scope_lifecycle,scope['cgroup'],scope['unit'])
        proof=post['separate_cpu_proof_scope']
        scope_empty(mods.scope_lifecycle,proof['cgroup'],proof['unit'])
        build=read(root/'evidence/BUILD.json')
        native=mods.strict_native.verify(root,arm,build)
        require(native==post['strict_native'],'native restoration/probe identity changed')
        require(sha(root/'evidence/BUILD.json')==post['build_manifest_sha256'],'BUILD changed')
        for rel,h in build['source_sha256'].items():
            require(sha(root/'third_party/sglang'/rel)==h,'canonical source changed: '+rel)
        require(sha(Path(build['wheel']))==build['wheel_sha256'],'canonical wheel changed')
        require(sha(root/'build/libkv_vmm.so')==build['kv_vmm_sha256']==post['kv_vmm_actual_sha256'],
                'canonical KV VMM library changed')
        require(sha(root/'serve.sh')==build['serve_sha256']==post['serve_actual_sha256'],
                'canonical launcher changed')
        if arm=='B':
            controller_post=read(campaign/'SOURCE_AFTER.json')
            require(controller_post.get('pass_gate') is True and
                    controller_post.get('root')==str(root) and
                    controller_post.get('build_manifest_sha256')==sha(root/'evidence/BUILD.json') and
                    controller_post.get('scope_empty_at_verification') is True,
                    'archived B controller post-source receipt differs')
        require(read(campaign/'checks/RESULT.json')['status']=='TOOLS_COMPLETE_PENDING_NUMERICAL_REVIEW',
                'canonical client incomplete')
        files=['RUN.json','SOURCE_AFTER.json','BOOT.json','OVERLAY.json','checks/RESULT.json',
               'server/server_info.json','server/runtime-before.json','server/runtime-after.json']
        artifacts={p:sha(campaign/p) for p in files}
        artifacts['INDEPENDENT_SOURCE_AFTER.json']=sha(post_path)
        result[arm]={'campaign':str(campaign),'root':str(root),
                     'artifacts':artifacts,'independent_post_source':str(post_path),
                     'boot_id':post['boot_id'],'native':native,'build':build,
                     'harness':mods.provenance[arm]}
        if arm=='B':
            require(run['baseline_A']['path']==str(a/'RUN.json'),'B baseline path differs')
            require(run['baseline_A']['sha256']==sha(a/'RUN.json'),'B belongs to another A')
            require(run['boot']['boot_id']==accepted['boot']['boot_id'],'canonical A/B boot differs')
    require(result['A']['boot_id']==result['B']['boot_id'],'post-source A/B boot differs')
    return result

def require_programme(path,prepared):
    """An evidence-linked internal review, not a new request for user permission."""
    review=read(path)
    require(review.get('status')=='REVIEWED_FEATURE_PROGRAMME','programme has not been reviewed')
    require(review.get('reviewer') and review.get('reviewed_utc') and review.get('rationale'),
            'programme review lacks explanation')
    require(review.get('prepared_sha256')==sha(prepared),'programme belongs to another preparation')
    data=read(prepared)
    require(review.get('canonical_artifacts')=={a:v['artifacts'] for a,v in data['canonical'].items()},
            'programme canonical evidence differs')
    require(review.get('numerical_review')=='A_AND_B_REVIEWED_FOR_FEATURE_TRIAL',
            'B script completion is not numerical acceptance')
    return review
