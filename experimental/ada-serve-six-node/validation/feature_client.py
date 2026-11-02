"""Bounded correctness-only sessions, queue and KV checks on a caller-owned server.

Adapted from the recorded historical session client. No server start/stop, no
performance acceptance, no numerical tolerance inferred from observed results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import signal
import threading
import time
import uuid
import hashlib
import sys
from contextlib import contextmanager
from feature_transport import Context, digest, source_inventory, validate_profile, process_provenance
from kv_checks import run_kv_checks, validate_profile as validate_kv_profile
from queue_checks import run_queue_checks

CTX=None


@contextmanager
def workers(count):
    pool=ThreadPoolExecutor(max_workers=count)
    try:
        yield pool
    except BaseException:
        CTX.close_connections()
        pool.shutdown(wait=False,cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def request(route,body=None,allow_reject=False):
    return CTX.request(route,route,body,allow_reject)


def body(ids,sid=None,tokens=1):
    result=dict(input_ids=list(ids),rid='session-'+uuid.uuid4().hex,
        sampling_params=dict(temperature=0,max_new_tokens=tokens,ignore_eos=True),
        return_logprob=True,logprob_start_len=-1,top_logprobs_num=5,
        return_text_in_logprobs=False,stream=False)
    if sid is not None:result['session_params']=dict(id=sid)
    return result


def summary(result):
    value=result['value'];meta=value['meta_info'];ids=value['output_ids']
    if result['status']!=200 or meta.get('finish_reason',{}).get('type')!='length':
        raise ValueError('expected a complete length-limited generation')
    logs=meta.get('output_token_logprobs') or []
    if not ids or len(ids)!=meta['completion_tokens'] or len(logs)!=len(ids):
        raise ValueError('missing output IDs/logprobs')
    if not all(type(i) is int and i>=0 for i in ids):raise ValueError('invalid output ID')
    if not all(len(x)>=2 and x[1]==i and isinstance(x[0],(int,float)) and not isinstance(x[0],bool)
               and math.isfinite(x[0]) and x[0]<=1e-3 for x,i in zip(logs,ids)):
        raise ValueError('nonfinite or misaligned selected-token logprob')
    cached=meta.get('cached_tokens')
    if type(cached)is not int or not 0<=cached<=meta['prompt_tokens']:
        raise ValueError('missing cache accounting')
    return dict(output_ids=ids,prompt_tokens=meta['prompt_tokens'],cached_tokens=cached,
        output_logprobs=[x[0] for x in logs],wall_s=result['wall_s'])


def compare(a,b):
    if len(a['output_ids'])!=len(a['output_logprobs']) or len(b['output_ids'])!=len(b['output_logprobs']):
        raise ValueError('unpaired output/logprob lengths')
    same=a['output_ids']==b['output_ids']
    return dict(output_ids_equal=same,
        max_selected_logprob_delta=max(abs(x-y) for x,y in zip(a['output_logprobs'],b['output_logprobs'])) if same else None)


def require_overlap_rejection(result):
    """An unrelated server/transport error is not an ownership witness."""
    message='Streaming session already has an active request.'
    value=result.get('value')
    if not isinstance(value,dict):
        raise AssertionError('overlap rejection lacks structured reason')
    reason=value.get('meta_info',{}).get('finish_reason') or {}
    native=(result['status']==200 and reason.get('type')=='abort'
            and reason.get('message')==message)
    error=value.get('error') or {}
    http=(result['status']==400 and isinstance(error,dict)
          and error.get('message')==message)
    if not (native or http):
        raise AssertionError('overlap did not reject the active-session conflict')


def require_abort_control(result):
    if result.get('status')!=200:
        raise AssertionError('owned cancellation was not acknowledged')


def require_aborted_result(result, *, queued=False):
    value=result.get('value') or {}
    meta=value.get('meta_info') or {}
    reason=meta.get('finish_reason') or {}
    if result.get('status')!=200 or reason.get('type')!='abort':
        raise AssertionError('cancelled generation did not report terminal abort')
    expected='Abort in waiting queue' if queued else 'Aborted'
    if reason.get('message')!=expected:
        raise AssertionError('generation aborted for an unrelated reason')
    if queued and (meta.get('completion_tokens')!=0 or value.get('output_ids')!=[]):
        raise AssertionError('queued cancellation started generating')


def require_one_eviction(rows, previous_lengths, *, radix_disabled, page_size):
    if len(rows)!=4 or len(previous_lengths)!=4:
        raise AssertionError('eviction witness requires four session rows')
    if any(k%page_size in (0,1) for k in previous_lengths):
        raise AssertionError('eviction fixture cannot distinguish retained and radix KV')
    lost=[]
    for index,(row,previous) in enumerate(zip(rows,previous_lengths)):
        cached=row['cached_tokens']
        if cached in (previous,previous-1):continue
        bound=0 if radix_disabled else (previous//page_size)*page_size
        if type(cached)is not int or not 0<=cached<=bound:
            raise AssertionError('evicted row cache exceeds possible radix fallback')
        lost.append(index)
    if len(lost)!=1:
        raise AssertionError(f'expected exactly one idle eviction, saw {len(lost)}')
    return lost[0]


def main():
    global CTX
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--host',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--model',required=True);p.add_argument('--context',required=True,type=int)
    p.add_argument('--profile',required=True,type=Path);p.add_argument('--source-root',required=True,type=Path)
    p.add_argument('--server-pid',required=True,type=int);p.add_argument('--deadline',type=int,default=1800)
    p.add_argument('--exclusive-caller-owned-server',action='store_true')
    a=p.parse_args()
    profile=validate_profile(json.loads(a.profile.read_text(encoding='utf-8')),model=a.model,context=a.context)
    if profile['kind']=='combined-prefix':validate_kv_profile(profile,a.deadline)
    sources=source_inventory(a.source_root,profile['source_sha256'])
    if not a.execute:
        print(json.dumps(dict(execute=False,cohorts=[1,2,4],turns=3,first_prompt_tokens=512,
            replay=True,idle_eviction=True,overlap_active_abort=True,queued_abort=True,
            fifo_503=True,kv=profile['kind']=='combined-prefix',deadline_s=a.deadline,
            profile_sha256=digest(profile),source_files=len(sources))))
        return 0
    if not a.exclusive_caller_owned_server or sys.platform!='linux':
        p.error('execution requires Linux and explicit exclusive caller-owned server authorization')
    a.output.mkdir(parents=True,exist_ok=False)
    CTX=Context(host=a.host,port=a.port,model=a.model,output=a.output,deadline_s=a.deadline)
    report=dict(status='RUNNING',requests=[],controls=[],cohorts=[],replays=[],loads=[],
        profile=profile,profile_sha256=digest(profile),source_before=sources,
        tool_sha256={x:hashlib.sha256(Path(__file__).with_name(x).read_bytes()).hexdigest()
                     for x in ('feature_client.py','feature_transport.py','kv_checks.py','queue_checks.py')},
        scope='Short synthetic correctness witnesses; not benchmark timing, broad quality or long-context qualification')
    lock=threading.RLock();opened=set();started=time.monotonic()
    def save():
        with lock:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def record(label,route,payload=None,allow_reject=False):
        result=request(route,payload,allow_reject)
        with lock:
            report['requests' if route=='/generate' else 'controls'].append(dict(
                label=label,route=route,request=payload,response=result));save()
        return result
    def tokenize(text):
        result=record('tokenize','/tokenize',dict(model=a.model,prompt=text,add_special_tokens=False))
        ids=result['value']['tokens']
        if not ids or any(type(x)is not int for x in ids):raise ValueError('invalid tokenizer output')
        return ids
    def open_session():
        result=record('open','/open_session',dict(capacity_of_str_len=100000,streaming=True,timeout=300))
        sid=result['value']
        if not isinstance(sid,str) or not sid:raise ValueError('invalid session ID')
        opened.add(sid);return sid
    def close_session(sid):
        record('close','/close_session',dict(session_id=sid));opened.discard(sid)
    def generate(ids,label,sid=None,tokens=1):
        result=record(label,'/generate',body(ids,sid,tokens));measured=summary(result)
        if len(measured['output_ids'])!=tokens:raise ValueError('unexpected output count')
        return measured
    def flush():CTX.flush()
    def wait_load(predicate,after,seconds=15):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            value=request('/v1/loads?include=all')['value']
            rows=value.get('loads',[])
            report['loads'].append(value);save()
            if len(rows)!=1:raise ValueError('expected one rank load snapshot')
            row=rows[0]
            if row['num_running_reqs']>4:raise AssertionError('active limit exceeded')
            if row['timestamp']>=after and predicate(row):return row
            CTX.wait(.25)
        raise TimeoutError('fresh load snapshot condition not reached')
    def expired(*_):raise TimeoutError('absolute feature-client deadline')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,a.deadline)
    signal.signal(signal.SIGTERM,expired)
    try:
        report['process_before']=process_provenance(a.server_pid,a.source_root,a.port,profile['environment'])
        info=request('/get_server_info')['value']
        server=info.get('server_args',info)
        for key,value in profile['server_args'].items():
            if key not in server or server[key]!=value or type(server[key]) is not type(value):
                raise ValueError('effective session option differs: '+key)
        report['server_info']=info
        models=request('/v1/models')['value']
        if a.model not in [x['id'] for x in models['data']]:raise ValueError('model not served')
        report['models']=models
        report['idle_entry']=CTX.idle();save()
        padding=tokenize('This synthetic notebook records a quiet room, a wooden table and a clock. Entries remain in chronological order. ')
        append=tokenize(' Continue the notebook with one more observation about the room. ')
        small=tokenize('The capital of France is')
        # Match the upstream kit's append protocol and inherited-cache contract.
        # Capture exact token arrays so fresh replay does not depend on retokenizing
        # generated text or regenerating an earlier continuation.
        for n in (1,2,4):
            flush();sessions=[];cohort=dict(n=n,turns=[],eviction=None)
            report['cohorts'].append(cohort)
            for i in range(n):
                prefix=tokenize(f'Notebook {chr(65+i)} describes location {i+1}. ')
                ids=(prefix+padding*((512//len(padding))+1))[:512]
                sessions.append(dict(id=open_session(),history=[],initial=ids,last=None))
            try:
                with workers(n) as executor:
                    for turn in range(3):
                        pending=[];t0=time.monotonic()
                        for i,s in enumerate(sessions):
                            delta=s['initial'] if turn==0 else append
                            full=s['history']+delta
                            pending.append((s,full,executor.submit(generate,delta,
                                f'N{n}-turn{turn}-session{i}',s['id'])))
                        rows=[]
                        for s,full,future in pending:
                            measured=future.result()
                            if measured['prompt_tokens']!=len(full):raise AssertionError('session history length differs')
                            if turn and measured['cached_tokens'] not in (len(s['history']),len(s['history'])-1):
                                raise AssertionError('partial-page inheritance missing')
                            if turn==2 and len(s['history'])%256 in (0,1):
                                raise AssertionError('fixture cannot distinguish retained KV from page-aligned radix')
                            s.update(history=full+measured['output_ids'],last=measured,last_input=full)
                            rows.append(measured)
                        cohort['turns'].append(dict(turn=turn,wall_s=time.monotonic()-t0,rows=rows));save()
                    if n==4:
                        ordinary=generate(small,'fifth-after-four-idle')
                        before=[len(s['history']) for s in sessions]
                        pending=[executor.submit(generate,append,f'eviction-resume-{i}',s['id']) for i,s in enumerate(sessions)]
                        resumed=[]
                        for s,future in zip(sessions,pending):
                            full=s['history']+append;measured=future.result()
                            if measured['prompt_tokens']!=len(full):raise AssertionError('eviction lost token history')
                            s.update(history=full+measured['output_ids'],last=measured,last_input=full)
                            resumed.append(measured)
                        lost=require_one_eviction(resumed,before,
                            radix_disabled=profile['server_args']['disable_radix_cache'],
                            page_size=profile['server_args']['page_size'])
                        cohort['eviction']=dict(ordinary=ordinary,previous_lengths=before,resumed=resumed,evicted_index=lost);save()
            finally:
                for s in sessions:close_session(s['id'])
            for i,s in enumerate(sessions):
                flush();cold=generate(s['last_input'],f'N{n}-fresh-replay-{i}')
                if cold['cached_tokens']!=0:raise AssertionError('fresh replay was cached')
                row=dict(n=n,session=i,session_result=s['last'],fresh=cold,comparison=compare(s['last'],cold))
                if n==1 and not profile['server_args']['disable_radix_cache']:
                    hot=generate(s['last_input'],'ordinary-radix-hot-control')
                    row['ordinary_radix_hot']=hot
                    row['radix_hot_comparison']=compare(s['last'],hot)
                report['replays'].append(row);save()

        flush();sid=open_session()
        first=generate(small,'abort-session-first',sid)
        committed=small+first['output_ids']
        active=body(append,sid,256);active['stream']=True
        stream=CTX.start_stream('active-session-turn',active)
        aborted=False
        try:
            while True:
                snap=stream.snapshot()
                if snap['errors']:raise ValueError('active stream transport failed: '+str(snap['errors']))
                if snap['first_token'] and not aborted:
                    if snap['finished']:raise ValueError('active stream ended before overlap checks')
                    rejected=record('overlap-rejection','/generate',body(small,sid),allow_reject=True)
                    require_overlap_rejection(rejected)
                    # A rejected turn must not release the still-running owner's
                    # busy flag. A second overlap detects that ownership leak.
                    rejected_again=record('second-overlap-rejection','/generate',body(small,sid),allow_reject=True)
                    require_overlap_rejection(rejected_again)
                    require_abort_control(CTX.abort(active['rid']));aborted=True
                if snap['finished']:break
                CTX.wait(.05)
        finally:stream.close()
        if not aborted:raise AssertionError('active cancellation not reached')
        finished=stream.snapshot()
        if finished['errors'] or not finished['chunks']:raise AssertionError('active cancellation stream incomplete')
        require_aborted_result(dict(status=finished['status'],value=finished['chunks'][-1]['value']))
        report['active_abort']=dict(request=active,stream=stream.snapshot());save()
        wait_load(lambda r:r['num_running_reqs']==0 and r['num_waiting_reqs']==0,time.time())
        recovered=generate(append,'after-active-abort',sid)
        if recovered['prompt_tokens']!=len(committed+append):raise AssertionError('aborted turn leaked into history')
        close_session(sid);flush()
        replay=generate(committed+append,'abort-recovery-fresh')
        report['active_abort']['recovery']=dict(result=recovered,fresh=replay,comparison=compare(recovered,replay));save()

        # A fifth session cancelled while queued must be usable again.
        sid=open_session();holder_bodies=[body(small,tokens=256) for _ in range(4)]
        queued=body(small,sid,1)
        with workers(5) as executor:
            mark=time.time()
            holders=[executor.submit(record,f'queue-holder-{i}','/generate',b,True) for i,b in enumerate(holder_bodies)]
            try:
                wait_load(lambda r:r['num_running_reqs']==4,mark)
                mark=time.time();waiting=executor.submit(record,'queued-session','/generate',queued,True)
                wait_load(lambda r:r['num_waiting_reqs']>=1,mark)
                require_abort_control(record('cancel-queued-session','/abort_request',dict(rid=queued['rid'])))
                cancelled=waiting.result(timeout=min(30,CTX.remaining()))
                require_aborted_result(cancelled,queued=True)
                report['queued_abort_result']=cancelled
            finally:
                for b in holder_bodies:record('stop-owned-holder','/abort_request',dict(rid=b['rid']))
                for future in holders:future.result(timeout=30)
        wait_load(lambda r:r['num_running_reqs']==0 and r['num_waiting_reqs']==0,time.time())
        retry=generate(small,'after-queued-abort',sid)
        if retry['prompt_tokens']!=len(small):raise AssertionError('cancelled first turn entered session history')
        report['queued_abort_retry']=retry;close_session(sid);flush()
        queue_profile=dict(profile,queue=dict(input_ids=small,hold_tokens=1024))
        report['queue']=run_queue_checks(CTX,profile=queue_profile,output_dir=a.output/'queue')
        if report['queue']['status']!='PASS':raise ValueError('queue checks incomplete or failed')
        if profile['kind']=='combined-prefix':
            report['kv']=run_kv_checks(CTX,profile=profile,output_dir=a.output/'kv')
            if report['kv']['status']!='PASS':raise ValueError('KV tier checks incomplete or failed')
        report.update(status='MEASURED',history_and_lifecycle_checks=True,
            all_replay_selected_ids_equal=(all(r['comparison']['output_ids_equal'] for r in report['replays'])
                and report['active_abort']['recovery']['comparison']['output_ids_equal']),
            numerical_acceptance='REVIEW_REQUIRED',
            quality='Short replay witness only; numerical differences recorded without universal bitwise requirement.')
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
    finally:
        report['close_errors']=[]
        try:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            signal.signal(signal.SIGINT,signal.SIG_IGN)
            signal.setitimer(signal.ITIMER_REAL,30)
            CTX.begin_cleanup(30)
            CTX.close_connections()
            for rid in list(CTX.owned):
                try:CTX.abort(rid)
                except Exception as error:report['close_errors'].append(repr(error))
            for sid in list(opened):
                try:close_session(sid)
                except Exception as error:report['close_errors'].append(repr(error))
            report['idle_exit']=CTX.idle(seconds=min(10,CTX.remaining()))
        except BaseException as error:
            report['close_errors'].append(repr(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:CTX.close_connections()
            except Exception as error:report['close_errors'].append(repr(error))
            try:report['reader_threads_remaining']=CTX.close_streams()
            except Exception as error:report['close_errors'].append(repr(error))
        try:
            report['source_after']=source_inventory(a.source_root,profile['source_sha256'])
            report['process_after']=process_provenance(a.server_pid,a.source_root,a.port,profile['environment'])
            if report.get('process_before')!=report['process_after']:
                raise ValueError('server process/provenance changed during checks')
        except Exception as error:
            report['close_errors'].append(repr(error))
        if report['close_errors'] or report.get('reader_threads_remaining'):
            report['status']='FAIL'
        report['seconds']=time.monotonic()-started;save()
    print(json.dumps({k:v for k,v in report.items() if k in ('status','error','seconds','history_and_lifecycle_checks','all_replay_selected_ids_equal','close_errors')}))
    return int(report['status']!='MEASURED')


if __name__=='__main__':raise SystemExit(main())
