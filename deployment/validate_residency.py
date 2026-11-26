#!/usr/bin/env python3
"""Validate co-residency; explicitly distinguish partial and full nine-model runs."""
import argparse
import concurrent.futures
import datetime as dt
import json
import time
import threading
import manage as m


def request(model,barrier=None):
    if barrier is not None: barrier.wait(timeout=30)
    start=time.perf_counter()
    body={'model':model['repo'],'messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],'temperature':0,'max_tokens':64}
    if model['repo'].startswith('Qwen/'): body['chat_template_kwargs']={'enable_thinking':False}
    try:
        response=m.api(model,'/v1/chat/completions',body)
        if not response['body']['choices'][0]['message'].get('content'): raise RuntimeError('Empty content')
        return {'key':model['key'],'seconds':time.perf_counter()-start,'response':response,'error':None}
    except Exception as e:
        return {'key':model['key'],'seconds':time.perf_counter()-start,'error':str(e)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available-only',action='store_true')
    parser.add_argument('--monitor-seconds',type=int,default=600)
    args=parser.parse_args()
    status=m.status()
    ready=[m.model_for(row['key']) for row in status['models'] if row['ready'] and row['process_verified']]
    if len(ready)!=9 and not args.available_only: raise SystemExit('Nine verified endpoints required; use --available-only for an explicitly partial test')
    if not ready: raise SystemExit('No verified ready endpoints')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=m.ROOT/'results'/f'residency-{stamp}.json'
    result={'started_at':m.now(),'scope':'all-nine' if len(ready)==9 else 'partial','models':[x['key'] for x in ready],
            'requested_monitor_seconds':args.monitor_seconds,'before':status,'context':[],'rounds':[],'telemetry':[],'health':[],'errors':[]}
    try:
        for model in ready:
            m.context_validation(model)
            result['context'].append(model['key'])
            m.save(path,result)
        for label,workers in [('one-per-endpoint',1),('concurrency4-round1',4),('concurrency4-round2',4),('concurrency4-round3',4)]:
            started=time.perf_counter()
            barrier=threading.Barrier(len(ready)*workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready)*workers) as pool:
                futures=[pool.submit(request,model,barrier) for model in ready for _ in range(workers)]
                responses=[f.result() for f in futures]
            result['rounds'].append({'label':label,'seconds':time.perf_counter()-started,'requests':responses,'inventory':m.gpu_inventory()})
            m.save(path,result)
            if any(r['error'] for r in responses): raise RuntimeError('Concurrent request error; stopping new load')
            print(label,'passed',len(responses),'requests',flush=True)
        started=time.monotonic()
        next_health=0
        while time.monotonic()-started<args.monitor_seconds:
            elapsed=time.monotonic()-started
            result['telemetry'].append(m.gpu_inventory())
            if elapsed>=next_health:
                for model in ready:
                    m.identity(model)
                    if not m.matches(m.load(m.state_path(model))): raise RuntimeError('Process identity lost: '+model['key'])
                result['health'].append({'timestamp':m.now(),'elapsed_seconds':elapsed,'all_selected_healthy':True})
                print('Health passed',len(ready),'endpoints;',round(elapsed),'seconds',flush=True)
                next_health+=30
            m.save(path,result)
            time.sleep(min(5,max(0,args.monitor_seconds-(time.monotonic()-started))))
        for model in ready: m.identity(model)
        result['monitor_elapsed_seconds']=time.monotonic()-started
        result['status']='passed' if len(ready)==9 else 'partial-passed'
    except BaseException as e:
        result['status']='failed'
        result['errors'].append({'type':type(e).__name__,'message':str(e)})
        raise
    finally:
        result['finished_at']=m.now()
        m.save(path,result)
        print('Saved',path,flush=True)

if __name__=='__main__': main()
