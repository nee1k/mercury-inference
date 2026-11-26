#!/usr/bin/env python3
"""User-owned nine-endpoint deployment; never reclaims unrelated GPU processes."""
import argparse
import concurrent.futures
import contextlib
import csv
import datetime as dt
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
PYTHON='/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python'
VLLM=str(Path(PYTHON).with_name('vllm'))
MODELS=json.loads((ROOT/'models.json').read_text())
sys.path.insert(0,str(ROOT.parent))
from benchmark import stream_request


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)
def load(path): return json.loads(path.read_text()) if path.exists() else None
def state_path(m): return ROOT/'state'/f"{m['key']}.json"
def model_for(key): return next(m for m in MODELS if m['key']==key)
def run(args): return subprocess.check_output(args,text=True,timeout=20)
def api(m,path,body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{m['port']}"+path,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180 if body else 5) as response:
        raw=response.read()
        return {'status':response.status,'body':json.loads(raw) if raw else None}
def process(pid):
    try:
        p=Path('/proc')/str(pid)
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'ppid':int(fields[1]),'start_ticks':fields[19], 'uid':p.stat().st_uid,
                'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip(), 'state':fields[0]}
    except (OSError,ValueError,IndexError): return None

def matches(state):
    current=process(state['pid']) if state else None
    return bool(current and current['state']!='Z' and current['uid']==os.getuid() and current['start_ticks']==state['start_ticks'] and current['command']==state['command_line'])

def gpu_inventory():
    gpu=list(csv.DictReader(io.StringIO(run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,nounits']))))
    apps=list(csv.DictReader(io.StringIO(run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,nounits']))))
    return {'timestamp':now(),'gpus':[{k.strip():v.strip() for k,v in r.items()} for r in gpu],
            'processes':[{k.strip():v.strip() for k,v in r.items()} for r in apps]}

def identity(m):
    response=api(m,'/v1/models')
    info=next((x for x in response['body']['data'] if x['id']==m['repo']),None)
    if info is None or info.get('max_model_len',4096)!=4096:
        raise RuntimeError('Endpoint identity/context mismatch')
    api(m,'/health')
    return response

def adopt_external(m):
    if not m.get('external'): return
    identity(m)
    found=[]
    for pid in Path('/proc').iterdir():
        if not pid.name.isdigit(): continue
        proc=process(pid.name)
        if proc and proc['uid']==os.getuid() and f"serve {m['repo']} " in proc['command'] and f"--port {m['port']} " in proc['command']:
            found.append(proc)
    if len(found)!=1: raise RuntimeError('Cannot uniquely verify original endpoint process')
    p=found[0]
    state={'key':m['key'],'external':True,'pid':p['pid'],'start_ticks':p['start_ticks'],'command_line':p['command'], 'adopted_at':now()}
    save(state_path(m),state)

def ancestor_owned(pid):
    allowed=[s for m in MODELS if (s:=load(state_path(m))) and matches(s)]
    seen=set()
    while pid and pid not in seen:
        seen.add(pid)
        proc=process(pid)
        if not proc or proc['uid']!=os.getuid(): return False
        if any(proc['pid']==s['pid'] and proc['start_ticks']==s['start_ticks'] for s in allowed): return True
        pid=proc['ppid']
    return False

@contextlib.contextmanager
def gpu_lock(m):
    with (ROOT/'state'/f"gpu-{m['gpu']}.lock").open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def build_command(m,snapshot):
    cmd=[PYTHON,VLLM,'serve',snapshot,'--served-model-name',m['repo'],'--host','127.0.0.1','--port',str(m['port']),
         '--dtype','bfloat16','--tensor-parallel-size','1','--gpu-memory-utilization',str(m['memory']),
         '--max-model-len','4096','--max-num-seqs','4','--max-num-batched-tokens','4096']
    if m.get('text_only'): cmd+=['--language-model-only']
    if m.get('mistral'): cmd+=['--tokenizer-mode','mistral','--config-format','mistral','--load-format','mistral']
    return cmd

def start(m):
    if m.get('external'):
        adopt_external(m)
        print(m['key'],'existing endpoint retained',flush=True)
        return
    with gpu_lock(m):
        current=load(state_path(m))
        if current and matches(current):
            print(m['key'],'already running',current['pid'],flush=True)
            return
        preflight=load(ROOT/'state/preflight.json')
        entry=next(x for x in preflight['models'] if x['key']==m['key'])
        if entry['access']!='available': raise RuntimeError('Repository access blocked; authenticate and rerun preflight')
        download=load(ROOT/'state'/f"{m['key']}-download.json")
        if not download or download['revision']!=entry['revision']: raise RuntimeError('Pinned download required first')
        with socket.socket() as s: s.bind(('127.0.0.1',m['port']))
        inventory=gpu_inventory()
        gpu=next(g for g in inventory['gpus'] if int(g['index'])==m['gpu'])
        unexpected=[p for p in inventory['processes'] if p['gpu_uuid']==gpu['uuid'] and not ancestor_owned(int(p['pid']))]
        if unexpected: raise RuntimeError('Unmanaged GPU allocations: '+json.dumps(unexpected))
        # Do not profile a neighbor while another managed process is still starting.
        for neighbor in MODELS:
            if neighbor['gpu']!=m['gpu'] or neighbor['key']==m['key']: continue
            ns=load(state_path(neighbor))
            if ns and matches(ns): identity(neighbor)
        if float(gpu['memory.free [MiB]']) < float(gpu['memory.total [MiB]'])*m['memory']:
            raise RuntimeError('Free memory is below the requested per-process budget')
        cmd=build_command(m,download['snapshot'])
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=str(m['gpu'])
        env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        env['LD_LIBRARY_PATH']='/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib:'+env.get('LD_LIBRARY_PATH','')
        env.setdefault('MAX_JOBS','8')
        env['HF_HUB_OFFLINE']='1'
        env['TRANSFORMERS_OFFLINE']='1'
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        log=ROOT/'logs'/f"{m['key']}-{stamp}.log"
        save(ROOT/'results'/f"{m['key']}-{stamp}-before.json",inventory)
        with log.open('x') as output:
            child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
        proc=process(child.pid)
        if not proc: raise RuntimeError('Server exited before process identity capture')
        state={'key':m['key'],'pid':child.pid,'start_ticks':proc['start_ticks'],'command_line':proc['command'],
               'command':cmd,'gpu_uuid':gpu['uuid'],'log':str(log),'started_at':now(),'revision':entry['revision'],'external':False}
        save(state_path(m),state)
        print(m['key'],'started',child.pid,'log',log,flush=True)

def wait_ready(m,timeout=3600):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        state=load(state_path(m))
        if not matches(state): raise RuntimeError('Server process exited; inspect '+str((state or {}).get('log')))
        try:
            identity(m)
            state.setdefault('ready_at',now())
            state['last_health_check_at']=now()
            save(state_path(m),state)
            print(m['key'],'ready',flush=True)
            return
        except Exception: time.sleep(5)
    raise TimeoutError('Server readiness timeout; process left untouched')

def stop(m):
    if m.get('external'): raise RuntimeError('Original Qwen3-8B is protected; this manager will not stop it')
    with gpu_lock(m):
        state=load(state_path(m))
        if not matches(state): raise RuntimeError('No verified live managed process; no signal sent')
        # pidfd binds the signal to this exact process, avoiding PID-reuse races.
        fd=os.pidfd_open(state['pid'])
        try:
            if not matches(state): raise RuntimeError('Process identity changed; no signal sent')
            signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally: os.close(fd)
        state['stop_requested_at']=now()
        save(state_path(m),state)
        print(m['key'],'SIGTERM sent to verified API process',state['pid'])

def basic_validation(m):
    models=identity(m)
    body={'model':m['repo'],'messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],'temperature':0,'max_tokens':128}
    if m['repo'].startswith('Qwen/'): body['chat_template_kwargs']={'enable_thinking':False}
    started=time.perf_counter()
    chat=api(m,'/v1/chat/completions',body)
    content=chat['body']['choices'][0]['message'].get('content')
    if not content: raise RuntimeError('Empty chat response')
    streamed=stream_request(f"http://127.0.0.1:{m['port']}",{**body,'stream':True,'stream_options':{'include_usage':True}})
    if streamed['error']: raise RuntimeError(streamed['error'])
    result={'timestamp':now(),'key':m['key'],'models':models,'chat':chat,'stream':streamed,'seconds':time.perf_counter()-started,'inventory':gpu_inventory()}
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    save(ROOT/'results'/f"{m['key']}-{stamp}-validation.json",result)
    print(m['key'],'HTTP models/chat/stream passed; answer',repr(content),flush=True)
    return result

def context_validation(m):
    state=load(ROOT/'state'/f"{m['key']}-download.json")
    if m.get('mistral'):
        # Ask vLLM's renderer for its actual Mistral tokenization, avoiding mismatched HF templates.
        render=lambda text: api(m,'/v1/chat/completions/render',{'model':m['repo'],'messages':[{'role':'user','content':text}],'max_tokens':128})
        rendered=render('Please reply OK.\n'+'sample ' * 3900)
        raw=rendered['body']
        def find_ids(value):
            if isinstance(value,dict):
                for key in ('prompt_token_ids','token_ids','input_ids'):
                    if key in value and isinstance(value[key],list): return value[key]
                for child in value.values():
                    found=find_ids(child)
                    if found is not None: return found
            if isinstance(value,list):
                for child in value:
                    found=find_ids(child)
                    if found is not None: return found
            return None
        ids=find_ids(raw)
        if ids is None: raise RuntimeError('Unable to inspect server renderer token IDs')
        # Use completions token IDs to test the engine context budget exactly.
        ids=ids[:3968]
    else:
        from transformers import AutoTokenizer
        checkpoint=state['snapshot'] if state else m['repo']
        revision=None if state else next(x['revision'] for x in load(ROOT/'state/preflight.json')['models'] if x['key']==m['key'])
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,revision=revision,local_files_only=True)
        ids=tokenizer.encode('Please reply OK.\n'+'sample ' * 4100,add_special_tokens=False)[:3968]
    if len(ids)<3900: raise RuntimeError('Context fixture token count too short')
    response=api(m,'/v1/completions',{'model':m['repo'],'prompt':ids,'temperature':0,'max_tokens':128})
    used=response['body']['usage']['prompt_tokens']
    if used!=len(ids): raise RuntimeError('Prompt count differs from supplied tokens')
    result={'key':m['key'],'timestamp':now(),'prompt_tokens':used,'max_tokens':128,'response':response}
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    save(ROOT/'results'/f"{m['key']}-{stamp}-context-validation.json",result)
    save(ROOT/'results'/f"{m['key']}-context-validation.json",result)
    print(m['key'],'context validation passed',used,'+128',flush=True)

def status():
    result={'timestamp':now(),'inventory':gpu_inventory(),'models':[]}
    preflight=load(ROOT/'state/preflight.json') or {'models':[]}
    for m in MODELS:
        state=load(state_path(m))
        row={**m,'process_verified':matches(state),'state':state}
        try: row['endpoint']=identity(m); row['ready']=True
        except Exception as e: row.update(ready=False,error=type(e).__name__)
        row['access']=next((x['access'] for x in preflight['models'] if x['key']==m['key']),'unchecked')
        result['models'].append(row)
        print(m['key'],f"GPU {m['gpu']} port {m['port']}",'READY' if row['ready'] else 'NOT READY','access='+row['access'],flush=True)
    save(ROOT/'results/status.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','download','start','wait','stop','validate','context','status','report','up'])
    parser.add_argument('key',nargs='?',choices=[m['key'] for m in MODELS])
    args=parser.parse_args()
    if args.action=='preflight': return subprocess.run([PYTHON,str(ROOT/'preflight.py')],check=True)
    if args.action=='status': return status()
    if args.action=='report': return subprocess.run([PYTHON,str(ROOT/'summarize.py')],check=True)
    if args.action=='up':
        failures=[]
        with (ROOT/'state/up.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for m in MODELS:
                try:
                    if not m.get('external'):
                        row=next(x for x in load(ROOT/'state/preflight.json')['models'] if x['key']==m['key'])
                        if row['access']!='available':
                            failures.append({'key':m['key'],'reason':'repository access blocked'})
                            print(m['key'],'SKIPPED: repository access blocked',flush=True)
                            continue
                        ds=load(ROOT/'state'/f"{m['key']}-download.json")
                        if not ds or ds['revision']!=row['revision']:
                            subprocess.run([PYTHON,str(ROOT/'download.py'),m['key']],check=True)
                    start(m)
                    wait_ready(m)
                    basic_validation(m)
                except Exception as e:
                    failures.append({'key':m['key'],'reason':str(e)})
                    print(m['key'],'FAILED:',e,flush=True)
            save(ROOT/'results/up-failures.json',failures)
        status()
        if failures: raise SystemExit(2)
        return
    if not args.key: parser.error('Model key required')
    m=model_for(args.key)
    if args.action=='download': subprocess.run([PYTHON,str(ROOT/'download.py'),m['key']],check=True)
    elif args.action=='start': start(m)
    elif args.action=='wait': wait_ready(m)
    elif args.action=='stop': stop(m)
    elif args.action=='validate': basic_validation(m)
    elif args.action=='context': context_validation(m)

if __name__=='__main__': main()
