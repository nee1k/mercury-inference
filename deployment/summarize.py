#!/usr/bin/env python3
"""Generate an observed deployment report; never imply nine-model completion."""
import json
from pathlib import Path
import re
import manage as m


def belongs(pid,state):
    seen=set()
    while pid and pid not in seen:
        seen.add(pid)
        p=m.process(pid)
        if not p: return False
        if p['pid']==state['pid'] and p['start_ticks']==state['start_ticks']: return True
        pid=p['ppid']
    return False


def main():
    status=m.status()
    rows=[]
    for row in status['models']:
        state=row.get('state')
        apps=[p for p in status['inventory']['processes'] if state and row['process_verified'] and belongs(int(p['pid']),state)]
        memory=sum(float(p.get('used_gpu_memory [MiB]',0)) for p in apps)
        row['observed_process_memory_mib']=memory if apps else None
        row['engine_processes']=apps
        details={}
        if state and state.get('log'):
            text=Path(state['log']).read_text()
            for key,pattern in {'model_memory_gib':r'Model loading took ([\d.]+) GiB',
                                'kv_cache_tokens':r'GPU KV cache size: ([\d,]+) tokens',
                                'engine_init_seconds':r'init engine .* took ([\d.]+) s',
                                'cuda_graph_memory_gib':r'Graph capturing finished in .* took ([\d.]+) GiB'}.items():
                match=re.search(pattern,text)
                if match: details[key]=float(match.group(1).replace(',',''))
        row['startup_observations']=details
        ready=row['ready'] and row['process_verified']
        flag='READY' if ready else 'ACCESS BLOCKED' if row['access']=='blocked' else 'NOT READY'
        rows.append(f"| {row['gpu']} | `{row['repo']}` | {row['port']} | {flag} | {memory:.0f}"+('' if apps else ' (no allocation attributed)')+' |')
    status['completion']='all-nine-ready' if all(x['ready'] and x['process_verified'] for x in status['models']) else 'partial'
    files=sorted((m.ROOT/'results').glob('residency-*.json'))
    residency=json.loads(files[-1].read_text()) if files else None
    if residency:
        status['latest_residency']={'file':str(files[-1]),'scope':residency['scope'],'status':residency.get('status','in-progress'),
            'models':residency['models'],'monitor_elapsed_seconds':residency.get('monitor_elapsed_seconds'),
            'requests':sum(len(r['requests']) for r in residency['rounds']),
            'request_errors':sum(bool(x['error']) for r in residency['rounds'] for x in r['requests']),
            'errors':residency['errors']}
    m.save(m.ROOT/'results/summary.json',status)
    lines=['# Observed Mercury deployment','',f"Snapshot: {status['timestamp']}",f"Completion: **{status['completion']}**",'',
           '| GPU | Model | Localhost port | Observed status | Process memory (MiB) |','|---|---|---|---|---|',*rows,'',
           'Readiness is distinct from full concurrent-load acceptance. See timestamped residency results for tested participants, rounds, context checks, and monitoring duration.','',
           '## GPU totals','']
    for gpu in status['inventory']['gpus']:
        lines.append(f"- GPU {gpu['index']}: {gpu['memory.used [MiB]']} MiB used, {gpu['memory.free [MiB]']} MiB free.")
    if residency:
        proof=status['latest_residency']
        lines+=['','## Latest co-residency check','',
                f"Scope: {proof['scope']}; participants: {len(proof['models'])}; status: {proof['status']}.",
                f"Concurrent-round requests: {proof['requests']}; request errors: {proof['request_errors']}.",
                f"Completed monitor duration: {proof['monitor_elapsed_seconds']} seconds.",
                f"Raw evidence: `{proof['file']}`.",
                'All nine must be available and pass the full test before nine-model acceptance can be claimed.']
    lines+=['','## Reproduce and resume','',
            'Use `~/mercury-inference/deployment/mercury status` for current observations. After authenticating locally with approved model access, rerun `mercury preflight`, then `mercury up`. Those commands preserve verified running endpoints.',
            '','Exact arguments, pinned revisions, process identity, logs, and startup times are retained under `state/`. No shared packages were changed.','']
    (m.ROOT/'results/summary.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
