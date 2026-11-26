#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, get_token
ROOT=Path(__file__).resolve().parent
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('key')
    args=parser.parse_args()
    report=json.loads((ROOT/'state/preflight.json').read_text())
    model=next(m for m in report['models'] if m['key']==args.key)
    if model['access']!='available': raise SystemExit('Checkpoint access blocked')
    required=model['weight_bytes']
    if shutil.disk_usage(Path.home()).free < required*1.2+10*2**30:
        raise SystemExit('Insufficient free cache storage with safety margin')
    patterns=model['weight_files']+['*.json','*.model','*.model.*','*.tiktoken','tekken*','*.txt','README.md','LICENSE*','*.jinja']
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    snapshot=snapshot_download(model['repo'],revision=model['revision'],allow_patterns=patterns,max_workers=4,token=get_token())
    state={'key':model['key'],'repo':model['repo'],'revision':model['revision'],'snapshot':snapshot,'started_at':start,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    target=ROOT/'state'/f"{model['key']}-download.json"
    temporary=target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state,indent=2)+'\n')
    temporary.replace(target)
    print(json.dumps(state),flush=True)
