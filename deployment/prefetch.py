#!/usr/bin/env python3
"""Bounded checkpoint prefetch; does not start GPU processes."""
import concurrent.futures
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python'

def download(key):
    with (ROOT/'logs'/f'{key}-download.log').open('a') as output:
        result=subprocess.run([PYTHON,str(ROOT/'download.py'),key],stdout=output,stderr=subprocess.STDOUT)
    print(key,'download exit',result.returncode,flush=True)
    return result.returncode

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(download,['qwen3-32b','mistral-small31','qwen3-30b-a3b']))
    raise SystemExit(1 if any(results) else 0)
