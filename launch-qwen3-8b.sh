#!/usr/bin/env bash
set -euo pipefail
# Keep E4S's required GCC module; override only this process's runtime libraries.
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module load gcc/11.4 cuda/13.2 mvapich/5.0.1-plus-cuda13.2 e4s/26.06/mvapich-5.0.1-zfp-cuda13.2
export LD_LIBRARY_PATH="/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
PYTHON=/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python
"$PYTHON" - <<'PY'
import socket, subprocess
with socket.socket() as s:
    s.bind(('127.0.0.1', 8001))
processes = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
if processes:
    raise SystemExit('GPU 0 has active compute processes; inspect before launching: ' + processes)
import triton, torch, vllm
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('Versions:', triton.__version__, torch.__version__, vllm.__version__, flush=True)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
exec /packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.30 --max-model-len 4096
