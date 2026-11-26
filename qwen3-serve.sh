#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ${1:-} != start ]]; then
  exec /usr/bin/python3 qwen3_manage.py "$@"
fi

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module load gcc/11.4 cuda/13.2 mvapich/5.0.1-plus-cuda13.2 e4s/26.06/mvapich-5.0.1-zfp-cuda13.2
export LD_LIBRARY_PATH="/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib:${LD_LIBRARY_PATH:-}"
# In Ray 2.55, disabling cluster discovery makes local IP resolution preserve loopback.
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0 RAY_USAGE_STATS_ENABLED=0
export PYTHONUNBUFFERED=1
unset RAY_ADDRESS
exec /packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python qwen3_manage.py "$@"
