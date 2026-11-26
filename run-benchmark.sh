#!/usr/bin/env bash
set -euo pipefail
# Client-only tooling; does not launch, restart, or reconfigure an endpoint.
export LD_LIBRARY_PATH="/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib:${LD_LIBRARY_PATH:-}"
exec /packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  "$(dirname "$(readlink -f "$0")")/benchmark.py" "$@"
