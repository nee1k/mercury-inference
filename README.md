# Mercury: Qwen3-8B endpoint

## Four-GPU Ray Serve endpoint

On Mercury, start the service in a persistent terminal:

```bash
~/mercury-inference/qwen3-serve.sh start
```

Use `http://127.0.0.1:8180/v1` as the OpenAI base URL and `Qwen/Qwen3-8B` as
the model. The router serves `GET /v1/models` and `POST /v1/chat/completions`
with regular JSON and streaming responses. The `X-Mercury-GPU` response
header identifies the physical GPU that handled the request.

```bash
~/mercury-inference/qwen3-serve.sh status
~/mercury-inference/qwen3-serve.sh release 2
~/mercury-inference/qwen3-serve.sh restore 2
~/mercury-inference/qwen3-serve.sh stop
```

Each H200 hosts a BF16 instance using the cached weights, a 4,096-token
context, and a 30% GPU-memory target. The launcher records the physical
GPU UUIDs and idle memory before starting. The dedicated Ray cluster
schedules against `mercury_gpu_0` through `mercury_gpu_3`.

`release` immediately removes the GPU from rotation and gives its active
requests up to 120 seconds to finish. You can lower this with `--timeout`.
If it times out, the GPU returns to rotation. A successful release means
the Serve app and vLLM processes have exited and VRAM is within 8 MiB of its
idle baseline, allowing for small driver allocations. `restore` checks for
compute processes and excess VRAM usage before loading the model, then waits
until it answers a health check.
Coordinate standalone CUDA jobs separately because they can claim a GPU
outside Ray's scheduling.

The API listens on loopback and management uses an owner-only Unix socket.
Run `status` to see the GPU UUIDs, compute PIDs, VRAM, routing state, and
active request counts. Worker and Ray logs are under
`/tmp/mercury-qwen3-$(id -u)/`.

## Single-GPU endpoint on port 8001

This deployment uses the existing E4S vLLM 0.21.0, PyTorch 2.11.0+cu130,
and Triton 3.6.0. No packages were installed or modified.

## Runtime fix

Keep the original GCC 11.4/CUDA 13.2/MVAPICH/E4S modules. Prepend
`/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib` to LD_LIBRARY_PATH only
inside the launch process. Triton needs GLIBCXX_3.4.30 **and** GLIBCXX_3.4.32;
GCC 12.5 passes the first requirement but fails the second. Imports were
verified with the actual loaded GCC 13.4 libstdc++.so.6.0.32 in /proc/self/maps.

## Launch

Run on Mercury in a terminal or your own persistent terminal session:

```bash
~/mercury-inference/launch-qwen3-8b.sh > ~/mercury-inference/logs/qwen3-8b.log 2>&1
```

The script refuses to start if port 8001 cannot be bound or GPU 0 already
has compute processes. This check is a snapshot, not a GPU reservation;
coordinate concurrent deployments with the lab.

Equivalent vLLM command after loading the original modules:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib:$LD_LIBRARY_PATH" \
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.30 --max-model-len 4096
```

## Validate

```bash
python3 ~/mercury-inference/validate-qwen3-8b.py
```

Validation records both HTTP responses, request duration, token usage,
and nvidia-smi memory/process observations under `results/`. The chat smoke
test disables Qwen thinking to obtain a short answer; this is a validation
request, not a benchmark. Run locally with GPU/localhost access.

The endpoint binds 127.0.0.1:8001 only. Port 8006 was already listening and
was left untouched; its owner was not established.

## Verified run: 2026-10-22

- Endpoint: http://127.0.0.1:8001; API PID 2565132, engine PID 2565787.
- API startup banner: 09:59:04 PDT; ready: 10:03:31 PDT (4 min 27 sec,
  excluding launcher preflight/imports). This first run included downloading weights.
- Model loading including download: 145.90 seconds; reported model memory: 15.27 GiB.
- Safetensors loading: 15.38 seconds; torch.compile: 22.06 seconds.
- Engine profile/KV-cache/warmup stage: 79.92 seconds, including compilation.
- CUDA graphs: 0.65 GiB; KV cache capacity: 172,016 tokens.
- vLLM estimated concurrency at 4096 tokens/request: 42x; this is a memory
  capacity estimate, not a measured throughput or concurrency guarantee.
- GPU 0 total memory after validation: 43,673 MiB (42.65 GiB).
  Engine process: 43,662 MiB. GPUs 1–3: 0 MiB.
- GET /v1/models: HTTP 200, model Qwen/Qwen3-8B.
- POST /v1/chat/completions: HTTP 200, answer `4.`, 23 prompt tokens and
  3 completion tokens, 0.889 seconds end-to-end for this single smoke test.
- First inference logged a Triton JIT compilation for the slot-mapping kernel;
  do warmup before collecting benchmark measurements.

The server was left running in the agent's execution session. This is not a
system-managed service and has no automatic restart. For future runs, use the
launcher in a persistent terminal/session appropriate to lab policy. Do not
launch a second copy while this endpoint is active. To stop this particular
run, first verify that PID 2565132 is still your Qwen3-8B API server, then send
it SIGTERM (`kill -TERM 2565132`); never use a broad process-name kill command.

## Benchmark pilot

See [PILOT.md](PILOT.md) for the dataset contract, preparation and benchmark
commands, safety gates, metric definitions, and report interpretation. The pilot
uses the existing endpoint without changing it. Real workload measurements await
the lab's held-out inputs, schema, and reference outputs.

## Nine-model residency deployment

See [deployment/README.md](deployment/README.md) for the nine-model manifest,
per-model localhost APIs, access prerequisites, launch/status/stop controls,
and co-residency validation. The live deployment status is recorded separately
from the original Qwen3-8B pilot results.
