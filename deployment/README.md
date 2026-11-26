# Mercury nine-endpoint deployment

All APIs bind 127.0.0.1. These are nine independent vLLM processes sharing four
GPUs, not a routing gateway or compute-isolated service. The manifest expresses
initial tested-or-pending budgets; `results/status.json` is the observed status.
Do not equate a manifest entry with a running endpoint.

| Key | Model/API model ID | GPU | Port | Initial memory fraction |
|---|---|---:|---:|---:|
| qwen3-8b | Qwen/Qwen3-8B | 0 | 8001 | 0.30, retained original |
| qwen3-4b | Qwen/Qwen3-4B | 0 | 8002 | 0.13 |
| qwen3-14b | Qwen/Qwen3-14B | 0 | 8003 | 0.27 |
| qwen3-32b | Qwen/Qwen3-32B | 1 | 8004 | 0.55 |
| gemma3-12b | google/gemma-3-12b-it | 1 | 8005 | 0.26 |
| gemma3-27b | google/gemma-3-27b-it | 2 | 8007 | 0.48 |
| llama31-8b | meta-llama/Llama-3.1-8B-Instruct | 2 | 8008 | 0.22 |
| mistral-small31 | mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 3 | 8009 | 0.42 |
| qwen3-30b-a3b | Qwen/Qwen3-30B-A3B | 3 | 8010 | 0.48 |

Port 8006 is an existing unrelated listener and is never touched. Budgets sum to
0.70/0.81/0.70/0.90 of each GPU's total memory. Actual fit and useful concurrency
require validation. Weights/KV cache stay BF16/auto; no automatic quantization,
CPU offload, or changes to GPU placement occur.

## Access and downloads

Authenticate using your own Mercury terminal and an account with accepted access
to Gemma/Llama repositories. Do not put tokens in scripts, chat, logs, or shell
command arguments. Use the locally installed Hugging Face login interface in your own terminal:

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/hf auth login
```

After login/access approval, rerun `mercury preflight` and `mercury up` below.

```bash
~/mercury-inference/deployment/mercury preflight
~/mercury-inference/deployment/mercury download qwen3-4b
```

Preflight resolves/pins immutable revisions, checks config and weight access,
and records checkpoint sizes, architecture metadata, license, and dependency
versions. `state/preflight.json` contains no authentication secrets. HTTP 401/403
is an access blocker; it is not permission to substitute another checkpoint.
Downloads select the required safetensors format and tokenizer/config artifacts,
not duplicate formats. The storage free-space check cannot infer every site's
NFS quota; filesystem quota/write failures are recorded as deployment failures.
Shared packages are never changed.

## Launch and inspect

```bash
~/mercury-inference/deployment/mercury up
~/mercury-inference/deployment/mercury status
~/mercury-inference/deployment/mercury report
```

`up` retains and verifies the original Qwen3-8B, skips access-blocked repositories,
and downloads/starts/validates the others sequentially. A partial deployment exits
with code 2 and records reasons under `results/up-failures.json`. Repeating `up`
reuses live managed processes and matching pinned downloads. Rerun preflight after
fixing model access, then rerun `up` to fill missing endpoints.

To control one new endpoint:

```bash
~/mercury-inference/deployment/mercury start qwen3-14b
~/mercury-inference/deployment/mercury wait qwen3-14b
~/mercury-inference/deployment/mercury validate qwen3-14b
~/mercury-inference/deployment/mercury context qwen3-14b
```

`start` requires the pinned download. Exact commands, GPU UUID, API PID and start
identity, revision, log filename, start/readiness timestamps are in `state/<key>.json`.
Readiness polling permits 60 minutes after launch; timeout leaves the process
untouched. Inspect logs before retrying. Detached processes survive the invoking
terminal, but this is not a system service and has no automatic restart.

Future managed launches default `MAX_JOBS=8` for native kernel compilation;
the first Qwen3-30B-A3B startup began before this cap and compiled its FlashInfer
CUTLASS kernels using the package default. No shared installation was changed.

New processes use the existing E4S Python/vLLM and isolated GCC 13.4 library paths,
TP=1, BF16, max context 4096, max sequences 4, and max batched tokens 4096. They
load pinned snapshots offline. CUDA visibility uses numeric IDs with PCI ordering
because this installed vLLM version rejects UUID-valued CUDA_VISIBLE_DEVICES;
the corresponding UUID is retained in each launch record. Gemma/Mistral use `--language-model-only`; Mistral
also uses its native tokenizer/config/load modes. Existing Qwen3-8B is unchanged.

Per-GPU launch locks prevent simultaneous managed startups on one GPU. A fresh
GPU/process inventory rejects unknown allocations, including those owned by your
account. Verified managed neighbors must be healthy before a new startup. Startup
assumes neighbors remain idle through profiling: coordinate external clients and
lab use; locks do not reserve hardware against other users or API clients.

## Stop and recovery

```bash
~/mercury-inference/deployment/mercury stop qwen3-14b
```

Stop verifies ownership, API command, and kernel process-start identity, then uses
a pidfd to send SIGTERM only to that API process. It never uses a broad kill,
SIGKILL, or another user's process. Wait for graceful exit/GPU release before
starting again. The original Qwen3-8B is protected and cannot be stopped by this
manager. Failed budgets do not cause neighbor termination, automatic memory
rebudgeting, precision changes, or a fallback installation.

## Co-residency acceptance

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  ~/mercury-inference/deployment/validate_residency.py
```

Requires all nine verified endpoints. It checks long context per model, then one
simultaneous request per endpoint, followed by three rounds of four concurrent
requests per endpoint, then monitors health every 30 seconds and GPU memory every
five seconds for ten minutes. A request error stops new validation load. Context
tests submit tokenizer-counted prompt IDs with a 128-token output allowance,
approaching the 4096-token context limit; these are engine-capacity tests, not
scientific quality evaluations. Normal text/chat is separately smoke-tested.

`--available-only` allows an explicitly **partial** co-residency test; passing it
never establishes nine-model completion. Results retain scope, exact participants,
raw responses, timing, errors, context checks, health observations, and GPU samples.
The default monitor duration is 600 seconds. No benchmark quality claims follow
from smoke tests, and residence does not guarantee good concurrent throughput.

## API usage

```bash
curl -sS http://127.0.0.1:8002/v1/models
curl -sS http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

Use the matching model ID and port from the table. Qwen supports the
`enable_thinking` option; omit it for other families. Add `"stream":true` and use
`curl -N` for streaming. Maximum input plus output is 4096 tokens. Image inputs are
disabled. Use an SSH tunnel for remote clients, rather than binding public ports.

## Tests

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  -m unittest discover -s ~/mercury-inference/deployment -v
```

These fixtures verify process/ownership guards, protected original service,
manifest invariants, and generated commands without changing live services.
