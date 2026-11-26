# Qwen3-8B pilot harness

The user selected **Qwen/Qwen3-8B, BF16, tensor parallel 1, GPU 0 only**. Other proposed models and the precision pair are not selected for execution. This single-model pilot cannot establish model-size or precision tradeoffs.

## Current stage

Harness implemented and validated for the selected Qwen3-8B configuration. Syntax checks and hardware-free tests passed ([test evidence](results/test-harness.json)). The live dry run, all function checks, saved-result parser/gate tests and real regression gate passed ([dry-run evidence](results/qwen3-8b-bf16-tp1/dry-run-20261120T201240.785325Z/dry-run.json)). Its server was gracefully stopped and GPU memory release verified. **Await the user's go at the phase-2 checkpoint.** The full workload sweep and natural reasoning comparison have not run. The live production API is still unidentified and is never a benchmark target.

The cached snapshot is pinned in `configs/qwen3-8b-bf16-tp1.yaml`. Serving and benchmarking run offline. Missing files fail rather than download. Metadata checks read the selected repository's public API and configuration/index JSON, never weights.

## Teaching: commands and flags

New test command:

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python -m unittest discover -s scripts -p test_harness.py -v
```

- `-m unittest`: run the standard-library test runner.
- `discover`: discover test cases.
- `-s scripts`: search the scripts directory.
- `-p test_harness.py`: select this test-file pattern.
- `-v`: display each test result.

The pilot's server command is:

```bash
setsid nohup /packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/vllm serve \
  --config /storage/users/neelk/mercury-inference/model-size-study/configs/qwen3-8b-bf16-tp1.yaml \
  --host 127.0.0.1 --port PORT
```

Do not run this bare command: use the guarded launcher or dry-run script. `PORT` represents a checked free study port; it is not a literal argument.

- `setsid`: detach into a new session.
- `nohup`: survive terminal hangup; the launcher explicitly redirects input and output.
- `serve`: start a plain vLLM API server, not Ray Serve.
- `--config`: read the pinned offline model configuration.
- `--host 127.0.0.1`: bind to loopback only.
- `--port`: use the selected free port in the permitted study range.

The YAML flags select the cached `model` and `tokenizer`, the public `served-model-name`, immutable `revision`, BF16 `dtype`, one `tensor-parallel-size`, `distributed-executor-backend: mp`, fixed context/memory/seed, disabled remote code, automatic tool choice with `hermes`, and `qwen3` reasoning parsing. Prefix caching is left at its installed default. All flags are recorded in the phase-0 installed help.

The launcher unsets `RAY_ADDRESS`, sets offline HF/Transformers operation, pins PCI-ordered CUDA visibility, and prepends the tested GCC runtime paths only inside child processes. No modules/packages/global configuration are changed.

GPU checks use `nvidia-smi -q -x -i 0`: `-q` requests detailed state, `-x` selects parseable XML, and `-i 0` selects GPU 0. XML includes all listed process types. Sample for ten seconds; require no processes, memory below 1 GiB and zero utilization throughout. These checks do not reserve hardware against other users.

Benchmark flags:

- `--backend openai-chat`: use streamed chat requests.
- `--base-url` / `--endpoint /v1/chat/completions`: target only the verified study endpoint.
- `--model` / `--tokenizer`: use the pinned local tokenizer; `--served-model-name` supplies the API's model ID.
- `--dataset-name random`: synthesize random text inputs.
- `--random-input-len`: target content tokens after subtracting measured chat-template overhead.
- `--random-output-len`: request the shape's exact completion length.
- `--random-range-ratio 0`: disable random variation of input/output lengths.
- `--num-prompts`: use the protocol's prompt count for this concurrency.
- `--max-concurrency`: bound simultaneous requests.
- `--request-rate inf`: supply requests without an artificial arrival-rate limit.
- `--seed 0`: reproduce dataset generation.
- `--ignore-eos`: force the requested output length.
- `--temperature 0.6`, `--top-p 0.95`, `--top-k 20`: record the selected snapshot's sampling defaults explicitly.
- `--extra-body`: disable Qwen thinking in each benchmark request.
- `--percentile-metrics ttft,tpot,itl,e2el`: include every requested latency metric.
- `--metric-percentiles 50,95,99`: retain all required percentiles.
- `--save-result` / `--save-detailed`: save aggregate and per-request evidence.
- `--result-dir` / `--result-filename`: isolate each run's raw result.

## Pilot and later execution

Run the following from the Mercury SSH terminal after reviewing the command explanations:

```bash
cd /storage/users/neelk/mercury-inference/model-size-study
setsid nohup \
  /packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  -u scripts/dry_run.py \
  > results/qwen3-8b-dry-run.log 2>&1 < /dev/null &
```

The pilot runs hardware-free tests, checks the selected model's API metadata, checks GPU idleness, starts the offline server, verifies health/first completion and required startup metrics, checks all five functions, discards warmup, measures the chat shape at concurrency one and four, tests summary/gate parsing on saved data, and reruns the real provisional gate. It then gracefully stops its server, verifies GPU memory release and stops at the phase-2 checkpoint. A failed check stops progression and retains evidence. No full sweep or natural reasoning run starts automatically.

Its memory budget is 90% while the server is active. No lab notification has been sent. Every startup has a fresh idle check; other workloads are untouched.

Inspect `results/qwen3-8b-dry-run.log` and the timestamped `dry-run.json`. Logs and result paths are never reused for unrelated configurations. `results/<config>/server.log` is append-only, with each launch's byte offset retained.

Later interfaces (only after the corresponding checkpoint):

```bash
bash scripts/serve.sh qwen3-8b-bf16-tp1 0 8100
bash scripts/bench.sh qwen3-8b-bf16-tp1 8100 1
bash scripts/gate.sh http://127.0.0.1:8100 Qwen/Qwen3-8B BASELINE_MANIFEST
bash scripts/stop.sh qwen3-8b-bf16-tp1
```

The shell wrappers accept configuration/GPU/port or gate arguments exactly as documented in the study protocol. They are also usable through `bash` without executable permission changes. Run long operations with the same background/logging pattern. Never benchmark an unmanaged or production endpoint. The gate baseline is a saved benchmark `manifest.json` containing both required chat cells.

The natural runner accepts `<base-url> <model>`, sends prompts sequentially, compares thinking-off with model-default thinking, saves raw stream chunks/usage, caps output and marks censorship. Timing starts before the HTTP request; role-only chunks do not count as tokens. Reasoning fields or a closing `</think>` delimiter identify the first answer; split delimiters are supported. Streaming measurements resolve chunk arrival, not individual token production inside a chunk. Tool-call deltas are recorded, never executed. Provisional prompts are short and use a conservative context allowance; replacing them with long workloads requires exact token-budget validation first.

## Evidence and interpretation

Every benchmark cell records the version, revision, command, launch evidence, template overhead, actual input/output lengths and GPU observations. Unrelated GPU arrivals invalidate the cell and stop the recorded benchmark subprocess; they are never terminated. Missing/error/length-mismatched evidence cannot enter a summary or pass a gate.

Pilot and gate cells never count as sweep/frontier repeats. The three concurrency-16 frontiers maximize total parameters while minimizing GPUs and p95 end-to-end latency; parameter count is not measured quality. A new deferred configuration requires recomputation and repeats for any new frontier member. With only one selected configuration, no comparative model recommendation follows.

The regression gate is provisional and can fail due to real measurement noise. Preserve that failure and investigate; do not weaken its threshold to obtain a pass. Server load time includes startup through the first valid completion with cached weights, not metadata collection or the pre-launch version command.
