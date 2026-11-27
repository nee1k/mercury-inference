# Two-GPU agent serving on Mercury

This study replays multi-turn agent sessions against two independent Qwen3-32B BF16 replicas, one per GPU. It compares round-robin routing, session affinity, FP8 KV storage and native host-memory offload. Only the study endpoints receive replay traffic.

| Config | Routing | KV storage | Full replay | First preemption |
|---|---|---|---|---|
| C1 | Round robin | Default BF16 | Complete | [48 sessions](results/C1/sweep-20261126T202631.742740Z.json) |
| C2 | Balanced session affinity | Default BF16 | Pending. Dry run passed. | Pending |
| C3 | Balanced session affinity | FP8 | Pending | Pending |
| C4 | Balanced session affinity | Default BF16 plus native host offload | Pending | Pending |

[Measured results and preemption thresholds](results/summary.md) · [Study decisions](STORY.md) · [GPU and runtime preflight](results/preflight/occupancy.json)

## C2 dry run

All eight turns completed with usage, and both replicas reported every required metric. Two sessions stayed on each replica for both turns. Sampled running requests peaked at two per replica and neither replica preempted. Both replicas and the router exited before the C1 rerun. [Dry-run result](results/C2/dry-run-S4-T2-20261126T202609.599273Z/result.json) · [Cleanup](results/C2/released-20261126T202631.383218Z.json)

## Current C1 full sweep

| Sessions | Turns 2–8 p95 TTFT (s) | End-to-end p95 (s) | Preemptions | Prefix hit rate | Peak KV | Peak waiting | Peak running on GPUs 2 / 3 | Adjacent turns that switched replicas | Evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 1.551 | 10.790 | 0 | 67.8% | 29.1% | 0 | 5 / 6 | 24/56 | [Result](results/C1/full-S8-T8-20261126T202940.659391Z/result.json) |
| 16 | 1.877 | 18.484 | 0 | 66.6% | 73.8% | 0 | 11 / 12 | 48/112 | [Result](results/C1/full-S16-T8-20261126T203053.918470Z/result.json) |
| 24 | 4.637 | 32.947 | 0 | 36.4% | 98.0% | 3 | 15 / 16 | 79/168 | [Result](results/C1/full-S24-T8-20261126T203237.179455Z/result.json) |
| 32 | 17.541 | 58.308 | 0 | 23.0% | 99.9% | 7 | 21 / 22 | 98/224 | [Result](results/C1/full-S32-T8-20261126T203504.619335Z/result.json) |
| 48 | 53.140 | 92.681 | 4 | 16.6% | 100.0% | 19 | 28 / 33 | 155/336 | [Result](results/C1/full-S48-T8-20261126T203839.705224Z/result.json) |

All 1,024 turns completed. Queueing first appeared at 24 sessions as the KV cache approached capacity. At 48 sessions, nearly half of adjacent turns switched replicas, and the prefix hit rate had fallen to 16.6%. This is the round-robin baseline for measuring how balanced session affinity changes cache reuse and later-turn latency. [Full sweep](results/C1/sweep-20261126T202631.742740Z.json) · [24-session result](results/C1/full-S24-T8-20261126T203237.179455Z/result.json) · [48-session result](results/C1/full-S48-T8-20261126T203839.705224Z/result.json)

Each replica reported 61.03 GiB of model memory at load, 251,104 GPU KV tokens, and an estimated 7.66 concurrent requests at a 32,768-token context. Both replicas and the router exited. [GPU 2 readiness](results/C1/gpu2-20261126T202631.742740Z-ready.json) · [GPU 3 readiness](results/C1/gpu3-20261126T202631.742740Z-ready.json) · [Cleanup](results/C1/released-20261126T204417.342512Z.json)

## Historical C1 dry run

| Sessions × turns | Turn 1 TTFT p50 / p95 | Later TTFT p50 / p95 | End-to-end p95 | Preemptions | Prefix hit rate | Peak KV | Errors | Final context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 × 2 | 0.369 / 0.371 s | 0.435 / 0.437 s | 6.762 s | 0 | 24.3% | 2.7% | 0 / 4 | 5,642–5,751 tokens |

Both sessions switched replicas between turns. The local tokenizer confirmed the shared system prompt and tool sample at 2,048 tokens each and the task sample at 1,024 tokens. The dry run confirms that the workload and telemetry work. The full sweep is needed to find the preemption threshold. [Metrics summary](results/summary.md) · [Per-turn traces](results/C1/dry-run-S2-T2-20261126T052917.085819Z/turns.jsonl) · [Workload calibration](results/C1/dry-run-S2-T2-20261126T052917.085819Z/run.json)

Each replica reported a 251,104-token KV cache and an estimated 7.66 concurrent requests at a 32,768-token context. Weights loaded in 53.83 seconds on each. vLLM's context-based concurrency figure is a capacity estimate. [GPU 2 startup](results/C1/gpu2-ready.json) · [GPU 3 startup](results/C1/gpu3-ready.json)

The router and both replicas stopped. GPUs 2 and 3 returned to idle with no compute PIDs and 5 MiB and 0 MiB used, respectively. [Cleanup result](results/C1/released-20261126T052938.924630Z.json)

## Historical C1 full sweep

These runs used the earlier harness. The current sweep tracks replica changes per session and derives its paper values from the rerun.

| Config | Run | Sessions | Turn 1 TTFT p50 / p95 | Later TTFT p50 / p95 | End to end p95 | Preemptions | Prefix hit rate | Peak KV | Peak waiting | Peak running per replica | Errors / attempted | Final context | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C1 | full | 8 | 0.619 / 0.770 | 0.466 / 1.992 | 11.284 | 0 | 68.3% | 33.3% | 0 | GPU 2 (5) / GPU 3 (6) | 0 / 64 | 19,657–19,911 | [complete](results/C1/full-S8-T8-20261126T053812.727870Z/run.json) · [turns](results/C1/full-S8-T8-20261126T053812.727870Z/turns.jsonl) · [metrics](results/C1/full-S8-T8-20261126T053812.727870Z/metrics.jsonl) |
| C1 | full | 16 | 1.004 / 1.029 | 0.689 / 1.985 | 18.094 | 0 | 67.3% | 72.8% | 0 | GPU 2 (11) / GPU 3 (12) | 0 / 128 | 19,873–19,911 | [complete](results/C1/full-S16-T8-20261126T053929.020067Z/run.json) · [turns](results/C1/full-S16-T8-20261126T053929.020067Z/turns.jsonl) · [metrics](results/C1/full-S16-T8-20261126T053929.020067Z/metrics.jsonl) |
| C1 | full | 24 | 1.129 / 1.488 | 0.589 / 3.320 | 29.231 | 0 | 57.8% | 99.7% | 3 | GPU 2 (17) / GPU 3 (18) | 0 / 192 | 19,756–19,911 | [complete](results/C1/full-S24-T8-20261126T054109.956456Z/run.json) · [turns](results/C1/full-S24-T8-20261126T054109.956456Z/turns.jsonl) · [metrics](results/C1/full-S24-T8-20261126T054109.956456Z/metrics.jsonl) |
| C1 | full | 32 | 1.630 / 2.012 | 1.125 / 19.879 | 58.340 | 0 | 19.7% | 99.7% | 8 | GPU 2 (22) / GPU 3 (23) | 0 / 256 | 19,483–19,911 | [complete](results/C1/full-S32-T8-20261126T054324.907749Z/run.json) · [turns](results/C1/full-S32-T8-20261126T054324.907749Z/turns.jsonl) · [metrics](results/C1/full-S32-T8-20261126T054324.907749Z/metrics.jsonl) |
| C1 | full | 48 | 2.104 / 2.970 | 1.866 / 53.689 | 95.962 | 1 | 16.7% | 99.7% | 20 | GPU 2 (29) / GPU 3 (35) | 0 / 384 | 19,444–19,911 | [complete](results/C1/full-S48-T8-20261126T054700.400354Z/run.json) · [turns](results/C1/full-S48-T8-20261126T054700.400354Z/turns.jsonl) · [metrics](results/C1/full-S48-T8-20261126T054700.400354Z/metrics.jsonl) |

The first observed preemption was at S=48, on GPU 3. All 1,024 turns completed successfully. Queueing first appeared at S=24, where sampled KV use reached 99.7%. The scheduler can defer new inputs that do not fit, which explains why near-full KV and queueing can appear before any running request is preempted. [Full results](results/summary.md) · [S=48 telemetry](results/C1/full-S48-T8-20261126T054700.400354Z/metrics.jsonl) · [Scheduler source](results/preflight/scheduler-policy.txt)

At higher load the prefix hit rate fell and later-turn p95 TTFT rose. The sweep changed load while keeping round-robin routing, so the observed slowdown includes queueing and changed cache reuse. C2 will provide the routing comparison. Both replicas and the router stopped and the selected GPUs returned to their pre-run idle memory baselines. [Full results](results/summary.md) · [Cleanup result](results/C1/released-20261126T055238.169142Z.json)

## Prefix-cache measurements

`request_hit_pct` is the percentage of successful prompt tokens served from cache. The replay sums the cached tokens in final stream usage across successful turns and divides by their total prompt tokens. A missing or null usage detail contributes zero cached tokens. Earlier traces lack this measurement. The counter-based `prefix_hit_pct` uses the two replicas’ hit and query deltas. In vLLM 0.21.0, repeat lookups for waiting requests can give those requests extra weight. [Upstream fix and explanation](https://github.com/vllm-project/vllm/pull/48860)

For C4, the request metric includes tokens loaded from the host tier. The scheduler passes the connector’s external token count into prefill statistics. vLLM adds local and external cached tokens, and chat streaming writes that total into usage. The aggregate alone cannot identify the host share. [v0.21.0 scheduler](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/core/sched/scheduler.py#L580-L614) · [Cached-token assignment](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/metrics/stats.py#L266-L273) · [Streaming usage](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/entrypoints/openai/chat_completion/serving.py#L949-L952)

## Method

- Each session sees the same system prompt, a seeded private task and a seeded tool output per follow-up turn. The replay measures text lengths with the local model tokenizer, appends the actual prior answer and captures usage from the stream. The synthetic workload measures serving behavior.
- The same session IDs and input seed are used at a given concurrency across configurations. Private text changes between concurrency levels. The shared system prefix can remain warm between levels within a configuration.
- Timing begins at the router request. TTFT uses the first generated token and end-to-end latency includes the stream terminator. Cache gauges and counter changes come from both replicas. Final context is the last prompt plus completion token count.
- Each full sweep uses 8, 16, 24, 32 and 48 concurrent sessions, with eight turns per session. The report gives peak running requests separately for each replica so routing balance can be compared with KV pressure. For round robin it also counts successful adjacent turn pairs that switched replicas. Sticky routing assigns each new session to the replica holding fewer sessions for that replay. Each run records the final assignments and the number of distinct sessions served per replica. [Sweep scope and predictions](STORY.md)
- The controller checks both selected GPUs before launching, rechecks each immediately before startup, and verifies that both release their memory after the run. The cached snapshot is pinned and launches are offline. Local result files and PID records are gitignored.

## Run a checkpoint

On Mercury, run one approved full configuration at a time from the repository root in the E4S Python environment. Send its output to a background log and monitor it while it runs.

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  -B -u agent-study/run_config.py C2 full
```

`-B` skips bytecode caches. `-u` flushes progress promptly. The controller defaults to GPUs 2 and 3 when both pass the idle sample and picks three free study ports. The optional `--gpus` takes two preferred indices. The optional `--ports` takes the two replica ports followed by the router port. For C3 and C4 the controller also runs the existing model-size study's protocol checks against one verified study replica.

Use `python -B agent-study/summarize.py` to rebuild the local summary from raw replay results. Current runs include a `result.json` with raw trace and metric links. Full-sweep checkpoint files link the per-load results, readiness, protocol checks where applicable, and cleanup. The controller refreshes [paper values](results/paper_values.json) using the current sweep evidence. The final two-GPU recommendation is pending the full measurements.
