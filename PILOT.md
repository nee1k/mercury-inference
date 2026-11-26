# Qwen3-8B pilot runbook

This tooling benchmarks the existing endpoint; it does not launch, restart,
terminate, or tune a server. No new packages are required. `run-benchmark.sh`
uses the existing E4S Python and a process-local GCC 13.4 runtime override.

## Current baseline and scope

The verified endpoint is `http://127.0.0.1:8001`, Qwen/Qwen3-8B, BF16,
TP=1, context 4096, GPU-memory utilization 0.30, on GPU 0. The 2026-10-22
inventory measured 43,673 MiB (42.65 GiB) allocated on GPU 0; GPUs 1–3
had no compute allocations. This is a timestamped observation, not a reservation.
The older ~128.4 GiB/GPU allocation snapshot is historical. A currently active
four-GPU Qwen3-235B deployment has not been verified.

This pilot measures the existing configuration. Larger models, engine tuning,
SGLang/Ollama comparisons, cold starts, service lifecycle changes, and nine-endpoint
residency are later experiments. Keep their results separate from this baseline.

## Supply held-out lab inputs

Prepare a self-contained JSON Schema and a UTF-8 JSONL file. Each nonblank line:

```json
{"id":"paper-001","input_text":"Scientific text to extract from...","reference_output":{"field":"reference value"}}
```

IDs must be unique nonempty strings. `input_text` must be a string and
`reference_output` a JSON object valid under the schema. The schema may use local
fragment references, but external references are rejected. Nonstandard NaN and
Infinity values are rejected. Choose and freeze the held-out set before comparing
configurations; do not tune prompts on those reference answers.

The bundled `fixtures/` files are software fixtures only. They provide no evidence
of scientific extraction quality. Real workload runs require your lab dataset,
schema, reference answers, and permission to use them.

The fixed prompt includes the schema and input text, requests only a JSON object,
uses temperature 0, and disables thinking. No constrained-decoding option is
added. Exact agreement ignores object key order, preserves array order, and
compares value types strictly. Transport errors and truncated outputs fail all
reported quality rates. Raw outputs remain available for later task-specific review.

## Prepare without sending requests

```bash
~/mercury-inference/run-benchmark.sh \
  --dataset /absolute/path/lab-heldout.jsonl \
  --schema /absolute/path/extraction-schema.json \
  --prepare-only
```

Preparation uses only the cached tokenizer at revision
`b968826d9c46dd6066d109eabc6255188de91218`; it does not download a model or contact
the endpoint. `--revision` can select another locally cached checkpoint, but it
must match the serving checkpoint. The API identifies the model, not its exact
revision: corroborate revision with the endpoint's launch provenance. The current
launcher was started without an explicit revision; the recorded SHA is the local
cache revision associated with that deployment, not an API-attested revision.

Inspect the printed run directory's `workload.json` and `exclusions.json`.
Rendered prompt lengths include schema and chat-template overhead:

- Short: ≤512 tokens.
- Medium: 513–1536 tokens.
- Long: >1536 tokens.

An example/output-cap pair is excluded when input plus output cap exceeds 4096.
Documents are never silently truncated. Missing groups are reported, not filled
with synthetic examples. Token-count mismatches between tokenizer and server usage
are counted in reports and require investigation before interpreting length groups.

## Initial measured pilot

Once inputs are ready, check lab use and run from a persistent terminal on Mercury:

```bash
~/mercury-inference/run-benchmark.sh \
  --dataset /absolute/path/lab-heldout.jsonl \
  --schema /absolute/path/extraction-schema.json \
  --concurrency 1 4
```

Defaults: three repetitions, output caps 128 and 512, seed 20261022,
180-second request timeout, output root `~/mercury-inference/results/pilot/`.
Omitting `--concurrency` defaults to 1. Each output cap gets five excluded warmup
requests per present length group at concurrency 1. Each measured stage traverses
all eligible examples once; each repetition uses a recorded deterministic shuffle.
Concurrency is sorted ascending. Workers issue their next request after the
previous one finishes, so this is a closed-loop benchmark, not an arrival-rate test.
The context length is fixed; caps are maximums, not forced output lengths.

For N examples fitting both caps, the default 1-and-4 pilot sends 12N measured
requests plus up to 30 warmup requests. Inspect preparation output first to estimate
the run size. Prefix-cache state is not reset: repeated examples can benefit from
caching. Results describe a warmed, cache-enabled service.

Concurrency 8, 16, or 32 requires an actually coordinated test window and the
explicit `--coordinated-window` flag. The flag records an assertion; it does not
reserve GPUs or arrange a window. Do not use it as a substitute for coordination.
GPU/process inventories are captured before the run, before every concurrency/cap
stage, and afterward. Record overlapping researchers' traffic in your interpretation;
the endpoint-wide metrics cannot attribute it to this client.

The runner stops issuing requests on reported OOM, endpoint unavailability/timeout,
three consecutive request errors, or >5% errors after at least 20 responses in a
stage. Already issued requests finish or reach their deadlines. Quality failures
and truncations are reported separately and do not trigger transport-error stops.
Ctrl-C stops further scheduling and allows in-flight work to wind down. No process
kill or server restart is performed. Read `status.json`; a stopped or interrupted
run is not a completed baseline.

## Results and interpretation

Every run has a unique directory. Inputs, schema, prompt/config hashes, source
snapshots, software versions, environment, inventories, raw request results, and
sampled telemetry are preserved. Existing smoke-test results are untouched.

`report.md` is the compact report; `summary.json` includes per-repetition and
per-length P50/P95 latency, TTFT, TPOT, decode time, quality rates, errors,
truncations, request throughput, token throughput, and sample counts. Runs with
fewer than 100 successful requests per group/condition are flagged as small samples.
P95 uses linear interpolation. There are no predetermined acceptance thresholds.

TTFT ends at the first nonempty content chunk. Decode time ends at the last content
chunk, excluding usage trailers; TPOT divides that interval by completion tokens
minus one when counts exist. These are client observations; chunks are not tokens.
End-to-end latency includes stream completion and usage trailers. Throughput uses
successful completions divided by measured stage wall time, including failures;
output token throughput is unavailable if any successful response lacks usage.
Per-length latency/quality groups share a mixed workload, so no per-length
throughput is inferred.

GPU/metrics sampling targets one-second intervals; slow or failed probes can reduce
cadence. Peak GPU memory is a sampled peak, not a guaranteed absolute peak. Raw
Prometheus snapshots preserve available queue/prefill/decode histograms; report
series summarize their observed first/last/min/max values. They are endpoint-wide
and may include other traffic. Client TTFT cannot separate queueing from prefill;
do not substitute subtraction-based estimates for unavailable server timings.

To regenerate reports from preserved artifacts:

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  ~/mercury-inference/report.py /absolute/path/to/run-directory
```

Select lab quality and latency targets after inspecting these results. Recommend
next experiments from quality/latency/throughput tradeoffs; do not interpret the
single model pilot as validating other models or simultaneous residency.

## Software checks

```bash
/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python \
  -m unittest discover -s ~/mercury-inference -v
```

Mocked transport tests exercise the whole run/report lifecycle without model traffic.
Set `MERCURY_SOCKET_TESTS=1` to additionally run a localhost HTTP streaming fixture;
it binds an ephemeral port and does not contact vLLM.
