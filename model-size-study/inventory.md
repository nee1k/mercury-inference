# Mercury study inventory — reviewed

Phase 0 collection and review are complete, with the live-deployment limitations below. Awaiting the user's **go** for phase 1. No study model was downloaded or served. Evidence was collected through commands submitted by the user; this does not establish unrestricted access from future Codex sandbox commands.

## Hardware and availability

The inventory observed four NVIDIA H200 NVL GPUs, each with 143,771 MiB total memory, driver 610.43.02 and MIG disabled. The driver advertises CUDA 13.3; this is the driver's capability, not the PyTorch runtime version. Sources: [GPU details](results/inventory/gpu-details.txt), [full nvidia-smi](results/inventory/nvidia-smi.txt).

| GPU | Memory used (MiB) | Utilization | Compute PID | Owner |
|---|---:|---:|---|---|
| 0 | 0 | 0% | None listed | — |
| 1 | 114,731 | 37% | 3020288 | yuwang |
| 2 | 5 | 0% | None listed | — |
| 3 | 116,531 | 100% | 117094 | OnAdditive1 |

Table sources: [GPU details](results/inventory/gpu-details.txt), [GPU processes](results/inventory/gpu-processes.txt), [untruncated process owners](results/inventory/gpu-process-owners-full.txt). These are observations at collection time, not reservations.

GPU 0 passed the full idle rule during the ten-second sample: no listed processes, memory used below 1 GiB, and zero utilization throughout. GPU 2 appeared idle in the snapshot but was not sampled for ten seconds; it is also outside the current GPU-0-only scope. Sources: [idle sample](results/inventory/gpu0-idle-sample.json), [GPU details](results/inventory/gpu-details.txt).

Every GPU pair reports NV6 connectivity. Source: [topology matrix and legend](results/inventory/topology.txt).

Repeat the complete idle check immediately before every model launch. TP2 and TP4 configurations are deferred under the user's current scope. Do not stop existing workloads.

## Access and runtime

All access checks passed: NVIDIA access, installed-environment torch/CUDA, a throwaway loopback HTTP server, an HF-cache test write/removal and HTTPS to Hugging Face. Source: [access-check.json](results/access-check.json).

Installed executable directory: `/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin`.

- vLLM reports **0.21.0**: [version with intended runtime](results/inventory/vllm-version-runtime.txt).
- PyTorch imports as **2.11.0+cu130**, Triton imports as **3.6.0**, and CUDA is available: [runtime import check](results/inventory/runtime-import-check.txt). Distribution metadata, including Ray and Transformers versions, is retained in [inventory.json](results/inventory/inventory.json).
- The initial benchmark CLI import found an older C++ library and disabled Triton: [original benchmark help](results/inventory/vllm-bench-serve-help.txt).
- Prepending `/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib` to **child-process-only** `LD_LIBRARY_PATH` resolved the import. The loaded library is `/storage/packages/gcc/13.4/lib64/libstdc++.so.6.0.32`: [runtime import check](results/inventory/runtime-import-check.txt). No packages or global configuration were changed.

Saved help: [serve](results/inventory/vllm-serve-help.txt), [all serve flags](results/inventory/vllm-serve-help-all.txt), [benchmark](results/inventory/vllm-bench-serve-help.txt), [all benchmark flags with the corrected runtime](results/inventory/vllm-bench-serve-help-all-runtime.txt).

Required benchmark flags are present, including extra-body, detailed results, output-length controls and all required percentile settings. Sources: [flag verification](results/inventory/supplement.json), [full benchmark help](results/inventory/vllm-bench-serve-help-all-runtime.txt). Use an explicit openai-chat backend and chat-completions endpoint when applying chat-template reasoning controls; validate actual prompt and output lengths in the dry run.

## Live Ray Serve deployment — not identified

`serve status` and `serve config` could not connect to the default `http://localhost:8265` dashboard. Sources: [status failure](results/inventory/ray-serve-status.txt), [config failure](results/inventory/ray-serve-config-redacted.txt). A command-name query found no matching Ray daemon names in its visible process namespace: [query output](results/inventory/ray-process-names.txt), [exit status](results/inventory/supplement.json). Neither observation proves that the production app is absent.

The read-only [TCP listener inventory](results/inventory/listening-tcp-sockets.txt) does not identify a production model API. Its URL, model ID and config remain **UNVERIFIED**. Do not guess a target for load, start Ray or change any deployment. Obtain the live endpoint URL/dashboard address before the permitted model-list query and concurrency-one natural runner. Candidate selection and isolated GPU-0 work can continue after the checkpoint.

## Storage and cached models

HF cache: `/home/users/neelk/.cache/huggingface/hub`. Study and cache filesystems each reported **20,281,143,853,056 free bytes**; this is filesystem free space, not an account quota guarantee. Source: [inventory.json](results/inventory/inventory.json), fields `hf_cache` and `disk`.

All indexed weight shards were present for Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-30B-A3B and Qwen/Qwen3-32B. Exact revisions, weight sizes and tokenizer/config observations are in [cached_models](results/inventory/inventory.json). This is a file-presence check, not model launch validation.

Mistral-Small-3.1-24B-Instruct-2503 contains a consolidated weight file while the HF index names absent split shards. Completeness for the native Mistral loading path remains unverified; do not label it uncached or unusable solely from the HF-index check. Source: [cached_models](results/inventory/inventory.json).

## Earlier small model

Historical launch/config evidence identifies a Qwen/Qwen3-8B deployment, but does not establish that it was the model the lab found too small. Preserve the identity as **UNCONFIRMED**. Sources: [historical_model_evidence](results/inventory/inventory.json), the referenced existing launcher and README.

## Checkpoint

Await **go** for phase 1 candidate proposals. No downloads, model launches, live inference requests or changes to other workloads have occurred in this phase. Function checks for every configuration, server load/KV/concurrency metrics, and an official BF16/FP8 dense-model pair remain required.
