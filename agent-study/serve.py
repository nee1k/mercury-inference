"""Launch a single pinned, offline Qwen3-32B replica and record readiness."""

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import time
import urllib.error

from study import (
    CONFIGS, HELP, MAX_CONTEXT, MODEL, PORTS, REQUEST_SETTINGS, RESULTS, REVISION,
    SNAPSHOT, VERSION, VLLM, base_url, launch, neighbors, request,
    same_process, sample_idle, save, stamp, stop,
)

FLAGS = (
    "--host", "127.0.0.1", "--served-model-name", MODEL, "--dtype", "bfloat16",
    "--tensor-parallel-size", "1", "--max-model-len", str(MAX_CONTEXT),
    "--gpu-memory-utilization", "0.90", "--seed", "0",
    "--distributed-executor-backend", "mp",
)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def verify_flags(flags):
    help_text = HELP.read_text()
    missing = [flag for flag in flags if flag.startswith("--") and
               not re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", help_text)]
    if missing:
        raise ValueError(f"Skip configuration: flags missing from saved vLLM help: {missing}")


def verify_cache():
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())
    files = set(index["weight_map"].values()) | {
        "model.safetensors.index.json", "config.json", "tokenizer.json",
        "tokenizer_config.json", "generation_config.json",
    }
    missing = [name for name in sorted(files) if not (SNAPSHOT / name).is_file()]
    if missing:
        raise ValueError(f"Pinned cache incomplete; download approval required: {missing}")
    return {name: (SNAPSHOT / name).stat().st_size for name in sorted(files)}


def verify_pci_order(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True, timeout=15,
    )
    devices = [line.split(",") for line in output.splitlines()]
    order = [int(index) for index, _ in sorted(devices, key=lambda item: item[1].strip())]
    if gpu >= len(order) or order[gpu] != gpu:
        raise RuntimeError("Selected numeric CUDA index does not match nvidia-smi PCI order")


def startup_evidence(log):
    lines = ANSI.sub("", log.read_text(errors="replace")).splitlines()
    load = [line for line in lines if re.search(
        r"loading weights took|model loading took|model loading.*seconds", line, re.I)]
    cache = [line for line in lines if "gpu kv cache size:" in line.lower()]
    concurrency = [line for line in lines if "maximum concurrency" in line.lower()]
    loading = [(line, match[1]) for line in load
               if (match := re.search(r"Model loading took (\d+(?:\.\d+)?) GiB\b", line))]
    cache_match = cache and re.search(r"([\d,]+)\s+tokens", cache[-1])
    concurrency_match = concurrency and re.search(
        r"Maximum concurrency for 32,768 tokens per request:\s+(\d+(?:\.\d+)?)x",
        concurrency[-1])
    if not (loading and cache_match and concurrency_match):
        raise RuntimeError("Replica startup log is missing model GiB, KV size or 32K concurrency")
    load_line, gib = loading[-1]
    return {
        "load_lines": load, "kv_cache_line": cache[-1],
        "model_loading_line": load_line, "model_loading_gib": float(gib),
        "max_concurrency_line": concurrency[-1],
        "kv_cache_tokens": int(cache_match[1].replace(",", "")),
        "max_concurrency": float(concurrency_match[1]),
    }


def serve(config, gpu, port, supplied):
    replica = f"gpu{gpu}"
    run_id = os.environ.get("MERCURY_AGENT_STUDY_RUN_ID") or stamp()
    if not re.fullmatch(r"\d{8}T\d{6}\.\d{6}Z", run_id):
        raise ValueError("Invalid study run ID")
    result_name = f"{replica}-{run_id}"
    folder = RESULTS / config
    log = folder / f"{result_name}.log"
    extra = CONFIGS[config][1]
    if supplied and tuple(supplied) != extra:
        raise ValueError("Extra flags must match the selected study configuration")
    flags = (*FLAGS, "--port", str(port), *extra)
    verify_flags(flags)
    version = importlib.metadata.version("vllm")
    if version != VERSION:
        raise RuntimeError(f"Expected vLLM {VERSION}, found {version}")
    cache = verify_cache()
    verify_pci_order(gpu)
    idle = sample_idle(gpu)
    save(folder / f"{result_name}-idle.json", idle)
    if not idle["idle"]:
        raise RuntimeError(f"GPU {gpu} did not pass its 10-second idle check")
    url = base_url(port)
    command = [VLLM, "serve", str(SNAPSHOT), *flags]
    start = time.monotonic()
    record = launch(
        replica, command, log, config=config, gpu=gpu, port=port, run_id=run_id,
        base_url=url, gpu_uuid=idle["samples"][-1]["uuid"],
        idle_used_mib=idle["samples"][-1]["used_mib"], model=MODEL,
        revision=REVISION, vllm_version=version,
    )
    print(f"Started {replica} PID {record['pid']} at {url}; log: {log}", flush=True)
    try:
        while time.monotonic() - start < 900:
            neighbors(record)
            try:
                with request(url, "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Replica did not become healthy")
        settings = {**REQUEST_SETTINGS, "stream": False}
        del settings["stream_options"]
        with request(url, "/v1/chat/completions", {
            **settings, "messages": [{"role": "user", "content": "What is 2 + 2? Reply briefly."}],
        }, timeout=120) as response:
            completion = json.load(response)
        if not completion["choices"][0]["message"].get("content") or not completion.get("usage"):
            raise RuntimeError("Readiness completion lacked content or usage")
        evidence = {
            **record, **startup_evidence(log), "ready_at": stamp(),
            "seconds_to_first_completion": time.monotonic() - start,
            "completion_usage": completion["usage"], "cache_files": cache,
            "gpu_at_ready": neighbors(record),
        }
        ready = folder / f"{result_name}-ready.json"
        save(ready, evidence)
        print(f"Ready {replica}; evidence: {ready}", flush=True)
    except BaseException:
        if same_process(record):
            stop(record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("gpu", type=int, choices=range(4))
    parser.add_argument("port", type=int, choices=PORTS)
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    serve(args.config, args.gpu, args.port, args.flags)
