"""Collect reproducible C1 setup and isolated measurements."""

import argparse
from importlib.metadata import version
import json
import platform
import random
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path

from study import MODEL, SNAPSHOT, RESULTS, gpu_snapshot, managed, neighbors, request, stamp
from runner import sse_events

REPORT = RESULTS / "setup" / "c1_rerun.md"
GPU_FIELDS = ("index,name,memory.total,driver_version,power.limit,pcie.link.gen.max,"
              "pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current")


def inventory():
    print("Python", platform.python_version())
    for package in ("torch", "vllm", "transformers"):
        print(package, version(package))
    print("snapshot", SNAPSHOT.name)
    config = json.loads((SNAPSHOT / "config.json").read_text())
    for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
                "head_dim", "intermediate_size", "vocab_size", "tie_word_embeddings"):
        print(key, json.dumps(config[key]))


def live(gpu):
    from replay import COUNTERS, scrape
    record = managed(f"gpu{gpu}")
    neighbors(record)
    base = record["base_url"]
    with request(base, "/metrics") as response:
        samples = [line for line in response.read().decode().splitlines()
                   if line.startswith("vllm:cache_config_info{")]
    if len(samples) != 1 or not re.search(r'\bblock_size="\d+"', samples[0]):
        raise ValueError("Expected one labeled cache configuration with block_size")
    print(samples[0])
    hits, queries = COUNTERS[1:]
    for length in (1024, 4096, 16384):
        for trial in range(1, 4):
            neighbors(record)
            rng = random.Random(length + trial)
            prompt = [length + trial] + rng.choices(range(1000, 20000), k=length - 1)
            body = dict(model=MODEL, prompt=prompt, max_tokens=1, temperature=0, stream=True,
                        stream_options={"include_usage": True}, return_token_ids=True)
            before = scrape(base)
            status = {}
            usage = {}
            ttft = None
            start = time.perf_counter()
            with request(base, "/v1/completions", body, timeout=180) as response:
                for event in sse_events(response, status):
                    if event.get("error"):
                        raise ValueError("Probe completion failed")
                    for choice in event.get("choices", []):
                        if ttft is None and (choice.get("token_ids") or choice.get("text")):
                            ttft = time.perf_counter() - start
                    usage = event.get("usage") or usage
            if not (status.get("done") and ttft and usage.get("prompt_tokens") == length
                    and usage.get("completion_tokens") == 1):
                raise ValueError("Probe length or streaming evidence incomplete")
            end = time.monotonic() + 15
            while True:
                after = scrape(base)
                delta = after[queries] - before[queries]
                if delta or time.monotonic() >= end:
                    break
                time.sleep(0.2)
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            hit_delta = after[hits] - before[hits]
            if cached not in (0, None) or hit_delta != 0 or delta != length:
                raise ValueError(f"Cache verification failed: length={length} hits={hit_delta} queries={delta}")
            # v0.21.0 omits prompt_tokens_details for zero in the completion stream.
            print(f"gpu={gpu} prompt_tokens={length} trial={trial} ttft_s={ttft:.9f} "
                  f"cached_tokens=0 usage_cached_tokens={json.dumps(cached)} "
                  f"prefix_hits_delta={hit_delta:g} prefix_queries_delta={delta:g}", flush=True)


def copies(gpu):
    def occupied():
        sample = gpu_snapshot(gpu)
        return bool(sample["pids"] or sample["used_mib"] >= 1024 or sample["gpu_util_percent"])

    if occupied():
        print(f"gpu={gpu} skipped=occupied")
        return
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    import torch
    if occupied():
        print(f"gpu={gpu} skipped=occupied")
        return
    size = 2**30
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty_like(host, device="cuda")
    for direction, dest, src in (("H2D", dev, host), ("D2H", host, dev)):
        rates = []
        for trial in range(1, 11):
            if set(gpu_snapshot(gpu)["pids"]) - {os.getpid()}:
                print(f"gpu={gpu} skipped=occupied")
                return
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            dest.copy_(src, non_blocking=True)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end)
            rates.append(size / ms / 1e6)
            print(f"gpu={gpu} direction={direction} trial={trial} bytes={size} milliseconds={ms:.6f}")
        print(f"gpu={gpu} direction={direction} median_GB_per_s={statistics.median(rates):.6f}")


def record(command, *, study_python=False, select=None):
    import sys
    import os
    from study import runtime_env
    actual = ([sys.executable, "-B"] if study_python else []) + command
    label = shlex.join((["python", "-B"] if study_python else []) + command)
    output = subprocess.check_output(actual, text=True, cwd=Path(__file__).parent,
                                     env={**runtime_env(), "LC_ALL": "C"})
    if select:
        output = "".join(line for line in output.splitlines(True) if re.search(select, line))
        label += " | grep -E " + shlex.quote(select)
    if not output.strip():
        raise ValueError("Empty setup observation")
    if re.search(r'(?:/home/|/storage/users/|(?:\d{1,3}\.){3}\d{1,3}\b)', output) or os.environ.get("USER", "\0") in output:
        raise ValueError("Unsafe setup observation")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("a") as handle:
        handle.write(f"\n```text\n$ {label}\n{output.rstrip()}\n```\n")


def gpu_facts(gpus):
    record(["nvidia-smi", "-i", ",".join(map(str, gpus)), "--query-gpu=" + GPU_FIELDS,
            "--format=csv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "inventory", "live", "copies"))
    parser.add_argument("gpu", nargs="?", type=int)
    args = parser.parse_args()
    if args.mode == "setup":
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("x") as report:
            report.write("# C1 rerun setup and measurements\n\n"
                         "Commands run from agent-study with the study Python and LC_ALL=C.\n")
        record(["date", "-u", "+%FT%TZ"])
        gpu_facts((2, 3))
        record(["nvidia-smi"], select="CUDA (UMD )?Version")
        record(["grep", "^MemTotal:", "/proc/meminfo"])
        record(["lscpu"], select=r"^(CPU\(s\):|Model name:|Core\(s\) per socket:|Socket\(s\):)")
        record(["c1_probe.py", "inventory"], study_python=True)
    elif args.mode == "inventory":
        inventory()
    elif args.mode == "live":
        live(args.gpu)
    else:
        copies(args.gpu)
