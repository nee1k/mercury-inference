"""Fixed-length benchmarks against verified study-owned servers only."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from common import (CONFIG, COUNTS, MODEL, OFF, ROOT, SHAPES, SNAPSHOT, VLLM,
                    check_neighbors, gpu_snapshot, http, load, owned_server,
                    register_child, runtime_env, save, stamp, terminate_child, version)


def validate_result(raw, count, output_tokens, input_tokens):
    problems = []
    if raw.get("completed") != count or raw.get("failed") != 0:
        problems.append("Request count/failure evidence is missing or not all requests succeeded")
    for key, target in (("output_lens", output_tokens), ("input_lens", input_tokens)):
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != count or any(value != target for value in values):
            problems.append(key + " does not match the requested fixed length for every request")
    errors = raw.get("errors")
    if not isinstance(errors, list) or len(errors) != count or any(errors):
        problems.append("Per-request error evidence missing or nonempty")
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for percentile in (50, 95, 99):
            value = raw.get("p%d_%s_ms" % (percentile, metric))
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                problems.append("Missing/invalid percentile: p%d_%s_ms" % (percentile, metric))
    if not isinstance(raw.get("output_throughput"), (int, float)) or not math.isfinite(raw["output_throughput"]) or raw["output_throughput"] <= 0:
        problems.append("Missing/invalid output throughput")
    return problems


def chat_overhead(base):
    prompt = "alpha"
    plain = http(base, "/tokenize", {"model": MODEL, "prompt": prompt})["count"]
    chat = http(base, "/tokenize", {"model": MODEL,
        "messages": [{"role": "user", "content": prompt}], **OFF})["count"]
    if chat < plain:
        raise ValueError("Invalid chat token overhead")
    return chat - plain


def cell(base, directory, shape, concurrency, count, kind, installed_version, output_override=None):
    state = owned_server(base)
    directory = Path(directory)
    directory.mkdir(parents=True)
    target_input, output = SHAPES[shape]
    if output_override is not None:
        output = output_override
    overhead = chat_overhead(base)
    content_tokens = target_input - overhead
    if content_tokens < 1:
        raise ValueError("Chat overhead exceeds requested input length")
    command = [VLLM, "bench", "serve", "--backend", "openai-chat", "--base-url", base,
        "--endpoint", "/v1/chat/completions", "--model", str(SNAPSHOT),
        "--served-model-name", MODEL, "--tokenizer", str(SNAPSHOT),
        "--dataset-name", "random", "--random-input-len", str(content_tokens),
        "--random-output-len", str(output), "--random-range-ratio", "0",
        "--num-prompts", str(count), "--max-concurrency", str(concurrency),
        "--request-rate", "inf", "--seed", "0", "--ignore-eos", "--temperature", "0.6",
        "--top-p", "0.95", "--top-k", "20", "--extra-body", '{"chat_template_kwargs":{"enable_thinking":false}}',
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
        "--save-result", "--save-detailed", "--result-dir", str(directory),
        "--result-filename", "raw.json"]
    help_text = (ROOT / "results" / "inventory" / "vllm-bench-serve-help-all-runtime.txt").read_text()
    if any(argument not in help_text for argument in command if argument.startswith("--")):
        raise ValueError("Benchmark command includes a flag absent from saved installed help")
    before = gpu_snapshot()
    problems = check_neighbors(state, before)
    record = {"config": CONFIG, "model": MODEL, "revision": state["revision"], "gpus": 1,
        "total_parameters_b": 8.2, "precision": "BF16", "reasoning": "off", "shape": shape,
        "concurrency": concurrency, "prompts": count, "input_tokens": target_input,
        "random_content_tokens": content_tokens, "chat_template_overhead_tokens": overhead,
        "output_tokens": output, "kind": kind, "vllm_version": installed_version,
        "version_command": [VLLM, "--version"], "launch_evidence": state["launch_directory"],
        "command": command, "timestamp": stamp(), "status": "RUNNING"}
    save(directory / "gpu-before.json", before)
    save(directory / "cell.json", record)
    if problems:
        record.update(status="INVALID", problems=problems)
        save(directory / "cell.json", record)
        raise RuntimeError("Unrelated GPU allocation before benchmark")
    print("Benchmark " + shape + " c=" + str(concurrency) + " (" + kind + ")", flush=True)
    process, child = None, None
    try:
        with (directory / "command.log").open("w") as log, (directory / "gpu-monitor.jsonl").open("w") as monitor:
            process = subprocess.Popen(command, env=runtime_env([0]), cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            child = register_child(process.pid, "benchmark", command, directory)
            while process.poll() is None:
                sample = gpu_snapshot()
                monitor.write(json.dumps(sample) + "\n")
                monitor.flush()
                arrivals = check_neighbors(state, sample)
                problems.extend(arrivals)
                if arrivals:
                    terminate_child(child)
                    process.wait(timeout=30)
                    break
                time.sleep(1)
            code = process.wait()
        after = gpu_snapshot()
        save(directory / "gpu-after.json", after)
        problems.extend(check_neighbors(state, after))
        if code:
            problems.append("Benchmark command exited " + str(code))
        raw = load(directory / "raw.json") if (directory / "raw.json").is_file() else {}
        problems.extend(validate_result(raw, count, output, target_input))
        record.update(status="VALID" if not problems else "INVALID", problems=sorted(set(problems)),
                      raw_file=str(directory / "raw.json"), exit_code=code)
        save(directory / "cell.json", record)
        if problems:
            raise RuntimeError("Benchmark invalid: " + "; ".join(sorted(set(problems))))
        return str(directory / "cell.json")
    except Exception as error:
        if process is not None and child is not None and process.poll() is None:
            try:
                terminate_child(child)
                process.wait(timeout=30)
            except Exception as cleanup_error:
                problems.append("Benchmark cleanup needs review: " + str(cleanup_error))
        record.update(status="INVALID", problems=sorted(set(problems + [str(error)])))
        save(directory / "cell.json", record)
        raise


def run(base, repeats=1, dry=False, kind="sweep"):
    owned_server(base)
    if not 1 <= repeats <= 3:
        raise ValueError("Repeats must be 1–3")
    installed_version = version()
    root = ROOT / "results" / CONFIG / (kind + "-" + stamp())
    root.mkdir()
    cells = []
    cell(base, root / "warmup-discarded", "chat", 1, 1, "warmup-discarded", installed_version, output_override=16)
    for repeat in range(repeats):
        for shape in (("chat",) if dry else tuple(SHAPES)):
            for concurrency in ((1, 4) if dry else tuple(COUNTS)):
                cells.append(cell(base, root / ("r%d-%s-c%d" % (repeat + 1, shape, concurrency)),
                                  shape, concurrency, COUNTS[concurrency], kind, installed_version))
    manifest = {"config": CONFIG, "kind": kind, "vllm_version": installed_version, "cells": cells}
    save(root / "manifest.json", manifest)
    return root / "manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=(CONFIG,))
    parser.add_argument("port", type=int)
    parser.add_argument("repeats", type=int)
    args = parser.parse_args()
    print(run("http://127.0.0.1:" + str(args.port), args.repeats))
