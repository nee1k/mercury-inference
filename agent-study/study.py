"""Local configuration and process safeguards for the two-GPU agent study."""

import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STATE = ROOT / "state"
PREVIOUS = ROOT.parent / "model-size-study"
sys.path.append(str(PREVIOUS / "scripts"))
from common import (  # Reuse Mercury's process identity and NVIDIA telemetry helpers.
    PYTHON, VLLM, descendant, gpu_snapshot, identity, runtime_env, same_process,
    save, stamp,
)

MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots" / REVISION
)
VERSION = "0.21.0"
HELP = PREVIOUS / "results/inventory/vllm-serve-help-all.txt"
MAX_CONTEXT = 32768
PORTS = range(8120, 8130)
CONFIGS = {
    "C1": ("rr", ()),
    "C2": ("sticky", ()),
    "C3": ("sticky", ("--kv-cache-dtype", "fp8")),
    "C4": ("sticky", ("--kv-offloading-size", "100", "--kv-offloading-backend", "native")),
}
REQUEST_SETTINGS = {
    "model": MODEL,
    "temperature": 0,
    "max_tokens": 300,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}


def base_url(port):
    if port not in PORTS:
        raise ValueError("Study ports must be in 8120–8129")
    return f"http://localhost:{port}"


def local_url(url):
    parsed = urlsplit(url)
    canonical = base_url(parsed.port)
    if parsed.hostname == "127.0.0.1" and url.rstrip("/") == canonical.replace("localhost", parsed.hostname):
        return url.rstrip("/")
    if url.rstrip("/") != canonical:
        raise ValueError("Expected a 127.0.0.1 study base URL")
    return canonical


def request(base, path, body=None, headers=None, timeout=30):
    payload = None if body is None else json.dumps(body).encode()
    req = Request(local_url(base) + path, data=payload,
                  headers={"Content-Type": "application/json", **(headers or {})})
    return build_opener(ProxyHandler({})).open(req, timeout=timeout)


def available_port(port):
    base_url(port)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def sample_idle(gpu):
    samples = []
    start = time.monotonic()
    for offset in range(11):
        time.sleep(max(0, start + offset - time.monotonic()))
        samples.append(gpu_snapshot(gpu))
    return {
        "idle": len({row["uuid"] for row in samples}) == 1 and all(
            not row["pids"] and row["used_mib"] < 1024 and row["gpu_util_percent"] == 0
            for row in samples
        ),
        "samples": samples,
    }


def pid_path(name):
    return STATE / f"{name}.json"


def managed(name):
    try:
        return json.loads(pid_path(name).read_text())
    except FileNotFoundError:
        return None


def launch(name, command, log, **fields):
    STATE.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with (STATE / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = managed(name)
        if existing and same_process(existing):
            raise RuntimeError(f"{name} is already running")
        active = [item for path in STATE.glob("*.json")
                  if (item := managed(path.stem)) and same_process(item)]
        if any(item["port"] == fields["port"] for item in active):
            raise RuntimeError("A study process already holds that port")
        if fields.get("gpu") is not None and any(
            item.get("gpu") == fields["gpu"] for item in active
        ):
            raise RuntimeError("Another study replica already holds that GPU")
        available_port(fields["port"])
        if fields.get("gpu") is not None:
            latest = gpu_snapshot(fields["gpu"])
            if (latest["pids"] or latest["used_mib"] >= 1024 or
                    latest["gpu_util_percent"] or latest["uuid"] != fields["gpu_uuid"]):
                raise RuntimeError(f"GPU {fields['gpu']} changed after the idle check")
        with log.open("xb") as output:
            # Persist only the three numerical startup messages needed by the study.
            pattern = (r"Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|"
                       r"GPU KV cache size: [0-9,]+ tokens|"
                       r"Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x")
            filtered = subprocess.Popen(["grep", "--line-buffered", "-Eo", pattern],
                                        stdin=subprocess.PIPE, stdout=output)
            gpu = fields.get("gpu")
            process = subprocess.Popen(
                ["setsid", "nohup", *command], stdin=subprocess.DEVNULL,
                stdout=filtered.stdin, stderr=subprocess.STDOUT,
                env=runtime_env([gpu]) if gpu is not None else None,
            )
            filtered.stdin.close()
        handle = os.pidfd_open(process.pid)
        try:
            record = {**identity(process.pid), **fields, "name": name,
                      "log": str(log.relative_to(ROOT)), "started_at": stamp()}
            with (STATE / "pids.txt").open("a") as pids:
                pids.write(f"{process.pid}\n")
                pids.flush()
                os.fsync(pids.fileno())
            save(pid_path(name), record)
        except BaseException:
            signal.pidfd_send_signal(handle, signal.SIGTERM)
            raise
        finally:
            os.close(handle)
        return record


def stop(record):
    if not record or not same_process(record):
        return
    recorded = (STATE / "pids.txt").read_text().splitlines()
    if (record.get("uid") != os.getuid() or str(record["pid"]) not in recorded or
            managed(record["name"]) != record):
        raise RuntimeError("Study ownership could not be verified; no signal sent")
    try:
        fd = os.pidfd_open(record["pid"])
    except ProcessLookupError:
        return
    try:
        if same_process(record):
            signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def neighbors(record):
    sample = gpu_snapshot(record["gpu"])
    if sample["uuid"] != record["gpu_uuid"] or not same_process(record):
        raise RuntimeError(f"GPU identity or study process changed for {record['name']}")
    foreign = [pid for pid in sample["pids"] if not descendant(pid, record["pid"])]
    if foreign:
        raise RuntimeError(f"Unrelated GPU processes on GPU {record['gpu']}: {foreign}")
    return sample
