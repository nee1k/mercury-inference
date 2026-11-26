"""Shared standard-library helpers. Never start or attach to Ray."""
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/packages/e4s/26.06/mvapich-5.0.1-zfp-cuda13.2/python/pkgs/python-3.12.13/bin/python"
VLLM = str(Path(PYTHON).with_name("vllm"))
MODEL = "Qwen/Qwen3-8B"
CONFIG = "qwen3-8b-bf16-tp1"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots" / REVISION
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
SHAPES = {"chat": (1024, 300), "long-doc": (16384, 500), "long-gen": (2048, 2000)}
COUNTS = {1: 10, 4: 40, 16: 64, 64: 256}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def runtime_env(gpus=None):
    environment = os.environ.copy()
    environment.pop("RAY_ADDRESS", None)
    for name in ("OPENAI_API_KEY", "VLLM_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        environment.pop(name, None)
    environment["LD_LIBRARY_PATH"] = "/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib:" + environment.get("LD_LIBRARY_PATH", "")
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MAX_JOBS="8",
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if gpus is not None:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    return environment


def version():
    result = subprocess.run([VLLM, "--version"], env=runtime_env(), text=True,
                            capture_output=True, timeout=120, check=True)
    return result.stdout.strip()


def local_url(base):
    parsed = urllib.parse.urlsplit(base.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Only local HTTP URLs are permitted")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/v1"):
        raise ValueError("Expected a base URL with no credentials, query or arbitrary path")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def http(base, endpoint, body=None, timeout=120):
    request = urllib.request.Request(local_url(base) + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def identity(pid):
    # Split after the final ')' because Linux comm may itself contain spaces.
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "ppid": int(stat[1]), "start_ticks": int(stat[19]),
            "uid": directory.stat().st_uid, "state": stat[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def same_process(record):
    try:
        current = identity(record["pid"])
        return all(current[key] == record[key] for key in ("pid", "start_ticks", "uid", "boot_id")) and current["state"] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def register_child(pid, kind, command, directory):
    record = identity(pid)
    record.update(kind=kind, command=command)
    with (ROOT / "state" / "pids.txt").open("a") as pids:
        pids.write(str(pid) + "\n")
        pids.flush()
        os.fsync(pids.fileno())
    save(Path(directory) / (kind + "-process.json"), record)
    return record


def terminate_child(record):
    recorded = {int(line) for line in (ROOT / "state" / "pids.txt").read_text().splitlines() if line.strip()}
    if record["pid"] not in recorded or record["uid"] != os.getuid() or not same_process(record):
        raise ValueError("Child process identity is not valid; no signal sent")
    descriptor = os.pidfd_open(record["pid"])
    try:
        if not same_process(record):
            raise ValueError("Child identity changed; no signal sent")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        try:
            pid = identity(pid)["ppid"]
        except (OSError, ValueError):
            return False
    return False


def gpu_snapshot(index=0):
    result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", str(index)],
                            text=True, capture_output=True, timeout=15, check=True)
    tree = ET.fromstring(result.stdout)
    gpu = tree.find("gpu")
    if gpu is None:
        raise ValueError("No GPU in nvidia-smi XML")
    def number(path):
        value = gpu.findtext(path, "")
        if not re.match(r"^\d+(?:\.\d+)?(?:\s|$)", value):
            raise ValueError("Missing numeric GPU observation: " + path)
        return float(value.split()[0])
    pids = []
    for process in gpu.findall("processes/process_info"):
        pid = process.findtext("pid", "")
        if not pid.isdigit():
            raise ValueError("Unidentifiable process on tested GPU")
        pids.append(int(pid))
    return {"timestamp": stamp(), "uuid": gpu.findtext("uuid"), "index": index,
            "used_mib": number("fb_memory_usage/used"),
            "gpu_util_percent": number("utilization/gpu_util"), "pids": pids}


def idle_sample():
    samples = []
    begin = time.monotonic()
    for index in range(11):
        time.sleep(max(0, begin + index - time.monotonic()))
        sample = gpu_snapshot()
        sample["elapsed_seconds"] = time.monotonic() - begin
        samples.append(sample)
    idle = all(not item["pids"] and item["used_mib"] < 1024 and item["gpu_util_percent"] == 0 for item in samples)
    idle = idle and len({item["uuid"] for item in samples}) == 1
    return {"idle": idle, "samples": samples}


def state_path(config=CONFIG):
    if config != CONFIG:
        raise ValueError("Configuration is not selected")
    return ROOT / "state" / (config + ".json")


def owned_server(base, model=MODEL):
    state = load(state_path())
    if local_url(base) != state["base_url"] or model != state["model"]:
        raise ValueError("Benchmark/function checks require this study's managed endpoint")
    recorded = {int(line) for line in (ROOT / "state" / "pids.txt").read_text().splitlines() if line.strip()}
    if state["pid"] not in recorded or not same_process(state) or state["uid"] != os.getuid():
        raise ValueError("Managed server identity is no longer valid")
    arguments = (Path("/proc") / str(state["pid"]) / "cmdline").read_bytes().split(b"\0")
    if b"serve" not in arguments or str(ROOT / "configs" / (CONFIG + ".yaml")).encode() not in arguments:
        raise ValueError("PID command is not the study server")
    return state


def check_neighbors(state, sample):
    if sample["uuid"] != state["gpu_uuid"]:
        return ["GPU identity changed"]
    if not same_process(state):
        return ["Study server exited or identity changed"]
    return ["Unrelated GPU PID " + str(pid) for pid in sample["pids"] if not descendant(pid, state["pid"])]


def verify_cache():
    index = load(SNAPSHOT / "model.safetensors.index.json")
    names = set(index["weight_map"].values()) | {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"}
    missing = sorted(name for name in names if not (SNAPSHOT / name).is_file())
    if missing:
        raise ValueError("Cache incomplete; download permission required: " + ", ".join(missing))
    return {"snapshot": str(SNAPSHOT), "revision": REVISION,
            "files": {name: (SNAPSHOT / name).stat().st_size for name in sorted(names)},
            "verification": "file presence and size, not checksum validation"}
