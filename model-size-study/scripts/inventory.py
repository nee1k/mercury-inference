"""Read-only phase-0 inventory; writes evidence only inside the study folder."""
import csv
import datetime
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "inventory"
BIN = Path(sys.executable).parent
OUT.mkdir(exist_ok=True)
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
report = {"timestamp": started, "python": sys.executable,
          "execution_context": "user Mercury terminal; not proof of Codex sandbox access",
          "commands": {}, "status": "RUNNING"}


def redact(text):
    text = re.sub(r"\bhf_[A-Za-z0-9]+", "[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*).*", r"\1[REDACTED]", text)
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[REDACTED]@", text)
    return text


def safe_config(text):
    # Keep structure and a small allowlist of serving fields. Environment values,
    # arbitrary user config and unrecognized scalars are never saved or printed.
    allowed = {"name", "import_path", "route_prefix", "host", "port", "model",
               "model_id", "model_source", "served_model_name", "revision", "dtype",
               "tensor_parallel_size", "pipeline_parallel_size", "num_gpus", "num_cpus",
               "accelerator_type", "max_model_len", "gpu_memory_utilization",
               "distributed_executor_backend", "num_replicas", "min_replicas", "max_replicas",
               "target_ongoing_requests", "max_ongoing_requests", "reasoning_parser",
               "tool_call_parser", "enable_auto_tool_choice", "trust_remote_code"}
    lines = []
    hidden_indent = None
    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        if hidden_indent is not None:
            if line.strip() and indent <= hidden_indent:
                hidden_indent = None
            else:
                continue
        match = re.match(r"^(\s*(?:-\s+)?)([\w.-]+):(?:\s+(.*))?$", line)
        if not match:
            if line.strip():
                lines.append(" " * indent + "[REDACTED unparsed line]")
            continue
        prefix, key, value = match.groups()
        if re.search(r"(?i)(env|secret|token|password|credential|header|auth|api.?key)", key):
            lines.append(prefix + key + ": '[REDACTED]' ")
            hidden_indent = indent
        elif value is None:
            lines.append(line)
        elif key in allowed and not value.startswith(("|", ">", "{", "[")):
            lines.append(redact(line))
        else:
            lines.append(prefix + key + ": '[REDACTED scalar]' ")
            if value.startswith(("|", ">")):
                hidden_indent = indent
    return "\n".join(lines) + "\n"


def run(name, args, timeout=45, config=False):
    print("Collecting " + name, flush=True)
    begin = time.monotonic()
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, code = "", "Timed out; no partial output retained", None
    except OSError as error:
        stdout, stderr, code = "", str(error), None
    filename = OUT / (name + ".txt")
    filename.write_text((safe_config(stdout) if config else redact(stdout))
                        + ("\nSTDERR:\n" + redact(stderr) if stderr else ""))
    report["commands"][name] = {"argv": args, "exit_code": code,
                               "seconds": round(time.monotonic() - begin, 3),
                               "result": str(filename.relative_to(ROOT))}
    return stdout if code == 0 else ""


access = json.loads((ROOT / "results" / "access-check.json").read_text())
if access.get("status") != "PASS":
    raise SystemExit("Run phase-0 access checks successfully first")

run("nvidia-smi", ["nvidia-smi"])
run("topology", ["nvidia-smi", "topo", "-m"])
run("gpu-details", ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,mig.mode.current,driver_version", "--format=csv"])
apps = run("gpu-processes", ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"])
pids = []
for row in csv.DictReader(io.StringIO(apps)):
    pid = row.get(" pid", row.get("pid", "")).strip()
    if pid.isdigit():
        pids.append(pid)
if pids:
    run("gpu-process-owners", ["ps", "-p", ",".join(pids), "-o", "pid=,ppid=,user=,comm="])

print("Sampling GPU 0 for 10 seconds (inventory only; repeat before every launch)", flush=True)
samples = []
begin = time.monotonic()
for index in range(11):
    delay = begin + index - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    gpu = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    processes = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    samples.append({"elapsed_seconds": round(time.monotonic() - begin, 3),
                    "gpu": gpu.stdout.strip(), "processes": processes.stdout.strip(),
                    "gpu_exit_code": gpu.returncode, "processes_exit_code": processes.returncode})
idle = True
for sample in samples:
    fields = [part.strip() for part in sample["gpu"].split(",")]
    try:
        valid = (len(fields) == 4 and int(fields[2]) < 1024 and int(fields[3]) == 0)
    except ValueError:
        valid = False
    idle = idle and valid and not sample["processes"] and sample["gpu_exit_code"] == 0 and sample["processes_exit_code"] == 0
(OUT / "gpu0-idle-sample.json").write_text(json.dumps({"idle": idle, "samples": samples}, indent=2) + "\n")
report["gpu0_idle_during_sample"] = idle

run("vllm-version", [str(BIN / "vllm"), "--version"], timeout=120)
run("vllm-serve-help", [str(BIN / "vllm"), "serve", "--help"], timeout=120)
run("vllm-serve-help-all", [str(BIN / "vllm"), "serve", "--help=all"], timeout=120)
run("vllm-bench-serve-help", [str(BIN / "vllm"), "bench", "serve", "--help"], timeout=120)
run("ray-serve-status", [str(BIN / "serve"), "status"], timeout=30)
run("ray-serve-config-redacted", [str(BIN / "serve"), "config"], timeout=30, config=True)

versions = {}
for package in ("vllm", "torch", "triton", "ray", "transformers", "huggingface-hub"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "NOT INSTALLED"
report["package_versions"] = versions

hf_home = Path(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
cache = Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))))
report["hf_cache"] = str(cache)
report["disk"] = {}
for name, path in (("study", ROOT), ("hf_cache", cache)):
    disk = shutil.disk_usage(path)
    report["disk"][name] = {"path": str(path), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free,
                             "note": "Filesystem free space is not an account quota guarantee"}
models = []
for directory in sorted(cache.glob("models--*")):
    entry = {"model": directory.name.removeprefix("models--").replace("--", "/"), "snapshots": []}
    for snapshot in sorted((directory / "snapshots").glob("*")):
        files = list(snapshot.rglob("*"))
        weights = [path for path in files if path.suffix == ".safetensors"]
        weight_names = {path.name for path in weights if path.is_file()}
        indexes = list(snapshot.glob("*.safetensors.index.json"))
        required = set()
        for path in indexes:
            try:
                required.update(json.loads(path.read_text()).get("weight_map", {}).values())
            except (OSError, ValueError):
                pass
        entry["snapshots"].append({"revision": snapshot.name, "path": str(snapshot),
                                   "weight_bytes_present": sum(path.stat().st_size for path in weights if path.is_file()),
                                   "weight_files_present": len(weight_names), "required_shards_from_index": len(required),
                                   "missing_index_shards": sorted(required - weight_names),
                                   "has_config": (snapshot / "config.json").is_file(),
                                   "has_tokenizer_config": (snapshot / "tokenizer_config.json").is_file(),
                                   "completeness": "index shards present" if required and required <= weight_names else "UNVERIFIED"})
    models.append(entry)
report["cached_models"] = models

# Read known historical deployment evidence; do not scan shell history, auth files,
# environments or unrestricted logs that could contain credentials.
historical = []
for relative in ("README.md", "launch-qwen3-8b.sh", "deployment/manifest.json"):
    path = ROOT.parent / relative
    if path.is_file():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            ids = re.findall(r"(?:Qwen|google|meta-llama|mistralai|openai)/[A-Za-z0-9_.-]+", line)
            if ids:
                historical.append({"path": str(path), "line": number, "model_ids": sorted(set(ids))})
report["historical_model_evidence"] = historical
report["earlier_too_small_model"] = "UNCONFIRMED: legacy Qwen3-8B deployment exists, but no evidence links it to the lab's 'too small' report"
report["status"] = "COLLECTED_REVIEW_REQUIRED"
(OUT / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
lines = ["# Mercury study inventory — phase 0 evidence collected", "",
         "Collected: " + started, "", "Review raw evidence before declaring the phase complete.", "",
         "- Access checks: [access-check.json](results/access-check.json) (user terminal).",
         "- Full inventory: [inventory.json](results/inventory/inventory.json).",
         "- GPU/process inventory: [nvidia-smi.txt](results/inventory/nvidia-smi.txt), [gpu-processes.txt](results/inventory/gpu-processes.txt).",
         "- GPU 0 idle sample: [gpu0-idle-sample.json](results/inventory/gpu0-idle-sample.json).",
         "- CLI help, topology and redacted Ray Serve observations are in `results/inventory/`.",
         "- Model downloads and study servers: none started by this collector.",
         "- Live endpoint URL/model discovery remains subject to review of the read-only Serve observations.",
         "- Earlier 'too small' model remains unconfirmed; historical model IDs are recorded as clues only."]
(ROOT / "inventory.md").write_text("\n".join(lines) + "\n")
print("Inventory collected; review results/inventory/inventory.json", flush=True)
