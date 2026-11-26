"""Collect missing CLI/runtime evidence without altering packages or services."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "inventory"
BIN = Path(sys.executable).parent
environment = os.environ.copy()
prefix = "/packages/gcc/13.4/lib64:/packages/gcc/13.4/lib"
environment["LD_LIBRARY_PATH"] = prefix + ":" + environment.get("LD_LIBRARY_PATH", "")
report = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "runtime_library_prefix": prefix, "commands": {}}


def run(name, arguments, timeout=120):
    print("Collecting " + name, flush=True)
    try:
        result = subprocess.run(arguments, capture_output=True, text=True,
                                timeout=timeout, env=environment)
        text = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
        code = result.returncode
    except subprocess.TimeoutExpired:
        text, code = "Timed out; no partial output retained\n", None
    except OSError as error:
        text, code = str(error) + "\n", None
    text = re.sub(r"\bhf_[A-Za-z0-9]+", "[REDACTED]", text)
    path = OUT / (name + ".txt")
    path.write_text(text)
    report["commands"][name] = {"exit_code": code, "result": str(path.relative_to(ROOT))}
    return text, code


diagnostic = '''import json
import torch
import triton
import triton.backends
from triton._C import libtriton
from pathlib import Path
print(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                  "cuda_available": torch.cuda.is_available(),
                  "libstdcpp": sorted(set(line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libstdc++" in line))}, indent=2))
assert torch.cuda.is_available(), "CUDA unavailable in this execution context"
'''
run("runtime-import-check", [sys.executable, "-c", diagnostic])
run("vllm-version-runtime", [str(BIN / "vllm"), "--version"])
help_text, help_code = run("vllm-bench-serve-help-all-runtime",
                           [str(BIN / "vllm"), "bench", "serve", "--help=all"])
flags = ("--extra-body", "--percentile-metrics", "--metric-percentiles", "--save-detailed",
         "--ignore-eos", "--random-input-len", "--random-output-len", "--save-result",
         "--max-concurrency", "--num-prompts")
report["benchmark_required_flags"] = {flag: flag in help_text for flag in flags}
report["benchmark_flags_verified"] = help_code == 0 and all(report["benchmark_required_flags"].values())
run("listening-tcp-sockets", ["ss", "-ltnp"], timeout=15)

# Query only owners and executable names; do not save arbitrary command arguments.
process_text = (OUT / "gpu-processes.txt").read_text()
pids = re.findall(r",\s*(\d+),", process_text)
if pids:
    run("gpu-process-owners-full", ["ps", "-p", ",".join(pids),
                                    "-o", "pid=,ppid=,user:32=,comm="], timeout=15)
run("ray-process-names", ["ps", "-C", "raylet,gcs_server,serve", "-o",
                           "pid=,ppid=,user:32=,comm="], timeout=15)
report["status"] = ("COLLECTED_REVIEW_REQUIRED" if report["benchmark_flags_verified"]
                    and report["commands"]["runtime-import-check"]["exit_code"] == 0 else "BLOCKED_REVIEW_REQUIRED")
(OUT / "supplement.json").write_text(json.dumps(report, indent=2) + "\n")
print("Supplement finished: " + report["status"], flush=True)
