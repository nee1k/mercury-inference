"""Phase-2 pilot only. No full sweep, natural runner or frontier repeats."""
import copy
from pathlib import Path
import subprocess

import benchmark
import function_checks
import gate
import serve
import summarize
from common import CONFIG, MODEL, PYTHON, ROOT, gpu_snapshot, load, same_process, save, stamp, state_path


def main():
    directory = ROOT / "results" / CONFIG / ("dry-run-" + stamp())
    directory.mkdir(parents=True)
    result = {"phase": 2, "config": CONFIG, "status": "RUNNING", "directory": str(directory)}
    save(directory / "dry-run.json", result)
    marker = stamp()
    launched = False
    try:
        print("Testing evidence parser, stream parser and gate before GPU work", flush=True)
        with (directory / "tests.log").open("w") as log:
            subprocess.run([PYTHON, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_harness.py", "-v"],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Checking selected model's public API metadata; no weight download", flush=True)
        with (directory / "metadata.log").open("w") as log:
            subprocess.run([PYTHON, "-u", "scripts/inspect_candidates.py"], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if state_path().exists() and same_process(load(state_path())):
            raise ValueError("A study server is already running; refusing to adopt or stop it")
        port = serve.free_port()
        state = serve.start(CONFIG, "0", port)
        launched = True
        metrics = state["server_metrics"]
        if metrics["kv_cache_tokens"] is None or metrics["maximum_concurrency"] is None or metrics["concurrency_context_tokens"] != 32768:
            raise ValueError("Required startup KV/concurrency evidence missing")
        functions = function_checks.run(state["base_url"], directory / "functions")
        result["function_checks"] = str(functions)
        if load(functions)["status"] != "PASS":
            raise ValueError("Function checks failed; no benchmark load started")
        manifest = benchmark.run(state["base_url"], dry=True, kind="pilot")
        result["baseline_manifest"] = str(manifest)
        baseline = gate.read_chat(manifest)
        regression = copy.deepcopy(baseline)
        regression[4]["p95_tpot_ms"] *= 1.21
        assertions = {"self_comparison": gate.compare(baseline, baseline, True)["status"] == "PASS",
                      "regression_rejected": gate.compare(baseline, regression, True)["status"] == "FAIL",
                      "missing_cell_rejected": gate.compare(baseline, {1: baseline[1]}, True)["status"] == "FAIL",
                      "failed_function_rejected": gate.compare(baseline, baseline, False)["status"] == "FAIL"}
        save(directory / "saved-evidence-gate-tests.json", assertions)
        if not all(assertions.values()):
            raise ValueError("Gate failed tests against saved pilot results")
        summary = summarize.run()
        pilot_rows = [row for row in summary["rows"] if row["scope"] == "pilot" and row["config"] == CONFIG]
        if {row["concurrency"] for row in pilot_rows} != {1, 4}:
            raise ValueError("Summary did not recover both saved pilot cells")
        gate_result = gate.run(state["base_url"], MODEL, manifest)
        result["gate_result"] = str(gate_result)
        result["status"] = "PASS" if load(gate_result)["status"] == "PASS" else "GATE_FAILED_REVIEW"
    except Exception as error:
        result.update(status="FAIL", error=type(error).__name__ + ": " + str(error))
        print(result["error"], flush=True)
    finally:
        if state_path().exists():
            state = load(state_path())
            # Also clean up a failed startup created by THIS invocation only.
            ours = launched or state.get("launched_at", "") >= marker
            if ours and same_process(state):
                try:
                    serve.stop(CONFIG)
                    result["cleanup"] = "Study server stopped; GPU memory release verified"
                except Exception as error:
                    result.update(status="CLEANUP_FAILED_REVIEW", cleanup_error=str(error))
            elif ours:
                try:
                    sample = gpu_snapshot()
                    save(directory / "cleanup-gpu.json", sample)
                    if sample["pids"] or sample["used_mib"] >= 1024:
                        raise RuntimeError("Server already exited but GPU memory release is unconfirmed")
                    result["cleanup"] = "Server already exited; GPU memory release verified"
                except Exception as error:
                    result.update(status="CLEANUP_FAILED_REVIEW", cleanup_error=str(error))
        save(directory / "dry-run.json", result)
    print("Dry run " + result["status"] + "; evidence " + str(directory / "dry-run.json"), flush=True)
    print("PHASE 2 CHECKPOINT: wait for review and the user's go", flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
